import cv2
import pytesseract
import pandas as pd
import numpy as np
import re
import os
import platform
from jiwer import cer, wer
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ==========================================
# KONFIGURASI TESSERACT
# ==========================================
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def proses_dan_evaluasi(path_gambar, teks_kunci_jawaban):
    print(f"Memproses: {path_gambar}...")
    
    # Baca gambar
    img_asli = cv2.imread(path_gambar)
    if img_asli is None:
        print(f"[ERROR] Gambar '{path_gambar}' tidak ditemukan.")
        return None

    # ==========================================
    # PIPELINE ENHANCEMENT (Sama persis dengan app.py)
    # ==========================================
    # 1. Upscaling 
    img_zoomed = cv2.resize(img_asli, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_zoomed, cv2.COLOR_BGR2GRAY)
    
    # 2. CamScanner Illumination Normalization
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel_bg)
    background = cv2.GaussianBlur(background, (21, 21), 0)
    normalized = cv2.divide(gray, background, scale=255)
    
    # 3. Unsharp Masking 
    gaussian_blur_temp = cv2.GaussianBlur(normalized, (0, 0), 2.0)
    sharpened = cv2.addWeighted(normalized, 1.5, gaussian_blur_temp, -0.5, 0)
    
    # 4. Kontras & Brightness
    gray_bright = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=0)
    
    # 5. Bilateral Filter
    blur = cv2.bilateralFilter(gray_bright, 9, 75, 75)
    
    # 6. Kalkulasi Matriks Parameter (Otomatis)
    img_width = img_zoomed.shape[1]
    mean_brightness = np.mean(gray_bright)
    
    auto_block = int((img_width / 1000.0) * 25)
    if auto_block % 2 == 0: auto_block += 1
    auto_block = max(11, min(199, auto_block))
    
    auto_c = int((mean_brightness / 255.0) * 15)
    auto_c = max(3, min(25, auto_c))
    
    # 7. Adaptive Thresholding 
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, auto_block, auto_c)
    
    # Median blur dimatikan untuk evaluasi (aman untuk huruf tipis)
    thresh_clean = thresh

    # ==========================================
    # MENGHITUNG EVALUASI CITRA (PSNR, SSIM, MSE)
    # ==========================================
    nilai_mse = mse(gray, blur)
    nilai_psnr = psnr(gray, blur, data_range=255)
    nilai_ssim = ssim(gray, blur, data_range=255)

    # ==========================================
    # EKSTRAKSI & PEMBERSIHAN TEKS (Regex)
    # ==========================================
    hasil_teks_kotor = pytesseract.image_to_string(thresh_clean, config='--psm 4').strip()
    hasil_teks = re.sub(r'[^A-Za-z0-9 \n:,\.\-\/]', '', hasil_teks_kotor)
    hasil_teks = '\n'.join([s for s in hasil_teks.splitlines() if s.strip()])

    if len(hasil_teks) == 0:
        hasil_teks = "[GAGAL MEMBACA]"
        nilai_cer = 1.0 
        nilai_wer = 1.0 
    else:
        nilai_cer = cer(str(teks_kunci_jawaban).lower(), hasil_teks.lower())
        nilai_wer = wer(str(teks_kunci_jawaban).lower(), hasil_teks.lower())

    return {
        "Nama File": path_gambar,
        "Kunci Jawaban (Asli)": teks_kunci_jawaban,
        "Hasil OCR Sistem": hasil_teks,
        "CER (Error Karakter)": round(nilai_cer, 3),
        "WER (Error Kata)": round(nilai_wer, 3),
        "MSE": round(nilai_mse, 3),
        "PSNR": round(nilai_psnr, 3),
        "SSIM": round(nilai_ssim, 3)
    }

# ==========================================
# PROGRAM UTAMA: MEMBACA SOAL DARI EXCEL
# ==========================================
if __name__ == '__main__':
    file_dataset = "data_uji.xlsx"
    
    print("="*60)
    print("[SISTEM EVALUASI OCR AKTIF]")
    print("="*60)
    
    # Mengecek apakah file data_uji.xlsx sudah dibuat oleh user
    if not os.path.exists(file_dataset):
        print(f"\n[INFO] File '{file_dataset}' tidak ditemukan.")
        print("[SISTEM] Membuat template dataset Excel baru...")
        
        # Membuat template excel dengan OTOMATIS membaca semua gambar di folder 'file'
        folder_dataset = "file"
        daftar_file = []
        if os.path.exists(folder_dataset):
            # Ambil semua nama file berakhiran jpg/jpeg/png/webp
            daftar_file = [f for f in os.listdir(folder_dataset) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
        if len(daftar_file) == 0:
            daftar_file = ["contoh.jpg"] # Jaga-jaga jika folder kosong
            
        df_template = pd.DataFrame({
            "Nama File": daftar_file,
            "Kunci Jawaban": ["[Ketik teks asli di sini]"] * len(daftar_file)
        })
        df_template.to_excel(file_dataset, index=False)
        print(f"[INFO] Template '{file_dataset}' berhasil dibuat.")
        print("[INFO] Harap isi dataset pada file tersebut sebelum melanjutkan eksekusi program.")
        
    else:
        print(f"\n[INFO] Membaca data dari '{file_dataset}'...")
        df_soal = pd.read_excel(file_dataset)
        
        semua_hasil = []
        for index, baris in df_soal.iterrows():
            nama_file = str(baris['Nama File'])
            kunci = str(baris['Kunci Jawaban'])
            
            # Menyambungkan otomatis ke folder 'file'
            path_lengkap = os.path.join("file", nama_file)
            
            hasil = proses_dan_evaluasi(path_lengkap, kunci)
            if hasil is not None:
                semua_hasil.append(hasil)
                
        if len(semua_hasil) > 0:
            df_hasil = pd.DataFrame(semua_hasil)
            df_hasil.to_excel("hasil_evaluasi_lengkap.xlsx", index=False)
            
            print("\n" + "="*60)
            print(f"[INFO] Proses evaluasi selesai untuk {len(semua_hasil)} data.")
            print("[INFO] Hasil ekspor: 'hasil_evaluasi_lengkap.xlsx'")
            print("="*60)