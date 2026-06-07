import streamlit as st
import cv2
import numpy as np
import pytesseract

# 1. Konfigurasi Lokasi Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 2. Pengaturan Halaman Web
st.set_page_config(
    page_title="OCR & Image Enhancement", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Custom CSS untuk UI Premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 30px;
    }
    
    .block-container {
        padding-top: 2rem !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HALAMAN UTAMA
# ==========================================
st.markdown('<p class="main-title">Digital Image OCR System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Restorasi & Ekstraksi Teks Dokumen Buram dengan Algoritma Adaptive</p>', unsafe_allow_html=True)

# Area Input Gambar
st.markdown("### 📥 Input Dokumen")
tab1, tab2 = st.tabs(["📂 Upload dari Galeri", "📸 Ambil dari Kamera"])

with tab1:
    uploaded_file = st.file_uploader("Pilih foto ijazah, KTP, atau arsip lama", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

with tab2:
    camera_file = st.camera_input("Jepret Langsung")

# Penentu file mana yang diproses (Kamera diprioritaskan jika dua-duanya diisi)
file_yang_diproses = camera_file if camera_file is not None else uploaded_file

if file_yang_diproses is None:
    # Sidebar kosong saat belum ada foto
    with st.sidebar:
        st.markdown("## 🎛️ Panel Kendali")
        st.caption("Sistem Peningkatan Kualitas Citra")
        st.divider()
        st.info("👈 Silakan upload atau jepret dokumen di halaman utama terlebih dahulu. Sistem akan mengkalkulasi resolusi foto dan mengatur parameter secara otomatis.")
        st.markdown("<br><br><br><br><br><p style='text-align: center; color: #666; font-size: 0.8rem;'>Skripsi OCR v2.0</p>", unsafe_allow_html=True)
else:
    # Membaca file yang diupload atau dijepret
    file_bytes = np.asarray(bytearray(file_yang_diproses.read()), dtype=np.uint8)
    img_asli = cv2.imdecode(file_bytes, 1)

    if img_asli is None:
        st.error("🚨 Gagal membaca gambar! Pastikan file yang diupload tidak rusak.")
        st.stop()

    with st.spinner('Sistem sedang mengkalkulasi dokumen dan menyesuaikan parameter... ⏳'):
        
        # 1. Upscaling 
        img_zoomed = cv2.resize(img_asli, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        # 2. Grayscale
        gray = cv2.cvtColor(img_zoomed, cv2.COLOR_BGR2GRAY)
        
        # ==========================================
        # RAHASIA CAMSCANNER: ILLUMINATION NORMALIZATION
        # Menghapus bayangan (shadow) gelap di sisi kiri kertas
        # ==========================================
        # Cari background dengan menghapus tulisan hitam menggunakan Dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        background = cv2.GaussianBlur(background, (21, 21), 0)
        
        # Bagi gambar asli dengan background. Bayangan gelap akan otomatis jadi putih!
        normalized = cv2.divide(gray, background, scale=255)
        
        # 3. Kontras & Brightness
        gray_bright = cv2.convertScaleAbs(normalized, alpha=1.1, beta=0)
        
        # 4. Bilateral Filter
        blur = cv2.bilateralFilter(gray_bright, 9, 75, 75)
        
        # ==========================================
        # ALGORITMA ADAPTIVE: AUTO-DETECT PARAMETER
        # ==========================================
        # Sistem menghitung karakteristik foto (resolusi & pencahayaan)
        img_width = img_zoomed.shape[1]
        mean_brightness = np.mean(gray_bright)
        
        # Rumus Pintar Block Size (Berdasarkan lebar gambar)
        auto_block = int((img_width / 1000.0) * 25)
        if auto_block % 2 == 0: auto_block += 1
        auto_block = max(11, min(199, auto_block)) # Batasi minimal 11 maksimal 199
        
        # Rumus Pintar Kepekaan Tinta C (Berdasarkan seberapa terang gambar)
        # Gambar gelap butuh C kecil, gambar terang butuh C besar
        auto_c = int((mean_brightness / 255.0) * 15)
        auto_c = max(3, min(25, auto_c))
        
        # Median Blur dimatikan secara default untuk menyelamatkan huruf tipis Ijazah
        auto_median = 1
        
        # ==========================================
        # PANEL KENDALI (DIPINDAH KE HALAMAN UTAMA AGAR TERLIHAT JELAS)
        # ==========================================
        st.divider()
        st.markdown("### 🎛️ Panel Kendali Parameter")
        st.success("⚡ **Algoritma Adaptive Aktif!** Sistem telah mengkalkulasi matriks dokumenmu dan memutar tuas parameter ke titik paling optimal secara otomatis.")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            block_size = st.slider("🔍 Area Deteksi (Block Size)", min_value=3, max_value=199, value=auto_block, step=2, help="Semakin besar, huruf yang bolong akan semakin padat.")
            c_value = st.slider("🧽 Kepekaan Tinta (C)", min_value=0, max_value=50, value=auto_c, step=1, help="Naikkan nilai ini jika background masih kotor (banyak bintik).")
        with col_s2:
            ketebalan = st.slider("✒️ Penebal Huruf (Morfologi)", min_value=0, max_value=3, value=0, step=1, help="Naikkan ke 1 atau 2 untuk menebalkan tulisan Ijazah yang sangat tipis/putus-putus.")
            median_filter = st.slider("🧹 Pembersih Bintik (Median)", min_value=1, max_value=7, value=auto_median, step=2, help="Angka 1 = Mati. Gunakan 3 atau 5 jika banyak bintik. Awas: Angka besar bisa membuat tulisan tipis putus.")
            
        with st.expander("💡 Panduan Penggunaan", expanded=False):
            st.markdown("""
            **1.** Jika huruf putus-putus, **Nyalakan Penebal Huruf** (Geser ke angka 1/2).\n
            **2.** Jika gambar penuh bercak/kotor, **NAIKKAN angka C**.\n
            **3.** Jika tengah huruf bolong besar, **NAIKKAN Area Deteksi**.
            """)

        # ==========================================
        # 3. JURUS 1: Unsharp Masking (Penajam Tepi Huruf Ekstrem)
        # Ini berfungsi menebalkan batas huruf sebelum dipotong menjadi hitam putih
        gaussian_blur_temp = cv2.GaussianBlur(normalized, (0, 0), 2.0)
        sharpened = cv2.addWeighted(normalized, 1.5, gaussian_blur_temp, -0.5, 0)
        
        # 4. Kontras & Brightness
        gray_bright = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=0)
        
        # 5. Bilateral Filter
        blur = cv2.bilateralFilter(gray_bright, 9, 75, 75)
        
        # 6. Adaptive Thresholding 
        thresh = cv2.adaptiveThreshold(
            blur, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            block_size, 
            c_value     
        )
        
        # 7. JURUS 2: Penebal Tinta (Morphological Erosion)
        # Mengikis warna putih di sekitar tulisan agar tinta hitam melebar/menebal
        if ketebalan > 0:
            kernel_tebal = np.ones((ketebalan + 1, ketebalan + 1), np.uint8)
            thresh = cv2.erode(thresh, kernel_tebal, iterations=1)
        
        # 8. Pembersih Bintik (Hanya aktif jika slider > 1)
        if median_filter > 1:
            thresh_clean = cv2.medianBlur(thresh, median_filter)
        else:
            thresh_clean = thresh
        
        import re
        
        # 7. Ekstraksi Teks (OCR)
        hasil_teks_kotor = pytesseract.image_to_string(thresh_clean, config='--psm 4').strip()
        
        # 8. JURUS 3: POST-PROCESSING (Pembersih Huruf Alien)
        # Menggunakan Regex untuk hanya mengizinkan Huruf, Angka, Spasi, Enter, dan tanda baca dasar
        # Karakter aneh seperti ~, ™, =, |, _, akan otomatis dibuang!
        hasil_teks = re.sub(r'[^A-Za-z0-9 \n:,\.\-\/]', '', hasil_teks_kotor)
        
        # Hapus baris kosong yang berlebihan akibat pembersihan
        hasil_teks = '\n'.join([s for s in hasil_teks.splitlines() if s.strip()])
        
        if len(hasil_teks) == 0:
            hasil_teks = "[Belum ada teks terdeteksi. Silakan atur ulang slider di panel kendali.]"

        st.divider()
        
        # LAYOUT KOLOM BERSEBELAHAN
        st.markdown("### 🖼️ Analisis Citra Visual")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**📸 Citra Asli (Original)**")
            img_rgb = cv2.cvtColor(img_asli, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            
        with col2:
            st.success(f"**✨ Citra Hasil Enhancement** (Block={block_size}, C={c_value})")
            st.image(thresh_clean, channels="GRAY", use_container_width=True)
            
        st.divider()
        
        # Menampilkan hasil teks OCR
        st.markdown("### 📝 Hasil Ekstraksi Teks (OCR)")
        st.caption("Teks di bawah ini dihasilkan otomatis oleh mesin Tesseract OCR berdasarkan citra hasil enhancement.")
        st.text_area("Hasil OCR", value=hasil_teks, height=250, label_visibility="collapsed")