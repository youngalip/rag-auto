import os
import re
import json
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image
import pytesseract
import PyPDF2
from pdf2image import convert_from_path

# Pastikan tokenizer tersedia
nltk.download('punkt')

# =====================================================================
# 1. LEXICON (KATA KUNCI UNTUK LABEL OTOMATIS)
# =====================================================================

lexicon = {
    "administrasi_lab": [
        "bebas lab", "laboratorium", "praktikum", "kalab", "kasublab"
    ],
    "nilai": [
        "khs", "transkrip", "ipk", "nilai", "ujian", "remedial"
    ],
    "administrasi_akademik": [
        "cuti kuliah", "krs", "registrasi", "pembatalan", "jadwal kuliah",
        "siakadu", "siani", "profil program studi", "melisa", "simlppm"
    ],
    "keuangan": [
        "ukt", "beasiswa", "pembayaran", "biaya", "angsuran", "pembiayaan"
    ],
    "perpustakaan": [
        "perpustakaan", "kartu perpustakaan", "opac", "e-library", "jurnal", "open journal"
    ],
    "sertifikasi": [
        "sertifikat", "sertifikasi", "lsp", "tep", "tes", "ujian bahasa",
        "sertifikat profesi", "sertifikat kompetensi"
    ],
    "sidang_yudisium": [
        "yudisium", "syarat yudisium", "wisuda", "skp", "skripsi", "tugas akhir"
    ],
    "informasi_umum": [
        "kalender akademik", "panduan", "tutorial", "manual pengguna", "user manual"
    ]
}

def assign_label(text):
    """Memberikan label berdasarkan kemunculan kata kunci dalam lexicon."""
    text_low = text.lower()
    for label, keywords in lexicon.items():
        for kw in keywords:
            if kw in text_low:
                return label
    return "lainnya"


# =====================================================================
# 2. EKSTRAKSI TEKS DARI DOKUMEN
# =====================================================================

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image, lang='ind')

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""

    except Exception:
        pass

    # Jika PDF hasil scan → OCR
    if not text.strip():
        print(f"[SCAN PDF] OCR: {os.path.basename(pdf_path)}")
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img, lang="ind")

    return text


# =====================================================================
# 3. PREPROCESSING (Cleaning + Tokenizing + Chunking)
# =====================================================================

def clean_text(text):
    """Membersihkan karakter tidak penting."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^\w\s.,?!-]', '', text)
    return text.strip()

def chunk_text(text, chunk_size=200):
    """Memecah teks menjadi chunk ukuran 200 kata."""
    words = word_tokenize(text)
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks


# =====================================================================
# 4. PIPELINE: EKSTRAKSI → LABEL → PREPROCESSING
# =====================================================================

def process_file(filepath):
    """Proses lengkap satu file: ekstraksi → cleaning → chunk → label"""
    ext = os.path.splitext(filepath)[1].lower()

    # --- EKSTRAKSI TEKS DULU ---
    if ext == ".txt":
        raw = extract_text_from_txt(filepath)
    elif ext in [".png", ".jpg", ".jpeg"]:
        raw = extract_text_from_image(filepath)
    elif ext == ".pdf":
        raw = extract_text_from_pdf(filepath)
    else:
        raise ValueError(f"Format tidak didukung: {filepath}")

    # --- CLEANING & CHUNKING ---
    cleaned = clean_text(raw)
    chunks = chunk_text(cleaned, chunk_size=200)

    # --- LABEL OTOMATIS (SETELAH DAPAT TEKS) ---
    output = []
    for i, ch in enumerate(chunks):
        output.append({
            "source": os.path.basename(filepath),
            "chunk_id": i,
            "content": ch,
            "label": assign_label(ch)  # label otomatis diberikan di sini
        })

    return output


# =====================================================================
# 5. PROSES SATU FOLDER
# =====================================================================

def process_folder(folder_path, output_json="hasil_preprocessing.json"):
    all_data = []
    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        try:
            chunks = process_file(path)
            all_data.extend(chunks)
            print(f"Diproses: {fname}")
        except Exception as e:
            print(f"Lewati {fname}: {e}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\nTotal chunk: {len(all_data)} → Disimpan ke '{output_json}'")


# =====================================================================
# 6. JALANKAN
# =====================================================================

if __name__ == "__main__":
    folder = "dataset layanan akademik"
    process_folder(folder)
