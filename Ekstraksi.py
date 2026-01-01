import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import os
import json

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image, lang='ind')

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text += extracted
    except Exception as e:
        print(f"Gagal ekstrak PDF ({pdf_path}): {e}")

    # Jika PDF adalah hasil scan → OCR
    if not text.strip():
        print(f"PDF {os.path.basename(pdf_path)} hasil scan → OCR...")
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img, lang="ind")

    return text

folder = "dataset layanan akademik"
results = []  

def process_file(filepath):
    print(f"MEMPROSES: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext in [".jpg", ".jpeg", ".png"]:
            raw_text = extract_text_from_image(filepath)

        elif ext == ".txt":
            raw_text = extract_text_from_txt(filepath)

        elif ext == ".pdf":
            raw_text = extract_text_from_pdf(filepath)

        else:
            print(f" FORMAT TIDAK DIDUKUNG: {filepath}")
            return
    except Exception as e:
        print(f" ERROR MEMPROSES {filepath}: {e}")
        return

    print(f" SUKSES: {filepath}")

    results.append({
        "filename": os.path.basename(filepath),
        "text": raw_text
    })

for root, dirs, files in os.walk(folder):
    for file in files:
        process_file(os.path.join(root, file))

with open("extracted.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Selesai! Hasil tersimpan di extracted.json")
