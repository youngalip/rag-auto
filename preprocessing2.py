import json
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

# =====================================================
# 1. CLEANING
# =====================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)          # hapus karakter aneh
    text = re.sub(r'[^a-z0-9\s.,?!-]', ' ', text)      # sisakan simbol dasar
    text = re.sub(r'\s+', ' ', text)                   # rapikan spasi
    return text.strip()

# =====================================================
# 2. CHUNKING
# =====================================================

def chunk_text(text, chunk_size=200):
    """
    Memecah teks menjadi beberapa chunk (berdasarkan jumlah kata)
    """
    words = word_tokenize(text)
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)

    return chunks

# =====================================================
# 3. PREPROCESS PIPELINE
# =====================================================

def preprocess(
    input_file="labeled.json",
    output_file="preprocessed.json",
    chunk_size=200
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []

    for item in data:
        raw_text = item["text"]
        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned, chunk_size)

        for i, ch in enumerate(chunks):
            processed_data.append({
                "filename": item["filename"],
                "chunk_id": i,
                "cleaned_text": ch,
                "label": item["label"]   # LABEL TETAP
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"Preprocessing + chunking selesai")
    print(f"Total chunk: {len(processed_data)}")
    print(f"Hasil disimpan di: {output_file}")

# =====================================================
# 4. RUN
# =====================================================

if __name__ == "__main__":
    preprocess(
        input_file="labeled.json",
        output_file="preprocessed.json",
        chunk_size=200
    )
