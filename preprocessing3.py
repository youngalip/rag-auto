# =====================================================
# preprocess_improved.py
# Improved Preprocessing for RAG System (Indonesian)
# =====================================================

import json
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =====================================================
# NLTK SETUP
# =====================================================
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# =====================================================
# SASTRAWI INITIALIZATION
# =====================================================
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# =====================================================
# 1. TEXT CLEANING
# =====================================================
def clean_text(text, keep_stopwords=True):
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep important punctuation for sentence structure
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Optional: remove stopwords (keep for context understanding)
    if not keep_stopwords:
        text = stopword_remover.remove(text)
    
    return text.strip()

# =====================================================
# 2. SMART CHUNKING (Sentence-aware)
# =====================================================
def smart_chunk(text, max_chunk_size=150):
    """
    Chunk text based on sentence boundaries
    to preserve semantic meaning
    """
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for sent in sentences:
        sent_len = len(word_tokenize(sent))

        if current_length + sent_len <= max_chunk_size:
            current_chunk.append(sent)
            current_length += sent_len
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_length = sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]

# =====================================================
# 3. PREPROCESSING PIPELINE
# =====================================================
def preprocess_improved(
    input_file="labeled.json",
    output_file="preprocessed.json",
    max_chunk_size=150
):
    """
    Full preprocessing pipeline for RAG
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    idx = 0

    stats = {
        "total_items": len(data),
        "total_chunks": 0,
        "multi_chunk_items": 0
    }

    for item in data:
        filename = clean_text(item.get("filename", "unknown"), keep_stopwords=True)
        text = clean_text(item["text"], keep_stopwords=True)

        chunks = smart_chunk(text, max_chunk_size=max_chunk_size)

        if len(chunks) > 1:
            stats["multi_chunk_items"] += 1

        stats["total_chunks"] += len(chunks)

        for i, chunk in enumerate(chunks):
            processed_data.append({
                "id": idx,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "filename": filename,
                "text": chunk,
                "label": item.get("label", "lainnya")
            })
            idx += 1

    # Save output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    # Stats
    print("=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)
    print(f"Total original items     : {stats['total_items']}")
    print(f"Items with multi chunks  : {stats['multi_chunk_items']} "
          f"({stats['multi_chunk_items']/stats['total_items']*100:.1f}%)")
    print(f"Total chunks generated   : {stats['total_chunks']}")
    print(f"Average chunks per item  : {stats['total_chunks']/stats['total_items']:.2f}")
    print(f"Output saved to          : {output_file}")
    print("=" * 60)

# =====================================================
# 4. MAIN
# =====================================================
if __name__ == "__main__":
    preprocess_improved(
        input_file="labeled.json",
        output_file="preprocessed.json",
        max_chunk_size=150
    )
