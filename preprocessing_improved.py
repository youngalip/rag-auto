import json
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# NLTK SETUP
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Sastrawi
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# ===== TEXT CLEANING (LESS AGGRESSIVE) =====
def clean_text(text, keep_stopwords=True):
    """
    FIXED: Less aggressive cleaning to preserve information
    """
    text = text.lower()
    
    # Remove URLs and emails only
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep most punctuation for sentence structure
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# ===== SMART CHUNKING =====
def smart_chunk(text, max_chunk_size=200):  # Increased from 150
    """
    FIXED: Larger chunks for better context
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

# ===== MAIN PREPROCESSING =====
def preprocess_improved(
    input_file="extracted.json",
    output_file="preprocessed.json",
    max_chunk_size=200
):
    """
    FIXED: Improved preprocessing with better labeling
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    idx = 0

    stats = {
        "total_items": len(data),
        "total_chunks": 0,
        "multi_chunk_items": 0,
        "label_distribution": {}
    }

    for item in data:
        filename = clean_text(item.get("filename", "unknown"), keep_stopwords=True)
        text = clean_text(item["text"], keep_stopwords=True)
        
        # FIXED: Improved labeling
        label = item.get("label", "lainnya")

        chunks = smart_chunk(text, max_chunk_size=max_chunk_size)

        if len(chunks) > 1:
            stats["multi_chunk_items"] += 1

        stats["total_chunks"] += len(chunks)
        stats["label_distribution"][label] = stats["label_distribution"].get(label, 0) + 1

        for i, chunk in enumerate(chunks):
            processed_data.append({
                "id": idx,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "filename": filename,
                "text": chunk,
                "label": label
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
    print(f"\nLabel Distribution:")
    for label, count in sorted(stats["label_distribution"].items()):
        print(f"  {label:30s}: {count:4d} ({count/stats['total_chunks']*100:.1f}%)")
    print(f"Output saved to          : {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    preprocess_improved(
        input_file="labeled.json",  # FIXED: Input from labeling.py
        output_file="preprocessed.json",
        max_chunk_size=200
    )