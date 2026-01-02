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

# ===== TEXT CLEANING (PRESERVE INFORMATION) =====
def clean_text(text, keep_stopwords=True):
    """
    Clean text while preserving important information
    """
    text = text.lower()
    
    # Remove URLs and emails
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep punctuation for sentence structure
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Keep stopwords for natural language
    # (Don't remove - they're important for context!)
    
    return text.strip()

# ===== SMART CHUNKING WITH OVERLAP =====
def smart_chunk(text, max_chunk_size=150, overlap=30):
    """
    Smart sentence-aware chunking with overlap
    
    Args:
        text: Text to chunk
        max_chunk_size: Maximum words per chunk
        overlap: Overlapping words between chunks
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return [text] if text.strip() else []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sent in sentences:
        sent_words = len(word_tokenize(sent))
        
        if current_length + sent_words <= max_chunk_size:
            current_chunk.append(sent)
            current_length += sent_words
        else:
            if current_chunk:
                # Save current chunk
                chunk_text = " . ".join(current_chunk) + " ."
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > 1:
                    # Keep last sentence for overlap
                    current_chunk = [current_chunk[-1], sent]
                    current_length = len(word_tokenize(current_chunk[-1])) + sent_words
                else:
                    current_chunk = [sent]
                    current_length = sent_words
            else:
                current_chunk = [sent]
                current_length = sent_words
    
    # Add remaining
    if current_chunk:
        chunk_text = " . ".join(current_chunk) + " ."
        chunks.append(chunk_text)
    
    return chunks if chunks else [text]

# ===== MAIN PREPROCESSING =====
def preprocess_improved(
    input_file="labeled.json",
    output_file="preprocessed.json",
    max_chunk_size=150,
    overlap=30
):
    """
    Improved preprocessing with smart chunking
    """
    print("="*60)
    print("IMPROVED PREPROCESSING")
    print("="*60)
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} documents...")
    
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
        label = item.get("label", "lainnya")
        
        # Skip if too short
        if len(text) < 20:
            continue
        
        # Chunk text
        chunks = smart_chunk(text, max_chunk_size=max_chunk_size, overlap=overlap)
        
        if len(chunks) > 1:
            stats["multi_chunk_items"] += 1
        
        stats["total_chunks"] += len(chunks)
        stats["label_distribution"][label] = stats["label_distribution"].get(label, 0) + len(chunks)
        
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
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED")
    print("="*60)
    print(f"Total original items     : {stats['total_items']}")
    print(f"Items with multi chunks  : {stats['multi_chunk_items']} "
          f"({stats['multi_chunk_items']/stats['total_items']*100:.1f}%)")
    print(f"Total chunks generated   : {stats['total_chunks']}")
    print(f"Average chunks per item  : {stats['total_chunks']/stats['total_items']:.2f}")
    
    print(f"\nLabel Distribution:")
    for label, count in sorted(stats["label_distribution"].items()):
        print(f"  {label:30s}: {count:4d} ({count/stats['total_chunks']*100:.1f}%)")
    
    print(f"\nOutput saved to: {output_file}")
    print("="*60)
    
    return processed_data

if __name__ == "__main__":
    preprocess_improved(
        input_file="labeled.json",
        output_file="preprocessed.json",
        max_chunk_size=150,
        overlap=30
    )