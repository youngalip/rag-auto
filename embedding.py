import json
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import faiss
from collections import Counter

# =====================================================
# IMPROVED WORD2VEC + FAISS INDEXING
# =====================================================

def train_word2vec(preprocessed_file="preprocessed.json"):
    """
    Train Word2Vec model with better parameters
    """
    print("="*60)
    print("TRAINING WORD2VEC MODEL")
    print("="*60)
    
    print("\nLoading preprocessed data...")
    with open(preprocessed_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Collect all text
    all_texts = [item["text"] for item in data]
    
    print(f"Tokenizing {len(all_texts)} documents...")
    tokenized_texts = [word_tokenize(text.lower()) for text in all_texts]
    
    # Vocabulary statistics
    all_words = [w for doc in tokenized_texts for w in doc]
    vocab_counter = Counter(all_words)
    
    print(f"\nVocabulary Statistics:")
    print(f"  Total tokens: {len(all_words)}")
    print(f"  Unique tokens: {len(vocab_counter)}")
    print(f"  Avg tokens per document: {len(all_words)/len(all_texts):.1f}")
    
    # Train Word2Vec
    print("\nTraining Word2Vec model...")
    w2v = Word2Vec(
        sentences=tokenized_texts,
        vector_size=200,           # Increased dimension
        window=7,                  # Larger context window
        min_count=2,               # Lower threshold
        workers=4,
        sg=1,                      # Skip-gram
        epochs=20,                 # More epochs
        negative=10,
        alpha=0.025,
        min_alpha=0.0001
    )
    
    w2v.save("word2vec.model")
    print(f"\n✓ Word2Vec model saved: word2vec.model")
    print(f"✓ Vocabulary size: {len(w2v.wv)}")
    
    return w2v, data

def build_faiss_index(w2v, data, index_file="faiss_index.idx"):
    """
    Build FAISS index with TF-IDF weighted vectors
    """
    print("\n" + "="*60)
    print("BUILDING FAISS INDEX")
    print("="*60)
    
    def doc_vector(text):
        """Create document vector with TF-IDF weighting"""
        tokens = word_tokenize(text.lower())
        vectors = []
        
        for token in tokens:
            if token in w2v.wv:
                # TF weighting
                weight = tokens.count(token) / len(tokens)
                vectors.append(w2v.wv[token] * weight)
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(w2v.vector_size)
    
    # Create embeddings
    print("\nCreating document embeddings...")
    doc_vectors = []
    
    for i, item in enumerate(data):
        vec = doc_vector(item["text"])
        doc_vectors.append(vec)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(data)} documents")
    
    doc_vectors = np.array(doc_vectors).astype("float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(doc_vectors)
    
    # Create FAISS index
    dimension = doc_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
    index.add(doc_vectors)
    
    # Save index
    faiss.write_index(index, index_file)
    
    print(f"\n✓ FAISS index saved: {index_file}")
    print(f"✓ Total vectors: {index.ntotal}")
    print(f"✓ Dimension: {dimension}")
    
    return index

def save_mapping(data, output_file="faiss_mapping.json"):
    """Save document mapping for retrieval"""
    print(f"\nSaving document mapping to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Mapping saved")

def verify_embeddings(w2v, sample_queries=None):
    """Verify embedding quality"""
    if sample_queries is None:
        sample_queries = [
            "cara mengajukan bebas lab",
            "prosedur cuti kuliah",
            "pembayaran ukt"
        ]
    
    print("\n" + "="*60)
    print("EMBEDDING QUALITY CHECK")
    print("="*60)
    
    for query in sample_queries:
        print(f"\nQuery: {query}")
        tokens = word_tokenize(query.lower())
        
        # Token coverage
        in_vocab = [t for t in tokens if t in w2v.wv]
        oov = [t for t in tokens if t not in w2v.wv]
        
        print(f"  In vocabulary: {len(in_vocab)}/{len(tokens)} tokens")
        if oov:
            print(f"  OOV tokens: {oov}")
        
        # Similar words
        if in_vocab:
            key_word = in_vocab[0]
            try:
                similar = w2v.wv.most_similar(key_word, topn=3)
                print(f"  Similar to '{key_word}':")
                for word, sim in similar:
                    print(f"    - {word} ({sim:.3f})")
            except:
                pass
    
    print("="*60)

def main():
    """Main pipeline"""
    print("="*60)
    print("IMPROVED EMBEDDING PIPELINE")
    print("="*60)
    
    # Step 1: Train Word2Vec
    w2v, data = train_word2vec("preprocessed.json")
    
    # Step 2: Build FAISS index
    index = build_faiss_index(w2v, data, "faiss_index.idx")
    
    # Step 3: Save mapping
    save_mapping(data, "faiss_mapping.json")
    
    # Step 4: Verify
    verify_embeddings(w2v)
    
    print("\n" + "="*60)
    print("EMBEDDING PIPELINE COMPLETED")
    print("="*60)
    print("\nGenerated files:")
    print("  - word2vec.model")
    print("  - faiss_index.idx")
    print("  - faiss_mapping.json")
    print("="*60)

if __name__ == "__main__":
    main()