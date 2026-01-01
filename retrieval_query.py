import faiss
import json
import os
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from collections import defaultdict
from difflib import SequenceMatcher

# =====================================================
# IMPROVED RETRIEVAL WITH RE-RANKING V3
# FIX #1: Deduplication
# FIX #2: Confidence Threshold
# FIX #3: Better top-k (5 instead of 3)
# =====================================================

class ImprovedRetriever:
    def __init__(self, 
                 index_path="faiss_index.idx",
                 mapping_path="faiss_mapping.json",
                 w2v_path="word2vec.model",
                 confidence_threshold=0.2):  # 🔧 Tuned: 0.3 → 0.2 (less aggressive)
        """
        Initialize improved retriever
        
        Args:
            confidence_threshold: Minimum score to consider results relevant (default: 0.3)
        """
        print("Loading retrieval components...")
        
        # Load FAISS index
        base_dir = os.path.dirname(__file__) if __file__ else "."
        self.index = faiss.read_index(os.path.join(base_dir, index_path))
        
        # Load document mapping
        with open(os.path.join(base_dir, mapping_path), "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        
        # Load Word2Vec
        self.w2v = Word2Vec.load(os.path.join(base_dir, w2v_path))
        
        # 🆕 Set confidence threshold
        self.confidence_threshold = confidence_threshold
        
        print(f"Loaded {len(self.documents)} documents")
        print(f"FAISS index size: {self.index.ntotal}")
        print(f"Word2Vec vocab size: {len(self.w2v.wv)}")
        print(f"⚠️ Confidence threshold: {self.confidence_threshold}")
    
    def encode_query(self, query):
        """
        Encode query with TF-IDF weighting (same as document encoding)
        """
        tokens = word_tokenize(query.lower())
        vectors = []
        
        for token in tokens:
            if token in self.w2v.wv:
                weight = tokens.count(token) / len(tokens)
                vectors.append(self.w2v.wv[token] * weight)
        
        if vectors:
            vec = np.mean(vectors, axis=0).reshape(1, -1).astype("float32")
            faiss.normalize_L2(vec)  # Normalize for cosine similarity
            return vec
        else:
            return np.zeros((1, self.w2v.vector_size), dtype="float32")
    
    def retrieve_basic(self, query, top_k=5):
        """
        Basic FAISS retrieval
        """
        q_vec = self.encode_query(query)
        distances, indices = self.index.search(q_vec, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'doc': doc,
                    'score': float(distances[0][i]),
                    'idx': int(idx)
                })
        
        return results
    
    def merge_chunks(self, results):
        groups = defaultdict(list)

        for r in results:
            key = r['doc'].get('filename', 'unknown')
            groups[key].append(r)

        merged = []
        for key, chunks in groups.items():
            chunks.sort(key=lambda x: x['doc'].get('chunk_id', 0))

            combined_text = " ".join(c['doc']['text'] for c in chunks)
            avg_score = np.mean([c['score'] for c in chunks])

            merged.append({
                'source': key,
                'answer': combined_text,
                'score': avg_score,
                'num_chunks': len(chunks),
                'chunk_ids': [c['doc'].get('chunk_id', 0) for c in chunks]
            })

        merged.sort(key=lambda x: x['score'], reverse=True)
        return merged

    
    # 🆕 FIX #1: DEDUPLICATION
    def calculate_similarity(self, text1, text2):
        """
        Calculate similarity between two texts using SequenceMatcher
        """
        return SequenceMatcher(None, text1[:200].lower(), text2[:200].lower()).ratio()
    
    def deduplicate_results(self, results, similarity_threshold=0.90):  # 🔧 Tuned: 0.85 → 0.90 (less aggressive)
        """
        🆕 FIX #1: Remove duplicate answers based on content similarity
        
        Args:
            results: List of result dictionaries
            similarity_threshold: Similarity above this is considered duplicate (default: 0.90)
        """
        if not results:
            return results
        
        unique_results = []
        seen_answers = []
        
        for result in results:
            answer = result['answer']
            
            # Check if answer is similar to any seen answer
            is_duplicate = False
            for seen in seen_answers:
                similarity = self.calculate_similarity(answer, seen)
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_answers.append(answer)
        
        return unique_results
    
    def rerank_bm25(self, query, candidates):
        """
        Simple BM25-like re-ranking based on term frequency
        """
        query_tokens = set(word_tokenize(query.lower()))
        
        reranked = []
        for candidate in candidates:
            # Calculate overlap score
            answer_tokens = set(word_tokenize(candidate['answer'].lower()))
            overlap = len(query_tokens & answer_tokens)
            coverage = overlap / len(query_tokens) if query_tokens else 0
            
            # Combine with original score
            combined_score = candidate['score'] * 0.7 + coverage * 0.3
            
            candidate['rerank_score'] = combined_score
            candidate['overlap'] = overlap
            candidate['coverage'] = coverage
            reranked.append(candidate)
        
        # Sort by re-ranked score
        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked
    
    # 🆕 FIX #2: CONFIDENCE THRESHOLD
    def filter_by_confidence(self, results):
        """
        🆕 FIX #2: Filter out low-confidence results
        
        Returns:
            tuple: (filtered_results, low_confidence_flag)
        """
        if not results:
            return [], True
        
        # Check best score
        best_score = results[0]['rerank_score']
        
        if best_score < self.confidence_threshold:
            return [], True  # All results below threshold
        
        # Keep results above threshold
        filtered = [r for r in results if r['rerank_score'] >= self.confidence_threshold]
        
        return filtered, False
    
    # 🆕 FIX #3: Increased top_k from 3 to 5
    def retrieve(self, query, top_k=5, merge_chunks=True, rerank=True):
        """
        Complete retrieval pipeline with all improvements
        
        🆕 CHANGES:
        - top_k increased from 3 to 5 (better context)
        - Added deduplication
        - Added confidence filtering
        
        Returns:
            List of result dicts (backward compatible, no metadata in return)
        """
        # Step 1: Basic FAISS retrieval (get more candidates for re-ranking)
        candidates = self.retrieve_basic(query, top_k=top_k*3)
        
        # Step 2: Merge chunks if needed
        if merge_chunks:
            candidates = self.merge_chunks(candidates)
        
        # Step 3: Re-rank
        if rerank:
            candidates = self.rerank_bm25(query, candidates)
        
        # Step 4: 🆕 Deduplicate results
        candidates = self.deduplicate_results(candidates, similarity_threshold=0.85)
        
        # Step 5: 🆕 Filter by confidence
        candidates, low_confidence = self.filter_by_confidence(candidates)
        
        # Step 6: Add metadata to results
        for r in candidates:
            r['low_confidence'] = low_confidence
        
        # Step 7: Return top-k (backward compatible format)
        return candidates[:top_k]
    
    def format_context(self, results, max_length=400):  # 🔧 Tuned: 500 → 400 (balanced)
        """
        Format retrieved results into context string
        """
        contexts = []
        current_length = 0
        
        for r in results:
            answer = r['answer']
            answer_length = len(word_tokenize(answer))
            
            if current_length + answer_length <= max_length:
                contexts.append(answer)
                current_length += answer_length
            else:
                # Truncate if needed
                remaining = max_length - current_length
                if remaining > 20:  # Only add if meaningful amount remains
                    words = word_tokenize(answer)[:remaining]
                    contexts.append(" ".join(words))
                break
        
        return " ".join(contexts)

# =====================================================
# STANDALONE FUNCTIONS FOR BACKWARD COMPATIBILITY
# =====================================================

# Global retriever instance
_retriever = None

def get_retriever():
    """Get or create global retriever instance"""
    global _retriever
    if _retriever is None:
        _retriever = ImprovedRetriever()
    return _retriever

def query_retrieval(query, top_k=5):  # 🆕 Changed from 3 to 5
    """
    Backward compatible function for existing code
    Returns: List of tuples (answer, score)
    """
    retriever = get_retriever()
    results = retriever.retrieve(query, top_k=top_k)
    
    # Format as (answer, score) tuples for backward compatibility
    return [(r['answer'], r['score']) for r in results]

def query_retrieval_detailed(query, top_k=5):  # 🆕 Changed from 3 to 5
    """
    New function that returns full result objects
    """
    retriever = get_retriever()
    return retriever.retrieve(query, top_k=top_k)

# =====================================================
# TESTING
# =====================================================

def test_retrieval():
    """
    Test retrieval with sample queries including problematic ones from eval
    """
    print("\n" + "="*60)
    print("TESTING IMPROVED RETRIEVAL V3")
    print("="*60)
    
    retriever = ImprovedRetriever()
    
    # Test queries including the problematic ones from your eval results
    test_queries = [
        "bagaimana cara mengajukan bebas lab?",
        "sidia itu singkatan dari apa?",  # ❌ Error dari eval (cosine sim: 0.0)
        "klo file saya 2mb bisa dipakai daftar lsp?",  # ❌ Error dari eval
        "alamat website iopac unesa di mana?",  # ❌ Error dari eval
        "prosedur pengajuan cuti kuliah",
        "cara bayar ukt"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        results = retriever.retrieve(query, top_k=3, merge_chunks=True, rerank=True)
        
        if not results:
            print("❌ NO RESULTS FOUND (all below confidence threshold)")
            continue
        
        # Check confidence
        low_conf = results[0].get('low_confidence', False)
        if low_conf:
            print("⚠️ WARNING: Low confidence results")
        
        print(f"\n📊 Found {len(results)} results:")
        for i, r in enumerate(results, 1):
            print(f"\n[Result {i}]")
            print(f"Score: {r['rerank_score']:.4f} (original: {r['score']:.4f})")
            print(f"Coverage: {r['coverage']:.2%}, Overlap: {r['overlap']} terms")
            if 'num_chunks' in r:
                print(f"Chunks: {r['num_chunks']}")
            print(f"Answer: {r['answer'][:200]}...")
    
    print("\n" + "="*60)
    print("TESTING DEDUPLICATION")
    print("="*60)
    
    # Test deduplication with a query that might return duplicates
    query = "bebas lab"
    print(f"\nQuery: {query}")
    
    results_before = retriever.retrieve_basic(query, top_k=10)
    print(f"\nBefore merge & dedup: {len(results_before)} results")
    
    results_after = retriever.retrieve(query, top_k=5)
    print(f"After merge & dedup: {len(results_after)} results")
    
    print("\n All tests completed!")

if __name__ == "__main__":
    test_retrieval()