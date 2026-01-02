import faiss
import json
import os
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from collections import defaultdict

# =====================================================
# IMPROVED RETRIEVAL WITH DEDUPLICATION & RE-RANKING
# =====================================================

class ImprovedRetriever:
    def __init__(self, 
                 index_path="faiss_index.idx",
                 mapping_path="faiss_mapping.json",
                 w2v_path="word2vec.model"):
        """Initialize retriever"""
        print("Loading retrieval components...")
        
        # Load FAISS index
        base_dir = os.path.dirname(__file__) if __file__ else "."
        self.index = faiss.read_index(os.path.join(base_dir, index_path))
        
        # Load document mapping
        with open(os.path.join(base_dir, mapping_path), "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        
        # Load Word2Vec
        self.w2v = Word2Vec.load(os.path.join(base_dir, w2v_path))
        
        print(f"Loaded {len(self.documents)} documents")
        print(f"FAISS index size: {self.index.ntotal}")
        print(f"Word2Vec vocab size: {len(self.w2v.wv)}")
    
    def encode_query(self, query):
        """Encode query with TF-IDF weighting"""
        tokens = word_tokenize(query.lower())
        vectors = []
        
        for token in tokens:
            if token in self.w2v.wv:
                weight = tokens.count(token) / len(tokens)
                vectors.append(self.w2v.wv[token] * weight)
        
        if vectors:
            vec = np.mean(vectors, axis=0).reshape(1, -1).astype("float32")
            faiss.normalize_L2(vec)
            return vec
        else:
            return np.zeros((1, self.w2v.vector_size), dtype="float32")
    
    def retrieve_basic(self, query, top_k=5):
        """Basic FAISS retrieval"""
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
        """Merge chunks from same document/label"""
        # Group by filename or label
        groups = defaultdict(list)
        
        for r in results:
            # Use filename as key, fallback to label
            key = r['doc'].get('filename', r['doc'].get('label', 'unknown'))
            groups[key].append(r)
        
        merged = []
        for key, chunks in groups.items():
            # Sort by chunk_id if available
            chunks.sort(key=lambda x: x['doc'].get('chunk_id', 0))
            
            # Combine text
            combined_text = " ".join([c['doc']['text'] for c in chunks])
            
            # Average score
            avg_score = np.mean([c['score'] for c in chunks])
            
            merged.append({
                'source': key,
                'answer': combined_text,
                'score': avg_score,
                'num_chunks': len(chunks),
                'chunk_ids': [c['doc'].get('chunk_id', 0) for c in chunks]
            })
        
        # Sort by score
        merged.sort(key=lambda x: x['score'], reverse=True)
        
        return merged
    
    def deduplicate(self, results, threshold=0.9):
        """Remove duplicate results"""
        if not results:
            return results
        
        unique = []
        seen_texts = []
        
        for r in results:
            text = r['answer'][:200].lower()  # Use first 200 chars
            
            # Check similarity with seen texts
            is_duplicate = False
            for seen in seen_texts:
                # Simple character overlap
                overlap = len(set(text) & set(seen)) / len(set(text)) if text else 0
                if overlap >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(r)
                seen_texts.append(text)
        
        return unique
    
    def rerank_bm25(self, query, candidates):
        """BM25-like re-ranking"""
        query_tokens = set(word_tokenize(query.lower()))
        
        reranked = []
        for candidate in candidates:
            answer_tokens = set(word_tokenize(candidate['answer'].lower()))
            
            # Calculate overlap
            overlap = len(query_tokens & answer_tokens)
            coverage = overlap / len(query_tokens) if query_tokens else 0
            
            # Combine scores
            combined_score = candidate['score'] * 0.7 + coverage * 0.3
            
            candidate['rerank_score'] = combined_score
            candidate['overlap'] = overlap
            candidate['coverage'] = coverage
            reranked.append(candidate)
        
        # Sort by re-ranked score
        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked
    
    def retrieve(self, query, top_k=3, merge_chunks=True, rerank=True):
        """
        Complete retrieval pipeline
        
        Returns: List of result dicts with 'answer', 'score', etc.
        """
        # Step 1: Get candidates (more than needed for re-ranking)
        candidates = self.retrieve_basic(query, top_k=top_k*3)
        
        # Step 2: Merge chunks
        if merge_chunks:
            candidates = self.merge_chunks(candidates)
        
        # Step 3: Deduplicate
        candidates = self.deduplicate(candidates, threshold=0.9)
        
        # Step 4: Re-rank
        if rerank:
            candidates = self.rerank_bm25(query, candidates)
        
        # Step 5: Return top-k
        return candidates[:top_k]
    
    def format_context(self, results, max_length=400):
        """Format results into context string"""
        contexts = []
        current_length = 0
        
        for r in results:
            answer = r['answer']
            answer_length = len(word_tokenize(answer))
            
            if current_length + answer_length <= max_length:
                contexts.append(answer)
                current_length += answer_length
            else:
                # Truncate
                remaining = max_length - current_length
                if remaining > 20:
                    words = word_tokenize(answer)[:remaining]
                    contexts.append(" ".join(words))
                break
        
        return " ".join(contexts)

# =====================================================
# BACKWARD COMPATIBLE FUNCTIONS
# =====================================================

_retriever = None

def get_retriever():
    """Get or create global retriever"""
    global _retriever
    if _retriever is None:
        _retriever = ImprovedRetriever()
    return _retriever

def query_retrieval(query, top_k=3):
    """
    Backward compatible function
    Returns: List of (answer, score) tuples
    """
    retriever = get_retriever()
    results = retriever.retrieve(query, top_k=top_k)
    
    return [(r['answer'], r['score']) for r in results]

def query_retrieval_detailed(query, top_k=3):
    """Returns full result objects"""
    retriever = get_retriever()
    return retriever.retrieve(query, top_k=top_k)

# =====================================================
# TESTING
# =====================================================

def test_retrieval():
    """Test retrieval"""
    print("\n" + "="*60)
    print("TESTING RETRIEVAL")
    print("="*60)
    
    retriever = ImprovedRetriever()
    
    test_queries = [
        "bagaimana cara mengajukan bebas lab?",
        "prosedur pengajuan cuti kuliah",
        "cara bayar ukt"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        results = retriever.retrieve(query, top_k=3)
        
        for i, r in enumerate(results, 1):
            print(f"\n[Result {i}]")
            print(f"Score: {r.get('rerank_score', r['score']):.4f}")
            print(f"Answer: {r['answer'][:200]}...")

if __name__ == "__main__":
    test_retrieval()