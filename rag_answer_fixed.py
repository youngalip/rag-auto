import torch
import torch.nn.functional as F
from retrieval_query import ImprovedRetriever
from training_fixed import ImprovedSeq2SeqRAG

# ===== IMPROVED RAG ANSWER GENERATION =====
class RAGAnswerGenerator:
    def __init__(self, 
                 model_path="model_rag_v2.pth",
                 max_len_output=200,  # FIXED: 150 → 200
                 device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Loading model on {self.device}...")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.word2idx = checkpoint['word2idx']
        self.idx2word = checkpoint['idx2word']
        
        self.model = ImprovedSeq2SeqRAG(vocab_size=len(self.word2idx))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from fold {checkpoint.get('fold', '?')}")
        print(f"Vocabulary size: {len(self.word2idx)}")
        
        print("Loading retriever...")
        self.retriever = ImprovedRetriever()
        
        self.max_len_output = max_len_output
    
    def encode_input(self, text, max_len=250):  # FIXED: 200 → 250
        tokens = text.lower().split()
        ids = [self.word2idx.get(tok, self.word2idx["<unk>"]) for tok in tokens]
        ids = ids[:max_len]
        ids += [self.word2idx["<pad>"]] * (max_len - len(ids))
        return torch.tensor(ids).unsqueeze(0).to(self.device)
    
    def decode_ids(self, ids):
        words = []
        for i in ids:
            if i == self.word2idx["<eos>"]:
                break
            w = self.idx2word.get(i, "<unk>")
            if w not in ["<pad>", "<sos>", "<sep>"]:
                words.append(w)
        return " ".join(words)
    
    def generate(self, query, top_k=5, temperature=0.7, use_retrieval=True, verbose=False):
        """
        FIXED: Better generation with longer outputs
        """
        if verbose:
            print("-"*60)
            print(f"Query: {query}")
        
        # Step 1: Retrieve context
        if use_retrieval:
            results = self.retriever.retrieve(query, top_k=5, merge_chunks=True, rerank=True)
            
            if not results or (results and results[0].get('low_confidence', False)):
                if verbose:
                    print(" LOW CONFIDENCE: No relevant documents found")
                return {
                    'answer': "Maaf, saya tidak menemukan informasi yang relevan. Bisakah Anda mengajukan pertanyaan dengan cara yang berbeda?",
                    'context': None,
                    'retrieved_docs': None,
                    'warning': 'low_confidence'
                }
            
            context = self.retriever.format_context(results, max_length=400)
            
            if verbose:
                print(f"\nRetrieved {len(results)} documents")
                print(f"Context preview: {context[:200]}...")
        else:
            context = ""
        
        # Step 2: Format input
        model_input = f"{query} <sep> {context}"
        
        # Step 3: Encode
        input_tensor = self.encode_input(model_input)
        
        # Step 4: Generate
        with torch.no_grad():
            h, c = self.model.encode(input_tensor)
            
            inputs = torch.tensor([[self.word2idx["<sos>"]]]).to(self.device)
            outputs = []
            
            for step in range(self.max_len_output):
                out, h, c = self.model.decode_step(inputs, h, c)
                
                # FIXED: Better sampling strategy
                logits = out.squeeze(1) / temperature
                
                # Top-k sampling
                topk_logits, topk_indices = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                topk_probs = F.softmax(topk_logits, dim=-1)
                
                pred_idx = torch.multinomial(topk_probs, num_samples=1)
                word_id = torch.gather(topk_indices, -1, pred_idx).item()
                
                if word_id == self.word2idx["<eos>"]:
                    break
                
                outputs.append(word_id)
                inputs = torch.tensor([[word_id]]).to(self.device)
        
        # Step 5: Decode
        answer = self.decode_ids(outputs)
        
        # FIXED: Less aggressive re-generation check
        word_count = len(answer.split())
        
        # Only regenerate if VERY short AND uninformative
        if word_count < 8:  # FIXED: 10 → 8
            if verbose:
                print(f" Very short answer ({word_count} words), regenerating...")
            
            with torch.no_grad():
                h, c = self.model.encode(input_tensor)
                inputs = torch.tensor([[self.word2idx["<sos>"]]]).to(self.device)
                outputs = []
                
                for _ in range(self.max_len_output):
                    out, h, c = self.model.decode_step(inputs, h, c)
                    
                    # Higher temperature for diversity
                    logits = out.squeeze(1) / 1.2
                    
                    k = min(8, logits.size(-1))
                    topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)
                    topk_probs = F.softmax(topk_logits, dim=-1)
                    pred_idx = torch.multinomial(topk_probs, num_samples=1)
                    word_id = torch.gather(topk_indices, -1, pred_idx).item()
                    
                    if word_id == self.word2idx["<eos>"]:
                        break
                    
                    outputs.append(word_id)
                    inputs = torch.tensor([[word_id]]).to(self.device)
                
                new_answer = self.decode_ids(outputs)
                
                if len(new_answer.split()) > len(answer.split()):
                    answer = new_answer
                    if verbose:
                        print(f"✓ Regenerated: {answer}")

        if verbose:
            print(f"\nGenerated answer ({len(answer.split())} words): {answer}")
            print("-"*60)

        return {
            'answer': answer,
            'context': context if use_retrieval else None,
            'retrieved_docs': results if use_retrieval else None
        }
    
    def batch_generate(self, queries, **kwargs):
        results = []
        for query in queries:
            result = self.generate(query, **kwargs)
            results.append(result)
        return results

# ===== BACKWARD COMPATIBLE =====
_generator = None

def get_generator():
    global _generator
    if _generator is None:
        _generator = RAGAnswerGenerator()
    return _generator

def ask_rag_answer(query, top_k=5, verbose=False):
    generator = get_generator()
    result = generator.generate(query, top_k=top_k, verbose=verbose)
    return result['answer']

# ===== TESTING =====
if __name__ == "__main__":
    test_queries = [
        "bagaimana cara mengajukan bebas lab?",
        "prosedur pengajuan cuti kuliah",
        "cara bayar ukt",
        "syarat yudisium"
    ]
    
    print("="*60)
    print("TESTING RAG ANSWER GENERATION")
    print("="*60)
    
    generator = RAGAnswerGenerator()
    
    for query in test_queries:
        result = generator.generate(query, verbose=True)
        print()