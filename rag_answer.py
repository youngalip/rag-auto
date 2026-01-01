import torch
import torch.nn.functional as F
from retrieval_query import ImprovedRetriever
from training import ImprovedSeq2SeqRAG

# =====================================================
# IMPROVED RAG ANSWER GENERATION
# =====================================================

class RAGAnswerGenerator:
    def __init__(self, 
                 model_path="model_rag_v2.pth",
                 max_len_output=150,
                 device=None):
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Loading model on {self.device}...")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.word2idx = checkpoint['word2idx']
        self.idx2word = checkpoint['idx2word']
        
        self.model = ImprovedSeq2SeqRAG(vocab_size=len(self.word2idx))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from fold {checkpoint.get('fold', '?')}")
        print(f"Vocabulary size: {len(self.word2idx)}")
        
        # Load retriever
        print("Loading retriever...")
        self.retriever = ImprovedRetriever()
        
        self.max_len_output = max_len_output
    
    def encode_input(self, text, max_len=200):
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
    
    def generate(self, query, top_k=5, temperature=0.8, use_retrieval=True, verbose=False):
        if verbose:
            print("-"*60)
            print(f"Query: {query}")
        
        # Step 1: Retrieve context
        if use_retrieval:
            # Increased top_k from to 5
            results = self.retriever.retrieve(query, top_k=5, merge_chunks=True, rerank=True)
            
            # Check if low confidence
            if not results or (results and results[0].get('low_confidence', False)):
                if verbose:
                    print(" LOW CONFIDENCE: No relevant documents found")
                return {
                    'answer': "Maaf, saya tidak menemukan informasi yang relevan. Bisakah Anda mengajukan pertanyaan dengan cara yang berbeda?",
                    'context': None,
                    'retrieved_docs': None,
                    'warning': 'low_confidence'
                }
            
            context = self.retriever.format_context(results, max_length=300)
            
            if verbose:
                print(f"\nRetrieved {len(results)} documents")
                print(f"Context length: {len(context.split())} words")
                print(f"Context preview: {context[:200]}...")
        else:
            context = ""
        
        # Step 2: Format input
        model_input = f"{query} <sep> {context}"
        
        if verbose:
            print(f"\nModel input: {model_input[:200]}...")
        
        # Step 3: Encode input
        input_tensor = self.encode_input(model_input)
        
        # Step 4: Generate answer
        with torch.no_grad():
            h, c = self.model.encode(input_tensor)
            
            inputs = torch.tensor([[self.word2idx["<sos>"]]]).to(self.device)
            outputs = []
            
            for _ in range(self.max_len_output):
                out, h, c = self.model.decode_step(inputs, h, c)
                
                # Apply temperature
                logits = out.squeeze(1) / temperature
                
                # Top-k sampling
                topk_logits, topk_indices = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                topk_probs = F.softmax(topk_logits, dim=-1)
                
                # Sample from top-k
                pred_idx = torch.multinomial(topk_probs, num_samples=1)
                word_id = torch.gather(topk_indices, -1, pred_idx).item()
                
                if word_id == self.word2idx["<eos>"]:
                    break
                
                outputs.append(word_id)
                inputs = torch.tensor([[word_id]]).to(self.device)
        
        # Step 5: Decode answer
        answer = self.decode_ids(outputs)

        word_count = len(answer.split())
        is_informative = any(keyword in answer.lower() for keyword in ['login', 'klik', 'menu', 'buka', 'akses', 'website', 'form', 'langkah'])

        if word_count < 10 and not is_informative:
            if verbose:
                print(f" Short uninformative answer detected ({word_count} words), regenerating with higher temperature...")
            
            # Regenerate dengan temperature LEBIH TINGGI dan top-k lebih besar
            with torch.no_grad():
                h, c = self.model.encode(input_tensor)
                inputs = torch.tensor([[self.word2idx["<sos>"]]]).to(self.device)
                outputs = []
                
                for _ in range(self.max_len_output):
                    out, h, c = self.model.decode_step(inputs, h, c)
                    
                    # 🔧 Use MUCH higher temperature for more diversity
                    logits = out.squeeze(1) / 1.5
                    
                    # 🔧 Increase top-k for more diversity
                    k = min(10, logits.size(-1))
                    topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)
                    topk_probs = F.softmax(topk_logits, dim=-1)
                    pred_idx = torch.multinomial(topk_probs, num_samples=1)
                    word_id = torch.gather(topk_indices, -1, pred_idx).item()
                    
                    if word_id == self.word2idx["<eos>"]:
                        break
                    
                    outputs.append(word_id)
                    inputs = torch.tensor([[word_id]]).to(self.device)
                
                new_answer = self.decode_ids(outputs)
                
                # Only use new answer if it's actually longer or different
                if len(new_answer.split()) > len(answer.split()):
                    answer = new_answer
                    if verbose:
                        print(f"✓ Regenerated answer ({len(answer.split())} words): {answer}")
                else:
                    if verbose:
                        print(f" Regeneration didn't improve, keeping original")
        elif word_count < 15 and not is_informative:
            if verbose:
                print(f" Short answer ({word_count} words) but will keep as-is")

        if verbose:
            print(f"\nGenerated answer: {answer}")
            print("-"*60)

        return {
            'answer': answer,
            'context': context if use_retrieval else None,
            'retrieved_docs': results if use_retrieval else None
        }
    
    def batch_generate(self, queries, **kwargs):
        """
        Generate answers for multiple queries
        """
        results = []
        for query in queries:
            result = self.generate(query, **kwargs)
            results.append(result)
        return results

# =====================================================
# BACKWARD COMPATIBLE FUNCTION
# =====================================================

_generator = None

def get_generator():
    """Get or create global generator"""
    global _generator
    if _generator is None:
        _generator = RAGAnswerGenerator()
    return _generator

def ask_rag_answer(query, top_k=5, verbose=False):
    generator = get_generator()
    result = generator.generate(query, top_k=top_k, verbose=verbose)
    return result['answer']

# =====================================================
# EVALUATION
# =====================================================

def evaluate_rag(test_file="preprocessed.json", num_samples=10):
    """
    Evaluate RAG model on test samples
    """
    import json
    import random
    
    print("="*60)
    print("RAG MODEL EVALUATION")
    print("="*60)
    
    # Load test data
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Group by question
    question_groups = {}
    for item in data:
        q = item['question']
        if q not in question_groups:
            question_groups[q] = []
        question_groups[q].append(item)
    
    # Sample questions
    test_questions = random.sample(list(question_groups.keys()), min(num_samples, len(question_groups)))
    
    # Initialize generator
    generator = RAGAnswerGenerator()
    
    # Evaluate
    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{num_samples}")
        print('='*60)
        
        # Get ground truth
        items = question_groups[question]
        ground_truth = " ".join([item['answer'] for item in sorted(items, key=lambda x: x['chunk_id'])])
        
        # Generate answer
        result = generator.generate(question, verbose=True)
        
        print(f"\nGround truth: {ground_truth[:200]}...")
        
        results.append({
            'question': question,
            'ground_truth': ground_truth,
            'generated': result['answer'],
            'context': result['context']
        })
    
    return results

# =====================================================
# TESTING
# =====================================================

if __name__ == "__main__":
    # Test with sample queries
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
    
    # Run evaluation
    # evaluate_rag(num_samples=5)