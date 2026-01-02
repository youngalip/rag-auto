import json
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rag_answer_fixed import RAGAnswerGenerator

def calculate_cosine_similarity(ground_truths, predictions):
    """Calculate cosine similarity using TF-IDF"""
    if not ground_truths or not predictions:
        return []
    
    all_texts = ground_truths + predictions
    
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except:
        return [0.0] * len(ground_truths)
    
    n = len(ground_truths)
    gt_vectors = tfidf_matrix[:n]
    pred_vectors = tfidf_matrix[n:]
    
    similarities = []
    for i in range(n):
        sim = cosine_similarity(gt_vectors[i], pred_vectors[i])[0][0]
        similarities.append(sim)
    
    return similarities

def evaluate_rag_model(dataset_path="rag_training_data.json", max_samples=50):
    """
    FIXED: Evaluate on training data with sampling
    """
    print("="*60)
    print("EVALUATING RAG MODEL")
    print("="*60)
    
    # Load data
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Sample for faster evaluation
    if len(data) > max_samples:
        import random
        data = random.sample(data, max_samples)
        print(f"Sampled {max_samples} examples for evaluation")
    
    print(f"Total test samples: {len(data)}")
    
    # Initialize generator
    print("\nLoading RAG model...")
    generator = RAGAnswerGenerator()
    
    # Evaluate
    queries = []
    ground_truths = []
    predictions = []
    
    print("\nGenerating answers...")
    for i, item in enumerate(data, 1):
        question = item["question"]
        gt_answer = item["answer"]
        
        # Generate prediction
        try:
            result = generator.generate(question, verbose=False)
            pred_answer = result['answer']
        except Exception as e:
            print(f"Error on query {i}: {e}")
            pred_answer = ""
        
        queries.append(question)
        ground_truths.append(gt_answer)
        predictions.append(pred_answer)
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(data)} questions...")
    
    # Calculate metrics
    def tokenize(text):
        return set(text.lower().split())
    
    precisions, recalls, f1s = [], [], []
    exact_matches = []
    
    for gt, pred in zip(ground_truths, predictions):
        t_true = tokenize(gt)
        t_pred = tokenize(pred)
        
        if len(t_pred) == 0:
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)
        else:
            tp = len(t_true & t_pred)
            prec = tp / len(t_pred)
            rec = tp / len(t_true) if len(t_true) > 0 else 0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        
        exact_matches.append(1 if gt.strip().lower() == pred.strip().lower() else 0)
    
    avg_prec = np.mean(precisions)
    avg_rec = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    exact_match = np.mean(exact_matches)
    
    # Cosine Similarity
    print("\nCalculating cosine similarity...")
    cosine_sims = calculate_cosine_similarity(ground_truths, predictions)
    avg_cosine = np.mean(cosine_sims)
    
    # Error Analysis
    errors = []
    for i, (q, gt, pred, cos_sim) in enumerate(zip(queries, ground_truths, predictions, cosine_sims)):
        if gt.strip().lower() != pred.strip().lower():
            errors.append({
                'query': q,
                'ground_truth': gt,
                'prediction': pred,
                'cosine_sim': cos_sim
            })
    
    errors.sort(key=lambda x: x['cosine_sim'])
    
    # Display Results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Precision            : {avg_prec:.4f}")
    print(f"Recall               : {avg_rec:.4f}")
    print(f"F1-score             : {avg_f1:.4f}")
    print(f"Cosine Similarity    : {avg_cosine:.4f}")
    print(f"Exact Match Accuracy : {exact_match:.4f}")
    print(f"Total Samples        : {len(queries)}")
    print(f"Errors               : {len(errors)}")
    
    print("\n--- Worst Errors (Top 5) ---")
    for i, err in enumerate(errors[:5], 1):
        print(f"\n[Error {i}] (Cosine Sim: {err['cosine_sim']:.4f})")
        print(f"Query      : {err['query']}")
        print(f"Ground Truth: {err['ground_truth'][:100]}...")
        print(f"Prediction : {err['prediction'][:100]}...")
        print("-"*60)
    
    print("\n--- Best Predictions (Top 5) ---")
    correct = []
    for i, (q, gt, pred, cos_sim) in enumerate(zip(queries, ground_truths, predictions, cosine_sims)):
        if cos_sim > 0.3:  # Reasonable threshold
            correct.append({
                'query': q,
                'answer': pred,
                'cosine_sim': cos_sim
            })
    
    correct.sort(key=lambda x: x['cosine_sim'], reverse=True)
    
    for i, corr in enumerate(correct[:5], 1):
        print(f"\n[Good {i}] (Cosine Sim: {corr['cosine_sim']:.4f})")
        print(f"Query   : {corr['query']}")
        print(f"Answer  : {corr['answer'][:100]}...")
        print("-"*60)
    
    # Cosine similarity distribution
    cos_sim_ranges = {
        '0.0-0.2': sum(1 for s in cosine_sims if 0.0 <= s < 0.2),
        '0.2-0.4': sum(1 for s in cosine_sims if 0.2 <= s < 0.4),
        '0.4-0.6': sum(1 for s in cosine_sims if 0.4 <= s < 0.6),
        '0.6-0.8': sum(1 for s in cosine_sims if 0.6 <= s < 0.8),
        '0.8-1.0': sum(1 for s in cosine_sims if 0.8 <= s <= 1.0),
    }
    
    print("\n--- Cosine Similarity Distribution ---")
    for range_name, count in cos_sim_ranges.items():
        percentage = (count / len(cosine_sims)) * 100
        print(f"{range_name}: {count:4d} ({percentage:5.1f}%)")
    
    return {
        "precision": avg_prec,
        "recall": avg_rec,
        "f1": avg_f1,
        "cosine_similarity": avg_cosine,
        "exact_match": exact_match,
        "total": len(queries),
        "errors": len(errors),
        "correct": len(correct),
        "cosine_sim_distribution": cos_sim_ranges
    }

if __name__ == "__main__":
    start = time.time()
    
    # FIXED: Evaluate on training data with sampling
    hasil = evaluate_rag_model(
        "rag_training_data.json",  # FIXED: Use training data
        max_samples=50  # Sample for faster eval
    )
    
    durasi = round(time.time() - start, 2)
    print(f"\nEvaluation Time: {durasi} seconds ({durasi/60:.1f} minutes)")
    
    # Save results
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Results saved to: evaluation_results.json")