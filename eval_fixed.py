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

def evaluate_rag_model(dataset_path="rag_training_data.json", sample_size=None):
    """
    Evaluate RAG model on training data
    
    Args:
        dataset_path: Path to training data
        sample_size: Number of samples to evaluate (None = all)
    """
    print("="*60)
    print("EVALUATING RAG MODEL")
    print("="*60)
    
    # Load data
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Sample if needed
    if sample_size and len(data) > sample_size:
        import random
        data = random.sample(data, sample_size)
        print(f"Sampled {sample_size} examples for evaluation")
    
    print(f"Total test samples: {len(data)}")
    
    # Initialize generator
    print("\nLoading RAG model...")
    try:
        generator = RAGAnswerGenerator()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure you have trained the model first!")
        return None
    
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
        
        if i % 20 == 0:
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
                'idx': i,
                'query': q,
                'ground_truth': gt,
                'prediction': pred,
                'cosine_sim': cos_sim,
                'f1': f1s[i]
            })
    
    # Sort errors by similarity (worst first)
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
    
    # Show worst errors
    print("\n--- Worst Errors (Top 5) ---")
    for i, err in enumerate(errors[:5], 1):
        print(f"\n[Error {i}] (Cosine Sim: {err['cosine_sim']:.4f}, F1: {err['f1']:.4f})")
        print(f"Query      : {err['query']}")
        print(f"Ground Truth: {err['ground_truth'][:150]}...")
        print(f"Prediction : {err['prediction'][:150]}...")
        print("-"*60)
    
    # Show good predictions
    correct = []
    for i, (q, gt, pred, cos_sim, f1) in enumerate(zip(queries, ground_truths, predictions, cosine_sims, f1s)):
        if f1 > 0.7:  # Good predictions
            correct.append({
                'idx': i,
                'query': q,
                'answer': pred,
                'cosine_sim': cos_sim,
                'f1': f1
            })
    
    correct.sort(key=lambda x: x['f1'], reverse=True)
    
    print("\n--- Good Predictions (Top 5) ---")
    for i, corr in enumerate(correct[:5], 1):
        print(f"\n[Good {i}] (Cosine Sim: {corr['cosine_sim']:.4f}, F1: {corr['f1']:.4f})")
        print(f"Query   : {corr['query']}")
        print(f"Answer  : {corr['answer'][:150]}...")
        print("-"*60)
    
    # Distribution of scores
    f1_ranges = {
        '0.0-0.2': sum(1 for f in f1s if 0.0 <= f < 0.2),
        '0.2-0.4': sum(1 for f in f1s if 0.2 <= f < 0.4),
        '0.4-0.6': sum(1 for f in f1s if 0.4 <= f < 0.6),
        '0.6-0.8': sum(1 for f in f1s if 0.6 <= f < 0.8),
        '0.8-1.0': sum(1 for f in f1s if 0.8 <= f <= 1.0),
    }
    
    print("\n--- F1-Score Distribution ---")
    for range_name, count in f1_ranges.items():
        percentage = (count / len(f1s)) * 100
        bar = "█" * int(percentage / 5)
        print(f"{range_name}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    # Summary verdict
    print("\n" + "="*60)
    print("PERFORMANCE VERDICT")
    print("="*60)
    
    if avg_f1 >= 0.70:
        verdict = "✅ EXCELLENT - Ready for production"
    elif avg_f1 >= 0.60:
        verdict = "✅ GOOD - Acceptable for production"
    elif avg_f1 >= 0.50:
        verdict = "⚠️ ACCEPTABLE - Needs improvement"
    elif avg_f1 >= 0.40:
        verdict = "⚠️ POOR - Major improvements needed"
    else:
        verdict = "❌ VERY POOR - System not working"
    
    print(f"F1-Score: {avg_f1:.4f}")
    print(f"Verdict: {verdict}")
    
    if avg_f1 < 0.60:
        print("\n💡 Suggestions for improvement:")
        print("  - Add more training data (current: {})".format(len(data)))
        print("  - Improve Q&A quality (check samples above)")
        print("  - Consider manual Q&A augmentation")
        print("  - Tune hyperparameters (temperature, top_k)")
    
    return {
        "precision": avg_prec,
        "recall": avg_rec,
        "f1": avg_f1,
        "cosine_similarity": avg_cosine,
        "exact_match": exact_match,
        "total": len(queries),
        "errors": len(errors),
        "correct": len(correct)
    }

if __name__ == "__main__":
    start = time.time()
    
    # Evaluate
    hasil = evaluate_rag_model(
        "rag_training_data.json",
        sample_size=None  # Use all data, or set to 50 for quick test
    )
    
    if hasil:
        durasi = round(time.time() - start, 2)
        print(f"\nEvaluation Time: {durasi} seconds ({durasi/60:.1f} minutes)")
        
        # Save results
        with open("evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(hasil, f, indent=2, ensure_ascii=False)
        
        print("\n✓ Results saved to: evaluation_results.json")