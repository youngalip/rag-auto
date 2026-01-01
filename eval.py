# ========================================== 
# eval_rag_model.py - Evaluate Improved RAG System
# ==========================================

import json
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rag_answer import RAGAnswerGenerator

def calculate_cosine_similarity(ground_truths, predictions):
    """
    Calculate cosine similarity between ground truths and predictions
    using TF-IDF vectorization
    """
    if not ground_truths or not predictions:
        return []
    
    # Combine all texts for vectorization
    all_texts = ground_truths + predictions
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except:
        # If vectorization fails (e.g., empty texts), return zeros
        return [0.0] * len(ground_truths)
    
    # Split back into ground truths and predictions
    n = len(ground_truths)
    gt_vectors = tfidf_matrix[:n]
    pred_vectors = tfidf_matrix[n:]
    
    # Calculate cosine similarity for each pair
    similarities = []
    for i in range(n):
        sim = cosine_similarity(gt_vectors[i], pred_vectors[i])[0][0]
        similarities.append(sim)
    
    return similarities

def evaluate_rag_model(dataset_path, query_col="question", answer_col="answer"):
    """
    Evaluate improved RAG model end-to-end
    """
    print("="*60)
    print("EVALUATING IMPROVED RAG MODEL")
    print("="*60)
    
    # Load test data
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Group by question to handle chunks
    question_groups = {}
    for item in data:
        q = item[query_col]
        if q not in question_groups:
            question_groups[q] = []
        question_groups[q].append(item)
    
    print(f"Total unique questions: {len(question_groups)}")
    
    # Initialize generator
    print("\nLoading RAG model...")
    generator = RAGAnswerGenerator()
    
    # Evaluate
    queries = []
    ground_truths = []
    predictions = []
    
    print("\nGenerating answers...")
    for i, (question, items) in enumerate(question_groups.items(), 1):
        # Get ground truth (merge chunks)
        gt_answer = " ".join([item[answer_col] for item in sorted(items, key=lambda x: x.get('chunk_id', 0))])
        
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
        
        if i % 100 == 0:
            print(f"  Processed {i}/{len(question_groups)} questions...")
    
    # ==========================================
    # Metrics Calculation
    # ==========================================
    def tokenize(text):
        return set(text.lower().split())
    
    precisions, recalls, f1s = [], [], []
    exact_matches = []
    
    for gt, pred in zip(ground_truths, predictions):
        t_true = tokenize(gt)
        t_pred = tokenize(pred)
        
        # Precision, Recall, F1
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
        
        # Exact match
        exact_matches.append(1 if gt.strip().lower() == pred.strip().lower() else 0)
    
    avg_prec = np.mean(precisions)
    avg_rec = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    exact_match = np.mean(exact_matches)
    
    # Calculate Cosine Similarity
    print("\nCalculating cosine similarity...")
    cosine_sims = calculate_cosine_similarity(ground_truths, predictions)
    avg_cosine = np.mean(cosine_sims)
    
    # ==========================================
    # Error Analysis
    # ==========================================
    errors = []
    for i, (q, gt, pred, cos_sim) in enumerate(zip(queries, ground_truths, predictions, cosine_sims)):
        if gt.strip().lower() != pred.strip().lower():
            errors.append({
                'query': q,
                'ground_truth': gt,
                'prediction': pred,
                'cosine_sim': cos_sim
            })
    
    # Sort errors by cosine similarity (lowest first = worst errors)
    errors.sort(key=lambda x: x['cosine_sim'])
    
    # ==========================================
    # Display Results
    # ==========================================
    print("\n" + "="*60)
    print("HASIL EVALUASI RAG MODEL")
    print("="*60)
    print(f"Precision            : {avg_prec:.4f}")
    print(f"Recall               : {avg_rec:.4f}")
    print(f"F1-score             : {avg_f1:.4f}")
    print(f"Cosine Similarity    : {avg_cosine:.4f}")
    print(f"Exact Match Accuracy : {exact_match:.4f}")
    print(f"Jumlah Data Uji      : {len(queries)}")
    print(f"Jumlah Errors        : {len(errors)}")
    
    print("\n--- Worst Errors (5 Terburuk berdasarkan Cosine Similarity) ---")
    for i, err in enumerate(errors[:5], 1):
        print(f"\n[Error {i}] (Cosine Sim: {err['cosine_sim']:.4f})")
        print(f"Query         : {err['query']}")
        print(f"Ground Truth  : {err['ground_truth'][:150]}...")
        print(f"Prediction    : {err['prediction'][:150]}...")
        print("-"*60)
    
    print("\n--- Contoh BENAR (5 Teratas) ---")
    correct = []
    for i, (q, gt, pred, cos_sim) in enumerate(zip(queries, ground_truths, predictions, cosine_sims)):
        if gt.strip().lower() == pred.strip().lower():
            correct.append({
                'query': q,
                'answer': pred,
                'cosine_sim': cos_sim
            })
    
    for i, corr in enumerate(correct[:5], 1):
        print(f"\n[Correct {i}] (Cosine Sim: {corr['cosine_sim']:.4f})")
        print(f"Query   : {corr['query']}")
        print(f"Answer  : {corr['answer'][:150]}...")
        print("-"*60)
    
    # Calculate cosine similarity distribution
    cos_sim_ranges = {
        '0.0-0.2': sum(1 for s in cosine_sims if 0.0 <= s < 0.2),
        '0.2-0.4': sum(1 for s in cosine_sims if 0.2 <= s < 0.4),
        '0.4-0.6': sum(1 for s in cosine_sims if 0.4 <= s < 0.6),
        '0.6-0.8': sum(1 for s in cosine_sims if 0.6 <= s < 0.8),
        '0.8-1.0': sum(1 for s in cosine_sims if 0.8 <= s <= 1.0),
    }
    
    print("\n--- Distribusi Cosine Similarity ---")
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
    
    # Evaluate on preprocessed data
    hasil = evaluate_rag_model(
        "preprocessed.json", 
        query_col="question", 
        answer_col="answer"
    )
    
    durasi = round(time.time() - start, 2)
    print(f"\nWaktu Evaluasi: {durasi} detik ({durasi/60:.1f} menit)")
    
    # Save results
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Results saved to: evaluation_results.json")