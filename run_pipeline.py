"""
MASTER PIPELINE SCRIPT FOR RAG OTOMATIS (IMPROVED)

Usage:
    python run_pipeline.py --all           # Run all steps
    python run_pipeline.py --from-step 3   # Start from step 3
    python run_pipeline.py --test-only     # Only test model
"""

import argparse
import os
import sys
import time

def check_file(filename):
    """Check if file exists"""
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return False
    return True

def run_step(step_num, step_name, command, input_files=None, output_files=None):
    """Run a pipeline step with validation"""
    print("\n" + "="*60)
    print(f"STEP {step_num}: {step_name}")
    print("="*60)
    
    # Check input files
    if input_files:
        for f in input_files:
            if not check_file(f):
                print(f"❌ Cannot proceed: Missing input file")
                return False
    
    # Run command
    print(f"Running: {command}")
    start = time.time()
    
    result = os.system(command)
    
    elapsed = time.time() - start
    
    if result != 0:
        print(f"❌ Step {step_num} failed!")
        return False
    
    # Check output files
    if output_files:
        for f in output_files:
            if not check_file(f):
                print(f"⚠️ Warning: Expected output file not found: {f}")
    
    print(f"✓ Step {step_num} completed in {elapsed:.1f}s")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run RAG Otomatis Pipeline')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--from-step', type=int, default=1, help='Start from step number')
    parser.add_argument('--test-only', action='store_true', help='Only test model')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("RAG OTOMATIS - IMPROVED PIPELINE")
    print("="*60)
    
    # Determine steps to run
    if args.test_only:
        steps = [8]
    elif args.eval_only:
        steps = [9]
    elif args.all:
        steps = list(range(1, 10))
    else:
        steps = list(range(args.from_step, 10))
    
    print(f"\nSteps to run: {steps}")
    print("\nPipeline:")
    print("  1. Extract documents (Ekstraksi.py)")
    print("  2. Label documents (labeling.py)")
    print("  3. Preprocess (preprocessing_improved.py)")
    print("  4. Build embeddings (embedding.py)")
    print("  5. Generate training data (generate_qa_training.py)")
    print("  6. Train model (training_fixed.py)")
    print("  7. Test retrieval (retrieval_query.py)")
    print("  8. Test generation (rag_answer_fixed.py)")
    print("  9. Evaluate (eval_fixed.py)")
    
    input("\nPress Enter to continue...")
    
    # Track total time
    total_start = time.time()
    
    # Step 1: Extract documents
    if 1 in steps:
        if not run_step(
            1, "EXTRACT DOCUMENTS",
            "python Ekstraksi.py",
            input_files=None,
            output_files=["extracted.json"]
        ):
            return
    
    # Step 2: Label documents
    if 2 in steps:
        if not run_step(
            2, "LABEL DOCUMENTS",
            "python labeling.py",
            input_files=["extracted.json"],
            output_files=["labeled.json"]
        ):
            return
    
    # Step 3: Preprocess
    if 3 in steps:
        if not run_step(
            3, "PREPROCESS",
            "python preprocessing_improved.py",
            input_files=["labeled.json"],
            output_files=["preprocessed.json"]
        ):
            return
    
    # Step 4: Build embeddings
    if 4 in steps:
        if not run_step(
            4, "BUILD EMBEDDINGS",
            "python embedding.py",
            input_files=["preprocessed.json"],
            output_files=["word2vec.model", "faiss_index.idx", "faiss_mapping.json"]
        ):
            return
    
    # Step 5: Generate training data
    if 5 in steps:
        if not run_step(
            5, "GENERATE TRAINING DATA",
            "python generate_qa_training.py",
            input_files=["preprocessed.json"],
            output_files=["rag_training_data.json"]
        ):
            return
    
    # Step 6: Train model
    if 6 in steps:
        print("\n⚠️ WARNING: Training will take several hours!")
        print("   Estimated time: 2-8 hours depending on data size")
        confirm = input("   Continue? (y/n): ")
        
        if confirm.lower() != 'y':
            print("Training skipped")
            return
        
        if not run_step(
            6, "TRAIN MODEL",
            "python training_fixed.py",
            input_files=["rag_training_data.json"],
            output_files=["model_rag_v2.pth"]
        ):
            return
    
    # Step 7: Test retrieval
    if 7 in steps:
        if not run_step(
            7, "TEST RETRIEVAL",
            "python retrieval_query.py",
            input_files=["word2vec.model", "faiss_index.idx", "faiss_mapping.json"],
            output_files=None
        ):
            return
    
    # Step 8: Test generation
    if 8 in steps:
        if not run_step(
            8, "TEST GENERATION",
            "python rag_answer_fixed.py",
            input_files=["model_rag_v2.pth", "word2vec.model", "faiss_index.idx"],
            output_files=None
        ):
            return
    
    # Step 9: Evaluate
    if 9 in steps:
        if not run_step(
            9, "EVALUATE MODEL",
            "python eval_fixed.py",
            input_files=["model_rag_v2.pth", "rag_training_data.json"],
            output_files=["evaluation_results.json"]
        ):
            return
    
    # Summary
    total_elapsed = time.time() - total_start
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    
    print("\nGenerated files:")
    files = [
        "extracted.json", "labeled.json", "preprocessed.json",
        "word2vec.model", "faiss_index.idx", "faiss_mapping.json",
        "rag_training_data.json", "model_rag_v2.pth",
        "evaluation_results.json"
    ]
    
    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f) / (1024*1024)  # MB
            print(f"  ✓ {f:30s} ({size:.2f} MB)")
    
    print("\nNext steps:")
    print("  - Check evaluation_results.json for metrics")
    print("  - Test with: python rag_answer_fixed.py")
    print("  - Deploy if F1-score > 0.60")

if __name__ == "__main__":
    main()