import subprocess
import os
import sys
import time

# =====================================================
# COMPLETE RAG PIPELINE AUTOMATION
# =====================================================

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🚀 {text}")
    print("="*60)

def run_command(cmd, description):
    """Run a command and handle errors"""
    print_header(description)
    print(f"Command: {cmd}")
    print("-"*60)
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ ERROR in: {description}")
        print(f"Command failed: {cmd}")
        print(f"Exit code: {result.returncode}")
        sys.exit(1)
    
    print(f"\n✅ {description} completed in {elapsed:.1f}s")
    return True

def check_file(filepath, description):
    """Check if file exists and show info"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        
        # Format size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size/1024:.1f} KB"
        else:
            size_str = f"{size/(1024*1024):.1f} MB"
        
        print(f"  ✓ {description}")
        print(f"    File: {filepath}")
        print(f"    Size: {size_str}")
        return True
    else:
        print(f"  ❌ Missing: {filepath}")
        return False

def verify_step(files, step_name):
    """Verify files for a step"""
    print(f"\n📋 Verifying {step_name}...")
    
    all_ok = True
    for filepath, description in files:
        if not check_file(filepath, description):
            all_ok = False
    
    if not all_ok:
        print(f"\n❌ {step_name} verification failed!")
        sys.exit(1)
    
    print(f"✅ {step_name} verified successfully")

def main():
    """
    Main pipeline execution
    """
    print("="*60)
    print("🔥 COMPLETE RAG PIPELINE - AUTOMATED EXECUTION")
    print("="*60)
    print("\nPipeline Steps:")
    print("  1. Text Extraction from Documents")
    print("  2. Automatic Labeling")
    print("  3. Text Preprocessing")
    print("  4. Building Embeddings (Word2Vec + FAISS)")
    print("  5. Generate Training Data")
    print("  6. Model Training (K-Fold)")
    print("  7. Ready for Testing")
    print("\n" + "="*60)
    
    input("\nPress ENTER to start pipeline...")
    
    # ==================================================
    # STEP 1: TEXT EXTRACTION
    # ==================================================
    run_command("python Ekstraksi.py", "STEP 1/6: Text Extraction")
    verify_step([
        ("extracted.json", "Extracted text data")
    ], "Step 1")
    
    # ==================================================
    # STEP 2: AUTOMATIC LABELING
    # ==================================================
    run_command("python labeling.py", "STEP 2/6: Automatic Labeling")
    verify_step([
        ("labeled.json", "Labeled data")
    ], "Step 2")
    
    # ==================================================
    # STEP 3: PREPROCESSING
    # ==================================================
    run_command("python preprocessing3.py", "STEP 3/6: Text Preprocessing")
    verify_step([
        ("preprocessed.json", "Preprocessed chunks")
    ], "Step 3")
    
    # ==================================================
    # STEP 4: BUILDING EMBEDDINGS
    # ==================================================
    run_command("python embedding.py", "STEP 4/6: Building Embeddings")
    verify_step([
        ("word2vec.model", "Word2Vec model"),
        ("faiss_index.idx", "FAISS index"),
        ("faiss_mapping.json", "FAISS mapping")
    ], "Step 4")
    
    # ==================================================
    # STEP 5: GENERATE TRAINING DATA
    # ==================================================
    run_command("python generate_training_data_fixed.py", "STEP 5/6: Generate Training Data")
    verify_step([
        ("rag_training_data.json", "RAG training data")
    ], "Step 5")
    
    # ==================================================
    # STEP 6: MODEL TRAINING
    # ==================================================
    run_command("python training.py", "STEP 6/6: Model Training (K-Fold)")
    verify_step([
        ("model_rag_v2.pth", "Trained RAG model")
    ], "Step 6")
    
    # ==================================================
    # PIPELINE COMPLETED
    # ==================================================
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📦 Generated Files:")
    print("-"*60)
    
    all_files = [
        ("extracted.json", "Raw extracted text"),
        ("labeled.json", "Labeled data"),
        ("preprocessed.json", "Preprocessed chunks"),
        ("word2vec.model", "Word2Vec embeddings"),
        ("faiss_index.idx", "FAISS vector index"),
        ("faiss_mapping.json", "Document mapping"),
        ("rag_training_data.json", "Training dataset"),
        ("model_rag_v2.pth", "Trained RAG model")
    ]
    
    for filepath, description in all_files:
        check_file(filepath, description)
    
    print("\n" + "="*60)
    print("🧪 NEXT STEPS:")
    print("="*60)
    print("\n1. Test the model:")
    print("   python test_rag_simple.py")
    print("\n2. Use in your application:")
    print("   from rag_answer_fixed import RAGAnswerGenerator")
    print("   generator = RAGAnswerGenerator()")
    print("   result = generator.generate('your question here')")
    print("\n3. Run evaluation:")
    print("   python rag_answer_fixed.py")
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)