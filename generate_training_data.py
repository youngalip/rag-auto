import json
import random
from retrieval_query import ImprovedRetriever


# =====================================================
# GENERATE TRAINING DATA FOR RAG MODEL (ADAPTED)
# =====================================================

def generate_rag_training_data(
    input_file="labeled.json",
    output_file="rag_training_data.json",
    use_retrieval=True,
    negative_samples_ratio=0.1
):
    print("="*60)
    print("GENERATING RAG DATA (WITHOUT LABELS)")
    print("="*60)

    # 1. Load data
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 2. Inisialisasi Retriever
    retriever = None
    if use_retrieval and ImprovedRetriever is not None:
        try:
            retriever = ImprovedRetriever()
        except Exception as e:
            print(f"Gagal memuat retriever: {e}")
            use_retrieval = False

    training_data = []

    # 3. Proses Data
    for item in raw_data:
        # Label tetap digunakan untuk membuat pertanyaan, tapi TIDAK disimpan ke JSON
        label_clean = item['label'].replace("_", " ")
        
        templates = [
            f"Bagaimana informasi terkait {label_clean}?",
            f"Jelaskan mengenai {label_clean} dalam dokumen ini.",
            f"Apa isi dari {item['filename']}?",
            f"Informasi apa yang tersedia tentang {label_clean}?"
        ]
        question = random.choice(templates)
        
        # --- STRATEGY 1: Oracle ---
        training_data.append({
            "question": question,
            "context": item['text'],
            "answer": item['text'],
            "type": "oracle"
        })

        # --- STRATEGY 2: Realistic (Dengan Retrieval) ---
        if use_retrieval and retriever:
            try:
                results = retriever.retrieve(question, top_k=1)
                if results:
                    training_data.append({
                        "question": question,
                        "context": results[0]['text'],
                        "answer": item['text'],
                        "type": "retrieved"
                    })
            except: pass

    # --- STRATEGY 3: Negative (Salah Konteks) ---
    all_texts = [d['text'] for d in raw_data]
    num_negative = int(len(training_data) * negative_samples_ratio)
    
    for _ in range(num_negative):
        target = random.choice(training_data)
        wrong_context = random.choice(all_texts)
        
        while wrong_context == target['answer']:
            wrong_context = random.choice(all_texts)

        training_data.append({
            "question": target['question'],
            "context": wrong_context[:1000],
            "answer": "Maaf, informasi tersebut tidak ditemukan dalam dokumen yang diberikan.",
            "type": "negative"
        })

    # Simpan ke File
    random.shuffle(training_data)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"Berhasil membuat {len(training_data)} sampel di {output_file}")
    return training_data

# =====================================================
# VERIFY DATA
# =====================================================

def verify_training_data(file_path="rag_training_data.json", num_samples=3):
    print("\n" + "="*60)
    print("VERIFIKASI DATA (TANPA LABEL)")
    print("="*60)
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = random.sample(data, min(num_samples, len(data)))
    
    for i, sample in enumerate(samples, 1):
        print(f"\n[Sample {i} - Type: {sample['type']}]")
        print(f"Question : {sample['question']}")
        print(f"Context  : {sample['context'][:100]}...")
        print(f"Answer   : {sample['answer'][:100]}...")
        print("-" * 40)

if __name__ == "__main__":
    generate_rag_training_data()
    verify_training_data()