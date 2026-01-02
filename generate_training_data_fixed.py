import json
import random

# =====================================================
# FIXED: Generate Training Data WITHOUT Retrieval
# =====================================================

def generate_rag_training_data(
    input_file="preprocessed.json",
    output_file="rag_training_data.json",
    samples_per_chunk=2,
    max_context_words=250,
    max_answer_words=120
):
    """
    Generate training data dari preprocessed chunks
    TANPA menggunakan retriever (karena belum di-build)
    """
    print("="*60)
    print("GENERATING RAG TRAINING DATA (FIXED)")
    print("="*60)
    
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    training_data = []
    
    # Template pertanyaan berdasarkan label
    question_templates = {
        'administrasi_akademik': [
            "Bagaimana cara {}?",
            "Apa langkah-langkah untuk {}?",
            "Jelaskan prosedur {}",
            "Dimana saya bisa melakukan {}?",
            "Siapa yang harus dihubungi untuk {}?"
        ],
        'keuangan': [
            "Bagaimana cara membayar {}?",
            "Apa saja syarat pembayaran {}?",
            "Berapa biaya untuk {}?",
            "Bagaimana prosedur angsuran {}?",
            "Dimana saya bisa membayar {}?"
        ],
        'perpustakaan': [
            "Bagaimana cara mengakses {}?",
            "Dimana saya bisa meminjam {}?",
            "Apa layanan {} yang tersedia?",
            "Bagaimana prosedur {}?",
            "Dimana lokasi {}?"
        ],
        'sertifikasi': [
            "Bagaimana cara mendaftar {}?",
            "Apa persyaratan untuk {}?",
            "Berapa biaya {}?",
            "Dimana lokasi tes {}?",
            "Kapan jadwal {}?"
        ],
        'sidang_yudisium': [
            "Apa syarat untuk {}?",
            "Bagaimana prosedur pendaftaran {}?",
            "Kapan jadwal {}?",
            "Dokumen apa saja yang diperlukan untuk {}?",
            "Dimana tempat {}?"
        ],
        'administrasi_lab': [
            "Bagaimana cara mengajukan {}?",
            "Apa syarat bebas {}?",
            "Dimana lokasi {}?",
            "Siapa yang harus saya hubungi untuk {}?",
            "Kapan saya bisa mengajukan {}?"
        ],
        'nilai': [
            "Bagaimana cara melihat {}?",
            "Dimana saya bisa mengakses {}?",
            "Apa syarat untuk mendapatkan {}?",
            "Bagaimana prosedur pengajuan {}?",
            "Kapan {} diumumkan?"
        ],
        'lainnya': [
            "Apa itu {}?",
            "Jelaskan tentang {}",
            "Informasi apa yang tersedia tentang {}?",
            "Bagaimana cara menggunakan {}?",
            "Dimana saya bisa menemukan {}?"
        ]
    }
    
    # Generate samples untuk setiap chunk
    for chunk in chunks:
        label = chunk.get('label', 'lainnya')
        text = chunk['text']
        filename = chunk.get('filename', 'dokumen')
        
        # Extract keywords dari filename dan text
        filename_clean = filename.replace('.pdf', '').replace('.txt', '').replace('_', ' ').lower()
        
        # Extract key phrases from text
        key_phrases = []
        keywords = ['login', 'pendaftaran', 'pembayaran', 'ujian', 'jadwal', 'krs', 'ukt', 'yudisium', 'bebas lab', 'cuti', 'perpustakaan', 'tep', 'sertifikat']
        for kw in keywords:
            if kw in text.lower():
                key_phrases.append(kw)
        
        if not key_phrases:
            key_phrases = [filename_clean]
        
        # Limit text length
        words = text.split()
        context = ' '.join(words[:max_context_words])
        answer = ' '.join(words[:max_answer_words])
        
        # Generate multiple questions per chunk
        templates = question_templates.get(label, question_templates['lainnya'])
        
        num_samples = min(samples_per_chunk, len(key_phrases))
        selected_phrases = random.sample(key_phrases, num_samples) if len(key_phrases) >= num_samples else key_phrases
        
        for phrase in selected_phrases:
            template = random.choice(templates)
            
            # Generate question
            if '{}' in template:
                question = template.format(phrase)
            else:
                question = f"{template} {phrase}"
            
            # Strategy 1: Oracle (perfect match)
            training_data.append({
                "question": question,
                "context": context,
                "answer": answer,
                "type": "oracle"
            })
    
    # Add negative samples (wrong context)
    num_negative = int(len(training_data) * 0.15)
    print(f"\nAdding {num_negative} negative samples...")
    
    all_contexts = [d['context'] for d in training_data]
    
    for _ in range(num_negative):
        target = random.choice(training_data)
        wrong_context = random.choice(all_contexts)
        
        # Ensure it's actually wrong
        attempts = 0
        while wrong_context == target['context'] and attempts < 10:
            wrong_context = random.choice(all_contexts)
            attempts += 1
        
        training_data.append({
            "question": target['question'],
            "context": wrong_context,
            "answer": "Maaf, informasi yang Anda cari tidak ditemukan dalam konteks ini. Silakan ajukan pertanyaan yang lebih spesifik atau coba kata kunci lain.",
            "type": "negative"
        })
    
    # Shuffle
    random.shuffle(training_data)
    
    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # Statistics
    print("\n" + "="*60)
    print("GENERATION COMPLETED")
    print("="*60)
    print(f"Total samples: {len(training_data)}")
    print(f"  - Oracle: {sum(1 for d in training_data if d['type'] == 'oracle')}")
    print(f"  - Negative: {sum(1 for d in training_data if d['type'] == 'negative')}")
    print(f"\nAverage lengths:")
    
    avg_q = sum(len(d['question'].split()) for d in training_data) / len(training_data)
    avg_c = sum(len(d['context'].split()) for d in training_data) / len(training_data)
    avg_a = sum(len(d['answer'].split()) for d in training_data) / len(training_data)
    
    print(f"  - Question: {avg_q:.1f} words")
    print(f"  - Context: {avg_c:.1f} words")
    print(f"  - Answer: {avg_a:.1f} words")
    
    print(f"\nSaved to: {output_file}")
    print("="*60)
    
    return training_data

# =====================================================
# VERIFY
# =====================================================

def verify_training_data(file_path="rag_training_data.json", num_samples=5):
    """
    Verify generated training data
    """
    print("\n" + "="*60)
    print("VERIFYING TRAINING DATA")
    print("="*60)
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Check required fields
    required_fields = ['question', 'context', 'answer']
    for field in required_fields:
        if not all(field in d for d in data):
            print(f"❌ ERROR: Missing field '{field}' in some samples!")
            return False
    
    print("✓ All required fields present")
    
    samples = random.sample(data, min(num_samples, len(data)))
    
    for i, sample in enumerate(samples, 1):
        print(f"\n[Sample {i}]")
        print(f"Type: {sample.get('type', 'N/A')}")
        print(f"Question: {sample['question']}")
        print(f"Context: {sample['context'][:100]}...")
        print(f"Answer: {sample['answer'][:100]}...")
        print(f"Q: {len(sample['question'].split())} words | "
              f"C: {len(sample['context'].split())} words | "
              f"A: {len(sample['answer'].split())} words")
        print("-"*60)
    
    return True

if __name__ == "__main__":
    # Generate
    data = generate_rag_training_data()
    
    # Verify
    if verify_training_data(num_samples=5):
        print("\n✅ Training data is ready for model training!")
    else:
        print("\n❌ Training data has issues, please fix!")