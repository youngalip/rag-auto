import json
import random

# ===== HIGH QUALITY TRAINING DATA GENERATION =====
def generate_high_quality_training_data(
    input_file="preprocessed.json",
    output_file="rag_training_data.json"
):
    """
    FIXED: Generate MUCH higher quality training data
    - Focus on ORACLE samples (exact context = answer)
    - Better question templates
    - NO retrieval noise
    - Very few negative samples
    """
    print("=" * 60)
    print("GENERATING HIGH QUALITY TRAINING DATA")
    print("=" * 60)

    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Better question templates per label
    question_templates = {
        "administrasi_akademik": [
            "Bagaimana cara melakukan {}?",
            "Langkah-langkah {} adalah?",
            "Prosedur untuk {} bagaimana?",
            "Cara mengakses {} gimana?",
            "Tutorial {} yang benar?",
            "Panduan {} lengkap?"
        ],
        "keuangan": [
            "Bagaimana cara membayar {}?",
            "Prosedur pembayaran {} gimana?",
            "Cara {} lewat bank?",
            "Tutorial bayar {} lengkap?",
            "Langkah-langkah {} adalah?"
        ],
        "perpustakaan": [
            "Cara menggunakan {} gimana?",
            "Bagaimana mengakses {}?",
            "Tutorial {} lengkap?",
            "Prosedur {} adalah?"
        ],
        "sidang_yudisium": [
            "Kapan jadwal {}?",
            "Syarat {} apa saja?",
            "Prosedur pendaftaran {} bagaimana?",
            "Cara daftar {} gimana?"
        ],
        "administrasi_lab": [
            "Bagaimana cara mengajukan {}?",
            "Prosedur {} lengkap?",
            "Syarat {} apa saja?",
            "Tutorial {} yang benar?"
        ],
        "sertifikasi": [
            "Cara daftar {} gimana?",
            "Prosedur pendaftaran {} adalah?",
            "Syarat {} apa saja?",
            "Tutorial {} lengkap?"
        ],
        "nilai": [
            "Syarat {} apa saja?",
            "Persyaratan {} adalah?",
            "Cara mengajukan {} bagaimana?",
            "IPK minimal untuk {} berapa?"
        ],
        "lainnya": [
            "Informasi tentang {} apa saja?",
            "Jelaskan tentang {}",
            "Apa itu {}?",
            "Detail tentang {} gimana?"
        ]
    }

    # Extract key terms from text
    def extract_key_terms(text, label):
        text_lower = text.lower()
        
        key_terms = {
            "administrasi_akademik": ["krs", "khs", "jadwal kuliah", "bimbingan dpa", "sinau digital"],
            "keuangan": ["ukt", "pembayaran", "angsuran", "virtual account"],
            "perpustakaan": ["perpustakaan", "opac", "kartu perpustakaan"],
            "sidang_yudisium": ["yudisium", "wisuda", "sidang"],
            "administrasi_lab": ["bebas lab"],
            "sertifikasi": ["tep", "lsp", "sertifikat"],
            "nilai": ["fast track", "ipk", "spk"],
            "lainnya": ["dokumen", "informasi"]
        }
        
        for term in key_terms.get(label, []):
            if term in text_lower:
                return term
        
        return label.replace("_", " ")

    training_data = []

    # ===== STRATEGY: 95% ORACLE + 5% NEGATIVE =====
    print("Generating ORACLE samples (exact match)...")
    
    # Oracle samples - exact context match
    for item in raw_data:
        label = item['label']
        text = item['text']
        
        # Extract key term
        key_term = extract_key_terms(text, label)
        
        # Generate 2-3 questions per document
        templates = question_templates.get(label, question_templates["lainnya"])
        
        for _ in range(2):  # 2 questions per doc
            template = random.choice(templates)
            
            try:
                question = template.format(key_term)
            except:
                question = f"Informasi tentang {key_term}?"
            
            training_data.append({
                "question": question,
                "context": text,
                "answer": text,
                "type": "oracle",
                "label": label
            })

    # Small number of negative samples (for robustness)
    print("Generating NEGATIVE samples...")
    
    all_texts = [d['text'] for d in raw_data]
    num_negative = int(len(training_data) * 0.05)  # Only 5%
    
    for _ in range(num_negative):
        target = random.choice(training_data)
        wrong_context = random.choice(all_texts)
        
        while wrong_context == target['answer']:
            wrong_context = random.choice(all_texts)

        training_data.append({
            "question": target['question'],
            "context": wrong_context[:300],
            "answer": "Maaf, informasi tidak ditemukan dalam dokumen.",
            "type": "negative",
            "label": "negative"
        })

    # Shuffle
    random.shuffle(training_data)
    
    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    # Stats
    print("\n" + "=" * 60)
    print("GENERATION COMPLETED")
    print("=" * 60)
    print(f"Total samples: {len(training_data)}")
    
    type_counts = {}
    for item in training_data:
        t = item['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print(f"\nType Distribution:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t:15s}: {count:5d} ({count/len(training_data)*100:.1f}%)")
    
    label_counts = {}
    for item in training_data:
        if item['type'] == 'oracle':
            l = item['label']
            label_counts[l] = label_counts.get(l, 0) + 1
    
    print(f"\nLabel Distribution (Oracle only):")
    for l, count in sorted(label_counts.items()):
        print(f"  {l:30s}: {count:4d} ({count/sum(label_counts.values())*100:.1f}%)")
    
    print(f"\nSaved to: {output_file}")
    print("=" * 60)

    return training_data

def verify_data(file_path="rag_training_data.json", num_samples=10):
    """Verify generated data"""
    print("\n" + "=" * 60)
    print("VERIFYING TRAINING DATA")
    print("=" * 60)
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = random.sample(data, min(num_samples, len(data)))
    
    for i, sample in enumerate(samples, 1):
        print(f"\n[Sample {i}]")
        print(f"Type     : {sample['type']}")
        print(f"Label    : {sample.get('label', 'N/A')}")
        print(f"Question : {sample['question']}")
        print(f"Context  : {sample['context'][:100]}...")
        print(f"Answer   : {sample['answer'][:100]}...")
        print("-" * 60)

if __name__ == "__main__":
    generate_high_quality_training_data()
    verify_data()