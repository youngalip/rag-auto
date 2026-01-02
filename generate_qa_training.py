import json
import re
import random
from collections import defaultdict

# =====================================================
# SMART Q&A GENERATION FROM PREPROCESSED DOCUMENTS
# This replaces generate_better_training_data.py
# =====================================================

class SmartQAGenerator:
    """
    Generate Q&A pairs from labeled & preprocessed documents
    """
    
    def __init__(self):
        # Natural question templates per label
        self.templates = {
            "administrasi_lab": {
                "action": "mengajukan bebas lab",
                "procedures": [
                    "Bagaimana cara mengajukan bebas lab?",
                    "Apa langkah-langkah mengajukan bebas lab?",
                    "Prosedur pengajuan bebas lab bagaimana?",
                    "Gimana cara mengajukan bebas lab?"
                ],
                "requirements": [
                    "Apa syarat mengajukan bebas lab?",
                    "Persyaratan bebas lab apa saja?"
                ],
                "info": [
                    "Siapa yang verifikasi bebas lab?",
                    "Gimana cara memantau status bebas lab?"
                ]
            },
            "administrasi_akademik": {
                "action": "mengajukan cuti kuliah",
                "procedures": [
                    "Bagaimana cara mengajukan cuti kuliah?",
                    "Prosedur pengajuan cuti kuliah?",
                    "Langkah-langkah cuti kuliah apa saja?"
                ],
                "requirements": [
                    "Syarat cuti kuliah apa saja?",
                    "Dokumen apa yang diperlukan untuk cuti?"
                ]
            },
            "keuangan": {
                "action": "membayar UKT",
                "procedures": [
                    "Bagaimana cara membayar UKT?",
                    "Cara bayar UKT gimana?",
                    "Prosedur pembayaran UKT?"
                ],
                "info": [
                    "Dimana bisa bayar UKT?",
                    "Bank apa saja yang bisa untuk bayar UKT?",
                    "Apa itu virtual account UKT?"
                ]
            },
            "sidang_yudisium": {
                "action": "mendaftar yudisium",
                "procedures": [
                    "Bagaimana cara mendaftar yudisium?",
                    "Prosedur pendaftaran yudisium?"
                ],
                "requirements": [
                    "Apa saja syarat yudisium?",
                    "Persyaratan yudisium apa saja?",
                    "Dokumen apa yang diperlukan untuk yudisium?"
                ]
            },
            "perpustakaan": {
                "action": "menggunakan perpustakaan",
                "procedures": [
                    "Bagaimana cara meminjam buku di perpustakaan?",
                    "Prosedur peminjaman buku?"
                ],
                "info": [
                    "Cara akses OPAC perpustakaan?",
                    "Gimana cara menggunakan perpustakaan digital?"
                ]
            }
        }
    
    def extract_procedural_steps(self, text):
        """Extract numbered steps from text"""
        # Pattern for numbered lists
        pattern = r'(\d+[\.\)])\s+([^\n]+(?:\n(?!\d+[\.\)]).*?)*)'
        matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
        
        steps = []
        for match in matches:
            num = match.group(1)
            step = match.group(2).strip()
            
            # Clean
            step = re.sub(r'\s+', ' ', step)
            step = step.replace('\n', ' ')
            
            if 15 < len(step) < 300:
                steps.append(f"{num} {step}")
        
        if len(steps) >= 2:  # At least 2 steps
            return " ".join(steps[:8])  # Max 8 steps
        
        return None
    
    def extract_requirements(self, text):
        """Extract requirements/syarat"""
        patterns = [
            r'(?:syarat|persyaratan)[:\s]+([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\Z)',
            r'(?:harus|wajib)[:\s]+([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                req = match.group(1).strip()
                req = re.sub(r'\s+', ' ', req)
                
                if 30 < len(req) < 500:
                    return req
        
        return None
    
    def has_procedural_content(self, text):
        """Check if text contains procedural steps"""
        # Check for numbered lists
        if re.search(r'\d+[\.\)]\s', text):
            return True
        
        # Check for procedural keywords
        keywords = ['langkah', 'cara', 'prosedur', 'tahap', 'login', 'klik']
        return any(kw in text.lower() for kw in keywords)
    
    def has_requirements_content(self, text):
        """Check if text contains requirements"""
        keywords = ['syarat', 'persyaratan', 'harus', 'wajib', 'diperlukan']
        return any(kw in text.lower() for kw in keywords)
    
    def generate_qa_from_chunk(self, chunk):
        """Generate Q&A pairs from a single chunk"""
        text = chunk['text']
        label = chunk['label']
        
        if label not in self.templates:
            return []
        
        template_data = self.templates[label]
        qa_pairs = []
        
        # Try to extract procedural steps
        if self.has_procedural_content(text):
            steps = self.extract_procedural_steps(text)
            
            if steps:
                # Generate Q&A for procedure questions
                for question in template_data.get('procedures', []):
                    qa_pairs.append({
                        "question": question,
                        "answer": steps,
                        "label": label,
                        "type": "procedure"
                    })
        
        # Try to extract requirements
        if self.has_requirements_content(text):
            requirements = self.extract_requirements(text)
            
            if requirements:
                # Generate Q&A for requirements questions
                for question in template_data.get('requirements', []):
                    qa_pairs.append({
                        "question": question,
                        "answer": requirements,
                        "label": label,
                        "type": "requirements"
                    })
        
        # If no structured extraction, use chunk text as-is for info questions
        if not qa_pairs and len(text) > 50:
            for question in template_data.get('info', [])[:1]:  # Just 1 info question
                qa_pairs.append({
                    "question": question,
                    "answer": text,
                    "label": label,
                    "type": "info"
                })
        
        return qa_pairs

def generate_rag_training_data(
    input_file="preprocessed.json",
    output_file="rag_training_data.json"
):
    """
    Main function: Generate RAG training data from preprocessed documents
    """
    print("="*60)
    print("GENERATING RAG TRAINING DATA")
    print("="*60)
    
    # Load preprocessed data
    with open(input_file, 'r', encoding='utf-8') as f:
        preprocessed = json.load(f)
    
    print(f"Loaded {len(preprocessed)} chunks")
    
    # Initialize generator
    generator = SmartQAGenerator()
    
    # Generate Q&A pairs
    all_qa = []
    label_counts = defaultdict(int)
    type_counts = defaultdict(int)
    
    for chunk in preprocessed:
        qa_pairs = generator.generate_qa_from_chunk(chunk)
        
        for qa in qa_pairs:
            all_qa.append(qa)
            label_counts[qa['label']] += 1
            type_counts[qa['type']] += 1
    
    print(f"\nGenerated {len(all_qa)} Q&A pairs")
    
    # Check if enough data
    if len(all_qa) < 200:
        print(f"\n⚠️ WARNING: Only {len(all_qa)} Q&A pairs generated")
        print("   This may not be enough for good training")
        print("   Suggestions:")
        print("   - Add more documents")
        print("   - Ensure documents have structured content (numbered steps, etc)")
        print("   - Consider data augmentation")
    
    # Deduplicate Q&A pairs
    seen = set()
    unique_qa = []
    
    for qa in all_qa:
        key = (qa['question'].lower().strip(), qa['answer'][:100].lower().strip())
        if key not in seen:
            seen.add(key)
            unique_qa.append(qa)
    
    print(f"After deduplication: {len(unique_qa)} unique pairs")
    
    # Create RAG format training data
    # Format: question <sep> context → answer
    rag_training = []
    
    # Strategy 1: Oracle samples (context = answer)
    for qa in unique_qa:
        rag_training.append({
            "question": qa['question'],
            "context": qa['answer'],  # Oracle: context is the answer
            "answer": qa['answer'],
            "type": "oracle",
            "label": qa['label']
        })
    
    # Strategy 2: Add some variations with partial context
    for qa in random.sample(unique_qa, min(len(unique_qa) // 3, 100)):
        # Use first half of answer as context
        words = qa['answer'].split()
        if len(words) > 20:
            half = len(words) // 2
            partial_context = " ".join(words[:half])
            
            rag_training.append({
                "question": qa['question'],
                "context": partial_context,
                "answer": qa['answer'],
                "type": "partial",
                "label": qa['label']
            })
    
    # Shuffle
    random.shuffle(rag_training)
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rag_training, f, indent=2, ensure_ascii=False)
    
    # Statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Total training samples: {len(rag_training)}")
    
    type_dist = defaultdict(int)
    for item in rag_training:
        type_dist[item['type']] += 1
    
    print(f"\nType distribution:")
    for t, count in sorted(type_dist.items()):
        print(f"  {t:15s}: {count:4d} ({count/len(rag_training)*100:.1f}%)")
    
    print(f"\nLabel distribution (original Q&A):")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:30s}: {count:4d}")
    
    print(f"\nQ&A type distribution (original):")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype:15s}: {count:4d}")
    
    print(f"\nSaved to: {output_file}")
    print("="*60)
    
    # Show samples
    print("\n--- Sample Training Data ---")
    for i, sample in enumerate(random.sample(rag_training, min(5, len(rag_training))), 1):
        print(f"\n[Sample {i}] Type: {sample['type']}, Label: {sample['label']}")
        print(f"Question: {sample['question']}")
        print(f"Context : {sample['context'][:100]}...")
        print(f"Answer  : {sample['answer'][:100]}...")
    
    return rag_training

if __name__ == "__main__":
    generate_rag_training_data()