import json

# ===== IMPROVED LEXICON WITH WEIGHTED SCORING =====
lexicon = {
    "administrasi_lab": {
        "strong": ["bebas lab", "laboratorium", "kalab", "kasublab", "teknisi lab", 
                   "pengajuan lab", "verifikasi lab"],
        "medium": ["praktikum"],
        "exclude": ["journal", "jurnal", "publication"],  # Exclude journal-related
        "weight": 3.5
    },
    "nilai": {
        "strong": ["ipk", "transkrip nilai", "fast track", "spk", "surat penetapan kelulusan",
                   "syarat spk", "persyaratan spk"],
        "medium": ["khs", "kartu hasil studi", "nilai semester"],
        "exclude": ["journal", "jurnal", "open journal"],
        "weight": 3.0
    },
    "administrasi_akademik": {
        "strong": ["krs", "kartu rencana studi", "jadwal kuliah", "bimbingan dpa", 
                   "sinau digital", "sidia", "cuti kuliah", "siakadu", "pengajuan cuti",
                   "registrasi", "daftar ulang", "kepenasihatan"],
        "medium": ["semester", "perkuliahan", "melisa", "siani", "mata kuliah"],
        "exclude": ["journal", "jurnal", "publication", "research"],
        "weight": 2.8
    },
    "keuangan": {
        "strong": ["ukt", "pembayaran ukt", "angsuran ukt", "penurunan ukt", "pembebasan ukt",
                   "virtual account", "simukt", "btn", "tagihan ukt", "bayar ukt"],
        "medium": ["pembayaran", "biaya kuliah", "beasiswa", "atm", "teller", "transfer"],
        "exclude": ["journal", "jurnal"],
        "weight": 3.5
    },
    "perpustakaan": {
        "strong": ["perpustakaan unesa", "perpus unesa", "opac integrated", 
                   "kartu perpustakaan elektronik", "peminjaman buku", "pengembalian buku",
                   "lobby perpus", "orientasi perpustakaan"],
        "medium": ["katalog buku", "koleksi perpustakaan"],
        "exclude": ["open journal", "journal system", "e-journal", "publication"],
        "weight": 3.0
    },
    "sertifikasi": {
        "strong": ["lsp", "lembaga sertifikasi profesi", "tep", "toefl", "ielts", 
                   "pendaftaran tep", "sertifikat tep", "tes bahasa inggris"],
        "medium": ["sertifikat kompetensi", "ujian bahasa"],
        "exclude": ["journal", "jurnal"],
        "weight": 3.0
    },
    "sidang_yudisium": {
        "strong": ["yudisium", "wisuda", "pendaftaran yudisium", "syarat yudisium",
                   "kalender akademik", "jadwal yudisium", "toga"],
        "medium": ["kelulusan", "sidang akhir"],
        "exclude": ["journal", "jurnal"],
        "weight": 3.0
    }
}

def assign_label(text, filename=""):
    """
    FIXED: Better labeling with multi-level scoring
    """
    text_low = text.lower()
    filename_low = filename.lower()
    combined = text_low + " " + filename_low
    
    scores = {}
    
    for label, data in lexicon.items():
        score = 0
        
        # Strong keywords (weight: 5)
        for kw in data.get("strong", []):
            if kw in combined:
                score += 5 * data["weight"]
        
        # Medium keywords (weight: 2)
        for kw in data.get("medium", []):
            if kw in combined:
                score += 2 * data["weight"]
        
        scores[label] = score
    
    # Get best label
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    
    # Threshold: at least one keyword match
    if best_score >= 5:  # At least one strong keyword
        return best_label
    elif best_score >= 2:  # Some medium keywords
        return best_label
    
    return "lainnya"

def labeling(input_file="extracted.json", output_file="labeled.json"):
    """
    FIXED: Labeling with filename consideration
    """
    print("=" * 60)
    print("AUTOMATIC LABELING")
    print("=" * 60)
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} documents...")
    
    labeled = []
    label_counts = {}
    
    for item in data:
        # Use both text and filename for better labeling
        label = assign_label(item["text"], item.get("filename", ""))
        
        labeled.append({
            "filename": item["filename"],
            "text": item["text"],
            "label": label
        })
        
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)
    
    # Stats
    print("\n" + "=" * 60)
    print("LABELING COMPLETED")
    print("=" * 60)
    print(f"Total documents: {len(labeled)}")
    print(f"\nLabel Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:30s}: {count:4d} ({count/len(labeled)*100:.1f}%)")
    print(f"\nResult saved to: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    labeling()