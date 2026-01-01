import json

lexicon = {
    "administrasi_lab": [
        "bebas lab", "laboratorium", "praktikum", "kalab", "kasublab"
    ],
    "nilai": [
        "khs", "transkrip", "ipk", "nilai", "ujian", "remedial"
    ],
    "administrasi_akademik": [
        "cuti kuliah", "krs", "registrasi", "pembatalan", "jadwal kuliah",
        "siakadu", "siani", "profil program studi", "melisa", "simlppm"
    ],
    "keuangan": [
        "ukt", "beasiswa", "pembayaran", "biaya", "angsuran", "pembiayaan"
    ],
    "perpustakaan": [
        "perpustakaan", "kartu perpustakaan", "opac", "e-library", "jurnal", "open journal"
    ],
    "sertifikasi": [
        "sertifikat", "sertifikasi", "lsp", "tep", "tes", "ujian bahasa",
        "sertifikat profesi", "sertifikat kompetensi"
    ],
    "sidang_yudisium": [
        "yudisium", "syarat yudisium", "wisuda", "skp", "skripsi", "tugas akhir"
    ],
    "informasi_umum": [
        "kalender akademik", "panduan", "tutorial", "manual pengguna", "user manual"
    ]
}

def assign_label(text):
    text_low = text.lower()
    scores = {label: 0 for label in lexicon}

    for label, keywords in lexicon.items():
        for kw in keywords:
            if kw in text_low:
                scores[label] += 1

    best_label = max(scores, key=scores.get)

    if scores[best_label] == 0:
        return "lainnya"

    return best_label


# =====================================================
# LABELING dari extracted.json → labeled.json
# =====================================================

def labeling(input_file="extracted.json", output_file="labeled.json"):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    labeled = []

    for item in data:
        labeled.append({
            "filename": item["filename"],
            "text": item["text"],
            "label": assign_label(item["text"])
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)

    print(f"Labeling selesai hasil disimpan di {output_file}")


if __name__ == "__main__":
    labeling()
