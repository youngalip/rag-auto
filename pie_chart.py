import json
from collections import Counter
import matplotlib.pyplot as plt

# === 1. Membaca file JSON (hasil preprocessing otomatis) ===
json_path = "hasil_preprocessing_otomatis.json"  # ganti sesuai nama file kamu

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# === 2. Mengambil daftar label ===
labels = [item["label"] for item in data]

# === 3. Hitung jumlah masing-masing label ===
label_count = Counter(labels)

# === 4. Membuat Pie Chart ===
plt.figure(figsize=(8, 8))
plt.pie(
    label_count.values(),
    labels=label_count.keys(),
    autopct='%1.1f%%',
    startangle=140
)
plt.title("Distribusi Label Otomatis")
plt.show()
