import csv
import random

data = []

with open('student-mat.csv', 'r') as file:
    reader = csv.DictReader(file, delimiter=';')
    for row in reader:
        data.append(row)

with open('student-por.csv', 'r') as file:
    reader = csv.DictReader(file, delimiter=';')
    for row in reader:
        data.append(row)

X = []
y = []
for row in data:
    fitur = [
        float(row['absences']),
        float(row['G1']),
        float(row['G2']),
        float(row['traveltime']),
        float(row['failures']),
        float(row['studytime'])
    ]
    X.append(fitur)
    label = 1 if float(row['G3']) < 10 else 0
    y.append(label)

n_fitur = len(X[0])
mins = [min(X[i][j] for i in range(len(X))) for j in range(n_fitur)]
maxs = [max(X[i][j] for i in range(len(X))) for j in range(n_fitur)]

for i in range(len(X)):
    for j in range(n_fitur):
        if maxs[j] > mins[j]:
            X[i][j] = (X[i][j] - mins[j]) / (maxs[j] - mins[j])
        else:
            X[i][j] = 0.0

dataset = list(zip(X, y))
random.seed(42)
random.shuffle(dataset)

split_idx = int(0.8 * len(dataset))
train_data = dataset[:split_idx]
test_data  = dataset[split_idx:]

X_train = [item[0] for item in train_data]
y_train = [item[1] for item in train_data]
X_test  = [item[0] for item in test_data]
y_test  = [item[1] for item in test_data]

def manhattan(x1, x2):
    return sum(abs(a - b) for a, b in zip(x1, x2))

def knn(X_train, y_train, x_baru, k=5):
    dists = sorted([(manhattan(x_baru, xt), yt) for xt, yt in zip(X_train, y_train)])
    labels = [l for _, l in dists[:k]]
    proba = sum(labels) / k
    pred  = 1 if proba >= 0.5 else 0
    return pred

benar = 0
for i in range(len(X_test)):
    prediksi = knn(X_train, y_train, X_test[i], k=5)
    if prediksi == y_test[i]:
        benar += 1

akurasi = benar / len(X_test)
print(f"Akurasi: {akurasi * 100:.2f}%")
# Menyiapkan papan skor Confusion Matrix
TP = 0  # True Positive  (Ditebak Gagal, Aslinya Gagal)
TN = 0  # True Negative  (Ditebak Lulus, Aslinya Lulus)
FP = 0  # False Positive (Ditebak Gagal, Aslinya Lulus) -> "Tuduhan Palsu"
FN = 0  # False Negative (Ditebak Lulus, Aslinya Gagal) -> "Kecolongan"

for i in range(len(X_test)):
    prediksi = knn(X_train, y_train, X_test[i], k=5)
    aktual = y_test[i]
    
    # Memilah tebakan ke dalam 4 kategori
    if prediksi == 1 and aktual == 1:
        TP += 1
    elif prediksi == 0 and aktual == 0:
        TN += 1
    elif prediksi == 1 and aktual == 0:
        FP += 1
    elif prediksi == 0 and aktual == 1:
        FN += 1

# Menghitung Metrik Evaluasi (dengan pengaman pembagian nol)
total_data = len(X_test)
akurasi = (TP + TN) / total_data

# Presisi: Dari semua yang kita cap "Gagal", berapa persen yang benar-benar Gagal?
presisi = TP / (TP + FP) if (TP + FP) > 0 else 0.0

# Recall: Dari semua anak yang aslinya "Gagal", berapa persen yang berhasil kita selamatkan/deteksi?
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

# F1-Score: Nilai keseimbangan antara Presisi dan Recall
f1_score = 2 * (presisi * recall) / (presisi + recall) if (presisi + recall) > 0 else 0.0

# Menampilkan Hasil
print("=== HASIL EVALUASI KINERJA MODEL ===")
print(f"Total Data Uji : {total_data} Siswa")
print(f"Akurasi        : {akurasi * 100:.2f}%")
print(f"Presisi        : {presisi * 100:.2f}%")
print(f"Recall         : {recall * 100:.2f}%")
print(f"F1-Score       : {f1_score * 100:.2f}%")
print("====================================")