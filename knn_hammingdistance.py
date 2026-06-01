import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print("PREDIKSI RISIKO PUTUS SEKOLAH - KNN HAMMING")

df = pd.read_csv("student-mat.csv", sep=';')
print(f"\nData dimuat: {len(df)} siswa")

df['risiko'] = df['G3'].apply(
    lambda x: "TINGGI" if x < 10 else ("SEDANG" if x < 15 else "RENDAH")
)

print(f"\nRisiko:")
print(df['risiko'].value_counts().to_string())

df_encoded = df.copy()

for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object' and col not in ['risiko']:
        enc = LabelEncoder()
        df_encoded[col] = enc.fit_transform(df_encoded[col].astype(str))


df_encoded = df_encoded.drop(['G1', 'G2', 'G3'], axis=1)

print(f"\ndata siap: {df_encoded.shape[1]-1} fitur")

X = df_encoded.drop('risiko', axis=1)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)
    
y = df_encoded['risiko']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining: {len(X_train)} siswa | Testing: {len(X_test)} siswa")

model = KNeighborsClassifier(n_neighbors=5, metric='hamming')
model.fit(X_train, y_train)

print("\nModel KNN (K=5) dilatih")

y_pred = model.predict(X_test)

akurasi = accuracy_score(y_test, y_pred)
print(f"\nAKURASI: {akurasi:.1%}")

y_semua = model.predict(X)

print(f"\nHASIL IDENTIFIKASI ({len(df)} siswa):")
for risiko in ['TINGGI', 'SEDANG', 'RENDAH']:
    jumlah = (y_semua == risiko).sum()
    persen = 100 * jumlah / len(df)
    print(f"   Risiko {risiko}: {jumlah} siswa ({persen:.1f}%)")

print("\n selesai \n")
