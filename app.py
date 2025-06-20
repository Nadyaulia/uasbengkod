import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import logging

st.title("Prediksi Kategori Obesitas")
st.write("Lengkapi data diri Anda untuk mengetahui kategori obesitas.")

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

@st.cache_resource
def load_model_and_scaler():
    try:
        saved = joblib.load("obesity_model.pkl")
        if isinstance(saved, dict):
            model = saved.get("model")
            scaler = saved.get("scaler")
            feature_names = saved.get("feature_names")
        else:
            # Jika model tanpa metadata
            model = saved
            scaler = joblib.load("scaler.pkl")  # Pastikan scaler.pkl ada
            feature_names = None

        if model is None or scaler is None:
            raise ValueError("Model atau scaler tidak berhasil dimuat.")

        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model/scaler: {e}")
        return None, None, None

# Tambahkan validasi di app.py
try:
    model, scaler, feature_names = load_model_and_scaler()
    st.success("Model dan scaler berhasil dimuat.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model/scaler: {e}")

# Input numerik
age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=25)
height = st.number_input("Tinggi Badan (meter)", min_value=0.5, max_value=2.5, value=1.7)
weight = st.number_input("Berat Badan (kg)", min_value=20, max_value=200, value=70)
fcvc = st.slider("Frekuensi makan sayur per minggu", min_value=0, max_value=10, value=2)
ncp = st.slider("Jumlah makan per hari", min_value=1, max_value=10, value=3)
ch2o = st.slider("Konsumsi air per hari (liter)", min_value=0, max_value=5, value=2)
faf = st.slider("Frekuensi aktivitas fisik per minggu", min_value=0, max_value=7, value=2)
tue = st.slider("Waktu layar per hari (jam)", min_value=0, max_value=5, value=2)

# Input kategorikal
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
favc = st.selectbox("Sering makan makanan tinggi kalori?", ["yes", "no"])
smoke = st.selectbox("Apakah Anda perokok?", ["yes", "no"])
calc = st.selectbox("Seberapa sering konsumsi alkohol?", ["no", "Sometimes", "Frequently", "Always"])
caec = st.selectbox("Seberapa sering ngemil di antara waktu makan?", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Jenis transportasi utama", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])
family_history = st.selectbox("Riwayat keluarga dengan obesitas?", ["yes", "no"])
scc = st.selectbox("Apakah Anda mencatat kalori yang dikonsumsi?", ["yes", "no"])

# Tombol prediksi
if st.button("Prediksi Sekarang"):
    # Validasi input
    if age < 1 or age > 120:
        st.error("Usia harus antara 1 hingga 120 tahun.")
    elif height < 0.5 or height > 2.5:
        st.error("Tinggi badan harus antara 0.5 hingga 2.5 meter.")
    elif weight < 20 or weight > 200:
        st.error("Berat badan harus antara 20 hingga 200 kg.")
    else:
        st.success("Input berhasil disimpan! Silakan lanjutkan ke proses prediksi.")

def preprocess_input(data):
    # Mapping kategorikal
    gender_map = {"Male": 0, "Female": 1}
    calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    favc_map = {"no": 0, "yes": 1}
    smoke_map = {"no": 0, "yes": 1}
    scc_map = {"no": 0, "yes": 1}
    caec_map = {"Sometimes": 0, "Frequently": 1, "Always": 2, "no": 3}
    mtrans_map = {
        "Public_Transportation": 0,
        "Automobile": 1,
        "Walking": 2,
        "Motorbike": 3,
        "Bike": 4
    }
    family_history_map = {"no": 0, "yes": 1}

    # Encode data kategorikal
    data['Gender'] = data['Gender'].map(gender_map).fillna(-1).astype(int)
    data['CALC'] = data['CALC'].map(calc_map).fillna(-1).astype(int)
    data['FAVC'] = data['FAVC'].map(favc_map).fillna(-1).astype(int)
    data['SCC'] = data['SCC'].map(scc_map).fillna(-1).astype(int)
    data['SMOKE'] = data['SMOKE'].map(smoke_map).fillna(-1).astype(int)
    data['family_history_with_overweight'] = data['family_history_with_overweight'].map(family_history_map).fillna(-1).astype(int)
    data['CAEC'] = data['CAEC'].map(caec_map).fillna(-1).astype(int)
    data['MTRANS'] = data['MTRANS'].map(mtrans_map).fillna(-1).astype(int)

    # Daftar fitur numerik yang harus dinormalisasi
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

    # Normalisasi fitur numerik
    if scaler is None:
        raise ValueError("Scaler belum dimuat. Tidak bisa melakukan transformasi.")

    data = scaler.transform(data)

    return data

if st.button("Lihat Hasil Prediksi"):
    # Daftar kolom sesuai saat model dilatih
    EXPECTED_COLUMNS = [
        'Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC',
        'NCP', 'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight',
        'FAF', 'TUE', 'CAEC', 'MTRANS'
    ]

    # Buat DataFrame dari input pengguna
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Height': [height],
        'Weight': [weight],
        'CALC': [calc],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'SCC': [scc],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'family_history_with_overweight': [family_history],
        'FAF': [faf],
        'TUE': [tue],
        'CAEC': [caec],
        'MTRANS': [mtrans]
    })

    # Pastikan urutan kolom benar
    input_data = input_data[EXPECTED_COLUMNS]

    # Debugging: Tampilkan struktur data
    st.write("Data input awal:")
    st.write(input_data)

    # Proses input
    try:
        processed_data = preprocess_input(input_data.copy())
        st.write("Data setelah preprocessing:")
        st.write(processed_data)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat preprocessing data: {e}")
        st.stop()

    # Lakukan prediksi
    try:
        prediction = model.predict(processed_data)[0]
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.stop()

    # Decode hasil prediksi
    categories = {
        0: "Underweight",
        1: "Normal Weight",
        2: "Overweight Level I",
        3: "Overweight Level II",
        4: "Obesity Type I",
        5: "Obesity Type II",
        6: "Obesity Type III"
    }
    result = categories.get(prediction, "Tidak Diketahui")

    # Tampilkan hasil
    st.success(f"Prediksi Kategori Obesitas: {result}")
