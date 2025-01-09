import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# Memuat model dan scaler
model_filename = 'mlp_fish_model.pkl'
scaler_filename = 'scaler_fish_data.pkl'

mlp_model = load(model_filename)
scaler = load(scaler_filename)

st.title("Prediksi Spesies Ikan Menggunakan Model Perceptron")

length = st.number_input('Panjang Ikan (cm)', min_value=0.0, step=0.1)
weight = st.number_input('Berat Ikan (gram)', min_value=0.0, step=0.1)
w_l_ratio = st.number_input('Rasio Berat dan Panjang (w_l_ratio)', min_value=0.0, step=0.01)

if st.button('Prediksi Spesies'):
    # Membuat DataFrame dari input pengguna
    input_data = pd.DataFrame([[length, weight, w_l_ratio]], columns=['length', 'weight', 'w_l_ratio'])
    
    # Menstandarisasi input data menggunakan scaler
    input_scaled = scaler.transform(input_data)
    
    # Melakukan prediksi dengan model MLP
    prediction = mlp_model.predict(input_scaled)
    
    # Menampilkan hasil prediksi
    st.write(f"Prediksi Spesies Ikan: {prediction[0]}")
