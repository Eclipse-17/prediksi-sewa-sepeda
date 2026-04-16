import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model_xgb.pkl')
training_columns = joblib.load('columns.pkl')

st.title("Prediksi Penyewaan Sepeda")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Input:", df.head())

    # preprocessing sederhana (harus sama kayak training)
    categorical_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']

    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Samakan kolom
    for col in training_columns:
        if col not in df_processed:
            df_processed[col] = 0

    df_processed = df_processed[training_columns]

    # Prediksi
    predictions = model.predict(df_processed)

    df['prediksi'] = predictions
    st.write("Hasil Prediksi:", df)

    # download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download hasil", csv, "hasil_prediksi.csv", "text/csv")