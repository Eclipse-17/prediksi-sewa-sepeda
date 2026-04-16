import streamlit as st
import pandas as pd
import joblib

# Config halaman
st.set_page_config(
    page_title="Prediksi Sepeda",
    page_icon="🚲",
    layout="wide"
)

# Load model
model = joblib.load('model_xgb.pkl')
training_columns = joblib.load('columns.pkl')

# Header
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>🚲 Prediksi Penyewaan Sepeda</h1>
    <p style='text-align: center;'>Upload data dan lihat prediksi secara instan</p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

# Layout utama
col1, col2 = st.columns(2)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    with col1:
        st.subheader("📄 Data Input")
        st.dataframe(df.head(), use_container_width=True)

    # preprocessing
    categorical_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    for col in training_columns:
        if col not in df_processed:
            df_processed[col] = 0

    df_processed = df_processed[training_columns]

    # prediksi
    predictions = model.predict(df_processed)
    df['prediksi'] = predictions

    with col2:
        st.subheader("📊 Hasil Prediksi")
        st.dataframe(df, use_container_width=True)

    # statistik ringkas
    st.markdown("### 📈 Ringkasan Prediksi")
    st.metric("Rata-rata", round(df['prediksi'].mean(), 2))
    st.metric("Maksimum", round(df['prediksi'].max(), 2))
    st.metric("Minimum", round(df['prediksi'].min(), 2))

    # download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Hasil",
        csv,
        "hasil_prediksi.csv",
        "text/csv"
    )

else:
    st.info("Silakan upload file CSV terlebih dahulu di sidebar.")
