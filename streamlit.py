import streamlit as st
import pandas as pd
import joblib

# Modeli yükle
model = joblib.load('lgbm_model_final.pkl')

# Başlık
st.title('Obezite Seviyesi Tahmin Uygulaması')

# Kullanıcı girişi için form
with st.form("my_form"):
    st.write("Lütfen bilgilerinizi girin:")
    
    # Örnek kullanıcı giriş alanları
    age = st.number_input('Yaş', min_value=0, max_value=100, value=25)
    gender = st.selectbox('Cinsiyet', ['Erkek', 'Kadın'])
    height = st.number_input('Boy (cm)', min_value=100, max_value=250, value=170)
    weight = st.number_input('Kilo (kg)', min_value=20, max_value=200, value=70)
    
    # Formu gönderme düğmesi
    submitted = st.form_submit_button("Tahmin Yap")

# Eğer form gönderilirse
if submitted:
    # Özellikleri DataFrame'e dönüştür
    input_data = pd.DataFrame([[age, gender, height, weight]], columns=['Age', 'Gender', 'Height', 'Weight'])
    
    # Tahmin yap
    prediction = model.predict(input_data)[0]
    
    # Tahmini göster
    st.write(f'Tahmin edilen obezite seviyesi: {prediction}')
