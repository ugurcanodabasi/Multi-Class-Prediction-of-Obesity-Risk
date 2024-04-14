import streamlit as st
import pandas as pd
import pickle
import lightgbm as lgb

# Modeli yükle
model_path = 'lgbm_model_final.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Sidebar - Kullanıcı girdileri
st.sidebar.header('Kullanıcı Girdi Özellikleri')

def user_input_features():
    age = st.sidebar.number_input('Yaş', min_value=1, max_value=100, value=25)
    height = st.sidebar.number_input('Boy (cm)', min_value=100, max_value=250, value=170)
    weight = st.sidebar.number_input('Kilo (kg)', min_value=30, max_value=200, value=70)
    daily_water_intake = st.sidebar.number_input('Günlük Su Tüketimi (litre)', min_value=0.0, max_value=10.0, value=2.0)
    gender = st.sidebar.selectbox('Cinsiyet', ('Erkek', 'Kadın'))
    data = {'age': [age],
            'height': [height],
            'weight': [weight],
            'daily_water_intake': [daily_water_intake],
            'gender': [gender]}
    features = pd.DataFrame(data)
    return features

input_df = user_input_features()

# Main Page
st.write("""
# Basit Obezite Tahmin Uygulaması
Bu uygulama, LightGBM modeli kullanarak obezite seviyenizi tahmin eder!
""")

# Tahminleri göster
st.subheader('Kullanıcı Girdi Özellikleri')
st.write(input_df)

st.subheader('Tahmin Edilen Obezite Seviyesi')
prediction = model.predict(input_df)
st.write(prediction[0])

# CSV dosyasını yükleme ve görselleştirme
st.subheader('Tahmin Edilen Obezite Seviyeleri (CSV)')
data_path = 'predicted_obesity_levels.csv'
data = pd.read_csv(data_path)
st.write(data)
