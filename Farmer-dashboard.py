#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import tensorflow as tf
import pickle
import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Page configuration

st.set_page_config(
    page_title="Smart Agriculture Dashboard",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
.result-box {
    background-color: #eaffea;
    border-left: 6px solid #2ecc71;
    padding: 10px 14px;
    font-size: 20px;
    font-weight: 700;
    color: #145a32;
    width: fit-content;
    border-radius: 6px;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)


# Load images for disease class mapping

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    'PlantDoc-Dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load models

model = pickle.load(open("crop_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))


# Weather API

base_url = "http://api.openweathermap.org/data/2.5/weather?"

def get_weather_data(api_key, city):
    url = f'{base_url}q={city}&appid={api_key}&units=metric&exclude=minutely,daily,alerts'
    response = requests.get(url)
    data = response.json()
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    rainfall = data.get('rain', {}).get('1h', 0)
    return temperature, humidity, rainfall


# Disease detection

def detect_disease(leaf_image):
    model_cnn = tf.keras.models.load_model('model.h5')
    img = tf.keras.preprocessing.image.load_img(leaf_image, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)/255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    predictions = model_cnn.predict(img_array)
    return predictions.argmax()


# Weather Section

st.header("📡 Current Weather")
api_key = "83721cca1b591c77ee895325000e6143"  
col_city, _ = st.columns([1,3]) # input only 25% width
with col_city:
    city = st.text_input("City Name")
if city:
    temp, humidity, rainfall = get_weather_data(api_key, city)
    st.write(f"Weather in {city}: Temperature: {temp}°C, Humidity: {humidity}%, Rainfall (last 1h): {rainfall}mm")
else:
    temp, humidity, rainfall = 25.0, 50.0, 0.0


# Crop Recommendation Section

st.header("🌱 Crop Recommendation System")

# Row 1 - 4 inputs
col1, col2,_ = st.columns([1,1,4])
with col1:
    n = st.number_input("🧪 Nitrogen (N)", 0, 140, key="n")
with col2:
    p = st.number_input("🧫 Phosphorus (P)", 5, 145, key="p")

col3, col4, _ = st.columns([1,1,4])
with col3:
    k = st.number_input("⚗️ Potassium (K)", 5, 205, key="k")
with col4:
    ph = st.number_input("🧪 Soil pH", 3.5, 9.5, key="ph")
    

col5, col6, _ = st.columns([1,1,4])
with col5:
    temp_input = st.number_input("🌡️ Temperature (°C)", value=temp, disabled=city is not None, key="temp_input")
with col6:
    humidity_input = st.number_input("💧 Humidity (%)", value=humidity, disabled=city is not None, key="humidity_input")


col7, _ = st.columns([1,4])
with col7:
    rainfall_input = st.number_input("🌧️ Rainfall (mm)", value=rainfall, key="rainfall_input")


if st.button("🌾 Recommend Crop"):
    sample = [[n, p, k,ph, temp_input, humidity_input, rainfall_input]]
    result = model.predict(sample)
    crop = encoder.inverse_transform(result)[0]

    col_out, _ = st.columns([1, 4])
    with col_out:
        st.markdown(
            f'<div class="result-box">🌱 Recommended Crop: {crop}</div>',
            unsafe_allow_html=True
        )
    
# Plant Disease Detection

st.header("🍂 Plant Disease Detection")
#leaf_image = st.file_uploader("Upload leaf image", type=["jpg", "png"])
col_up, _ = st.columns([1, 4])  # small width

with col_up:
    leaf_image = st.file_uploader(
        "Leaf Image",
        type=["jpg", "png"]
    )

if leaf_image is not None:
    st.image(leaf_image, caption="Leaf Image", width =200)
    disease = detect_disease(leaf_image)
    class_labels = {v: k for k, v in train_generator.class_indices.items()}
    predicted_class = class_labels[disease]
    st.write(f"Disease Detected: **{predicted_class}**")


# In[ ]:




