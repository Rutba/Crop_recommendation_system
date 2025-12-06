#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import tensorflow as tf
import pickle
import os
import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------
# Load images from directories
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    horizontal_flip=True,
    vertical_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'PlantDoc-Dataset/train',  # Your local dataset folder
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# -------------------------
# Load model and encoder
# -------------------------
model = pickle.load(open("crop_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# -------------------------
# Weather API Integration
# -------------------------
base_url = "http://api.openweathermap.org/data/2.5/weather?"

def get_weather_data(api_key, city):
    url = f'{base_url}q={city}&appid={api_key}&units=metric&exclude=minutely,daily,alerts'
    response = requests.get(url)
    weather_data = response.json()
    temperature = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    rainfall = weather_data.get('rain', {}).get('1h', 0)
    return temperature, humidity, rainfall

# -------------------------
# Disease Detection
# -------------------------
def detect_disease(leaf_image_path):
    model_cnn = tf.keras.models.load_model('model.h5')
    img = tf.keras.preprocessing.image.load_img(leaf_image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    predictions = model_cnn.predict(img_array)
    return predictions.argmax()

# -------------------------
# Streamlit Dashboard
# -------------------------
st.title("🌾 Smart Agriculture Dashboard")

# Weather Section
api_key = "83721cca1b591c77ee895325000e6143"
city = st.text_input("City Name")
if city:
    temp, humidity, rainfall = get_weather_data(api_key, city)
    st.subheader("📡 Current Weather Conditions")
    st.write(f"Weather in {city}:")
    st.write(f"Temperature: {temp} °C")
    st.write(f"Humidity: {humidity} %")
    st.write(f"Rainfall (last 1h): {rainfall} mm")

# Crop Recommendation Section
st.title("🌱 Crop Recommendation System")
st.markdown("Weather values are auto-filled using OpenWeatherMap 🌦️")

n = st.number_input("Nitrogen", 0, 140)
p = st.number_input("Phosphorus", 5, 145)
k = st.number_input("Potassium", 5, 205)
temp = st.number_input("Temperature (°C)", value=temp if city else 25.0, disabled=city is not None)
humidity = st.number_input("Humidity (%)", value=humidity if city else 50.0, disabled=city is not None)
ph = st.number_input("Soil pH", 3.5, 9.5)
rainfall = st.number_input("Rainfall (mm)", value=rainfall if city else 0.0)

if st.button("🌾 Recommend Crop"):
    sample = [[n, p, k, temp, humidity, ph, rainfall]]
    result = model.predict(sample)
    st.success(f"Recommended Crop: {encoder.inverse_transform(result)[0]}")

# Plant Disease Detection Section
st.title("🍂 Plant Disease Detection")
leaf_image = st.file_uploader("Upload leaf image for disease detection", type=["jpg", "png"])
if leaf_image is not None:
    st.image(leaf_image, caption="Leaf Image", use_container_width=True)
    disease = detect_disease(leaf_image)
    class_labels = {v: k for k, v in train_generator.class_indices.items()}
    predicted_class = class_labels[disease]
    st.write(f"Disease Detected: {predicted_class}")


# In[ ]:




