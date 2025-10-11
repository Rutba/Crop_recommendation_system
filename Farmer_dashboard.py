#!/usr/bin/env python
# coding: utf-8

# Objective: Display crop recommendations, weather data, and disease detection results on a farmer-friendly dashboard.
# 
# Steps:
# 
# Create a Dashboard using Streamlit:
# 
# Display weather data, recommendations, and disease detection results on an interactive dashboard.
# 
# Example integration with weather data and disease detection results:

# In[5]:


import streamlit as st
import tensorflow as tf
import requests
import pickle
import os
import zipfile
import gdown


from tensorflow.keras.preprocessing.image import ImageDataGenerator

def download_and_extract_dataset():
    url = "https://www.dropbox.com/scl/fi/3fqzgib170fjdmqi9j067/PlantDoc-Dataset.zip?rlkey=dj5we8s990p22gc4u006mlx43&st=9re4c86q&dl=1"
    zip_path = "PlantDoc-Dataset.zip"
    extract_path = "PlantDoc-Dataset"

    if not os.path.exists(extract_path):
        st.info("Downloading dataset... This may take a few minutes.")

        # Download the file
        try:
            response = requests.get(url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            st.error(f"Download failed: {e}")
            return

        # Check file size — basic validation
        file_size = os.path.getsize(zip_path)
        st.write(f"Downloaded file size: {file_size / (1024*1024):.2f} MB")

        if file_size < 100_000:  # less than 100KB
            st.error("Downloaded file seems too small. Likely an HTML error, not a real ZIP.")
            return

        # Try extracting
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            st.success("Dataset extracted successfully.")
        except zipfile.BadZipFile:
            st.error("Downloaded file is not a valid ZIP file. Check the Dropbox link.")

            
# Call the function at app start
download_and_extract_dataset()


# Load images from directories with subdirectories as categories
# Set up the ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255.0, horizontal_flip=True, vertical_flip=True)
train_generator = train_datagen.flow_from_directory( 
    '/PlantDoc-Dataset/train',  # Specify the dataset path
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='categorical'
)




# Load model and encoder
model = pickle.load(open("crop_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))


# OpenWeatherMap URL for current weather
base_url = "http://api.openweathermap.org/data/2.5/weather?"

# Weather API Integration
def get_weather_data(api_key, city):
    url=f'{base_url}q={city}&appid={api_key}&units=metric&exclude=minutely,daily,alerts'
    #url = f'http://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={api_key}&units=metric'
    response = requests.get(url)
    weather_data = response.json()
    print(weather_data)
    temperature = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    rainfall = weather_data.get('rain', {}).get('1h', 0)
    return temperature, humidity, rainfall

# Disease Detection (with CNN model)
def detect_disease(leaf_image_path):
    model = tf.keras.models.load_model('model.h5')
    img = tf.keras.preprocessing.image.load_img(leaf_image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    predictions = model.predict(img_array)
    return predictions.argmax()

# Streamlit Dashboard
st.title("🌾 Smart Agriculture Dashboard")

# Get weather data
api_key = "83721cca1b591c77ee895325000e6143"
#lat, lon = 23.7104, 90.4125  # Example coordinates
# City for which you want the weather data
#city = "Dhaka"
city = st.text_input("City_Name")
if city:
    temp, humidity, rainfall = get_weather_data(api_key,city)
    st.subheader("📡 Current Weather Conditions")
    st.write(f"Weather in {city}:")
    st.write(f"Temperature: {temp} °C")
    st.write(f"Humidity: {humidity} %")
    st.write(f"Rainfall (last 1h): {rainfall} mm")




st.title("🌱 Crop Recommendation System")
st.markdown("Weather values are auto-filled using OpenWeatherMap 🌦️")

# User soil and nutrient inputs
n = st.number_input("Nitrogen", 0, 140)
p = st.number_input("Phosphorus", 5, 145)
k = st.number_input("Potassium", 5, 205)
temp = st.number_input("Temperature (°C)",value=temp, disabled=True)
humidity = st.number_input("Humidity (%)", value=humidity, disabled=True)
ph = st.number_input("Soil pH", 3.5, 9.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0)


if st.button(" 🌾 Recommend Crop"):
    sample = [[n, p, k, temp, humidity, ph, rainfall]]
    result = model.predict(sample)
    st.success(f"Recommended Crop: {encoder.inverse_transform(result)[0]}")



# Upload leaf image for disease detection
st.title("🍂 Plant Disease Detection")
leaf_image = st.file_uploader("Upload leaf image for disease detection", type=["jpg", "png"])
if leaf_image is not None:
    st.image(leaf_image, caption="Leaf Image", use_column_width=True)
    disease = detect_disease(leaf_image)
    # Get class labels (only if you have train_generator)
    class_labels = {v: k for k, v in train_generator.class_indices.items()}
    predicted_class = class_labels[disease]
    st.write(f"Disease Detected: {predicted_class}")
    
    


# In[ ]:




