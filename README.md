# Smart Agriculture Dashboard for Crop Recommendation & Disease Detection
We developed an intelligent, AI-powered dashboard designed to assist farmers in making data-driven decisions for improving agricultural productivity. The system empowers farmers to make informed agricultural decisions by recommending the best crop to plant and detecting crop diseases from leaf images. It integrates user-provided soil data with real-time environmental inputs to deliver personalized insights that enhance crop yield and reduce losses.

**Key Features & Functionalities:**

**1. Crop Recommendation System (CRS)**
A machine learning model predicts the most suitable crop to plant based on a combination of:
* User-Provided Soil Parameters:
  * Nitrogen (N)
  * Phosphorus (P)
  * Potassium (K)
  * pH Level
* Automatically Fetched Environmental Data:
  * Temperature
  * Humidity
  * Rainfall

**2. Plant Disease Detection**
A Convolutional Neural Network (CNN) is used to detect plant diseases from leaf images uploaded by users.
Image Analysis Workflow:
  * Users upload a photo of an infected plant leaf.
  * The CNN model processes the image and identifies potential diseases (e.g., Blight, Rust, Mosaic).
  * The dashboard displays the disease diagnosis 


**Machine Learning Workflow:**
 1. Data Preprocessing
    * Soil and weather data normalized and cleaned
    * Leaf images resized, augmented, and labeled
 2. Model Training & Evaluation
    * Crop Recommendation: Trained using supervised learning (Random Forest)
    * Disease Detection: CNN model trained on plant disease datasets (e.g., PlantVillage)
    * Evaluated using metrics like accuracy, F1-score, and confusion matrix
 3. Prediction & Output
    * Real-time predictions for best crop selection
    * Instant disease detection upon image upload


**User Interaction & Dashboard Interface:**
  * Farmers input soil parameters (N, P, K, pH) through a simple form
  * Weather data is filled in automatically via Weather API
  * Leaf image uploads are processed instantly for disease detection
  * The dashboard provides all these insights in an easy-to-understand interface.
  
**Impact & Benefits:**
  * Helps farmers make data-driven crop choices
  * Reduces crop failure with early disease detection
  * Promotes sustainable and precision farming

