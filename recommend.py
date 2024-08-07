
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Read the dataset
@st.cache  # Cache the dataset to avoid reading it multiple times
def load_dataset():
    return pd.read_csv('Crop_recommendation.csv')

dataset = load_dataset()

# Train the model using the Random Forest Classifier algorithm
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X, y)

# Crop images dictionary 
crop_images = {
    'apple': './crop img/apple.jpg',
    'banana': './crop img/banana.jpg',
    'blackgram':'./crop img/blackgram.jpg',
    'chickpea':'./crop img/chickpea.jpg',
    'coconut':'./crop img/coconut.jpg',
    'coffee':'./crop img/coffee.jpg',
    'cotton':'./crop img/cotton.jpg',
    'grapes':'./crop img/grapes.jpg',
    'jute':'./crop img/jute.jpg',
    'kidneybeans':'./crop img/kidneybean.jpeg',
    'lentil':'./crop img/lentil.jpeg',
    'maize':'./crop img/maize.jpeg',
    'mango':'./crop img/mango.jpg',
    'mothbeans':'./crop img/mothbeans.jpg',
    'mungbean':'./crop img/mungbean.jpg',
    'muskmelon':'./crop img/msukmelon.jpg',
    'orange':'./crop img/orange.jpg',
    'papaya':'./crop img/papaya.jpeg',
    'pigeonpeas':'./crop img/pigeonpeas.jpg',
    'pomegranate':'./crop img/pomegranate.jpg',
    'rice':'./crop img/rice.jpg',
    'watermelon':'./crop img/watermelon.jpg'

}

# Streamlit app
st.title('Crop Prediction Based on Soil Parameters')

# User input for soil parameters
st.sidebar.header('Input Soil Parameters')
st.sidebar.subheader('Select the values for each parameter:')

# Input labels for each parameter
input_labels = {
    'Nitrogen': 'Amount of Nitrogen in soil (kg/ha)',
    'Phosphorus': 'Amount of Phosphorus in soil (kg/ha)',
    'Potassium': 'Amount of Potassium in soil (kg/ha)',
    'Temperature': 'Temperature (°C)',
    'Humidity': 'Humidity (%)',
    'pH': 'pH Value',
    'Rainfall': 'Rainfall (mm)'
}

# Sliders for each parameter
n_params = st.sidebar.slider(input_labels['Nitrogen'], min_value=0, max_value=200, step=1)
p_params = st.sidebar.slider(input_labels['Phosphorus'], min_value=0, max_value=100, step=1)
k_params = st.sidebar.slider(input_labels['Potassium'], min_value=0, max_value=100, step=1)
t_params = st.sidebar.slider(input_labels['Temperature'], min_value=0.0, max_value=50.0, step=0.1)
h_params = st.sidebar.slider(input_labels['Humidity'], min_value=0.0, max_value=100.0, step=1.0)
ph_params = st.sidebar.slider(input_labels['pH'], min_value=0.0, max_value=14.0, step=0.1)
r_params = st.sidebar.slider(input_labels['Rainfall'], min_value=0.0, max_value=300.0, step=0.1)

# Store user inputs in a numpy array
user_input = np.array([[n_params, p_params, k_params, t_params, h_params, ph_params, r_params]])

# Make predictions using the trained model
prediction = classifier.predict(user_input)

# Display the predicted crop
crop_name = prediction[0]
st.write(f'Predicted Crop: {crop_name}')

# Debugging: Print the file path of the image
if crop_name in crop_images:
    
    st.image(crop_images[crop_name], caption=crop_name, use_column_width=True)
else:
    st.write("Image not available for this crop.")
