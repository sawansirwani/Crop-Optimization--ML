# Crop Recommendation Project 
This project aimed to build a model to recommend crops based on environmental factors. The data consisted of various features likely influencing crop yield,  including:

NPK (Nitrogen, Phosphorus, Potassium) content in the soil
Temperature
Humidity
Rainfall
pH
The target variable was the recommended crop type (e.g., rice, wheat, etc.).

# Model Building and Evaluation

## Data Preprocessing:

Missing values were handled (likely removed).
Model Training:

## Two machine learning models were explored:
Decision Tree Classifier

Random Forest Classifier (used for final analysis)

## Model Evaluation:

The Random Forest Classifier achieved a high accuracy of 99% on the test data.
A confusion matrix visualized the model's performance on each crop type.
Feature importance analysis identified the most influential factors for crop recommendation:
Humidity
Potassium (K) content
Phosphorus (P) content
Rainfall
Nitrogen (N) content

## Additional Notes:

using Streamlit for data output, likely creating a web app for users to interact with the model.
Shap analysis was potentially used to understand feature dependencies, providing further insights into how the model makes predictions.
Overall, this project successfully built a crop recommendation model with high accuracy based on environmental factors.
