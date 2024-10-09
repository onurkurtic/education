import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load the dataset from the 'original' sheet in the Excel file
df = pd.read_excel('career_data.xlsx', sheet_name='original')

# Clean the dataset by removing newline characters from 'Job profession'
df['Job profession'] = df['Job profession'].str.replace('\n', '')

# Select relevant feature columns and ensure they are numeric
features = ['Linguistic', 'Musical', 'Bodily', 'Logical - Mathematical', 
            'Spatial-Visualization', 'Interpersonal', 'Intrapersonal', 'Naturalist']

# Ensure the feature columns are numeric, and handle any non-numeric values by filling with 0
df[features] = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)

X = df[features]  # Feature set
y = df['Job profession']  # Target variable

# Label encode the target variable (Job Profession)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X, y_encoded)

# Streamlit interface
st.title("Career Path Guidance AI")
st.write("This app will suggest a career path based on your skills!")

# Get user input for the different skills
user_input = {}
for feature in features:
    user_input[feature] = st.slider(f"Rate your skill in {feature}:", 0, 20, 10)

# Convert user input into a DataFrame
user_input_df = pd.DataFrame([user_input])

# Process input and predict career path
if st.button('Predict Career Path'):
    prediction = model.predict(user_input_df)
    predicted_job = le.inverse_transform(prediction)

    st.write(f"Based on your skills, a suitable career path for you might b
