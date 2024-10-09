import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Sample data for demonstration purposes
data = {'Skills': ['Linguistic', 'Musical', 'Logical-Mathematical'],
        'Job Profession': ['Writer', 'Musician', 'Scientist']}
df = pd.DataFrame(data)

# Features and target
X = df['Skills'].values.reshape(-1, 1)
y = df['Job Profession']

# Label encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X, y_encoded)

# Streamlit interface
st.title("Career Path Guidance AI")
st.write("This app will suggest a career path based on your skills!")

# Get user input
user_input = st.text_input("Enter your top skill (e.g., Linguistic, Musical, Logical-Mathematical):")

# Process input and predict career path
if user_input:
    user_input_array = np.array([user_input]).reshape(-1, 1)
    prediction = model.predict(user_input_array)
    predicted_job = le.inverse_transform(prediction)

    st.write(f"Based on your input, a suitable career path for you might be: {predicted_job[0]}")
