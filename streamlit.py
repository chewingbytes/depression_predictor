import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    model = joblib.load("trained_model_here.pk1")
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'trained_model_here.pk1' is in the correct directory.")
    st.stop()

# Title and description
st.title("Depression Prediction App")
st.write("**Are you depressed?** Fill out the form on the sidebar to find out.")

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 22)
    academic_pressure = st.sidebar.slider('Academic Pressure (0 = None, 10 = Extreme)', 0.0, 10.0, 5.0)
    cgpa = st.sidebar.slider('CGPA (0.0 to 10.0)', 0.0, 10.0, 3.0)
    study_satisfaction = st.sidebar.slider('Study Satisfaction (0 = Very Dissatisfied, 10 = Very Satisfied)', 0.0, 10.0, 5.0)
    work_study_hours = st.sidebar.slider('Work/Study Hours per Day', 0.0, 24.0, 8.0)
    financial_stress = st.sidebar.slider('Financial Stress (0 = None, 10 = Extreme)', 0.0, 10.0, 5.0)
    dietary_habits_healthy = st.sidebar.selectbox('Healthy Dietary Habits (1 = Yes, 0 = No)', [1, 0])
    dietary_habits_unhealthy = st.sidebar.selectbox('Unhealthy Dietary Habits (1 = Yes, 0 = No)', [0, 1])
    suicidal_thoughts_no = st.sidebar.selectbox('Have you had suicidal thoughts? (0 = No, 1 = Yes)', [0, 1])
    suicidal_thoughts_yes = 1 - suicidal_thoughts_no  # Automatically set the opposite value

    # Map inputs to a dictionary
    data = {
        'Age': age,
        'Academic Pressure': academic_pressure,
        'CGPA': cgpa,
        'Study Satisfaction': study_satisfaction,
        'Work/Study Hours': work_study_hours,
        'Financial Stress': financial_stress,
        'Dietary Habits_Healthy': dietary_habits_healthy,
        'Dietary Habits_Unhealthy': dietary_habits_unhealthy,
        'Have you ever had suicidal thoughts ?_No': suicidal_thoughts_no,
        'Have you ever had suicidal thoughts ?_Yes': suicidal_thoughts_yes
    }

    # Return as DataFrame and a list of feature values
    return pd.DataFrame([data]), list(data.values())

# Get user input and display it
user_data, input_features = user_input_features()
st.subheader('User Input Parameters')
st.write(user_data)

# Make predictions
if st.button("Predict Depression Likelihood"):
    try:
        prediction = model.predict([input_features])[0]
        st.subheader('Prediction')
        st.success(f"Predicted Depression Likelihood: {prediction:.2f}/1")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
