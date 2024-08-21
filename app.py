import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('ridge_regression_model.pkl')

# Define the Streamlit app
def main():
    st.title("Fitness Tracker Model")

    st.write("Enter the details below to predict calories burned:")

    # Input fields for user data
    steps = st.number_input("Steps", min_value=0)
    distance_km = st.number_input("Distance (km)", min_value=0.0)
    active_minutes = st.number_input("Active Minutes", min_value=0)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0)
    heart_rate_avg = st.number_input("Average Heart Rate", min_value=0)
    workout_type = st.selectbox("Workout Type", options=['Walking', 'Cycling', 'Yoga', 'Swimming'])

    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'steps': [steps],
        'distance_km': [distance_km],
        'active_minutes': [active_minutes],
        'sleep_hours': [sleep_hours],
        'heart_rate_avg': [heart_rate_avg],
        'workout_type': [workout_type]
    })

    if st.button("Predict"):
        # Make prediction
        prediction = model.predict(input_data)
        st.write(f"Predicted Calories Burned: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()

