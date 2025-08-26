import streamlit as st
import joblib
import numpy as np

# Load the trained model and the scaler
# Ensure these files are in the same directory as this app.py file
try:
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'rf_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Set the title of the Streamlit app
st.title("Financial Fraud Detection App")
st.markdown("Enter transaction details to predict if it is fraudulent.")

# Create input fields for the user to enter data
st.header("Transaction Details")
amount = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")
location_lat = st.number_input("Location Latitude", format="%.4f")
location_long = st.number_input("Location Longitude", format="%.4f")
time_of_day = st.slider("Time of Day (Hour)", 0, 23)
day_of_week = st.selectbox("Day of Week", options=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

# Map day of week to numerical value (0=Monday, 6=Sunday)
day_of_week_map = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
day_of_week_num = day_of_week_map[day_of_week]

# The prediction button
if st.button("Predict"):
    # Create a numpy array from the user's input
    input_data = np.array([[amount, location_lat, location_long, time_of_day, day_of_week_num]])

    # Scale the input data using the loaded scaler
    scaled_input_data = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(scaled_input_data)
    prediction_proba = model.predict_proba(scaled_input_data)

    # Display the result to the user
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("This transaction is predicted to be **FRAUDULENT**.")
        st.write(f"Confidence: {prediction_proba[0][1]:.2%}")
    else:
        st.success("This transaction is predicted to be **LEGITIMATE**.")
        st.write(f"Confidence: {prediction_proba[0][0]:.2%}")