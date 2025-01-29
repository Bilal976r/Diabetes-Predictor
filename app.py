# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import preprocess_input  # Import preprocessing function

# Define the path to the trained model
MODEL_PATH = 'models/svm_model.pkl'

# Load the trained model using pickle
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Display the app header with a round logo and app name
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px;">
        <img src="assets/logo.png" 
             alt="logo" style="border-radius: 50%; width: 50px; height: 50px;">
        <h1 style="display: inline; margin: 0; font-size: 2.5rem;">Diabetes Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# App description
st.markdown(
    """
    Welcome to the Diabetes Predictor App! This tool uses a machine learning model 
    to predict the likelihood of diabetes based on user input or uploaded data. 
    Navigate through different sections using the buttons below.
    """
)

# Navigation buttons
section = st.sidebar.radio(
    "Choose Section:",
    ("Predict Diabetes", "Visualize Data", "Data Summary")
)

# Section: Predict Diabetes
if section == "Predict Diabetes":
    st.header("Diabetes Prediction")

    # Input choice: Manual or file upload
    input_choice = st.radio(
        "Choose Input Method:",
        ("Manual Input", "Upload CSV File")
    )

    if input_choice == "Manual Input":
        st.subheader("Enter Patient Details")

        # Default manual input with placeholders
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100, step=1)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=80, step=1)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=20, step=1)
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=30, step=1)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)

        if st.button("Predict"):
            # Collect the input data as a numpy array
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

            # Preprocess the input data
            processed_data = preprocess_input(input_data)

            # Make a prediction using the loaded model
            prediction = model.predict(processed_data)

            # Convert prediction to a human-readable result
            result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

            # Display the prediction result
            st.subheader(f"Prediction: {result}")
            if result == "Diabetic":
                st.warning("The patient is likely diabetic. Please consult a healthcare professional.")
            else:
                st.success("The patient is not diabetic.")

    elif input_choice == "Upload CSV File":
        st.subheader("Upload Patient Data")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            # Read the uploaded file
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(data)

            # Check for missing values
            if data.isnull().any().any():
                missing_count = data.isnull().sum().sum()
                st.error(f"The uploaded file contains {missing_count} missing values. Please fill in the missing data and re-upload the file.")
            else:
                # Check for required columns
                required_columns = [
                    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
                ]
                if all(col in data.columns for col in required_columns):
                    # Preprocess the input data
                    processed_data = preprocess_input(data[required_columns].values)

                    # Make predictions
                    predictions = model.predict(processed_data)

                    # Add predictions to the dataset
                    data['Prediction'] = ["Diabetic" if pred == 1 else "Non-Diabetic" for pred in predictions]

                    st.write("Predictions:")
                    st.dataframe(data)
                else:
                    st.error("Uploaded CSV is missing required columns.")

# Section: Visualize Data
elif section == "Visualize Data":
    st.header("Data Visualizations")
    try:
        dataset = pd.read_csv('data/diabetes.csv')

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Outcome distribution
        st.subheader("Outcome Distribution")
        fig, ax = plt.subplots()
        dataset['Outcome'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Distribution of Diabetic vs Non-Diabetic")
        st.pyplot(fig)

    except FileNotFoundError:
        st.error("Dataset file not found for visualization.")

# Section: Data Summary
elif section == "Data Summary":
    st.header("Data Summary")
    try:
        dataset = pd.read_csv('data/diabetes.csv')
        st.write("Dataset Summary:")
        st.dataframe(dataset.describe())

    except FileNotFoundError:
        st.error("Dataset file not found.")