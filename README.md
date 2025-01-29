# Diabetes Predictor

An interactive web application built with [Streamlit](https://streamlit.io/) to predict the likelihood of diabetes using a trained Support Vector Machine (SVM) model. This project includes functionalities for manual predictions, batch predictions from CSV files, data preprocessing, and visual insights into the dataset.

---

## Features
1. **Manual Input**: Predict diabetes by entering patient details manually.
2. **Batch Predictions**: Upload a CSV file to predict diabetes for multiple patients simultaneously.
3. **Data Visualization**:
   - Correlation heatmaps.
   - Outcome distributions.
4. **Error Handling**: Alerts for missing values or invalid input files.
5. **Unit Testing**: Robust tests for all critical components.

---

## Project Structure
Diabetes_Predictor/
│
├── app.py                       # Main Streamlit app
├── requirements.txt             # Dependencies for the app
├── README.md                    # Project description and instructions
├── data/
│   └── diabetes.csv             # Dataset file
├── models/
│   └── svm_model.pkl            # Serialized trained model
├── notebooks/
│   └── Project_3_Diabetes_Prediction.ipynb  # Original notebook for analysis
├── utils/
│   └── preprocessing.py         # Preprocessing and utility functions
├── tests/
│   └── test_app.py              # Unit tests for the app and its components
└── assets/
    └── logo.png                 # Optional branding/logo for the app


---

## Installation

### Prerequisites
- Python 3.8 or higher
- Pip (Python package manager)

### Steps
1. Clone this repository:
   git clone https://github.com/your-repo/diabetes-predictor.git

2. Navigate to the project directory:
   cd diabetes-predictor
3. Install the dependencies:
   pip install -r requirements.txt

4. Run the application:
   streamlit run app.py


# Input Data Requirements
## Manual Input:
The following details must be provided:

Pregnancies
Glucose Level
Blood Pressure
Skin Thickness
Insulin Level
BMI (Body Mass Index)
Diabetes Pedigree Function
Age

## CSV File:
The uploaded CSV file must contain these columns:

Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age

If any values are missing, the app will alert the user to correct the data.

# Technologies Used
Streamlit: Interactive user interface framework.
Scikit-learn: Machine learning for model training and predictions.
Pandas: Data manipulation and analysis.
Numpy: Numerical computations.
Matplotlib & Seaborn: For visualizations.

# Dataset
The app uses the "PIMA Diabetes Dataset" for training and predictions.

