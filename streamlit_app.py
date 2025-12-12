import streamlit as st
import pandas as pd
import joblib
import os
# Import numpy just in case it's needed by the underlying pipeline components
import numpy as np 
# The ColumnTransformer uses classes from sklearn.compose, but we need to ensure 
# the entire package's structure is available for joblib to load the pipeline.
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- Model Loading ---
# The model file 'employee_attrition_model.joblib' is created by the notebook.
MODEL_FILE = 'employee_attrition_model.joblib'

# Use st.cache_resource to cache the model loading for efficiency.
# This ensures the model is loaded only once across all user sessions.
@st.cache_resource
def load_model(filename):
    """Loads the pre-trained machine learning pipeline."""
    if not os.path.exists(filename):
        # In a real deployment, you would ensure the file exists.
        st.error(f"Model file '{filename}' not found. Please ensure it is in the same directory.")
        return None
    try:
        # Load the pipeline which includes the preprocessor and the classifier
        loaded_pipeline = joblib.load(filename)
        st.success("✅ Model loaded successfully!")
        return loaded_pipeline
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model pipeline
model_pipeline = load_model(MODEL_FILE)

# --- Application Interface ---
st.title("🧑‍💼 Employee Attrition Prediction App")
st.markdown("Predict whether an employee is likely to **Leave (1)** or **Stay (0)** based on their profile using a pre-trained Random Forest model.")

if model_pipeline is not None:
    # --- Feature Definitions (Derived from df.head() and df.info() in the notebook) ---
    
    # Categorical features and their unique values (from notebook output)
    education_options = ('Bachelors', 'Masters', 'PHD')
    city_options = ('Bangalore', 'Pune', 'New Delhi')
    gender_options = ('Male', 'Female')
    everbenched_options = ('No', 'Yes')
    payment_tier_options = (1, 2, 3) # Note: Used as a categorical feature in the preprocessor

    # Numerical features (based on df.describe() for ranges)
    joining_year_min, joining_year_max = 2012, 2018
    age_min, age_max = 22, 41
    experience_min, experience_max = 0, 7

    st.header("Employee Profile Input")

    # Layout inputs in two columns for better look and feel
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographic & Education")
        education = st.selectbox("Education", options=education_options, index=1, help="Highest level of education.")
        city = st.selectbox("City", options=city_options, index=0, help="Employee's city of work.")
        gender = st.selectbox("Gender", options=gender_options, index=0, help="Employee's gender.")
        
    with col2:
        st.subheader("Work Details")
        payment_tier = st.selectbox("Payment Tier", options=payment_tier_options, index=0, help="1 (Highest) to 3 (Lowest). Used as a categorical feature in model.")
        ever_benched = st.selectbox("Ever Benched", options=everbenched_options, index=0, help="Has the employee been benched? ('No'/'Yes').")
        
        # Numerical inputs
        joining_year = st.number_input("Joining Year", min_value=joining_year_min, max_value=joining_year_max, value=2015, step=1, help=f"Year of joining (Range: {joining_year_min}-{joining_year_max}).")
        age = st.number_input("Age", min_value=age_min, max_value=age_max, value=30, step=1, help=f"Employee's age (Range: {age_min}-{age_max}).")
        experience = st.number_input("Experience in Current Domain", min_value=experience_min, max_value=experience_max, value=3, step=1, help=f"Years of experience in the current domain (Range: {experience_min}-{experience_max}).")
    
    st.markdown("---")
    
    # --- Prediction Logic ---
    if st.button("🔮 Predict Attrition"):
        # 1. Collect inputs into a dictionary
        input_data = {
            'Education': education,
            'JoiningYear': joining_year,
            'City': city,
            'PaymentTier': payment_tier,
            'Age': age,
            'Gender': gender,
            'EverBenched': ever_benched,
            'ExperienceInCurrentDomain': experience
        }
        
        # 2. Convert to DataFrame (must match the format used for training)
        input_df = pd.DataFrame([input_data])
        
        # 3. Make prediction using the loaded pipeline
        try:
            # The pipeline handles scaling and one-hot encoding internally
            prediction = model_pipeline.predict(input_df)[0]
            probabilities = model_pipeline.predict_proba(input_df)[0]
            
            # Probability of leaving (Class 1)
            leave_probability = probabilities[1] * 100
            
            st.subheader("Prediction Result")
            
            # 4. Display the result
            if prediction == 1:
                st.error(f"Outcome: The model predicts the employee is likely to **LEAVE**.")
            else:
                st.success(f"Outcome: The model predicts the employee is likely to **STAY**.")
                
            st.info(f"Confidence (Probability of Leaving): **{leave_probability:.2f}%**")
            
            st.markdown("---")
            st.caption("A prediction of 'Leave' (1) suggests higher risk of attrition.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check your input values. Non-numeric or out-of-range values can cause errors.")