import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import base64

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "trained_model.pkl"
FEATURE_COLS = ['Applicant Income', 'Co-Applicant Income', 'Loan Amount', 'Loan Term Months', 'Credit History']
TARGET_COL = 'Application successful'

# -------------------------------
# Styling
# -------------------------------
def load_css():
    css_file = '.streamlit/style.css'
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Provide default CSS or load a simplified style
        st.markdown("""
            <style>
            body {
                font-family: sans-serif;
            }
            </style>
        """, unsafe_allow_html=True)
        st.warning("Custom CSS file not found. Using default styling.")



# Add custom styling section
def custom_streamlit_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');
        .file-uploader-container {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .file-uploader-container label {
            display: block;
            margin-bottom: 5px;
        }
        .file-uploader-container input[type="file"] {
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .upload-status {
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        .upload-status.success {
            background-color: #81C995;
            color: #202124;
        }
        .upload-status.error {
            background-color: #F28B82;
            color: #202124;
        }

        </style>
    """, unsafe_allow_html=True)
    load_css()

# -------------------------------
# Utility: Load/Save Model
# -------------------------------
def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)

# -------------------------------
# Utility: Train Model
# -------------------------------
@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    df_model = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    X = df_model[FEATURE_COLS]
    y = df_model[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    save_model(model)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

# -------------------------------
# Input Validation
# -------------------------------
def validate_inputs(applicant_income, coapplicant_income, loan_amount, loan_term):
    if loan_amount > (applicant_income + coapplicant_income) * 10:
        return False, "Loan amount too high relative to income"
    if loan_amount < 17000 or loan_amount > 495500:
        return False, "Loan amount should be between $17,000 and $495,500"
    if loan_term < 12 or loan term > 84:
        return False, "Loan term should be between 12 and 84 months"
    return True, ""

# -------------------------------
# Sample Data Info
# -------------------------------
def show_sample_data_format():
    st.sidebar.markdown("""
    ### Sample CSV Format:
    Your CSV should have these columns:
    - Applicant Income (numeric)
    - Co-Applicant Income (numeric)
    - Loan Amount (numeric)
    - Loan Term Months (numeric)
    - Credit History (0 or 1)
    - Application successful (0 or 1)

    Example row:
    5000,2000,150000,360,1,1
    """)

# -------------------------------
# Main Streamlit App
# -------------------------------
def main():
    st.set_page_config(
        page_title="Loan Eligibility Predictor",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    custom_streamlit_style()

    # Centered title with Google AI Studio style
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-family: "Google Sans", sans-serif; font-size: 2.5rem; font-weight: 500;'>
                Loan Eligibility Predictor 🏦
            </h1>
            <p style='font-family: "Google Sans", sans-serif; color: #E8EAED; font-size: 1.1rem;'>
                Enter customer details to predict loan eligibility using machine learning
            </p>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------
    # File Uploader Section
    # -------------------------------
    st.sidebar.header("1. Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"], accept_multiple_files=False)
    upload_status = ""

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            upload_status = f"<div class='upload-status success'>File uploaded successfully!</div>"
        except Exception as e:
            upload_status = f"<div class='upload-status error'>Error uploading file: {e}</div>"
        st.markdown(upload_status, unsafe_allow_html=True)


    # Check if the model exists or data was uploaded
    model_trained = os.path.exists(MODEL_PATH) or uploaded_file is not None

    # -------------------------------
    # Sidebar: Model Training or Loading
    # -------------------------------
    if not model_trained:
        show_sample_data_format()
        if uploaded_file is None:
            st.sidebar.warning("Please upload a CSV dataset to proceed.")
            return
        else:
            try:
                with st.spinner('Training model...'):
                    model, test_accuracy = train_model(data)
                st.sidebar.success(f"Model trained & saved! Test Accuracy: {test_accuracy * 100:.2f}%")
                model_trained = True
            except Exception as e:
                st.sidebar.error(f"Training failed: {e}")
                return
    else:
        model = load_model()
        st.sidebar.success("Pre-trained model loaded!")

    # -------------------------------
    # Input Features for Prediction
    # -------------------------------
    st.sidebar.header("2. Customer Details")
    with st.sidebar:
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
        coapplicant_income = st.number_input("Co-Applicant Income ($)", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=100000)
        loan_term = st.number_input("Loan Term (Months)", min_value=12, max_value=360, value=360)
        credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good" if x == 1.0 else "Poor")

    # Validate inputs
    is_valid, error_message = validate_inputs(applicant_income, coapplicant_income, loan_amount, loan_term)

    if not is_valid:
        st.error(error_message)
        return

    input_data = pd.DataFrame([{
        'Applicant Income': applicant_income,
        'Co-Applicant Income': coapplicant_income,
        'Loan Amount': loan_amount,
        'Loan Term Months': loan_term,
        'Credit History': credit_history
    }])

    # -------------------------------
    # Make Prediction
    # -------------------------------
    st.write("### Prediction Details")
    st.write(input_data)

    st.markdown("""
        <div class="loan-info" style='background-color: #303134; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;'>
            <h3 style='color: #E8EAED; margin-bottom: 1rem;'>Loan Information</h3>
            <ul style='color: #E8EAED; list-style-type: none; padding-left: 0;'>
                <li style='margin: 0.5rem 0;'>✦ Minimum Loan Amount: $17,000</li>
                <li style='margin: 0.5rem 0;'>✦ Maximum Loan Amount: $495,500</li>
                <li style='margin: 0.5rem 0;'>✦ Loan Term Range: 12-84 months</li>
                <li style='margin: 0.5rem 0;'>✦ Most Common Term: 36 months</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Predict Loan Eligibility", key="predict_button"):
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class='success-msg' style='background-color: #81C995; color: #202124; padding: 1rem; border-radius: 8px;'>
                        <span style='font-size: 1.2rem; font-weight: 500;'>✓ Loan Approved!</span><br>
                        Approval Probability: {probability * 100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='error-msg' style='background-color: #F28B82; color: #202124; padding: 1rem; border-radius: 8px;'>
                        <span style='font-size: 1.2rem; font-weight: 500;'>✕ Loan Rejected</span><br>
                        Approval Probability: {probability * 100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Add download button for the zip file
    st.markdown("---")
    st.markdown("### Download Application")

    if os.path.exists('loan_predictor.zip'):
        with open('loan_predictor.zip', 'rb') as f:
            zip_contents = f.read()
        st.download_button(
            label="📥 Download Application Files",
            data=zip_contents,
            file_name="loan_predictor.zip",
            mime="application/zip",
            help="Download the application files including code, requirements, and documentation"
        )
    else:
        st.warning("Download package is being prepared. Please try again in a moment.")

if __name__ == "__main__":
    main()
