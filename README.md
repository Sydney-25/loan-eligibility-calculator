

# Loan Eligibility Predictor

An interactive web application that leverages machine learning to predict loan eligibility.  
Built with Streamlit and scikit-learn, the tool processes applicant data to determine whether a loan application is likely to be approved or rejected.

---

## Table of Contents

- Overview
- Features
- Installation
- Usage
- Project Structure
- Customization
- Contributing
- Contact

---

## Overview

The Loan Eligibility Predictor is designed to help financial institutions and individuals make informed decisions by evaluating key financial parameters from loan applicants. By uploading a CSV file with applicant details, the application:

- Processes and validates input data.
- Trains a logistic regression model (or loads a pre-trained model) to predict loan approval.
- Provides real-time prediction feedback along with an approval probability.
- Offers an option to download the packaged application files.

---

## Features

- **Data Upload:**  
  Easily upload CSV files containing applicant data including:
  - Applicant Income
  - Co-Applicant Income
  - Loan Amount
  - Loan Term Months
  - Credit History
  - Application successful

- **Model Training & Loading:**  
  The app trains a logistic regression model on your data or loads an existing model if available.

- **Input Validation:**  
  Built-in checks ensure that loan amounts and terms are within realistic and predefined ranges.

- **Real-Time Prediction:**  
  Provides immediate feedback on loan eligibility, displaying both a decision (approved/rejected) and the associated approval probability.

- **Custom Styling:**  
  Leverages custom CSS to enhance the user interface and overall look of the application.

- **Download Application Files:**  
  A downloadable ZIP package is offered, containing all key application files.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/loan-eligibility-predictor.git
cd loan-eligibility-predictor
```

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have the following libraries installed:
- Streamlit
- Pandas
- Scikit-learn
- Joblib

---

## Usage

Start the application by running:

```bash
streamlit run loan.py
```

1. **Upload Your Data:**  
   Use the sidebar file uploader to load a CSV file with the required columns.

2. **Input Customer Details:**  
   Enter values for applicant income, co-applicant income, loan amount, loan term, and select the credit history.

3. **Train or Load Model & Predict:**  
   The app will train a new logistic regression model (if data is uploaded) or load an existing one.  
   Click the "Predict Loan Eligibility" button to view the prediction and its approval probability.

4. **Download Application Files:**  
   Once available, use the download button to retrieve a ZIP file of the application package.

---

## Project Structure

```
loan-eligibility-predictor/
├── loan.py                # Main Streamlit application for loan eligibility prediction
├── .streamlit/
│   └── style.css          # Custom CSS styling for the application
├── trained_model.pkl      # Saved logistic regression model (generated after training)
└── loan_predictor.zip     # Packaged application files for download
```

Ensure the directory structure is maintained so that the application and its modules can be correctly imported and executed.

---

## Customization

- **Styling:**  
  Modify the `.streamlit/style.css` file to change the appearance of the application.

- **Model & Data Processing:**  
  Update the logic in `loan.py` to fine-tune model training, prediction, or input validation as per your needs.

- **Input Validation:**  
  Adjust the rules in the `validate_inputs` function to better suit different loan criteria or business logic.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (e.g., `git checkout -b feature/your-feature`).
3. Commit your changes.
4. Push the branch and open a pull request.


---

## Contact

For any questions or suggestions, please reach out to:
sydney.abuto@gmail.com
