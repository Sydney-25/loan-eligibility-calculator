streamlit==1.31.1
pandas==2.1.4
scikit-learn==1.3.2
joblib==1.3.2
```

## Loan Parameters
- Minimum Loan Amount: $17,000
- Maximum Loan Amount: $495,500
- Loan Term: 12-84 months
- Required Information:
  - Applicant Income
  - Co-Applicant Income (if applicable)
  - Credit History
  - Loan Amount
  - Loan Term

## Installation
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install streamlit pandas scikit-learn joblib
```

3. Run the application:
```bash
streamlit run main.py
```

## Usage
1. Launch the application
2. Upload a CSV file with training data or use the pre-trained model
3. Enter the required loan application details
4. Click "Predict Loan Eligibility" to see results

## Dataset Format
Your CSV file should include these columns:
- Applicant Income (numeric)
- Co-Applicant Income (numeric)
- Loan Amount (numeric)
- Loan Term Months (numeric)
- Credit History (0 or 1)
- Application successful (0 or 1)

Example row:
```
5000,2000,150000,360,1,1