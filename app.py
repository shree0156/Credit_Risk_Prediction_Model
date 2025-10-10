import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load model and data
model = pickle.load(open("credit_model.pkl", "rb"))
augmented_df = pd.read_csv("augmented_credit_data.csv")

# Prepare label encoders using the augmented dataset
label_encoders = {}
categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for col in categorical_columns:
    le = LabelEncoder()
    le.fit(augmented_df[col])
    label_encoders[col] = le

st.title("Credit Risk Prediction App")

st.sidebar.header("Enter Applicant Information")

# Sidebar inputs
age = st.sidebar.slider("Age", 18, 80, 35)
income = st.sidebar.number_input("Annual Income ($)", 10000, 200000, 50000)
emp_length = st.sidebar.slider("Employment Length (Years)", 0, 40, 5)
home_ownership = st.sidebar.selectbox("Home Ownership", label_encoders['person_home_ownership'].classes_)
loan_intent = st.sidebar.selectbox("Loan Intent", label_encoders['loan_intent'].classes_)
loan_grade = st.sidebar.selectbox("Loan Grade", label_encoders['loan_grade'].classes_)
loan_amnt = st.sidebar.number_input("Loan Amount", 500, 50000, 10000)
interest_rate = st.sidebar.slider("Loan Interest Rate (%)", 2.0, 30.0, 10.0)
percent_income = st.sidebar.number_input("Loan % of Income", 0.0, 1.0, 0.2)
default_on_file = st.sidebar.selectbox("Default on File", label_encoders['cb_person_default_on_file'].classes_)
cred_hist_len = st.sidebar.slider("Credit History Length", 0, 40, 5)

# Combine inputs
input_data = pd.DataFrame({
    'person_age': [age],
    'person_income': [income],
    'person_home_ownership': [home_ownership],
    'person_emp_length': [emp_length],
    'loan_intent': [loan_intent],
    'loan_grade': [loan_grade],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [interest_rate],
    'loan_percent_income': [percent_income],
    'cb_person_default_on_file': [default_on_file],
    'cb_person_cred_hist_length': [cred_hist_len]
})

# Apply label encoding
for col in categorical_columns:
    input_data[col] = label_encoders[col].transform(input_data[col])

# Prediction
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)[0]
    result = "High Risk (Likely to Default)" if prediction == 1 else "Low Risk (Likely to Repay)"
    st.subheader(f"Prediction: {result}")

# Visualization
st.subheader("Data Overview & Insights")
st.write("Sample of augmented dataset:")
st.dataframe(augmented_df.head())

fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(data=augmented_df, x="loan_status", palette="coolwarm", ax=ax)
st.pyplot(fig)
