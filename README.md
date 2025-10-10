# Credit Risk Prediction Model

This project develops a **Credit Risk Prediction Model** to identify potential loan defaulters using machine learning and synthetic data generation. The model leverages **XGBoost** trained on a combination of real and synthetic (CTGAN) data, addressing class imbalance and improving predictive performance. A **Streamlit web application** is deployed for real-time predictions.

---

## **Features**

- Predict whether a loan will be **fully paid** or **defaulted**.
- Handles **class imbalance** using CTGAN synthetic data generation.
- Provides **high performance**:
  - ROC-AUC: 0.98
  - Recall improvement for minority class: +17%
- Interactive **Streamlit dashboard** for real-time prediction.
- Visual insights with **PCA and risk distribution plots**.

---

## **Project Structure**

├── data/
│ ├── augmented_credit_data.csv # Preprocessed + synthetic dataset
├── app.py # Streamlit app
├── credit_model.pkl # Trained XGBoost model
├── notebooks/ # EDA and model development notebooks
├── README.md
├── requirements.txt # Python dependencies

## Working of App
Run the following code locally, 
**streamlit run app.py**
in cmd prompt
make sure u have saved the above files i.e app.py, credit_model.pkl, and augmented_credit_data.csv in a folder.

### Usage
Enter borrower details in the sidebar (income, loan amount, interest rate, etc.).

Click Predict to get the loan risk status.

View prediction probabilities and PCA-based risk distribution visualizations.

## Technologies Used
XGBoost – Gradient boosting classifier for robust predictions.

CTGAN – Synthetic data generation to balance classes.

Streamlit – Web app for interactive predictions.

Python Libraries: pandas, scikit-learn, numpy, matplotlib, seaborn, pickle.

## Future Work
Integrate real-time financial data from credit bureaus.

Explore deep learning & ensemble models for further accuracy.

Deploy through APIs or cloud platforms (AWS, Azure, GCP).

Bias and fairness analysis for synthetic data usage.

Shreeja M
