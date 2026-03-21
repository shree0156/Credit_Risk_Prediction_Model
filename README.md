# Credit Risk Prediction Engine

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red?style=flat-square)
![CTGAN](https://img.shields.io/badge/CTGAN-Synthetic%20Data-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-ff4b4b?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Pipeline-orange?style=flat-square)

> Predicts loan default risk in real time using an XGBoost classifier trained on CTGAN-augmented data — achieving ROC-AUC 0.98 with a +17% recall improvement on the minority class.

---

## The Problem

Credit default prediction suffers from severe class imbalance — defaulters are a small minority of borrowers, so standard models learn to ignore them. Missing a true defaulter is the most expensive error a lender can make. This project directly tackles that imbalance using synthetic data generation, not just oversampling.

---

## How It Works

```
Raw Credit Dataset (imbalanced — few defaulters)
         │
         ▼
  EDA & Feature Engineering
  ─ debt-to-income ratio
  ─ repayment history flags
  ─ credit utilisation bands
         │
         ▼
  CTGAN Synthetic Data Generation
  ─ trains a GAN on the minority class
  ─ generates realistic synthetic defaulters
  ─ augments training data without duplication
         │
         ▼
  XGBoost Classifier
  ─ trained on augmented dataset
  ─ hyperparameter tuned
         │
         ▼
  Model Evaluation
  ─ ROC-AUC: 0.98
  ─ Recall improvement (minority class): +17%
  ─ PCA-based risk distribution visualisation
         │
         ▼
  Streamlit Web App
  ─ real-time borrower risk scoring
  ─ prediction probability display
  ─ PCA risk distribution plot
```

---

## Results

| Metric | Before Augmentation | After CTGAN Augmentation |
|--------|--------------------|-----------------------|
| ROC-AUC | baseline | **0.98** |
| Minority class recall | baseline | **+17% improvement** |
| Class balance | Imbalanced | Augmented with synthetic defaulters |

**Why CTGAN over SMOTE?**
SMOTE creates synthetic samples by interpolating between existing minority samples — it stays close to the original data distribution. CTGAN trains a Generative Adversarial Network specifically on tabular data, learning the underlying statistical relationships and generating more realistic, diverse synthetic samples. For financial data with complex feature correlations, CTGAN produces higher-quality augmentation.

---

## App Demo

<img width="1722" height="1022" alt="app credit" src="https://github.com/user-attachments/assets/a29e3179-d77c-4ce1-915c-9fee08bed206" />

**How to use the app:**
1. Enter borrower details in the sidebar (income, loan amount, interest rate, employment length, etc.)
2. Click **Predict** to get the loan risk status
3. View prediction probability and PCA-based risk distribution

---

## Project Structure

```
Credit_Risk_Prediction_Model/
│
├── Synthetic_Credit_Scoring.ipynb         # Full notebook: EDA, CTGAN, XGBoost, evaluation
├── app.py                                  # Streamlit web application
├── credit_model.pkl                        # Trained XGBoost model (serialised)
├── augmented_credit_data.csv               # Preprocessed + CTGAN-augmented dataset
├── credit_risk_dataset.csv                 # Original raw dataset
├── Credit Risk Prediction Model.pdf        # Project report
├── Credit Risk Prediction Model ppt.pptx  # Project presentation
└── README.md
```

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/shree0156/Credit_Risk_Prediction_Model.git
cd Credit_Risk_Prediction_Model
```

**2. Install dependencies**
```bash
pip install streamlit pandas scikit-learn xgboost numpy matplotlib seaborn
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

> **Note:** `app.py`, `credit_model.pkl`, and `augmented_credit_data.csv` must all be in the same directory.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| ML model | XGBoost (gradient boosting classifier) |
| Synthetic data | CTGAN (Conditional Tabular GAN) |
| Evaluation | Scikit-learn (ROC-AUC, Precision, Recall, F1) |
| Dimensionality reduction | PCA (Scikit-learn) |
| Web application | Streamlit |
| Model serialisation | Pickle (.pkl) |
| Data handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |

---

## Key Design Decisions

**Why XGBoost?** Gradient boosting consistently outperforms linear models on structured/tabular financial data. It handles missing values natively, is robust to outliers, and its tree structure captures non-linear interactions between features like income × loan amount.

**Why CTGAN over SMOTE?** SMOTE interpolates between existing minority samples — staying too close to known data points. CTGAN learns the full joint distribution of the minority class and generates genuinely novel realistic samples, reducing the risk of overfitting to interpolated artefacts.

**Why PCA visualisation in the app?** Loan risk isn't a binary yes/no in practice — it exists on a spectrum. The PCA scatter plot shows where a new borrower falls *relative to the full population* of defaulters and payers, giving the analyst spatial context beyond just a probability score.

---

## Future Improvements

- [ ] Integrate real-time financial data from credit bureau APIs
- [ ] Add SHAP explainability to show which features drove each prediction
- [ ] Explore deep learning and neural network ensembles for further accuracy gains
- [ ] Deploy to cloud (AWS / Azure / GCP) with an API endpoint
- [ ] Add bias and fairness analysis for synthetic data usage

---

## Author

**Shreeja Maiya**

---

*If you found this useful, consider giving it a ⭐*

