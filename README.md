# M.Tech (AIML) - Machine Learning Assignment 2

## BITS Pilani - Work Integrated Learning Programmes (WILP)

---

### 👥 Student Details
- **Student Name:** Ravi Bararawa
- **GitHub Username:** Bararawa
- **Repository Link:** https://github.com/Bararawa/ML-Assignment-2
- **Live Streamlit App Link:** https://ml-assignment-2-portal.streamlit.app
- **Submission Date:** 07-Nov-2026

---

## 1. Project Deliverables Checklist
- [x] GitHub Repository created and active with clean commit history
- [x] Pre-trained models serialized and saved in `/model` folder
- [x] Test dataset exported separately as `test_data.csv` 
- [x] Fully interactive Streamlit dashboard (`app.py`) deployed on Streamlit Community Cloud
- [x] Final Word report (`Assignment_2_Submission_Report.docx`) with comparisons produced

---

## 2. Project Directory Structure
```
project-folder/
├── app.py                      # Streamlit application entry point
├── requirements.txt            # System and app dependencies
├── README.md                   # Project description and evaluation matrix
├── test_data.csv               # 200 instances used in evaluating models
├── Assignment_2_Submission_Report.docx # Complete final report document
└── model/
    ├── Logistic_Regression.joblib
    ├── Decision_Tree.joblib
    ├── KNN.joblib
    ├── Naive_Bayes.joblib
    └── Support_Vector_Machine.joblib
```

---

## 3. Dataset Description
- **Dataset Source:** Programmatically generated synthetic high-dimensional customer metrics to represent customer churn behavior.
- **Total Instances:** 1,000 (800 rows for model training, 200 rows held out for test-set evaluation in `test_data.csv`).
- **Total Features:** 12 numerical input features (named `feature_1` to `feature_12`).
- **Target Class:** Binary (`0` = Loyal/Stayed, `1` = Churned/Left).

---

## 4. Models Implemented
The following five classification models were fully trained on identical training sets and evaluated on the test set:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **Naive Bayes (Gaussian)**
5. **Support Vector Machine (SVM)**

---

## 5. Performance Comparison Matrix
Evaluation metrics calculated on the 200-row `test_data.csv` partition without retraining:

| Model Name | Accuracy | AUC Score | Precision | Recall | F1-Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.6750 | 0.7810 | 0.6733 | 0.6800 | 0.6766 | 0.3500 |
| **Decision Tree** | 0.7500 | 0.7500 | 0.7232 | 0.8100 | 0.7642 | 0.5036 |
| **KNN** | 0.9100 | 0.9515 | 0.8796 | 0.9500 | 0.9135 | 0.8226 |
| **Naive Bayes** | 0.7100 | 0.7807 | 0.7143 | 0.7000 | 0.7071 | 0.4201 |
| **Support Vector Machine** | 0.9050 | 0.9453 | 0.8716 | 0.9500 | 0.9091 | 0.8133 |

---

## 6. Performance Observations & Analysis
- **K-Nearest Neighbors (KNN)** achieved the highest performance metrics, closely followed by **Support Vector Machine (SVM)**, which demonstrates outstanding efficacy in parsing complex non-linear decision spaces.
- **Decision Tree Classifier** achieved moderate metrics but displays expected boundary splits.
- **Naive Bayes** provides an excellent, fast probability benchmark but is constrained by independent probability assumptions on correlated features.
- **Logistic Regression** represents the baseline classifier displaying boundaries built primarily on linear margins.

---

## 7. App Deployment & Execution
### Prerequisites
- Python 3.8+
- Package dependencies listed in `requirements.txt`

### Running Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/Bararawa/ML-Assignment-2.git
   cd ML-Assignment-2
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the dashboard application:
   ```bash
   streamlit run app.py
   ```
4. Open your browser to the local address specified, drag and drop `test_data.csv` into the file uploader widget, choose your target classifier, and analyze performance reports!
