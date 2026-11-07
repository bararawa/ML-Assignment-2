import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="ML Classifier Evaluation Portal", layout="wide")

st.title("🧠 ML Classifier Evaluation Portal")
st.markdown('''
Welcome to the Classifier Evaluation Portal! This interactive dashboard loads pre-trained machine learning models
and evaluates their performance on your uploaded test datasets on the fly.
''')

# Sidebar options
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV Data", type=["csv"])

model_option = st.sidebar.selectbox(
    "Select Classifier Model",
    ["Logistic_Regression", "Decision_Tree", "KNN", "Naive_Bayes", "Support_Vector_Machine"]
)

# Cache model loading to optimize performance
@st.cache_resource
def load_model(name):
    model_path = f"model/{name}.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

if uploaded_file is not None:
    # Load uploaded dataset
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")

    # Show dataset preview
    st.subheader("📊 Test Dataset Preview")
    st.dataframe(df.head())

    # Features and target columns validation
    if 'target' in df.columns:
        X_test = df.drop(columns=['target'])
        y_test = df['target']

        # Load the selected model
        model = load_model(model_option)

        if model is not None:
            # Make predictions without retraining
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)

            # Evaluate metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            st.subheader(f"🎯 Evaluation Results: {model_option.replace('_', ' ')}")

            # Metrics layout
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")

            # Show Classification Report & Confusion Matrix side-by-side
            st.write("---")
            layout_col1, layout_col2 = st.columns(2)

            with layout_col1:
                st.subheader("📝 Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df)

            with layout_col2:
                st.subheader("📊 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'], ax=ax)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                st.pyplot(fig)
        else:
            st.error(f"Could not load {model_option}.joblib. Please check the model directory.")
    else:
        st.error("The dataset must contain a column named 'target' containing your ground-truth labels.")
else:
    st.info("💡 Please upload the 'test_data.csv' file from the sidebar to visualize and interact with the metrics.")
