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
