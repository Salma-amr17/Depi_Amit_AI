import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visuals import distribution, evaluate, feature_plot

# Streamlit page setup
st.set_page_config(page_title="ML Visualization Dashboard", layout="wide")

st.title("📊 Machine Learning Visualization Dashboard")

# Sidebar menu
st.sidebar.title("Options")
choice = st.sidebar.selectbox(
    "Select Visualization Type:",
    ["Distribution", "Model Evaluation", "Feature Importance"]
)

# ---- Distribution Visualization ----
if choice == "Distribution":
    st.header("Feature Distribution")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Preview of the dataset:")
        st.dataframe(data.head())

        st.write("Generating distribution plots for capital-gain and capital-loss columns...")
        distribution(data)
        st.pyplot(plt)

    else:
        st.info("Please upload a CSV file to display distributions.")

# ---- Model Evaluation Visualization ----
elif choice == "Model Evaluation":
    st.header("Model Evaluation")
    st.write("Example visualization for comparing multiple supervised models.")
    
    # Example (dummy) data for demonstration
    results = {
        'Model A': [{'train_time': 1, 'acc_train': 0.9, 'f_train': 0.8,
                     'pred_time': 0.2, 'acc_test': 0.85, 'f_test': 0.75}] * 3,
        'Model B': [{'train_time': 0.8, 'acc_train': 0.88, 'f_train': 0.77,
                     'pred_time': 0.1, 'acc_test': 0.83, 'f_test': 0.73}] * 3,
        'Model C': [{'train_time': 0.5, 'acc_train': 0.86, 'f_train': 0.74,
                     'pred_time': 0.05, 'acc_test': 0.8, 'f_test': 0.7}] * 3,
    }
    accuracy = 0.6
    f1 = 0.5

    evaluate(results, accuracy, f1)
    st.pyplot(plt)

# ---- Feature Importance Visualization ----
elif choice == "Feature Importance":
    st.header("Feature Importance")
    st.write("Displays the top 5 most important features based on their normalized weights.")
    
    # Example dummy data
    X_train = pd.DataFrame(np.random.rand(100, 10), columns=[f"Feature {i}" for i in range(10)])
    y_train = np.random.randint(0, 2, size=100)
    importances = np.random.rand(10)
    
    feature_plot(importances, X_train, y_train)
    st.pyplot(plt)

st.markdown("---")
st.caption("Developed with ❤ using Streamlit and Matplotlib.")