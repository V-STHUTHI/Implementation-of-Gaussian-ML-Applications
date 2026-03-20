import streamlit as st
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np

# --- 1. SYSTEM PATH SETUP ---
# This ensures the frontend can "see" the algos, viz, and prep folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 2. MODULAR IMPORTS ---
try:
    from data.dataset_loader import load_mnist_dataset
    from prep.normalize import z_score_normalization, scale_pixels
    from algos import pca, gmm, log_reg, lin_reg, anomaly
    from viz import plots
except ImportError as e:
    st.error(f"❌ Module Import Error: {e}. Check your folder structure!")

# --- 3. PAGE CONFIG ---
st.set_page_config(page_title="MNIST ML Analysis", layout="wide", page_icon="🔢")

# --- 4. DATA ENGINE (Cached) ---
@st.cache_data
def get_data_bundle():
    # Load and Subsample for speed
    X_raw, y = load_mnist_dataset()
    X_raw, y = X_raw[:1000], y[:1000] 
    
    # Preprocess
    X_norm = z_score_normalization(X_raw)
    X_scaled = scale_pixels(X_raw)
    
    return X_raw, X_norm, X_scaled, y

# Load everything once
X_raw, X_norm, X_scaled, y = get_data_bundle()

# --- 5. NAVIGATION LOGIC ---
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

def navigate_to(page_name):
    st.session_state.page = page_name

# --- 6. VIEW: MAIN DASHBOARD ---
if st.session_state.page == "Dashboard":
    st.title("🔢 MNIST Machine Learning Workbench")
    st.write("Welcome, Sthuthi! Explore the mathematical properties of the MNIST dataset through various ML lenses.")

    # Data Inspection Section
    st.header("Data Intelligence")
    tab1, tab2, tab3 = st.tabs(["📄 Raw (0-255)", "✨ Scaled (0-1)", "📊 Statistics"])
    
    with tab1:
        st.dataframe(pd.DataFrame(X_raw).head(10).style.background_gradient(cmap='Greys'), use_container_width=True)
    with tab2:
        st.dataframe(pd.DataFrame(X_scaled).head(10).style.background_gradient(cmap='Blues'), use_container_width=True)
    with tab3:
        st.table(pd.DataFrame({
            "Metric": ["Mean", "Std Dev", "Min", "Max"],
            "Raw Data": [X_raw.mean(), X_raw.std(), X_raw.min(), X_raw.max()],
            "Scaled Data": [X_scaled.mean(), X_scaled.std(), X_scaled.min(), X_scaled.max()]
        }))

    st.divider()

    # Visualizations Grid
    st.header("🎨 Research Visualizations")
    st.write("Click a button to view the high-resolution analysis in a new page.")
    
    c1, c2, c3 = st.columns(3)
    if c1.button("📈 Normal Distribution", use_container_width=True): navigate_to("Normal")
    if c2.button("📦 Pixel Boxplot", use_container_width=True): navigate_to("Boxplot")
    if c3.button("🔍 Anomaly Detection", use_container_width=True): navigate_to("Anomaly")

    c4, c5, c6 = st.columns(3)
    if c4.button("💠 PCA Projection", use_container_width=True): navigate_to("PCA")
    if c5.button("🧬 GMM Clustering", use_container_width=True): navigate_to("GMM")
    if c6.button("📉 Linear Regression", use_container_width=True): navigate_to("Linear")

    if st.button("🎯 Logistic Regression Matrix", use_container_width=True): navigate_to("Logistic")

# --- 7. VIEW: INDIVIDUAL GRAPH PAGES ---
else:
    if st.button("⬅️ Back to Dashboard"):
        navigate_to("Dashboard")
        st.rerun()

    current_page = st.session_state.page
    st.header(f"Analysis: {current_page}")
    st.divider()

    # Determine which modular function to trigger
    if current_page == "Normal":
        plots.plot_normal_distribution(X_scaled.flatten())
    elif current_page == "Boxplot":
        plots.plot_pixel_boxplot(X_scaled, start_pixel=380)
    elif current_page == "PCA":
        res = pca.apply_pca(X_norm, n_components=2)
        plots.plot_pca(res[0] if isinstance(res, tuple) else res, y)
    elif current_page == "GMM":
        res = pca.apply_pca(X_norm, n_components=2)
        X_red = res[0] if isinstance(res, tuple) else res
        plots.plot_gmm_clusters(X_red, gmm.train_gmm(X_red, 10).predict(X_red))
    elif current_page == "Logistic":
        plots.plot_logistic_results(y, log_reg.train_logistic_regression(X_norm, y).predict(X_norm))
    elif current_page == "Linear":
        plots.plot_linear_regression(y, lin_reg.train_linear_regression(X_norm, y).predict(X_norm))
    elif current_page == "Anomaly":
        plots.plot_anomalies(X_raw, anomaly.detect_anomalies(X_norm))

    # Display the plot in high-res full width
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close()