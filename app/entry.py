import streamlit as st
import numpy as np
import pandas as pd
from viz import plots  # Ensure your folder structure has __init__.py files

st.set_page_config(page_title="Normal Distribution ML", layout="wide")

st.title("📊 MNIST & Normal Distribution Analysis")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select Analysis", ["Normal Distribution", "MNIST Visuals", "Model Results"])

if page == "Normal Distribution":
    st.header("Normal Distribution Lab")
    
    col1, col2 = st.columns(2)
    with col1:
        u_val = st.slider("Mean (μ)", -10.0, 10.0, 0.0)
        s_val = st.slider("Std Dev (σ)", 0.1, 5.0, 1.0)
    
    with col2:
        # Calling the displot fix
        fig = plots.plot_normal_distribution(mean=u_val, std=s_val)
        st.pyplot(fig)

elif page == "MNIST Visuals":
    st.header("Data Exploration")
    
    # Create dummy data for demonstration if real data isn't loaded
    # Replace this with your actual data loading logic
    X_dummy = np.random.rand(100, 784) 
    
    st.subheader("Pixel Intensity Distribution")
    fig_box = plots.plot_pixel_boxplot(X_dummy, n_pixels=15)
    st.pyplot(fig_box)

elif page == "Model Results":
    st.header("Algorithm Performance")
    
    # Example: Confusion Matrix
    y_t = [0, 1, 0, 1, 1]
    y_p = [0, 1, 1, 1, 0]
    fig_cm = plots.plot_logistic_results(y_t, y_p)
    st.pyplot(fig_cm)