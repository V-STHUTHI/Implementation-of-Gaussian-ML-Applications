import numpy as np
from data.dataset_loader import load_mnist_dataset
from prep.normalize import z_score_normalization, scale_pixels
from algos.log_reg import train_logistic_regression
from algos.lin_reg import train_linear_regression
from algos.gmm import train_gmm
from algos.pca import apply_pca
from algos.anomaly import detect_anomalies
from viz.plots import *

def run_experiments():
    print("--- 📥 PHASE 1: LOADING DATA ---")
    X, y = load_mnist_dataset()
    # Subsampling 2000 images makes it fast and accurate
    X, y = X[:2000], y[:2000]
    
    print("--- 🧹 PHASE 2: PREPROCESSING ---")
    X_norm = z_score_normalization(X)
    X_scaled = scale_pixels(X)

    print("--- 📊 PHASE 3: RUNNING PLOTS ---")
    
    # 1. Distribution
    plot_normal_distribution(X_scaled.flatten())
    
    # 2. Boxplot (Focusing on center pixels to avoid flat lines)
    plot_pixel_boxplot(X_scaled, start_pixel=380, n_pixels=30)
    
    # 3. PCA
    res = apply_pca(X_norm, n_components=2)
    X_red = res[0] if isinstance(res, tuple) else res
    plot_pca(X_red, y)
    
    # 4. GMM
    model_gmm = train_gmm(X_red, n_components=10)
    plot_gmm_clusters(X_red, model_gmm.predict(X_red))
    
    # 5. Logistic
    model_log = train_logistic_regression(X_norm, y)
    plot_logistic_results(y, model_log.predict(X_norm))
    
    # 6. Linear
    model_lin = train_linear_regression(X_norm, y)
    plot_linear_regression(y, model_lin.predict(X_norm))
    
    # 7. Anomalies
    anoms = detect_anomalies(X_norm, threshold=3)
    plot_anomalies(X, anoms)

    print("\n✅ All 7 experiments complete. Windows closed. Check /graphs!")

if __name__ == "__main__":
    run_experiments()