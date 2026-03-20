import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# Create graphs folder if missing
if not os.path.exists("graphs"):
    os.makedirs("graphs")

def save_and_show(title):
    plt.tight_layout()
    plt.savefig(f"graphs/{title.replace(' ', '_').lower()}.png")
    print(f"显示窗口: {title} (Close window to see next plot)")
    plt.show()

def plot_normal_distribution(data):
    plt.figure(figsize=(8, 5))
    sns.histplot(data, kde=True, bins=30, color='royalblue')
    plt.title("Pixel Intensity Distribution")
    save_and_show("Normal Distribution")

def plot_pixel_boxplot(X, start_pixel=350, n_pixels=30):
    plt.figure(figsize=(12, 6))
    # We slice the middle of the image to avoid empty borders
    sns.boxplot(data=X[:, start_pixel : start_pixel + n_pixels])
    plt.title(f"Boxplot of Pixels {start_pixel} to {start_pixel + n_pixels}")
    plt.xlabel("Pixel Index")
    plt.ylabel("Scaled Intensity")
    save_and_show("Pixel Boxplot")

def plot_pca(X_reduced, y):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab10', s=5, alpha=0.6)
    plt.colorbar(scatter, label="Digit Label")
    plt.title("PCA Projection (2D)")
    save_and_show("PCA Result")

def plot_gmm_clusters(X_reduced, labels):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', s=5)
    plt.title("GMM Clustering Results")
    save_and_show("GMM Clusters")

def plot_logistic_results(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_and_show("Logistic Results")

def plot_linear_regression(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.2, color='darkgreen')
    plt.plot([0, 9], [0, 9], 'r--', label="Perfect Prediction")
    plt.title("Linear Regression: True vs Predicted Digit")
    plt.xlabel("True Label")
    plt.ylabel("Continuous Prediction")
    plt.legend()
    save_and_show("Linear Results")

def plot_anomalies(X, anomalies):
    if len(anomalies) == 0:
        print("No anomalies to plot!")
        return
    n = min(20, len(anomalies))
    rows = (n // 10) + 1
    plt.figure(figsize=(12, 2 * rows))
    for i, idx in enumerate(anomalies[:n]):
        plt.subplot(rows, 10, i + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle("MNIST Anomaly Detection (Top 20)")
    save_and_show("Anomalies")