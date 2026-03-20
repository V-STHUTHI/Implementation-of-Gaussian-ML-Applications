import numpy as np

def detect_anomalies(X, threshold=3):
    """
    Detect anomalies using the empirical rule (±3σ).
    Returns indices of anomalous samples.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Z-score for each sample
    z_scores = (X - mean) / (std + 1e-8)

    # Flag anomalies if any feature exceeds threshold
    anomalies = np.where(np.abs(z_scores).max(axis=1) > threshold)[0]
    return anomalies
