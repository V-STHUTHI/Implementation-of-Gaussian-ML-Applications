import numpy as np

def z_score_normalization(X):
    """Apply z-score normalization to dataset."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)

def scale_pixels(X):
    """Scale MNIST pixel values to [0,1]."""
    return X / 255.0

def crop_borders(X, crop_size=4):
    """
    Crop borders from MNIST images.
    Default removes 4 pixels from each side (28x28 -> 20x20).
    """
    X_images = X.reshape(-1, 28, 28)
    X_cropped = X_images[:, crop_size:28-crop_size, crop_size:28-crop_size]
    return X_cropped.reshape(-1, (28 - 2*crop_size) ** 2)

def remove_low_variance_pixels(X, threshold=50):
    """
    Remove pixels with variance below threshold.
    Returns reduced dataset with only informative pixels.
    """
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return X[:, mask]
