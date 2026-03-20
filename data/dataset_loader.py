from sklearn.datasets import fetch_openml

def load_mnist_dataset():
    # Fetch MNIST dataset (70,000 samples, 784 features)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"].astype(int)
    return X, y

