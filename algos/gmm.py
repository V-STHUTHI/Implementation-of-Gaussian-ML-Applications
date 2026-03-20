from sklearn.mixture import GaussianMixture

def train_gmm(X, n_components=10):
    model = GaussianMixture(n_components=n_components, covariance_type='full')
    model.fit(X)
    return model
