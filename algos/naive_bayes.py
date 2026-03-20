from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X, y):
    model = GaussianNB()
    model.fit(X, y)
    return model
