from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X, y):
    # Simplified for compatibility across sklearn versions
    model = LogisticRegression(max_iter=200, solver='lbfgs')
    model.fit(X, y)
    return model


