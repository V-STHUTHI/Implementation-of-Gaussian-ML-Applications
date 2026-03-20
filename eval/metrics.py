from sklearn.metrics import accuracy_score, f1_score

def evaluate_classification(model, X, y):
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)

def evaluate_f1(model, X, y):
    y_pred = model.predict(X)
    return f1_score(y, y_pred, average='weighted')
