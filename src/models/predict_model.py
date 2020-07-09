import pandas as pd
from sklearn.metrics import classification_report


def build_classification_report(model, y_test, X_test):
    return classification_report(y_true=y_test, y_pred=model.predict(X_test))


def make_predictions(clf, sample_set, y_actual):
    probs = clf.predict_proba(sample_set)[:, 1]
    probs = ['{:.0%}'.format(prob) for prob in probs]
    res = pd.DataFrame(data={
            'Prediction probability user uses this offer': probs,
            'Whether the offer is actually used': y_actual.astype(bool)})
    return res
