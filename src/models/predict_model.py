import pandas as pd
from sklearn.metrics import classification_report


def build_classification_report(model, y_test, X_test):
    """
    Build a text report showing the main classification metrics.

    Parameters
    ----------
    model: classifier
        The dataset to be split into train and test set
    y_test: numpy.ndarray
        Correct target values.
    X_test: numpy.ndarray
        Data set used to estimate targets.

    Returns
    -------
    string
        Text summary of the precision, recall, F1 score for each class.
    """

    return classification_report(y_true=y_test, y_pred=model.predict(X_test))


def make_predictions(clf, sample_set, y_actual):
    """
    Build a text report showing the main classification metrics.

    Parameters
    ----------
    clf: classifier
        The trained model.
    sample_set: numpy.ndarray
        Data set used to estimate targets.
    y_actual: numpy.ndarray
        Correct target values.

    Returns
    -------
    pandas.DataFrame
        Sample predictions and acutal values.
    """

    probs = clf.predict_proba(sample_set)[:, 1]
    probs = ['{:.0%}'.format(prob) for prob in probs]
    res = pd.DataFrame(data={
            'Prediction probability user uses this offer': probs,
            'Whether the offer is actually used': y_actual.astype(bool)})
    return res
