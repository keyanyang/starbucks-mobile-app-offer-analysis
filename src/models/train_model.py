import logging
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def data_preparation(df, y, test_size=0.2, random_state=0):
    """Standardize data then permute and split DataFrame index into train and test.
    Parameters
    ----------
    df: pandas.DataFrame
    y: str
    test_size: float
        Fraction between 0.0 and 1.0
    random_state: int

    Returns
    -------
    tuple of numpy.ndarray,
        X_train, X_test, y_train, y_test
    """

    logging.info("Splitting the data-frame into train and test parts")

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop([y], axis=1).values,
        df[y].values,
        test_size=test_size,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


class MajorityVoteClassifier:
    """Majority Vote Classifier
    This class contains the `fit` and `predict` methods that are compatible
    with the SciKit-Learn model classes.
    """

    def __init__(self):
        self.majority_vote = None

    def fit(self, X, y):
        from statistics import mode

        self.majority_vote = mode(y)
        return self

    def predict(self, X):
        if self.majority_vote is None:
            raise ValueError("The majority vote classifier has to be trained \
                                before making predictions")
        return [self.majority_vote] * len(X)


def run_majority_vote(X_train, y_train):
    """Use the majority vote to predict survival.
    Parameters
    ----------
    X_train: numpy.ndarray
    y_train: numpy.ndarray

    Returns
    -------
    classifier: Fitted estimator
    """

    logging.info("Running the majority vote classifier")

    majority_vote_classifier = MajorityVoteClassifier()
    majority_vote_classifier.fit(X_train, y_train)

    return majority_vote_classifier


def run_logistic_regression(X_train, y_train):
    """Use ridge logistic regression to train model.
    The ridge parameter is found using 5-fold cross-validation.
    Parameters
    ----------
    X_train: numpy.ndarray
    y_train: numpy.ndarray

    Returns
    -------
    classifier: Fitted estimator
    """

    logging.info("Running the ridge logistic regression classifier")

    from sklearn.linear_model import LogisticRegression

    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear', random_state=0))
    ])

    param_grid = {
        'clf__penalty': ['l1', 'l2'],
        'clf__C': np.logspace(-4, 4, 20)
    }

    cv = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv = 5,
        verbose=False)

    cv.fit(X_train, y_train)

    return cv


def run_random_forest(X_train, y_train):
    """Use random forest to train model.
    The max_features parameter is found using 10-fold cross-validation.
    Parameters
    ----------
    X_train: numpy.ndarray
    y_train: numpy.ndarray

    Returns
    -------
    classifier: Fitted estimator
    """

    logging.info("Running the random forest classifier")

    from sklearn.ensemble import RandomForestClassifier

    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('clf', RandomForestClassifier(criterion='gini', random_state=0))
    ])

    param_grid = [
        {'clf' : [RandomForestClassifier()],
        'clf__n_estimators' : [10, 50, 100]}
    ]

    cv = GridSearchCV(
        pipeline,
        param_grid = param_grid,
        cv=5,
        verbose=False)

    cv.fit(X_train, y_train)

    return cv
