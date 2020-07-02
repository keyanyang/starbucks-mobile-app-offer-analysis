import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def data_preparation(df, test_size, random_state=0):
    """Standardize data then permute and split DataFrame index into train and test.
    Parameters
    ----------
    df: pandas.DataFrame
    test_size: float
        Fraction between 0.0 and 1.0
    random_state: int

    Returns
    -------
    tuple of numpy.ndarray,
        X_train, X_test, y_train, y_test
    """

    scl = StandardScaler()
    independent_vars = df.columns.drop(['is_fully_paid', 'id'])
    df_clean = pd.DataFrame(scl.fit_transform(df[independent_vars]),
                            columns=independent_vars)
    df_clean['id'] = df['id']
    df_clean['is_fully_paid'] = df['is_fully_paid']

    logging.info("Splitting the data-frame into train and test parts")

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        df_clean.drop(['is_fully_paid', 'id'], axis=1).values,
        df_clean['is_fully_paid'].values,
        df_clean['id'].values,
        test_size=test_size,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test, idx_train, idx_test


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
    The ridge parameter is found using 10-fold cross-validation.
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
    from sklearn.model_selection import GridSearchCV

    lr = LogisticRegression(random_state=0, solver='lbfgs')

    param_range = [2 ** x for x in range(-10, 10)]

    gs = GridSearchCV(
        estimator=lr,
        param_grid={'C': param_range},
        cv=10
    )

    gs.fit(X_train, y_train)

    return gs


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
    from sklearn.model_selection import RandomizedSearchCV

    rf = RandomForestClassifier(criterion='gini', random_state=0)

    rs = RandomizedSearchCV(
        estimator=rf,
        param_distributions={'n_estimators': [10, 100]},
        cv=10,
        random_state=10)

    rs.fit(X_train, y_train)

    return rs
