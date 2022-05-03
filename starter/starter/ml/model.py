import pickle

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.svm import SVC
import pandas as pd
from starter.ml import data
import os
from pathlib import Path


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = SVC()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_model_metrics_sliced(trained_model, test_data, categorical_features, encoder, lb, slice_features):
    """
    source: Exercise Solution in Chapter 2-10
    https://classroom.udacity.com/nanodegrees/nd0821/parts/cd0582/modules/3a477328-761d-425d-9a83-49ae5ac95bab/lessons/bc211316-f7d5-49a8-8649-66981cc48ac9/concepts/edbbba60-38af-4bd5-8cbb-e1f193a86bd7
    """
    from aequitas.group import Group

    X_test, y_test, encoder, lb = data.process_data(
        test_data, categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    pred = inference(trained_model, X_test)

    df_aq = test_data.reset_index(drop=True).copy()
    df_aq['label_value'] = y_test
    df_aq['score'] = pred

    group = Group()
    xtab, idxs = group.get_crosstabs(df_aq, attr_cols=slice_features)

    return xtab


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model(trained_model, encoder, lb, path):
    Path(path).mkdir(exist_ok=True)

    with open(os.path.join(path, "model.pkl"), "wb") as model_file:
        pickle.dump(trained_model, model_file)

    with open(os.path.join(path, "encoder.pkl"), "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)

    with open(os.path.join(path, "lb.pkl"), "wb") as lb_file:
        pickle.dump(lb, lb_file)


def load_model(path):
    with open(os.path.join(path, "model.pkl"), "rb") as model_file:
        model = pickle.load(model_file)

    with open(os.path.join(path, "encoder.pkl"), "rb") as encoder_file:
        encoder = pickle.load(encoder_file)

    with open(os.path.join(path, "lb.pkl"), "rb") as lb_file:
        lb = pickle.load(lb_file)

    return model, encoder, lb
