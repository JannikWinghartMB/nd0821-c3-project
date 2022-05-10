import os
from pathlib import Path
import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.svm import SVC
import pandas as pd

from . import data


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
    slice_metrics_report = pd.DataFrame(columns=["feature", "value", "precision", "recall", "fbeta"])
    for slice_feature in slice_features:
        for value in list(test_data[slice_feature].unique()):
            slice_test_data = test_data[test_data[slice_feature] == value]

            X_test_slice, y_test_slice, encoder, lb = data.process_data(
                slice_test_data, categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb
            )

            y_pred_slice = inference(trained_model, X_test_slice)

            slice_metrics = compute_model_metrics(y_test_slice, y_pred_slice)

            slice_metrics_report = pd.concat(
                [
                    slice_metrics_report,
                    pd.DataFrame([{
                        "feature": slice_feature,
                        "value": value,
                        "precision": slice_metrics[0],
                        "recall": slice_metrics[1],
                        "fbeta": slice_metrics[2]
                    }])
                ],
                ignore_index=True
            )
    return slice_metrics_report


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
