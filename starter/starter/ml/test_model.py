import model
import pytest
import numpy as np
from sklearn.svm import SVC


@pytest.fixture()
def x_train():
    x_train = np.array([[1, 1], [1, 1], [2, 2], [2, 2]])
    return x_train


@pytest.fixture()
def y_train():
    y_train = np.array([0, 0, 1, 1])
    return y_train


def test_train_model_typecheck(x_train, y_train):
    # Arange

    # Act
    returned_value = model.train_model(x_train, y_train)

    # Assert
    assert isinstance(returned_value, SVC)


def test_compute_model_metrics_same_value():
    # Arange
    y = np.array([0, 0, 1, 1])
    preds = np.array([0, 0, 1, 1])

    # Act
    precision, recall, fbeta = model.compute_model_metrics(y, preds)

    # Assert
    assert precision == 1
    assert recall == 1
    assert fbeta == 1


def test_compute_model_metrics_different_value():
    # Arange
    y = np.array([0, 0, 1, 1])
    preds = np.array([1, 1, 0, 0])

    # Act
    precision, recall, fbeta = model.compute_model_metrics(y, preds)

    # Assert
    assert precision < 1
    assert recall < 1
    assert fbeta < 1


def test_inference_typecheck(x_train, y_train):
    # Arange
    trained_model = model.train_model(x_train, y_train)
    x_test = np.array([[1, 1], [2, 2]])

    # Act
    predictions = model.inference(trained_model, x_test)

    # Assert
    assert len(predictions) == 2
