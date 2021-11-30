"""
Unit tests for ml model
"""
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import pytest
import ml
from joblib import load


@pytest.fixture
def data():
    """
    Get dataset
    """
    df = pd.read_csv("data/census_clean.csv")
    return df


def test_process_data(data):
    """
    Check split have same number of rows for X and y
    """
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    X_test, y_test, _, _ = ml.process_data(
        data,
        categorical_features=ml.get_cat_features(),
        label="salary", encoder=encoder, lb=lb, training=False)

    assert len(X_test) == len(y_test)


def test_process_encoder(data):
    """
    Check splits have same number of rows for X and y
    """
    encoder_test = load("model/encoder.joblib")
    lb_test = load("data/model/lb.joblib")

    _, _, encoder, lb = ml.process_data(
        data,
        categorical_features=ml.get_cat_features(),
        label="salary", training=True)

    _, _, _, _ = ml.process_data(
        data,
        categorical_features=ml.get_cat_features(),
        label="salary", encoder=encoder_test, lb=lb_test, training=False)

    assert encoder.get_params() == encoder_test.get_params()
    assert lb.get_params() == lb_test.get_params()


def test_inference():
    """
    Check inference performance
    """
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    array = np.array([[
                     32,
                     "Private",
                     "Some-college",
                     "Married-civ-spouse",
                     "Exec-managerial",
                     "Husband",
                     "Black",
                     "Male",
                     80,
                     "United-States"
                     ]])
    df_temp = DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    X, _, _, _ = ml.process_data(
                df_temp,
                categorical_features=ml.get_cat_features(),
                encoder=encoder, lb=lb, training=False)
    pred = ml.inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == ">50K"