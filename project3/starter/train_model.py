"""
Train model procedure
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import ml


def train_test_model():
    """
    Execute model training
    """
    df = pd.read_csv("project3/data/census_clean.csv", index_col=False)
    train, _ = train_test_split(df, test_size=0.20)

    X_train, y_train, encoder, lb = ml.process_data(
        train, categorical_features= ml.get_cat_features(),
        label="salary", training=True
    )

    model = ml.train_model(X_train, y_train)

    with open("../model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("../model/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    with open("../model/labeler.pkl", "wb") as f:
        pickle.dump(lb, f)