"""
Check Score procedure
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
import ml
import logging


def check_score():
    """
    Execute score checking
    """
    df = pd.read_csv("data/census_clean.csv")
    _, test = train_test_split(df, test_size=0.20)

    trained_model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    slice_values = []

    for cat in ml.get_cat_features():
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = ml.process_data(
                df_temp,
                categorical_features=ml.get_cat_features(),
                label="salary", encoder=encoder, lb=lb, training=False)

            y_preds = trained_model.predict(X_test)

            prc, rcl, fb = ml.compute_model_metrics(y_test,
                                                    y_preds)

            line = "[%s->%s] Precision: %s " \
                   "Recall: %s FBeta: %s" % (cat, cls, prc, rcl, fb)
            logging.info(line)
            slice_values.append(line)

    with open('data/model/slice_output.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')