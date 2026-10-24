import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import compute_model_metrics, inference
from ml.data import process_data


def test_compute_model_metrics():
    """
    This test verifies evaluation metrics.  it checks if precision, recall, and F-beta fall
    within the standard valid range of 0-1.
    """
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_inference():
    """
    This test creates a small pipeline to test the results of the inference pipeline
    and ensure that predictions are returned in the expected format.
    """
    # set small model
    model = RandomForestClassifier(n_estimators=2, random_state=42)
    X = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array([0, 1, 0])
    model.fit(X, y)
    preds = inference(model, X)
    assert len(preds) == 3
    assert set(preds).issubset({0, 1})


def test_process_data():
    """
    This test is to verify that the proces_data function is working as expected.
    It sets up a mock dataset, processes it, and assert that the resulting features, labels,
    transformers match
    """
    df = pd.DataFrame({
        "age": [25, 40],
        "workclass": ["Private", "self-employed"],
        "salary": ["<=50K", ">50K"]
    })

    X, y, encoder, lb = process_data(
        df, categorical_features=["workclass"], label="salary", training=True
    )
    assert X.shape[0] == 2
    assert y.shape == (2,)
    assert set(y).issubset({0, 1})
    assert encoder is not None
