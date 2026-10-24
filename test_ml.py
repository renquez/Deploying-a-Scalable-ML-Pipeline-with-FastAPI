import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import compute_model_metrics, inference
from ml.data import process_data


def test_compute_model_metrics():
    """
    This test verifies evaluation metrics.  It checks if compute_mopdel_metrics returns the correct precision,
    recall, and F1-beta for known label/ prediction pairs
    """
    # Symmetric case
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(0.5)

    # Asymmetric case
    # TP=2, FP=1, FN=0 -- precision=2/3, recall=1.0, F1=0.8
    y2 = np.array([1, 1, 0, 0])
    preds2 = np.array([1, 1, 1, 0])
    precision2, recall2, fbeta2 = compute_model_metrics(y2, preds2)
    assert precision2 == pytest.approx(2/3)
    assert recall2 == pytest.approx(1.0)
    assert fbeta2 == pytest.approx(0.8)


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
