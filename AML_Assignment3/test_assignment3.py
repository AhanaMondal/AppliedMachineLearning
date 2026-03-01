import pytest
import joblib
import requests
import os
import time
from score import score

MODEL_PATH = r"C:\Users\Ahana\Documents\AppliedMachineLearning\AML_Assignment2\notebooks\mlruns\302143816419166244\models\m-c24d00be04c04ce39af7bda701c7ca18\artifacts\model.pkl"

model = joblib.load(MODEL_PATH)


# -----------------------
# UNIT TESTS
# -----------------------

def test_score_smoke():
    pred, prop = score("hello world", model, 0.5)
    assert pred is not None
    assert prop is not None


def test_score_format():
    pred, prop = score("hello world", model, 0.5)
    assert isinstance(pred, bool)
    assert isinstance(prop, float)

def test_prediction_binary():
    pred, _ = score("hello world", model, 0.5)
    assert pred in [True, False]


def test_propensity_range():
    _, prop = score("hello world", model, 0.5)
    assert 0.0 <= prop <= 1.0


def test_threshold_zero():
    pred, _ = score("hello world", model, 0.0)
    assert pred is True


def test_threshold_one():
    pred, _ = score("hello world", model, 1.0)
    assert pred is False


def test_typical_spam():
    pred, prop = score(
        "Congratulations! You won a free lottery. Click now!",
        model,
        0.5
    )
    assert 0.0 <= prop <= 1.0

def test_typical_nonspam():
    pred, _ = score("Let's schedule our meeting for tomorrow.", model, 0.5)
    assert pred is False


# -----------------------
# INTEGRATION TEST
# -----------------------
from app import app

def test_flask():
    client = app.test_client()

    response = client.post(
        "/score",
        json={"text": "Win money now!", "threshold": 0.5}
    )

    assert response.status_code == 200

    data = response.get_json()
    assert "prediction" in data
    assert "propensity" in data