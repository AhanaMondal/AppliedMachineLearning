import joblib
import numpy as np
from sklearn.base import BaseEstimator


def score(text: str, model: BaseEstimator, threshold: float) -> tuple[bool, float]:
    """
    Scores a trained sklearn model on a single text input.

    Args:
        text (str): input text
        model (sklearn estimator): trained model
        threshold (float): classification threshold between 0 and 1

    Returns:
        prediction (bool): True if predicted spam
        propensity (float): probability of positive class
    """

    if not isinstance(text, str):
        raise ValueError("Input text must be a string")

    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1")

    # Model expects iterable input
    probs = model.predict_proba([text])[0]
    propensity = float(probs[1])  # probability of positive class

    prediction = propensity >= threshold

    return prediction, propensity