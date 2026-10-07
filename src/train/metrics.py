import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def calculate_metrics(outputs, targets, threshold=0.5):
    """
    Multi-label classification metrics.
    """

    probabilities = 1 / (1 + np.exp(-outputs))

    predictions = (probabilities >= threshold).astype(int)

    metrics = {}

    metrics["precision"] = precision_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    metrics["recall"] = recall_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    metrics["f1_macro"] = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    metrics["f1_micro"] = f1_score(
        targets,
        predictions,
        average="micro",
        zero_division=0
    )

    try:
        metrics["roc_auc"] = roc_auc_score(
            targets,
            probabilities,
            average="macro"
        )
    except ValueError:
        metrics["roc_auc"] = np.nan

    return metrics