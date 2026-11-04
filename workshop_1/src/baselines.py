import time

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)


def get_baseline_models(random_state=42):
    """
    Return the baseline classification models.
    """

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    }


def train_and_evaluate_models(
    X_train,
    y_train,
    X_val,
    y_val,
    X_train_scaled,
    X_val_scaled
):
    """
    Train baseline models and evaluate them on validation data.

    Returns:
        trained_models
        results DataFrame
    """

    models = get_baseline_models()

    trained_models = {}
    results = []

    for name, model in models.items():

        start_time = time.time()

        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_val_scaled)
            probabilities = model.predict_proba(X_val_scaled)[:, 1]

        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            probabilities = model.predict_proba(X_val)[:, 1]

        elapsed_time = time.time() - start_time

        trained_models[name] = model

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_val, predictions),
            "Precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "F1-score": f1_score(
                y_val, predictions, zero_division=0
            ),
            "PR-AUC": average_precision_score(
                y_val, probabilities
            ),
            "Training Time (s)": elapsed_time
        })

    return trained_models, pd.DataFrame(results)