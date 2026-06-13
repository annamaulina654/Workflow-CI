import pandas as pd
import mlflow
import mlflow.sklearn
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load Dataset
df = pd.read_csv("telco_churn_processed.csv")

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

mlflow.autolog()

with mlflow.start_run():

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        "model.pkl"
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # log metric manual
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # simpan model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model"
    )

    # artifact tambahan
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    mlflow.log_artifact("metrics.json")

print("Accuracy:", accuracy)