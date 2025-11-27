import os
import pickle
import pandas as pd
import mlflow

from sklearn.metrics import accuracy_score, classification_report


def main():
    # Keep experiment name consistent with training
    mlflow.set_experiment("iris_classification")

    model_path = os.path.join("models", "model.pkl")
    test_path = os.path.join("data", "processed", "test.csv")
    report_dir = os.path.join("reports")
    os.makedirs(report_dir, exist_ok=True)
    metrics_path = os.path.join(report_dir, "metrics.txt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}. Run data processing first.")

    # Load test data
    df = pd.read_csv(test_path)

    # Features are all columns except last; target is last
    X_test = df.iloc[:, :-1]
    y_test = df.iloc[:, -1]

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict
    preds = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    cls_report = classification_report(y_test, preds)

    # Write metrics report
    with open(metrics_path, "w") as f:
        f.write(f"accuracy: {acc}\n")
        f.write("\nclassification_report:\n")
        f.write(cls_report)

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("model_path", model_path)
        mlflow.log_metric("test_accuracy", acc)
        # Save the textual report as an artifact
        mlflow.log_artifact(metrics_path)

    print(f"Evaluation finished. Accuracy: {acc}")
    print(f"Wrote metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
