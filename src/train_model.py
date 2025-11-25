import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    # Set MLflow experiment name
    mlflow.set_experiment("iris_classification")

    train_path = os.path.join("data", "processed", "train.csv")
    df = pd.read_csv(train_path)

    X = df.iloc[:, :-1]  # features
    y = df.iloc[:, -1]   # target

    model = LogisticRegression(max_iter=200)

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 200)

        # Train the model
        model.fit(X, y)

        # Evaluate accuracy
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        mlflow.log_metric("train_accuracy", acc)

        # Log model to MLflow Model Registry
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="iris_model"
        )

        print(f"Logged model to MLflow Registry: iris_model")
        print(f"Training accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
