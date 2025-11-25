import os
import pandas as pd
import pickle
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    # MLflow experiment name
    mlflow.set_experiment("iris_classification")

    train_path = os.path.join("data", "processed", "train.csv")
    df = pd.read_csv(train_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = LogisticRegression(max_iter=200)

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 200)

        model.fit(X, y)

        # Accuracy
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        mlflow.log_metric("train_accuracy", acc)

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "model.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        mlflow.log_artifact(model_path)  # log model

        print(f"Model saved to {model_path}")
        print(f"Logged to MLflow. Accuracy: {acc}")

if __name__ == "__main__":
    main()
