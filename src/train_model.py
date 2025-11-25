import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

def main():
  train_path = os.path.join("data", "processed", "train.csv")
  df = pd.read_csv(train_path)

  # X = all features except the last column (target)
  X = df.iloc[:, :-1]
  y = df.iloc[:, -1]

  model = LogisticRegression(max_iter=200)
  model.fit(X, y)

  # Save model
  os.makedirs("models", exist_ok=True)
  model_path = os.path.join("models", "model.pkl")
  with open(model_path, "wb") as f:
    pickle.dump(model, f)

  print(f"Model saved to {model_path}")

if __name__ == "__main__":
  main()