import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    raw_path = os.path.join("data", "raw", "iris.csv")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(raw_path)
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

if __name__ == "__main__":
    main()
