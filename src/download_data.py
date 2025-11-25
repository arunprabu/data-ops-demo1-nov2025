import os
from sklearn.datasets import load_iris
import pandas as pd


def main():
    # Where to save data
    data_dir = os.path.join("data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "iris.csv")

    # Load iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Save to csv
    df.to_csv(output_path, index=False)
    print(f"Saved iris dataset to {output_path}")


if __name__ == "__main__":
    main()
