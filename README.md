

1. Discriminative AI 
    ML 
      1. Recommendation (products, movies)
      2. Classification (dog breed, flower classification)
      3. Prediction (house price next year, weather prediction)
      4. Clustering (grouping -- age, demography, category)
    * output will be deterministic (boolean, string, object, array of object, lists, number, etc)

2. Generative AI 
    LLM 
      generating content (text, code, data, image, audio, video)
    
      * output will be non-deterministic 


Devops 
====
  will deal with code 


MLOps 
===
  deals with code, data, models 


# Minimal MLOps Data Versioning Project — DVC + Git

## 📁 Project Structure

```
data-ops-demo1/
│── src/
│   ├── download_data.py
│   └── process_data.py
│── data/
│   ├── raw/           ← DVC tracked
│   └── processed/     ← generated via pipeline
│── .gitignore
│── requirements.txt
│── dvc.yaml
│── dvc.lock
│── README.md          ← YOU are here
```

## 1️⃣ Setup Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Verify:**

```bash
which python
# MUST show: /path/to/project/.venv/bin/python
```

## 2️⃣ Install Dependencies

**requirements.txt:**

```
dvc
pandas
scikit-learn
```

```bash
pip install -r requirements.txt
```

## 3️⃣ Initialize Git & DVC

```bash
git init
dvc init
```

**Add and commit:**

```bash
git add .
git commit -m "Initial setup with git + dvc"
```

## 4️⃣ Data Collection Script

**src/download_data.py:**

```python
import os
from sklearn.datasets import load_iris
import pandas as pd

def main():
    data_dir = os.path.join("data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "iris.csv")

    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(output_path, index=False)
    print(f"Saved iris dataset to {output_path}")

if __name__ == "__main__":
    main()
```

## 5️⃣ Create DVC Stage — get_data

```bash
dvc stage add \
  -n get_data \
  -d src/download_data.py \
  -o data/raw/iris.csv \
  python src/download_data.py
```

**Commit:**

```bash
git add dvc.yaml
git commit -m "Add get_data stage to pipeline"
```

## 6️⃣ Generate dvc.lock

```bash
dvc repro
```

**Then commit:**

```bash
git add dvc.lock
git commit -m "Generate dvc.lock"
```
