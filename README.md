

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
python -m venv .venv
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

## 7️⃣ Data Processing Stage (train/test split)

**src/process_data.py:**

```python
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
```

## 8️⃣ Add Stage — process_data

```bash
dvc stage add \
  -n process_data \
  -d src/process_data.py \
  -d data/raw/iris.csv \
  -o data/processed/train.csv \
  -o data/processed/test.csv \
  python src/process_data.py
```

**Commit:**

```bash
git add dvc.yaml
git commit -m "Add process_data stage to pipeline"
```

## 9️⃣ Run FULL Pipeline

```bash
rm -rf data/processed
dvc repro
```

**Then:**

```bash
git add dvc.lock
git commit -m "Execute full pipeline"
```

## 🔍 Reproducing Everything from Scratch

```bash
git clone <repo-url>
cd data-ops-demo1
pip install -r requirements.txt
dvc repro
```


This will regenerate ALL data from raw → processed.

Time to add the model training stage. We’ll take `train.csv` → train a simple classifier → save `model.pkl`. No fluff. Straight to the point.

## 📁 Step 1: Create training script

Create `src/train_model.py`:

```python
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
```

Add and commit:

```bash
git add src/train_model.py
git commit -m "Add training script"
```

## ⚙️ Step 2: Create DVC Stage for training

```bash
dvc stage add \
  -n train_model \
  -d src/train_model.py \
  -d data/processed/train.csv \
  -o models/model.pkl \
  python src/train_model.py
```

Then commit:

```bash
git add dvc.yaml
git commit -m "Add train_model stage to pipeline"
```

## 🧪 Step 3: Run Full Pipeline

```bash
rm -rf data/processed models
dvc repro
```

This will run:

- `get_data`
- `process_data`
- `train_model`

If it finishes without errors, then your pipeline is officially complete.

## 📝 Step 4: Commit Final Output

```bash
git add dvc.lock models/model.pkl
git commit -m "Run pipeline and generate trained model"
```


## 📌 Some DVC Commands

1. **Check what DVC is tracking**

  ```bash
  dvc list .
  ```

  Shows files/data tracked by DVC in the project.

2. **Show version history of a tracked file** 📦

  For example: `model.pkl`

  ```bash
  dvc diff
  ```

  Compares current state vs last commit.

  To compare between two Git commits:

  ```bash
  dvc diff HEAD~1 HEAD
  ```

  This tells you:

  - What changed in data
  - What changed in pipeline
  - What got added/removed

3. **Show tracked data across commits** 🧠

  ```bash
  git log --oneline
  ```

  Pick two commit hashes. Then run:

  ```bash
  dvc diff <commit1> <commit2>
  ```

  Example:

  ```bash
  dvc diff 9123abc 4f89d21
  ```

  This shows data/model differences even if files are not stored in Git.

4. **Show a model from a past commit** 🔗

  ```bash
  git checkout <old-commit>
  dvc checkout
  ```

  This restores the old `model.pkl` and old data from that commit. Pure reproducibility. That’s real MLOps.

5. **Show pipeline visualisation** 📜

  ```bash
  dvc dag
  ```

  This prints the flow of your pipeline:

  - `get_data` → `process_data` → `train_model`

6. **Show cache location** 📂

  ```bash
  dvc cache dir
  ```

  All past versions of data/model live in the DVC cache.

7. **Explicitly list tracked versions** 🚀

  ```bash
  dvc gc --all-branches --all-tags --dry
  ```

  Shows what versions are stored — but doesn’t delete yet.

## 🧨 Final Punchline

To reproduce **any** version:

```bash
git checkout <old-commit>
dvc checkout
dvc repro
```

You get the exact old dataset and model. That is the whole point of DVC. That is what real MLOps looks like.

Do this now:

```bash
dvc dag
```

You can see the pipeline.