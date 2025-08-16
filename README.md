# MINST Digit recognition using basic ML

**Notebook:** `MINST_With_Basic_ML.ipynb`

This project builds and compares multiple classifiers with a **precision/recall** emphasis, logging experiments to **MLflow** and visualizing **ROC** and **Precision–Recall** curves. The dataset consists of flattened image features (handwritten digits–style), and the workflow includes resampling with **SMOTE** for class balance during training.

---

## 🎯 Objectives
- Prepare a digit‑classification style dataset (flattened images) for supervised learning.
- Train and compare multiple models with hyperparameter search.
- Emphasize **precision and recall** (and F1) for evaluation; plot ROC and PR curves.
- Track experiments and artifacts using **MLflow**.

---

## 🧪 Data
- Flattened image features with shapes like (1600, 784) train and (400, 784) test detected in outputs.
- Labels show roughly even per‑class proportions in sample output.  
- Train/test split is present (random_state=42).
- **Imbalance handling:** `imblearn` with **SMOTE** used during training.

> If you use a different source (e.g., `sklearn.datasets.load_digits`, Keras MNIST, or a CSV), update this section with the exact loader/path.

---

## 🤖 Models & Tuning
The notebook references and/or trains the following classifiers:
- **SVM (SVC)**
- **MLP (Neural Network)**
- **Random Forest**
- **Decision Tree**
- **Gradient Boosting**
- **AdaBoost**

**Search strategy:** `GridSearchCV` (5‑fold) is used for hyperparameter tuning. Best parameters and scores are printed to the console and logged to MLflow.

**Preprocessing:** Experiments include SMOTE for class balance. If scaling is required (e.g., for SVM/MLP), consider wrapping preprocessing + model in a `Pipeline` and tuning jointly.

---

## 📈 Evaluation
The notebook computes (or logs) the following metrics:
- `precision_score`, `recall_score`, `f1_score`
- Confusion matrix (in code) and **ROC** / **Precision–Recall** curves


---

## 🧪 Experiment Tracking (MLflow)
- The notebook initializes an MLflow experiment (e.g., **“Assignment 10 F1 Precision”**).
- Runs log parameters, metrics, and models/artifacts per trial.
- To view the UI locally:
  ```bash
  mlflow ui
  ```
  Then open the printed tracking URI in your browser.

---

## 📁 Repository Structure
```text
├── MINST_With_Basic_ML.ipynb   # Main notebook
└── README.md         # This file
```

> If you export model artifacts/plots, create an `artifacts/` or `images/` folder and reference them here.

---

## ⚙️ Setup & Run
**Core libraries:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `mlflow`, `imblearn`

Install:
```bash
pip install -U numpy pandas scikit-learn matplotlib mlflow imbalanced-learn
```

Run:
```bash
jupyter notebook "MINST_With_Basic_ML.ipynb"
```

Optional (to view MLflow UI):
```bash
mlflow ui
```

---

## 💡 Tips & Next Steps
- For SVM/MLP, add **feature scaling** (e.g., `StandardScaler`) in a `Pipeline` and tune `C`, `gamma`, `hidden_layer_sizes`, etc.
- Use **stratified** CV for balanced folds; double‑check that SMOTE is applied **inside** CV to avoid leakage (e.g., `Pipeline([('smote', SMOTE()), ('clf', ...)])`).  
- Add **calibration** (e.g., `CalibratedClassifierCV`) if you need better probability estimates for PR/ROC analysis.
- Log **confusion matrices**, **ROC/PR curves**, and **classification reports** as MLflow artifacts for easy comparison.
