# Companies Bankruptcy Prediction

A notebook-based machine-learning study of **company bankruptcy classification from financial indicators**, with a focus on class imbalance and the trade-offs between identifying bankrupt firms and raising false alarms.

**[Explore the notebook](ML_Project.ipynb)** · **[Dataset](data.csv)** · **[Original project summary](README.docx)**

## Project overview

The project compares four classification algorithms—Logistic Regression, Decision Tree, Random Forest and XGBoost—using SMOTE-balanced training data. Each algorithm is evaluated both with the original retained features and with standardized, PCA-transformed features.

The workflow covers exploratory analysis, class-distribution checks, feature-importance visualization, training-set resampling, dimensionality reduction and model evaluation through classification reports and confusion matrices.

## Data

The repository includes `data.csv`. The saved notebook describes:

| Property | Value |
|---|---:|
| Records | 6,819 |
| Original columns | 96, including the target |
| Target | `Bankrupt?` |
| Class 0 records | 6,599 |
| Class 1 records | 220 |
| Minority-class share | Approximately 3.23% |

The predictors include profitability, liquidity, leverage, cash-flow and other financial indicators. The notebook removes ` Liability-Assets Flag` before modeling, leaving 94 predictors. Preserve the CSV's original column names, including their leading spaces.

An upstream dataset citation is not recorded in the notebook or original project summary; the committed CSV is the input used here.

## Method

1. Inspect data types, descriptive statistics, missing values, class balance and correlations.
2. Visualize Random Forest feature importances before the split and again on resampled training data.
3. Split the data into 70% training and 30% test sets using `random_state=42`.
4. Apply `SMOTE(random_state=42)` to the training set only.
5. Fit each classifier directly on the resampled training features.
6. For the PCA variants, fit `StandardScaler` and `PCA(n_components=0.95)` on the resampled training set, then transform the test set.
7. Compare test-set accuracy, class-specific precision, recall and F1, alongside confusion matrices.

The split is not stratified. The test set in the saved outputs contains 2,046 records, including 78 class-1 examples.

## Results recorded in the notebook

These figures are transcribed from the committed notebook's saved outputs; model training has not been independently rerun for this documentation update. Precision, recall and F1 refer to **class 1** and retain the notebook's two-decimal reporting precision.

| Model | Training representation | Test accuracy | Class-1 precision | Class-1 recall | Class-1 F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | SMOTE | 76.44% | 0.08 | 0.47 | 0.13 |
| Logistic Regression | SMOTE + scaling + PCA | 87.29% | 0.20 | 0.77 | 0.32 |
| Decision Tree | SMOTE | 93.79% | 0.27 | 0.37 | 0.31 |
| Decision Tree | SMOTE + scaling + PCA | 91.20% | 0.20 | 0.44 | 0.27 |
| Random Forest | SMOTE | 95.65% | 0.44 | 0.51 | 0.47 |
| Random Forest | SMOTE + scaling + PCA | 94.57% | 0.31 | 0.35 | 0.33 |
| XGBoost | SMOTE | 96.48% | 0.55 | 0.42 | 0.48 |
| XGBoost | SMOTE + scaling + PCA | 94.48% | 0.33 | 0.44 | 0.38 |

XGBoost without PCA has the highest recorded accuracy and class-1 F1 among these runs. PCA improves the Logistic Regression results but reduces the recorded F1 for the three tree-based models.

**Accuracy needs context:** predicting class 0 for every test example would achieve approximately 96.19% accuracy. Class-1 precision, recall and F1 therefore provide essential context for assessing the models.

## Run locally

Clone the repository and install the packages imported by the notebook:

```bash
git clone https://github.com/haseebkn/Companies-Bankruptcy-Prediction.git
cd Companies-Bankruptcy-Prediction
python -m pip install numpy pandas matplotlib seaborn plotly scikit-learn imbalanced-learn xgboost notebook
jupyter notebook ML_Project.ipynb
```

Keep `data.csv` beside the notebook, start Jupyter from the repository directory and run the cells in order. The package name is `imbalanced-learn`; its Python import is `imblearn`.

The original package environment is not version-pinned, so results may vary with library versions.

## Repository contents

| File | Purpose |
|---|---|
| `ML_Project.ipynb` | Analysis, preprocessing, model comparisons and saved outputs |
| `data.csv` | Financial-indicator dataset |
| `README.md` | GitHub-rendered project overview |
| `README.docx` | Original Word-format summary |

## Evaluation scope

This is an exploratory model-comparison project. The notebook uses one random holdout split; it does not execute cross-validation or hyperparameter search, despite importing related utilities. The initial feature-importance visualization uses the full dataset and should be treated as descriptive analysis, not independent test evidence. The saved evaluation cells do not report probability-based ROC-AUC or PR-AUC.

The repository contains no exported trained model or deployment service. Its results describe this dataset and split, rather than a validated operational bankruptcy-risk system.
