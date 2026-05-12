# Titanic Survival Prediction with SVM

## Project Overview

This project uses a **Support Vector Machine (SVM)** classifier to predict passenger survival on the Titanic. The model is trained on passenger features such as age, gender, passenger class, and fare. The complete pipeline includes data preprocessing, training multiple SVM kernels, hyperparameter tuning with GridSearchCV, and cross-validation.

## Dataset

- **Source:** OpenML (`fetch_openml('titanic', version=1)`)
- **Target Column:** `survived` (0 = Died, 1 = Survived)
- **Dropped Columns:** `name`, `ticket`, `cabin`, `boat`, `body`, `home.dest` (low value or unstructured)

## Preprocessing & Encoding

| Data Type | Preprocessing Steps |
|-----------|----------------------|
| **Numeric** | Missing values → Median, then StandardScaler |
| **Categorical** | Missing values → Most frequent, then OneHotEncoder |

The preprocessing is wrapped in a `ColumnTransformer` with two parallel pipelines.

## Model & Kernels

Three SVM kernels were compared:

| Kernel | Characteristics |
|--------|-----------------|
| **Linear** | Fast, simple, good for linear data |
| **RBF** | Handles non-linear data, flexible |
| **Poly** | Captures complex relationships (degree-dependent) |

## Results & Kernel Comparison

Based on the classification report and confusion matrices:

- **RBF kernel** generally provides the highest accuracy and F1-score due to its ability to model non-linear relationships in the data.
- Linear kernel performs reasonably well but may miss complex patterns.
- Poly kernel performance depends heavily on degree setting; sometimes underperforms.

## Hyperparameter Tuning (GridSearchCV)

GridSearchCV was applied with 5-fold cross-validation (scoring = F1) to optimize:

- `C` (regularization): [0.1, 1, 10]
- `gamma`: ['scale', 'auto'] (for RBF)
- `degree`: [2, 3] (for Poly)
- `kernel`: ['linear', 'rbf', 'poly']

**Best Model:** Optimized RBF (or kernel chosen by GridSearchCV).  
**Best CV F1-score:** printed after tuning.

## Cross-Validation Results

5-Fold Cross Validation on the best model provides:

- Mean Accuracy (with ±2σ range)
- Mean F1-score (with ±2σ range)

Low variance across folds indicates **good model stability** and performance not dependent on a specific train/test split.

## Why Pandas DataFrame?

The dataset is loaded as a Pandas DataFrame (via `fetch_openml(as_frame=True)` or `pd.read_csv`) because:

| Advantage over NumPy arrays |
|-----------------------------|
| Easy column selection and dropping |
| Simple handling of missing values (`SimpleImputer`) |
| Readable code (column names instead of indices) |
| Seamless integration with `ColumnTransformer` |
| Better for mixed numeric/categorical data |

## How to Run

1. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

2. Run the script. It will:
   - Download the Titanic dataset (cached as `titanic.csv`)
   - Preprocess data (numeric + categorical)
   - Train linear, RBF, and poly SVM kernels
   - Display metrics and confusion matrices
   - Run GridSearchCV to find the best model
   - Perform 5-fold cross-validation on the final model


![Training Curve](assets/Figure1.png)
![Training Curve](assets/Figure2.png)
![Training Curve](assets/Figure3.png)
![Training Curve](assets/Figure4.png)

## Key Outputs

- Performance comparison of three kernels (Accuracy, Precision, Recall, F1)
- Confusion matrices for each kernel
- Best hyperparameters from GridSearchCV
- Final optimized model performance on test set
- Cross-validation mean scores with standard deviation

## Conclusion

This project demonstrates a complete SVM-based classification pipeline. With proper preprocessing and tuning, SVM (especially RBF kernel) achieves strong performance on the Titanic dataset. The use of ColumnTransformer, Pipeline, and GridSearchCV ensures reproducibility and efficiency.
