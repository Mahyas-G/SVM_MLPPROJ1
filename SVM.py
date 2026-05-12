import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

"""
titanic = fetch_openml('titanic', version=1, as_frame=True)
df = titanic.frame.copy()
"""

import os

if os.path.exists("titanic.csv"):
    df = pd.read_csv("titanic.csv")
else:
    titanic = fetch_openml('titanic', version=1, as_frame=True)
    df = titanic.frame.copy()
    df.to_csv("titanic.csv", index=False)


print("Original Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

target = 'survived'

cols_to_drop = ['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

X = df.drop(columns=[target])
y = df[target].astype(int)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Size:", X_train.shape)
print("Test Size:", X_test.shape)

kernels = ['linear', 'rbf', 'poly']
results = {}

for kernel in kernels:
    print(f"\n========== Kernel: {kernel.upper()} ==========")

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel=kernel, random_state=42))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    results[kernel] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1
    }

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Died (0)', 'Survived (1)'],
        yticklabels=['Died (0)', 'Survived (1)']
    )
    plt.title(f'Confusion Matrix - {kernel.upper()} Kernel')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

comparison_df = pd.DataFrame(results).T
print("\n===== Kernel Comparison =====")
print(comparison_df)

print("\n===== GridSearchCV for Best SVM =====")

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

param_grid = [
    {
        'classifier__kernel': ['linear'],
        'classifier__C': [0.1, 1, 10]
    },
    {
        'classifier__kernel': ['rbf'],
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto']
    },
    {
        'classifier__kernel': ['poly'],
        'classifier__C': [0.1, 1, 10],
        'classifier__degree': [2, 3]
    }
]

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation F1-score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_best_pred = best_model.predict(X_test)

print("\n===== Final Optimized Model Performance =====")
print("Accuracy :", accuracy_score(y_test, y_best_pred))
print("Precision:", precision_score(y_test, y_best_pred))
print("Recall   :", recall_score(y_test, y_best_pred))
print("F1-score :", f1_score(y_test, y_best_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_best_pred))

cm_best = confusion_matrix(y_test, y_best_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm_best,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Died (0)', 'Survived (1)'],
    yticklabels=['Died (0)', 'Survived (1)']
)
plt.title('Confusion Matrix - Optimized Best Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\n===== 5-Fold Cross Validation on Best Model =====")

cv_accuracy = cross_val_score(
    best_model,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

cv_f1 = cross_val_score(
    best_model,
    X,
    y,
    cv=5,
    scoring='f1'
)

print(f"Mean Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std()*2:.4f})")
print(f"Mean F1-score: {cv_f1.mean():.4f} (+/- {cv_f1.std()*2:.4f})")
