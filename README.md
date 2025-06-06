# Credit Card Fraud Detection Project

## Overview

This project aims to detect fraudulent transactions using anonymized credit card transaction data. It involves building and comparing multiple machine learning models including **XGBoost**, **Random Forest**, and **CatBoost** to determine the best-performing model for fraud detection. The final model is selected based on key metrics and saved for future use.

## Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Saving and Loading Models](#saving-and-loading-models)
- [How to Use](#how-to-use)
- [Installation](#installation)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Dataset Description

The dataset contains over 10,000 credit card transactions made by European cardholders in 2023. The dataset has been anonymized, and each transaction includes the following features:
- **id**: Unique identifier for each transaction.
- **V1-V28**: Anonymized features representing various transaction attributes.
- **Amount**: The transaction amount.
- **Class**: Binary label where `1` indicates a fraudulent transaction and `0` indicates a non-fraudulent transaction.

There are no missing values in the dataset.

## Project Workflow

1. **Data Loading and Exploration**: The dataset is loaded, and basic exploratory analysis is performed to check for missing values, data types, and distribution of the target variable.
2. **Data Preprocessing**: Features are scaled and processed as needed. The dataset is divided into training and testing sets.
3. **Model Building**: Various models, including XGBoost, Random Forest, and CatBoost, are trained and evaluated.
4. **Hyperparameter Tuning**: GridSearchCV is used to fine-tune model parameters for optimal performance.
5. **Evaluation and Comparison**: Models are evaluated using accuracy, precision, recall, F1-score, and AUC.
6. **Model Saving**: The best-performing model is saved using `joblib` for future use.
7. **Conclusion**: The project concludes by selecting the best model and documenting its performance.

## Technologies Used

- Python
- Pandas, NumPy for data manipulation
- Scikit-learn for model building and evaluation
- XGBoost, CatBoost for model training
- Matplotlib, Seaborn for data visualization
- Joblib for model saving

## Models

The following models are implemented in this project:

- **Random Forest Classifier**
- **XGBoost Classifier**
- **CatBoost Classifier**

Each model is trained on the dataset and evaluated to determine the best performance.

## Evaluation Metrics

The following metrics are used to evaluate the performance of the models:

- **Accuracy**: The overall correctness of the model.
- **Precision**: The ratio of true positive predictions to the total predicted positives.
- **Recall**: The ability of the model to detect positive instances (fraud).
- **F1-score**: The harmonic mean of precision and recall.
- **AUC (Area Under the Curve)**: Measures the ability of the classifier to distinguish between classes.

For imbalanced datasets like fraud detection, **Precision**, **Recall**, and **AUC** are the primary metrics used.

## Hyperparameter Tuning

Hyperparameter tuning is done using **GridSearchCV** to find the optimal combination of parameters for each model. Here are the main parameters tuned for each model:

- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **XGBoost**: `learning_rate`, `n_estimators`, `max_depth`, `subsample`
- **CatBoost**: `iterations`, `learning_rate`, `depth`

### Example of Hyperparameter Tuning Code

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
