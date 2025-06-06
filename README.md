
# 💳 Credit Card Fraud Detection using ML

## 🚀 Overview
This project tackles the critical challenge of detecting fraudulent credit card transactions using supervised machine learning models. The dataset is heavily imbalanced, which is addressed using SMOTE. Models are trained, evaluated, and explained using a full ML pipeline with visualization and interpretability techniques.

## 📚 Dataset
- **Rows:** 9,999 transactions
- **Features:** 28 anonymized PCA components (`V1`–`V28`), `Amount`, and target `Class`
- **Target:**
  - `0`: Non-Fraud (majority class)
  - `1`: Fraud (minority class, ~0.38%)

## 🧪 Project Pipeline

1. **Exploratory Data Analysis**
   - Class imbalance visualization
   - Distribution plots of transaction amounts
   - Correlation heatmaps and boxplots

2. **Preprocessing**
   - Handled extreme class imbalance using **SMOTE**
   - Standardized features using `StandardScaler`

3. **Modeling**
   - Trained 3 classifiers:
     - `RandomForestClassifier`
     - `XGBClassifier`
     - `CatBoostClassifier`
   - Hyperparameter tuning with `GridSearchCV`

4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Visuals: Confusion Matrix, ROC Curve
   - Model performance:
     - All three models achieved nearly perfect accuracy and AUC (~1.0)

5. **Model Explainability**
   - Feature importances extracted from Random Forest
   - Top features: `V14`, `V10`, `V12`, `V4`, `V3`

6. **Deployment Ready**
   - Final model saved via `joblib` as `Best_random_forest.pkl`

## 📊 Sample Evaluation Output

| Metric      | Random Forest | XGBoost | CatBoost |
|-------------|---------------|---------|----------|
| Accuracy    | 1.00          | 1.00    | 0.999    |
| Precision   | 1.00          | 1.00    | 1.00     |
| Recall      | 1.00          | 1.00    | 1.00     |
| ROC-AUC     | 1.00          | 1.00    | 0.999    |

## 🛠 Technologies Used
- Python, Pandas, NumPy, Seaborn, Matplotlib
- Machine Learning:
  - `sklearn`
  - `xgboost`
  - `catboost`
  - `imblearn` (SMOTE)
- Model Export: `joblib`

## 📂 Folder Structure

├── Credit Card Fraud Detection.ipynb
├── creditcard_2023_New.csv
├── Best_random_forest.pkl
└── README.md

## 📌 How to Run

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

	3.	Open the notebook and run all cells:

Credit Card Fraud Detection.ipynb


	4.	Evaluate the saved model from Best_random_forest.pkl.

🔮 Future Improvements
	•	Add SHAP visualizations for deeper explainability
	•	Deploy as a real-time fraud detection API (FastAPI or Flask)
	•	Integrate into a larger fraud analytics dashboard

📜 License

This project is licensed under the MIT License.

---

Let me know if you'd like me to generate a `requirements.txt` file or convert this into a LaTeX `\section{}` style block for inclusion in your resume PDF.
