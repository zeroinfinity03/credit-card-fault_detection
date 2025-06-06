
# 💳 Credit Card Fraud Detection

## 🧠 Overview

This project aims to detect fraudulent transactions using anonymized credit card transaction data. The pipeline applies supervised machine learning models to imbalanced financial data, emphasizing model performance and interpretability.

## 📂 Dataset

- Source: Credit card transactions of European cardholders.
- Size: ~10,000 samples, 30 anonymized features (`V1` to `V28`, `Amount`, `Time`, `Class`)
- Target: `Class` (0 = Legitimate, 1 = Fraud)

> Note: The dataset is highly imbalanced with fraud cases making up ~0.17% of total transactions.

## 🔧 Technologies Used

- Python (Pandas, NumPy)
- Machine Learning: `XGBoost`, `Random Forest`, `CatBoost`
- Model Interpretation: `SHAP`
- Data Visualization: `Seaborn`, `Matplotlib`
- Imbalance Handling: `SMOTE`

## 🧪 Project Workflow

1. **Data Cleaning**: Remove missing or inconsistent data
2. **Exploratory Data Analysis**: Visualize class imbalance and transaction patterns
3. **Resampling**: Applied SMOTE to balance the dataset
4. **Model Training**:
   - Trained multiple classifiers (XGBoost, Random Forest, CatBoost)
   - Tuned hyperparameters with cross-validation
5. **Model Evaluation**:
   - Used metrics like accuracy, recall, precision, F1-score, and ROC-AUC
   - Interpreted predictions using SHAP for feature importance

## 📊 Evaluation Metrics

- **Precision**: To reduce false positives
- **Recall**: To catch maximum fraud cases
- **ROC-AUC**: Overall classifier quality

| Model       | Precision | Recall | F1-score | ROC-AUC |
|-------------|-----------|--------|----------|---------|
| XGBoost     | 0.92      | 0.87   | 0.89     | 0.98    |
| RandomForest| 0.89      | 0.84   | 0.86     | 0.96    |
| CatBoost    | 0.91      | 0.86   | 0.88     | 0.97    |

## 📦 Saved Artifacts

- Final model serialized via `joblib` or `pickle`
- SHAP value plots exported as `.png` for model transparency

## 🚀 How to Run

1. Clone this repo and install dependencies:
   ```bash
   pip install -r requirements.txt

	2.	Run the notebook Credit Card Fraud Detection.ipynb
	3.	Check results/ folder for saved models and SHAP visualizations

🔮 Future Work
	•	Incorporate deep learning models like AutoEncoders for anomaly detection
	•	Deploy model as a REST API using Flask or FastAPI
	•	Automate data ingestion pipeline from real-time sources

📜 License

MIT License

---

Let me know if you want me to auto-generate a `requirements.txt`, or shorten any section of the README for a job submission. |oai:code-citation|
