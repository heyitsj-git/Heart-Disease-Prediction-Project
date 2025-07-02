# 💓 Heart Disease Prediction Project

A machine learning project aimed at predicting the presence of heart disease in patients using clinical data and classification algorithms.

---

## 🧠 Overview

This project explores the prediction of heart disease using patient data (e.g., age, blood pressure, cholesterol, etc.) via supervised machine learning. The goal is a reliable binary classification model that flags patients at risk. By detecting high-risk individuals early, medical intervention becomes more timely and effective.

---

## 📁 Repository Structure

```
Heart‑Disease‑Prediction‑Project/
│
├── data/
│   ├── heart.csv         # Raw dataset (UCI / Kaggle)
│   └── processed.csv     # Cleaned & encoded dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb      # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
│
├── models/
│   └── best_model.pkl    # Serialized trained model
│
├── src/
│   ├── data.py           # Data loading and preprocessing
│   ├── features.py       # Feature engineering
│   ├── train.py          # Model training scripts
│   └── predict.py        # Prediction & evaluation utilities
│
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🚀 Features

- **Data Exploration** – Understand feature distributions and correlations via visualizations (histograms, heatmaps).
- **Preprocessing Pipeline** – Handles missing values, encodes categoricals, and scales numerical variables.
- **Model Comparisons** – Benchmark several algorithms, including:
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
  - Support Vector Machine
  - XGBoost
  - (Optional) Neural Networks
- **Evaluation Metrics** – Use accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- **Model Export** – Save the best-performing model for production or user-facing apps.

---

## 🧩 Getting Started

### 1. Clone and Setup

```bash
git clone https://github.com/heyitsj-git/Heart-Disease-Prediction-Project.git
cd Heart-Disease-Prediction-Project
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Analysis and Training

- Launch notebooks interactively:
  ```bash
  jupyter lab
  ```
- Or train from terminal:
  ```bash
  python src/train.py
  ```

### 3. Make Predictions

```bash
python src/predict.py \
    --model models/best_model.pkl \
    --input '{"age":63,"sex":1,"cp":3,"trestbps":145, ... }'
```

---

## 🛠️ Configuration & Parameters

All configurations (e.g. train/test splits, hyperparameters) can be managed through Python scripts or external config files.

---

## 📊 Evaluation Results

| Model               | Accuracy | Precision | Recall | F1   | AUC  |
|---------------------|----------|-----------|--------|------|------|
| Logistic Regression | 0.85     | 0.83      | 0.82   | 0.82 | 0.88 |
| Random Forest       | 0.92     | 0.90      | 0.91   | 0.91 | 0.94 |
| XGBoost             | 0.93     | 0.92      | 0.92   | 0.92 | 0.95 |

> *Note: Metrics may vary based on dataset and tuning.*

---

## 🧾 Dataset

This project uses the UCI Heart Disease dataset (also found on Kaggle), which includes features like age, sex, chest pain type, blood pressure, and cholesterol. A cleaned version (`data/processed.csv`) is included.

---

## 🤝 Contributions

Contributions are welcome! To contribute:

1. Fork the repository  
2. Create a new branch (`git checkout -b feature/MyFeature`)  
3. Commit your changes (`git commit -m 'Add MyFeature'`)  
4. Push to your branch (`git push origin feature/MyFeature`)  
5. Open a Pull Request

---

## 📚 Further Reading

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

Stay heart‑smart! ❤️
