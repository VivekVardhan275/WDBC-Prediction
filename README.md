# 🧠 Breast Cancer Prediction using Machine Learning Models

This repository contains a comparative analysis of several supervised machine learning algorithms applied to the **UCI Breast Cancer Wisconsin Dataset**. The primary goal is to evaluate the performance of different classifiers and identify the best-performing model for accurate diagnosis of breast cancer (benign vs malignant).

---

## 📌 Project Overview

- **Dataset**: [UCI Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Models Implemented**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Naive Bayes
  - k-Nearest Neighbors (KNN)
  - Random Forest Classifier
  - XGBoost Classifier

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## ✅ Final Results Summary

| Model                   | Best Params                                                                                                                                   | CV Accuracy | Test Accuracy | Precision (0 / 1) | Recall (0 / 1) | F1-Score (0 / 1) | Confusion Matrix     |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------- | ----------------- | -------------- | ---------------- | -------------------- |
| **Logistic Regression** | `{'C': 1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'sag'}`                                                                                 | 0.9736      | 0.9737        | 0.96 / 1.00       | 1.00 / 0.93    | 0.98 / 0.96      | `[[72, 0], [3, 39]]` |
| **SVM**                 | `{'C': 1, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf', 'shrinking': True}`                                                                 | 0.9758      | 0.9737        | 0.96 / 1.00       | 1.00 / 0.93    | 0.98 / 0.96      | `[[72, 0], [3, 39]]` |
| **Naive Bayes**         | `{'var_smoothing': 1e-09}`                                                                                                                    | 0.9385      | 0.9211        | 0.92 / 0.92       | 0.96 / 0.86    | 0.94 / 0.89      | `[[69, 3], [6, 36]]` |
| **KNN**                 | `{'metric': 'euclidean', 'n_neighbors': 3, 'p': 1, 'weights': 'uniform'}`                                                                     | 0.9692      | 0.9386        | 0.92 / 0.97       | 0.99 / 0.86    | 0.95 / 0.91      | `[[71, 1], [6, 36]]` |
| **Random Forest**       | `{'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}`            | 0.9626      | 0.9737        | 0.96 / 1.00       | 1.00 / 0.93    | 0.98 / 0.96      | `[[72, 0], [3, 39]]` |
| **XGBoost**             | `{'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.2, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.8}` | 0.9758      | **0.9825**    | 0.97 / 1.00       | 1.00 / 0.95    | 0.99 / 0.98      | `[[72, 0], [2, 40]]` |


> 🔍 **XGBoost** achieved the best overall performance on both accuracy and ROC AUC.

---

## 📈 Visualizations

- ROC AUC curves for all models
- Accuracy comparison bar chart
- Confusion matrices per model


