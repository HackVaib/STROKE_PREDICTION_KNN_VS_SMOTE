#  Stroke Prediction using KNN (With and Without SMOTE)

This project predicts if a person may get a stroke or not using K-Nearest Neighbors (KNN) algorithm.

We compare the model results:
1. Without SMOTE (original imbalanced data)
2. With SMOTE (balanced data using SMOTE technique)

## Dataset

- Source: Kaggle (Stroke Prediction Dataset)
- Features: age, hypertension, heart disease, bmi, smoking status, etc.
- Target: stroke (0 = no stroke, 1 = stroke)

## Problem

The dataset is imbalanced. Most records are of people without stroke. This affects the model's ability to predict stroke cases.

## Solution

We create two models:
- First model is trained on the original data.
- Second model is trained after applying SMOTE to balance the data.

## Steps

1. Load dataset and remove 'id' column
2. Fill missing 'bmi' values with mean
3. Label encode categorical columns
4. Split data into train and test
5. Scale the features
6. Train KNN without SMOTE and evaluate
7. Apply SMOTE, train KNN again and evaluate
8. Compare both results

## Results

### Without SMOTE

- Accuracy: 0.8189
- Classification Report:
  - Class 0: Precision = 0.96, Recall = 0.84, F1 = 0.90
  - Class 1: Precision = 0.16, Recall = 0.45, F1 = 0.23

- Confusion Matrix:

[[809 151]
[ 34 28]]


### With SMOTE

- Accuracy: 0.94
- Classification Report:
- Class 0: Precision = 0.94, Recall = 1.00, F1 = 0.97
- Class 1: Precision = 0.00, Recall = 0.00, F1 = 0.00

- Confusion Matrix:
[[960 0]
[ 62 0]]