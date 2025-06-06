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

Results Summary


Without SMOTE:

Accuracy is high (~94%), but the model fails to detect any stroke cases (precision and recall for stroke = 0), showing poor performance on the minority class due to data imbalance.

With SMOTE:

Accuracy decreases (~82%), but detection of stroke cases improves significantly (precision ~16%, recall ~45%), making the model more effective in identifying minority class instances despite lower overall accuracy.
