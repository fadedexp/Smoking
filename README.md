# Smoking Prediction (Kaggle Competition)

## Overview
This project applies feature engineering, statistical correlation analysis, and ensemble machine learning techniques to predict a target variable (`smoking`). The code is organized into three primary modules:
1. **Feature Engineering and Transformation**
2. **Correlation and Statistical Analysis**
3. **Model Training and Evaluation**

## Folder Structure
- **`data/`**: Folder containing `train.csv` and `test.csv`.
- **`utils.py`**: Utility file with helper functions and classes for feature engineering and statistical analysis.
- **`smoking_prediction.ipynb`**: Main file for data loading, preprocessing, model training, and evaluation.

## Requirements
- Python 3.8 or higher
- Key libraries: `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, `phik`, `scipy`

# Project Modules Overview

## 1. Feature Engineering and Transformation (Feature_engineering and WoE Classes)

The **Feature Engineering** module is responsible for cleaning, creating, and transforming features to enhance model performance. It includes the following classes:

### Feature Engineering (feature_engineering)

This class performs various transformations and creates new features:

- **Derived Features**: Creates new features such as `BMI`, `Cholesterol_HDL_Ratio`, `Liver_Function_Score`, and more.
- **Blood Pressure Categorization**: Uses a custom `bp_category` function to categorize blood pressure levels into different categories.

### Weight of Evidence (WoE)

The `WoE` class bins continuous variables to create ordinal categories based on their predictive power for the target variable:

- **Weight of Evidence Binning**: Transforms features by categorizing values based on their likelihood of predicting the target.
- **Visualization with plot_woe**: Includes a `plot_woe` function to visualize Weight of Evidence values for various categories, aiding interpretability.

---

## 2. Correlation and Statistical Analysis (Correlation_methods and Plots Classes)

This module calculates different types of correlations to help with feature selection and data analysis.

### Correlation Calculation (Correlation_methods)

The `Correlation_methods` class supports several correlation measures:

- **Point-Biserial (r_pb)**: Calculates the correlation between binary and continuous variables.
- **Cramer’s V (cramers_v)**: Measures the association between binary and categorical variables.
- **Phi Coefficient**: Computes the correlation between two binary variables.

### Visualization

- **heatmap_phik**: Plots a heatmap to visualize Phik correlation coefficients between variables, providing insight into feature relationships.
- **plot_learning_curve**: Plots the learning curve for a model, using cross-validation to display performance trends across different training set sizes.

---

## 3. Model Training and Evaluation (in smoking_prediction.ipynb)

This script handles data loading, preprocessing, and training of ensemble models. Key components include:

### Data Splitting and Preprocessing

- **Data Splitting**: The dataset is split into training and test sets, with stratification to preserve the target distribution.
- **ColumnTransformer**: Applies scaling to continuous columns only, leaving categorical columns unscaled.

### Model Pipelines

Several pipelines are defined for different classifiers:

- **LogisticRegression Pipeline**
- **DecisionTreeClassifier Pipeline**
- **BaggingClassifier Pipeline**
- **RandomForestClassifier Pipeline**
- **SVC Pipeline**

Each pipeline combines scaling (if required) with a classifier, enabling modular and reusable models.

### Stacked Ensemble

The **StackingClassifier** combines the above models using a `RandomForestClassifier` as the final estimator. Cross-validation is used to compute the mean and standard deviation of ROC-AUC scores.

---

## Evaluation

The ensemble model’s performance is evaluated using the **ROC-AUC** metric:

- **Cross-Validation Scores**: The script outputs cross-validation scores, along with the mean and standard deviation of ROC-AUC values.

---
