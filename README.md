# Machine Learning GUI – Homework 1

**Course**: MKT3434
**Student**: Harun Akgün  
**Instructor**: Ertuğrul Bayraktar  
**Due Date**: March 27, 2025

---

## Project Overview

This project is a graphical user interface (GUI) for training and testing machine learning models for both regression and classification tasks. It supports model selection, loss function configuration, missing data handling, and result visualization directly through the interface.

---

## How to Run

1. Clone this repository or download the file `20067026.py`.
2. Make sure you have Python 3 and the following packages installed:
   ```
   pip install pandas numpy scikit-learn matplotlib
   ```
3. Run the GUI with:
   ```
   python 20067026d.py
   ```

---

## Features

### Supported Models

**Regression Tab**
- Linear Regression
- Support Vector Regression (SVR)

**Classification Tab**
- Logistic Regression
- Support Vector Machine (SVM)

---

### Loss Functions

| Task | Loss Functions |
|------|----------------|
| Regression | MSE, MAE, Huber |
| Classification | Cross-Entropy, Hinge |

---

### Missing Data Handling

- Mean Imputation
- Interpolation
- Forward Fill
- Backward Fill

---

### SVR/SVM Hyperparameters

- Kernel selection: `linear`, `rbf`, `poly`
- Adjustable `C` and `epsilon` values

---

## Example Datasets

- `boston.csv`: Boston Housing dataset for regression
- `iris.csv`: Iris dataset for classification

---

## Report

Screenshots, performance comparisons, and explanations are provided in the final report:  
📁 `20067026_ML_GUI_Homework_1_Report.pdf`

