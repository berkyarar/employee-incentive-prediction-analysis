# Polynomial vs Linear Regression on Customer Satisfaction

This project compares **Linear Regression** and **Polynomial Regression** to predict `Incentive` from a single feature, `Customer Satisfaction`, using scaling, model training, evaluation, and visualization. It also includes a **Pipeline** to test multiple polynomial degrees.

---

## 📌 Features
- Data loading & cleaning (`drop` unused column, null checks, describe/info)
- **EDA:** scatter plot of `Customer Satisfaction` vs `Incentive`
- **Preprocessing:** `StandardScaler` on X
- **Models:**  
  - Linear Regression (baseline)  
  - Polynomial Regression (degree=2) with `PolynomialFeatures`
- **Evaluation:** MAE, MSE, RMSE, R², Adjusted R² (test set)
- **Visualization:** model fits and comparison plots
- **Generalization:** predict on **new incoming data** (`3-newdatas.csv`) with the same scaler + polynomial transform
- **Experimentation:** `Pipeline` to sweep polynomial **degrees 1–10** and print test R²

---

## 📂 Dataset
-  data: `3-customersatisfaction.csv`  
  - Target: `Incentive`  
  - Feature: `Customer Satisfaction`
- New data for inference: `3-newdatas.csv` (single column input)


---

## 🛠 Tech Stack
- Python, NumPy, pandas, matplotlib  
- scikit-learn: `StandardScaler`, `LinearRegression`, `PolynomialFeatures`, `Pipeline`, metrics, `train_test_split`

---


