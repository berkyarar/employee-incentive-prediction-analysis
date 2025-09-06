# Employee Incentive Regression Analysis

This project analyzes the relationship between **customer satisfaction** and **employee incentives (bonuses)**.  
It compares **Linear Regression** and **Polynomial Regression** models, including preprocessing, model training, evaluation, and visualization. The workflow also demonstrates testing different polynomial degrees using a **Pipeline**.

---

## 📌 Features
- **Data Preparation**
  - Load and clean dataset (`3-customersatisfaction.csv`)
  - Drop unnecessary columns, check for missing values, descriptive statistics
- **Exploratory Data Analysis (EDA)**
  - Scatter plot of `Customer Satisfaction` vs `Incentive`
- **Preprocessing**
  - Feature scaling with `StandardScaler`
- **Modeling**
  - Linear Regression (baseline model)
  - Polynomial Regression (degree=2) using `PolynomialFeatures`
- **Evaluation**
  - Metrics: MAE, MSE, RMSE, R², Adjusted R² (test set)
- **Visualization**
  - Compare linear vs polynomial fits on the same data
- **Generalization**
  - Predict incentives on new incoming data (`3-newdatas.csv`)  
- **Experimentation**
  - Pipeline implementation to test polynomial degrees 1–10 and compare R² scores

---

## 📂 Dataset
-  `3-customersatisfaction.csv`  
  - Feature: `Customer Satisfaction`  
  - Target: `Incentive` (bonus given to employees)  
- **New data:** `3-newdatas.csv` (single column input for prediction)

---

## 🛠 Tech Stack
- Python  
- NumPy, pandas, matplotlib  
- scikit-learn: `StandardScaler`, `LinearRegression`, `PolynomialFeatures`, `Pipeline`, metrics,`train_test_split`

---

