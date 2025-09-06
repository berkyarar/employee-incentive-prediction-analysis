import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

df = pd.read_csv("3-customersatisfaction.csv")
print(df.head().to_string())
df.drop("Unnamed: 0", axis=1, inplace=True)
print(df.head().to_string())

print(df.info(max_cols=None))
print(df.describe().to_string())
print(df.isnull().sum().to_string())

plt.scatter(x=df["Customer Satisfaction"], y=df["Incentive"], color="r")
plt.xlabel("Customer Satisfaction")
plt.ylabel("Incentive")
plt.show()

# DEPENDENT AND INDEPENDENT FEATURES

X = df[["Customer Satisfaction"]]
y = df["Incentive"]
print(X.head())

# SPLİT PROCESS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
print(y_train)

# SCALE

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# TRAİN

regression_linear = LinearRegression()
regression_linear.fit(X_train, y_train)
print("Coefficient: ", regression_linear.coef_)
print("Intercept: ", regression_linear.intercept_)

# PREDİCTİON AND TEST METRİC

y_prediction = regression_linear.predict(X_test)

mae = mean_absolute_error(y_test, y_prediction)
mse = mean_squared_error(y_test, y_prediction)
rmse = np.sqrt(mse)
r_score = r2_score(y_test, y_prediction)
adjusted_r_score = 1 - (1 - r_score) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

print("mae: ", mae)
print("mse: ", mse)
print("rmse: ", rmse)
print("r2 score: ", r_score)
print("adjusted r2 score :", adjusted_r_score)

# VİSUALIZE

plt.scatter(X_train, y_train)
plt.plot(X_train, regression_linear.predict(X_train), "r")
plt.show()

# CONVERT POLYNOMİAL

# POLYNOMIAL FEATURES

poly = PolynomialFeatures(degree=2, include_bias=True)  # default value
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# TRAİN

regression_poly = LinearRegression()
regression_poly.fit(X_train_poly, y_train)

# PREDİCTİON AND TEST METRİC

y_pred_poly = regression_poly.predict(X_test_poly)

mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r_score_poly = r2_score(y_test, y_pred_poly)
adjusted_r_score_poly = 1 - (1 - r_score_poly) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

print("mae: ", mae_poly)
print("mse: ", mse_poly)
print("rmse: ", rmse_poly)
print("r2 score: ", r_score_poly)
print("adjusted r2 score :", adjusted_r_score_poly)

plt.scatter(X_train, y_train)
plt.scatter(X_train, regression_poly.predict(X_train_poly))
plt.show()

plt.subplot(2, 1, 1)
plt.scatter(X_train, y_train)
plt.scatter(X_train, regression_linear.predict(X_train))
plt.subplot(2, 1, 2)
plt.scatter(X_train, y_train)
plt.scatter(X_train, regression_poly.predict(X_train_poly))
plt.show()

# NEW DATA

new_df = pd.read_csv("3-newdatas.csv")
print(new_df.info(max_cols=None))
print(new_df.describe().to_string())
print(new_df.isnull().sum())
print(new_df.head().to_string())

new_df.rename(columns={"0": "Customer Satisfaction"}, inplace=True)
print(new_df.columns)
print(new_df.head().to_string())

X_new = new_df[["Customer Satisfaction"]]
X_new = scaler.transform(X_new)
X_new_poly = poly.transform(X_new)

y_new_prediction = regression_poly.predict(X_new_poly)

plt.plot(X_new, y_new_prediction, "b", label="New Predictions")
plt.scatter(X_train, y_train, color="r", label="Training Points")
plt.legend()
plt.show()


# PIPELINE

X_for_pipe = df[["Customer Satisfaction"]]
y_for_pipe = df["Incentive"]

X_for_pipe_train, X_for_pipe_test, y_for_pipe_train, y_for_pipe_test = train_test_split(X_for_pipe, y_for_pipe, test_size=0.2, random_state=15)

def poly_regression(degree):
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    lin_reg = LinearRegression()
    scaler_for_pipe = StandardScaler()
    pipeline = Pipeline([
        ("scaler", scaler_for_pipe),
        ("poly_features", poly_features),
        ("lin_reg", lin_reg),
    ])

    pipeline.fit(X_for_pipe_train, y_for_pipe_train)
    score = pipeline.score(X_for_pipe_test, y_for_pipe_test)
    print(f" for {degree}. degree r2 score: {score}")


for degree in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    poly_regression(degree)
