import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression

# Dataset
X = np.array([1, 2, 3, 4]).reshape(-1, 1)
y = np.array([2, 4, 6, 8])

# Models
lin_reg = LinearRegression()
ridge_reg = Ridge(alpha=1.0)       
lasso_reg = Lasso(alpha=5.0)       
# Fit models
lin_reg.fit(X, y)
ridge_reg.fit(X, y)
lasso_reg.fit(X, y)

# Predictions
X_test = np.linspace(0, 5, 100).reshape(-1, 1)
y_lin = lin_reg.predict(X_test)
y_ridge = ridge_reg.predict(X_test)
y_lasso = lasso_reg.predict(X_test)

# ======== Plotting ========
plt.figure(figsize=(8,5))
plt.scatter(X, y, color="black", label="Data Points")
plt.plot(X_test, y_lin, color="blue", label="Linear Regression")
plt.plot(X_test, y_ridge, color="green", linestyle="--", label="Ridge Regression")
plt.plot(X_test, y_lasso, color="red", linestyle=":", label="Lasso Regression")

plt.title("comparison: Linear vs Ridge vs Lasso")
plt.xlabel("X (years of experience)")
plt.ylabel("y (salary)")
plt.legend()
plt.grid(True)
plt.show()

# coefficients
print("Linear Regression Coefficients:", lin_reg.coef_, lin_reg.intercept_)
print("Ridge Regression Coefficients:", ridge_reg.coef_, ridge_reg.intercept_)
print("Lasso Regression Coefficients:", lasso_reg.coef_, lasso_reg.intercept_)
