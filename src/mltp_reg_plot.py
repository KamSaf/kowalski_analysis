import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = "car_prices.csv"
data = pd.read_csv(file_path)
cleaned_data = data.dropna(subset=["condition", "odometer", "mmr"])
# Display the first few rows and summary of the dataset
print(data.head())
print(data.info())
# Multiple Regression Analysis Plots
model = sm.OLS(
    cleaned_data["sellingprice"],
    sm.add_constant(cleaned_data[["year", "condition", "odometer", "mmr"]]),
).fit()

# 1. Residual Plot
residuals = model.resid
fitted_values = model.fittedvalues

plt.figure(figsize=(12, 8))
plt.scatter(fitted_values, residuals, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# 2. Q-Q Plot
fig, ax = plt.subplots(figsize=(12, 8))
sm.graphics.qqplot(residuals, line="45", fit=True, ax=ax)
plt.title("Q-Q Plot of Residuals")
plt.show()

# 3. Actual vs Predicted Plot
plt.figure(figsize=(12, 8))
plt.scatter(cleaned_data["sellingprice"], fitted_values, alpha=0.5)
plt.plot(
    [cleaned_data["sellingprice"].min(), cleaned_data["sellingprice"].max()],
    [cleaned_data["sellingprice"].min(), cleaned_data["sellingprice"].max()],
    "r--",
)
plt.xlabel("Cena sprzedaży")
plt.ylabel("Oszacowana cena")
plt.title("Cena sprzedaży vs Oszacowana cena")
plt.show()

# Print regression summary statistics
print(
    "\
Regression Summary Statistics:"
)
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
print(f"F-statistic: {model.fvalue:.4f}")
print(f"Prob (F-statistic): {model.f_pvalue:.4e}")
