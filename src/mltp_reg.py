import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

file_path = "car_prices.csv"
data = pd.read_csv(file_path)
cleaned_data = data.dropna(subset=["condition", "odometer", "mmr"])
print(data.head())
print(data.info())
model = sm.OLS(
    cleaned_data["sellingprice"],
    sm.add_constant(cleaned_data[["year", "condition", "odometer", "mmr"]]),
).fit()

residuals = model.resid
fitted_values = model.fittedvalues
plt.figure(figsize=(12, 8))
plt.scatter(
    cleaned_data["sellingprice"],
    fitted_values,
    alpha=0.5,
    label="Dane (cena oszacowana vs sprzedaży)",
)
plt.plot(
    [cleaned_data["sellingprice"].min(), cleaned_data["sellingprice"].max()],
    [cleaned_data["sellingprice"].min(), cleaned_data["sellingprice"].max()],
    "r--",
    label="Idealne dopasowanie",
)
plt.xlabel("Cena sprzedaży (USD)")
plt.ylabel("Oszacowana cena (USD)")
plt.title(
    "Porównanie ceny sprzedaży i \
oszacowanej ceny na podstawie modelu regresji"
)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
