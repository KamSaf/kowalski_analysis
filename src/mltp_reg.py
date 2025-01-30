import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Wczytaj dane
file_path = "car_prices.csv"
data = pd.read_csv(file_path)

# 2. Oczyść dane z brakujących wartości
cleaned_data = data.dropna(subset=["condition", "odometer", "mmr", "body"])
make_dummies = pd.get_dummies(cleaned_data["body"], drop_first=True, dtype=int)
X = pd.concat(
    [cleaned_data[["year", "condition", "odometer", "mmr"]], make_dummies], axis=1
)
print("Typy danych po konwersji:\n", X.dtypes)


# 3. Dopasuj model regresji
X = sm.add_constant(cleaned_data[["year", "condition", "odometer", "mmr"]])
y = cleaned_data["sellingprice"]
model = sm.OLS(y, X).fit()

# 4. Oblicz odległość Cooka
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]

# 5. Próg odległości Cooka (np. 4/n lub 3 standardowe odchylenia)
threshold = 4 / len(cleaned_data)
cleaned_data = cleaned_data[cooks_d < threshold]

# 6. Przeprowadź ponowną regresję na oczyszczonych danych
X_cleaned = sm.add_constant(cleaned_data[["year", "condition", "odometer", "mmr"]])
y_cleaned = cleaned_data["sellingprice"]
final_model = sm.OLS(y_cleaned, X_cleaned).fit()

# 7. Wyświetl tabelę podsumowującą (identyczną jak na obrazie)
print(final_model.summary())

# 8. Analiza wizualna reszt
residuals = final_model.resid
fitted_values = final_model.fittedvalues

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
plt.savefig("multiple_regression_plot_cleaned.jpg")
plt.show()


# Histogram reszt

sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Reszty")
plt.ylabel("Liczność")
plt.title("Histogram reszt")
plt.show()
