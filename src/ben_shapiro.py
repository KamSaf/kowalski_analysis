import statsmodels.api as sm
import pandas as pd
from scipy.stats import shapiro

# 1. Wczytaj dane
file_path = "car_prices.csv"
data = pd.read_csv(file_path)
data = data.head(50)

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
X_cleaned = sm.add_constant(cleaned_data[["year", "condition", "odometer"]])
y_cleaned = cleaned_data["sellingprice"]
final_model = sm.OLS(y_cleaned, X_cleaned).fit()

# 7. Wyświetl tabelę podsumowującą (identyczną jak na obrazie)
print(final_model.summary())

# 8. Analiza wizualna reszt
residuals = final_model.resid
fitted_values = final_model.fittedvalues


# analiza reszt
stat, p = shapiro(residuals)
print(f"Test Shapiro-Wilka: stat={stat:.4f}, p={p:.4f}")

if p > 0.05:
    print("Brak dowodów na odchylenie od normalności.")
else:
    print("Reszty nie są normalne.")
