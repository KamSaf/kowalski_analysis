import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

# from scipy import stats
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

file_path = "car_prices.csv"
data = pd.read_csv(file_path)


names = [
    "year",
    "body",
    "make",
    "model",
    "trim",
    "transmission",
    "vin",
    "state",
    "color",
    "interior",
    "seller",
    "saledate",
]


df = data.drop(names, axis=1)

df = df.dropna(subset=["age", "condition", "odometer", "mmr"])

df = df.rename(
    columns={
        "age": "Wiek",
        "condition": "Stan techniczny",
        "odometer": "Przebieg",
        "mmr": "Cena rynkowa",
        "sellingprice": "Cena sprzedaży",
    }
)


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.values)

cleaned_data = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

print(data.head())
print(data.info())
model = sm.OLS(
    cleaned_data["Cena sprzedaży"],
    sm.add_constant(
        cleaned_data[
            [
                "Wiek",
                "Stan techniczny",
                "Przebieg",
                "Cena sprzedaży",
            ]
        ]
    ),
).fit()


numerical_data = cleaned_data.select_dtypes(include=[np.number])

correlation_matrix = numerical_data.corr(method="pearson")
print("Macierz korelacji:")
print(correlation_matrix)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Macierz korelacji")
plt.show()
