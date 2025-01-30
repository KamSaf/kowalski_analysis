import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler


# Load the dataset
file_path = "car_prices.csv"
data = pd.read_csv(file_path)
cleaned_data = data.dropna(subset=["condition", "odometer", "mmr"])
# Display the first few rows and summary of the dataset
print(cleaned_data.head())
print(cleaned_data.info())


# Perform PCA Analysis

cluster_features = ["age", "condition", "odometer", "mmr", "sellingprice"]
X_cluster = cleaned_data[cluster_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

X_pca = X_scaled

# Create and fit PCA
pca = PCA()
X_pca_transformed = pca.fit_transform(X_pca)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(explained_variance_ratio) + 1),
    np.cumsum(explained_variance_ratio),
    "bo-",
    label="Skumulowana Wariancja",
)
plt.xlabel("Liczbna głównych składowych")
plt.ylabel("Skumulowana wyjaśniona wariancja")
plt.title("PCA - Skumulowany współczynnik wyjaśnionej wariancji")
plt.grid(True)
plt.legend(loc="best")
plt.savefig("pca_plot.jpg")
plt.show()

# Print explained variance ratio for each component
print(
    "\
Wyjaśniona Wariancja dla każdej składowej:"
)
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.4f}")

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i + 1}" for i in range(len(pca.components_))],
    index=cluster_features,
)

# Dane wejściowe (top_loadings_df)
top_loadings_df = pd.DataFrame({
    "PC1": [0.501587, 0.500450, 0, 0, 0],
    "PC2": [0, -0.307301, 0.911898, 0, 0],
}, index=["sellingprice", "mmr", "condition", "odometer", "age"])

# Filtrowanie cech, których wszystkie wartości są równe 0
filtered_loadings = top_loadings_df.loc[(top_loadings_df != 0).any(axis=1)]
print("Największe ładunki dla każdej składowej PCA:")
print(filtered_loadings)

# Przygotowanie danych do wykresu
features = filtered_loadings.index  # Nazwa cech (features)
pc1_values = filtered_loadings["PC1"]  # Wartości dla PC1
pc2_values = filtered_loadings["PC2"]  # Wartości dla PC2

# Parametry dla wykresu
x = np.arange(len(features))  # Pozycje słupków na osi X (dla każdej cechy)
width = 0.35  # Szerokość słupków

# Tworzenie wykresu
fig, ax = plt.subplots(figsize=(10, 6))

# Słupki dla PC1
bars_pc1 = ax.bar(x - width / 2, pc1_values, width, label="PC1", color="blue", alpha=0.7)

# Słupki dla PC2
bars_pc2 = ax.bar(x + width / 2, pc2_values, width, label="PC2", color="green", alpha=0.7)

# Dodanie etykiet cech na osi X
ax.set_xlabel("Feature (Cecha)")
ax.set_ylabel("Loading Value (Ładunek)")
ax.set_title("Wykres Słupkowy Ładunków PCA - PC1 i PC2")
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha="right")  # Nazwy cech na osi X z obrotem 45°

# Dodanie siatki na wykresie dla czytelności
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Legenda
ax.legend(title="Principal Component")

# Poprawki dotyczące układu
plt.tight_layout()
plt.show()

# Stworzenie wykresu łokcia (scree plot)
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(explained_variance_ratio) + 1),
    explained_variance_ratio,
    "ro-",
    label="Wyjaśniona wariancja"
)
plt.xlabel("Liczba głównych składowych")
plt.ylabel("Wariancja wyjaśniona")
plt.title("PCA - Wykres łokcia (Scree Plot)")
plt.grid(True)
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.legend(loc="best")

# Wyświetlenie wykresu
plt.tight_layout()
plt.show()