# Load the data and inspect it
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

# Get component loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(cluster_features))],
    index=cluster_features,
)
print(
    "\
Ładunki składowych PCA:"
)
print(loadings)
