from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = "car_prices.csv"
data = pd.read_csv(file_path)
cleaned_data = data.dropna(subset=["condition", "odometer", "mmr"])
# Display the first few rows and summary of the dataset
print(cleaned_data.head())
print(cleaned_data.info())


# Select features for clustering
cluster_features = ["year", "condition", "odometer", "mmr", "sellingprice"]
X_cluster = cleaned_data[cluster_features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Determine optimal number of clusters using elbow method
inertias = []
K = range(1, 11)
for k in K: 
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K, inertias, "bx-")
plt.xlabel("Ilość klastrów")
plt.ylabel("Inertia - Miara spójności wewnątrz klastra")
plt.title("Metoda łokcia dla optymalnej ilości klastrów")
plt.show()

# Fit K-means with optimal number of clusters (k=4 based on elbow curve)
kmeans = KMeans(n_clusters=4, random_state=42)
cleaned_data["Cluster"] = kmeans.fit_predict(X_scaled)

# Analyze cluster characteristics
print(
    "\
Cluster Characteristics:"
)
print(cleaned_data.groupby("Cluster")[cluster_features].mean())
