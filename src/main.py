# Load the data and inspect it
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

# Load the dataset
file_path = "car_prices.csv"
data = pd.read_csv(file_path)
cleaned_data = data.dropna(subset=["condition", "odometer", "mmr"])
# Display the first few rows and summary of the dataset
print(cleaned_data.head())
print(cleaned_data.info())

###
# Perform ANOVA analysis to check if there are significant differences in selling price based on car make
# Group data by 'make' and extract selling prices
groups = [
    group["sellingprice"].dropna() for name, group in cleaned_data.groupby("make")
]

# Perform one-way ANOVA
anova_result = stats.f_oneway(*groups)

# Display the ANOVA result
print("ANOVA Result:")
print("F-statistic:", anova_result.statistic)
print("p-value:", anova_result.pvalue)

###
# Perform multiple regression analysis to predict selling price based on other variables
# Select relevant features for regression analysis
features = ["year", "condition", "odometer", "mmr"]
X = cleaned_data[features]
y = cleaned_data["sellingprice"]

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Display the regression summary
print(model.summary())

###
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
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
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


###
# Perform PCA Analysis
# Standardize the features for PCA
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
)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("PCA - Cumulative Explained Variance Ratio")
plt.grid(True)
plt.show()

# Print explained variance ratio for each component
print(
    "\
Explained Variance Ratio for each component:"
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
PCA Component Loadings:"
)
print(loadings)
