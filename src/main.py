import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load the data
df = pd.read_csv("road_accident_dataset.csv")

# Select numerical columns for analysis
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
df_numerical = df[numerical_cols]

print("Numerical columns selected for analysis:")
print(numerical_cols)

###


# One-way ANOVA test comparing Number of Injuries across different Road Types
road_type_groups = [group for _, group in df.groupby("Road Type")["Number of Injuries"]]
f_stat, p_val = stats.f_oneway(*road_type_groups)

print("One-way ANOVA Results for Road Type vs Number of Injuries:")
print("F-statistic:", f_stat)
print("p-value:", p_val)

# Visualize the relationship
plt.figure(figsize=(10, 6))
sns.boxplot(x="Road Type", y="Number of Injuries", data=df)
plt.xticks(rotation=45)
plt.title("Number of Injuries by Road Type")
plt.tight_layout()
plt.show()

# Multiple Regression Analysis: Predicting Number of Injuries
# Selecting relevant features for regression
features = [
    "Visibility Level",
    "Number of Vehicles Involved",
    "Speed Limit",
    "Driver Alcohol Level",
    "Driver Fatigue",
    "Pedestrians Involved",
    "Cyclists Involved",
    "Emergency Response Time",
    "Traffic Volume",
]
X = df[features]
y = df["Number of Injuries"]

# Handling missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Adding a constant for statsmodels
X = sm.add_constant(X)

# Fitting the regression model
model = sm.OLS(y, X).fit()

# Summary of the regression model
print(model.summary())


# K-means Clustering
# Selecting features for clustering
cluster_features = [
    "Number of Injuries",
    "Number of Fatalities",
    "Emergency Response Time",
    "Medical Cost",
]
X_cluster = df[cluster_features].fillna(df[cluster_features].mean())

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Determining optimal number of clusters using elbow method
inertias = []
K = range(1, 10)
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

# Fitting K-means with optimal k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Visualizing clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, x="Number of Injuries", y="Medical Cost", hue="Cluster", palette="deep"
)
plt.title("Clusters of Accidents based on Injuries and Medical Cost")
plt.show()


# PCA Analysis
# Standardizing the numerical features

scaler = StandardScaler()
X_scaled_pca = scaler.fit_transform(df_numerical.fillna(df_numerical.mean()))

# Applying PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled_pca)

# Creating a DataFrame for the principal components
pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

# Visualizing the PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_df["PC1"], y=pca_df["PC2"], alpha=0.6)
plt.title("PCA Analysis: First Two Principal Components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Explained variance ratio
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
