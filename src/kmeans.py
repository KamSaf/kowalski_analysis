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
