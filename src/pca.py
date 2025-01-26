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
