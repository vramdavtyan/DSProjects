# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Load Dataset
data = pd.read_csv("airpollution.csv")

# 3. Descriptive analysis
# Head of the dataset
print(data.head())

# Structure of the dataset
print(data.info())

# Dimensions of the dataset
print(data.shape)

# 4. Localization measures (Summary statistics)
print(data.describe())

# 5. Dispersion measures (Standard deviation for numeric columns)
print(data.select_dtypes(include=np.number).std())

# 6. Select only numeric columns for PCA and K-means (ignoring non-numeric columns like city names)
numeric_data = data.select_dtypes(include=np.number)

# Check if the data now only contains numeric columns
print(numeric_data.head())

# 7. Correlation matrix for numeric columns
cor_data = numeric_data.corr()
print(cor_data)

# 8. Eigenvalues and Eigenvectors (for PCA)
eigenvalues, eigenvectors = np.linalg.eig(cor_data)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

# 9. Perform PCA
# Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# PCA with correlation matrix (standardized data)
pca = PCA()
pca.fit(scaled_data)

# Variance explained by each component
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Components (loadings)
print("Principal Components (Loadings):")
print(pca.components_)

# 10. Scree plot (Variance explained)
plt.figure(figsize=(8,6))
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

# 11. Identifying variables that contribute more to the components
pc_scores = pca.transform(scaled_data)
correlation = np.corrcoef(scaled_data.T, pc_scores.T)[:numeric_data.shape[1], numeric_data.shape[1]:]
print("Correlation between original variables and principal components:")
print(correlation)

# Optional: Plot PCA biplot (scatter plot with component loadings)
plt.figure(figsize=(8,6))
sns.scatterplot(x=pc_scores[:,0], y=pc_scores[:,1])
plt.title('PCA Biplot')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# 12. Elbow Method for Optimal Clusters
wss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=25, random_state=123)
    kmeans.fit(scaled_data)
    wss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(8,6))
plt.plot(range(1, 11), wss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WSS)')
plt.show()

# Find the optimal k (Elbow point)
optimal_k = np.diff(np.diff(wss)).argmin() + 2  # Add 2 because of the diff operation
print(f"Optimal number of clusters: {optimal_k}")

# 13. K-means Clustering
kmeans = KMeans(n_clusters=optimal_k, n_init=25, random_state=123)
kmeans.fit(scaled_data)

# Cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Assign clusters to the original dataset
data['Cluster'] = kmeans.labels_

# Visualize the clustering results on the first two principal components
plt.figure(figsize=(8,6))
sns.scatterplot(x=pc_scores[:, 0], y=pc_scores[:, 1], hue=data['Cluster'], palette='Set1', s=100)
plt.title('K-means Clustering (PCA components)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# 14. Cluster Sizes (Ensuring to only consider the numeric columns when calculating mean)
cluster_sizes = data['Cluster'].value_counts()
print("Cluster Sizes:")
print(cluster_sizes)

# 15. Cluster Means (Only considering numeric columns for aggregation)
cluster_means = data.groupby('Cluster').mean(numeric_only=True)  # Only numeric columns are aggregated
print("Cluster Means:")
print(cluster_means)

# 16. Cluster Descriptions (Summary statistics for each cluster)
for cluster in range(optimal_k):
    print(f"\nCluster {cluster} Description:")
    # Filtering out non-numeric columns and only summarizing numeric data for each cluster
    cluster_data = data[data['Cluster'] == cluster].select_dtypes(include=np.number)
    print(cluster_data.describe())
