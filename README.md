# ================================
# Customer Segmentation using K-Means
# Wholesale Customers Dataset
# ================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# 2. Load Dataset
df = pd.read_csv("Wholesale customers data.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# 3. Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


# 4. Elbow Method to find optimal K
inertia = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()


# 5. Apply K-Means (choose k = 3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

df["Cluster"] = clusters


# 6. Evaluate using Silhouette Score
sil_score = silhouette_score(scaled_data, clusters)
print("Silhouette Score:", sil_score)


# 7. PCA for 2D Visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure()
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Customer Segmentation using PCA")
plt.show()


# 8. Cluster Interpretation
cluster_summary = df.groupby("Cluster").mean()
print("\nCluster Summary (Mean Spending):")
print(cluster_summary)
