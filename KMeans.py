import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Reduce dimensions to 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

k_values = [2, 3, 4]

# Create plots for each K
plt.figure(figsize=(15, 4))

for i, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    score = silhouette_score(X, labels)
    
    # Plot clusters in PCA-reduced space
    plt.subplot(1, 3, i+1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=40)
    plt.title(f'K={k} | Silhouette Score={score:.2f}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

plt.suptitle('K-Means Clustering on Iris Dataset')
plt.tight_layout()
plt.show()
