import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Avoid Tkinter GUI errors
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Load dataset
df = pd.read_csv(r'C:\Users\ankit\Downloads\aryan\Mall_Customers.csv')
print("First 5 rows:\n", df.head())

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- AGNES ----------------
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram - AGNES (Agglomerative)')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('agnes_dendrogram.png')

agnes = AgglomerativeClustering(n_clusters=4)
labels_agnes = agnes.fit_predict(X_scaled)

plt.figure(figsize=(5, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_agnes, cmap='rainbow')
plt.title('AGNES Clustering Output')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.grid(True)
plt.tight_layout()
plt.savefig('agnes_clusters.png')

# ---------------- Simulated DIANA using Recursive KMeans ----------------
def recursive_kmeans(X, depth=2):
    labels = np.zeros(len(X), dtype=int)
    current_label = 0
    clusters = [np.arange(len(X))]

    for _ in range(depth):
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            kmeans = KMeans(n_clusters=2, random_state=42).fit(X[cluster])
            for i in [0, 1]:
                indices = cluster[kmeans.labels_ == i]
                labels[indices] = current_label
                new_clusters.append(indices)
                current_label += 1
        clusters = new_clusters
    return labels

labels_diana = recursive_kmeans(X_scaled, depth=2)

plt.figure(figsize=(5, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_diana, cmap='rainbow')
plt.title('DIANA (Simulated) Clustering Output')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.grid(True)
plt.tight_layout()
plt.savefig('diana_clusters.png')

# ---------------- Silhouette Scores ----------------
score_agnes = silhouette_score(X_scaled, labels_agnes)
score_diana = silhouette_score(X_scaled, labels_diana)

print(f'Silhouette Score - AGNES: {score_agnes:.2f}')
print(f'Silhouette Score - DIANA (Simulated): {score_diana:.2f}')
