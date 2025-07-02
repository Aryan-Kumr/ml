import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Mall_Customers.csv')

# two features for 2D clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# AGNES (Bottom-Up) Dendrogram
linkage_agnes = linkage(X_scaled, method='ward')
plt.figure()
dendrogram(linkage_agnes)
plt.title("Dendrogram - AGNES (Bottom-Up)")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# AGNES Clustering
agnes = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_agnes = agnes.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_agnes, cmap='rainbow')
plt.title("Clusters by AGNES")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.show()

# DIANA (Top-Down, simulated using complete linkage)
linkage_diana = linkage(X_scaled, method='complete')
plt.figure()
dendrogram(linkage_diana)
plt.title("Dendrogram - DIANA (Top-Down, Simulated)")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# DIANA Clustering (simulated)
diana = AgglomerativeClustering(n_clusters=4, linkage='complete')
labels_diana = diana.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_diana, cmap='rainbow')
plt.title("Clusters by DIANA (Simulated)")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.show()


# ---------------------------------------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

data = {'X': [1, 2, 5, 6, 8, 9, 10, 12], 
        'Y': [2, 1, 7, 6, 8, 9, 10, 11]}
df = pd.DataFrame(data)

# AGNES (Bottom-Up)
linkage_matrix = linkage(df, method='ward')
plt.figure()
dendrogram(linkage_matrix)
plt.title("Dendrogram - AGNES (Bottom-Up)")
plt.show()

# Agglomerative Clustering
agnes = AgglomerativeClustering(n_clusters=2)
labels_agnes = agnes.fit_predict(df)

plt.scatter(df['X'], df['Y'], c=labels_agnes, cmap='rainbow')
plt.title("Clusters by AGNES")
plt.show()

# DIANA (Top-Down) - simulated using 'complete' linkage
linkage_matrix_diana = linkage(df, method='complete')
plt.figure()
dendrogram(linkage_matrix_diana)
plt.title("Dendrogram - DIANA (Top-Down, Simulated)")
plt.show()

diana = AgglomerativeClustering(n_clusters=2, linkage='complete')
labels_diana = diana.fit_predict(df)

plt.scatter(df['X'], df['Y'], c=labels_diana, cmap='rainbow')
plt.title("Clusters by DIANA")
plt.show()
