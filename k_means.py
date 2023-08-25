from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generating random sample data for demonstration
data = np.random.rand(100, 2) * 15  # 100 data points in 2D space

# Using k-means with 4 clusters
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)

# Getting centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Plotting
colors = ["g.", "r.", "b.", "y."]

plt.figure(figsize=(10, 6))
for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], colors[labels[i]], markersize=20)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
plt.title("KMeans Clustering with 4 Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

