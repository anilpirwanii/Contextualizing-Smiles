import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load AU features
df = pd.read_csv("../data/openface_output/smile_frames_au.csv")

# Select AU columns (keep only *_r columns)
au_features = [col for col in df.columns if "_r" in col]
X = df[au_features]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering (try k=5)
kmeans = KMeans(n_clusters=5, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

# DBSCAN clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pc1'] = X_pca[:, 0]
df['pc2'] = X_pca[:, 1]

# Plotting K-means
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='pc1', y='pc2', hue='kmeans_cluster', palette='tab10')
plt.title("Smile Clusters (K-Means, PCA-Reduced)")
plt.savefig("../figures/kmeans_clusters.png")
plt.show()

# Save cluster-labeled data
df.to_csv("../data/smile_clusters.csv", index=False)
print("Clustered data saved to smile_clusters.csv")
