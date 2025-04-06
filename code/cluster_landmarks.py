import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("../data/features_landmarks.csv", error_bad_lines=False, engine='python')

# Drop non-feature column
df = df.drop(columns=["filename"], errors='ignore')

df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Normalize
X_scaled = StandardScaler().fit_transform(df)

# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df["landmark_cluster"] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
df["pc1"] = pcs[:, 0]
df["pc2"] = pcs[:, 1]

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="pc1", y="pc2", hue="landmark_cluster", palette="Set1")
plt.title("Smile Clusters (Landmark Features, K-Means + PCA)")
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.legend(title="landmark_cluster")
plt.tight_layout()
plt.savefig("../figures/landmark_clusters.png")
plt.show()
