import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from PIL import Image

# Load data
df = pd.read_csv("../data/features_landmarks.csv", error_bad_lines=False, engine='python')

# Store filename separately
filenames = df["filename"].values if "filename" in df.columns else None

# Drop non-feature column
df = df.drop(columns=["filename"], errors='ignore')
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Normalize features
X_scaled = StandardScaler().fit_transform(df)

#Determine optimal K for K-Means clustering
elbow_scores = []
silhouette_scores = []
K_range = range(2, 11)
cluster_models = {}

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    preds = kmeans.fit_predict(X_scaled)
    cluster_models[k] = kmeans
    elbow_scores.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, preds))

# Plot Elbow & Silhouette Scores
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, elbow_scores, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='o')
plt.title("Silhouette Scores")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")

plt.tight_layout()
plt.savefig("../figures/landmarks_elbow_silhouette.png", dpi=300)
print("Saved elbow/silhouette plot to ../figures/landmarks_elbow_silhouette.png")

# K-Means clustering with k=5
k=5
kmeans = KMeans(n_clusters=5, random_state=42)
df["landmark_cluster"] = kmeans.fit_predict(X_scaled)

# Add back filename and PCA
if filenames is not None:
    df["filename"] = filenames

# PCA for visualization
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
df["pc1"] = pcs[:, 0]
df["pc2"] = pcs[:, 1]

# Plot clusters
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

#Display PCA + K-Means Clustering results at the top
ax_top = plt.subplot(gs[0])
sns.scatterplot(
    data=df,
    x="pc1",
    y="pc2",
    hue="landmark_cluster",
    palette="Set1",
    s=70,
    alpha=0.8,
    ax=ax_top
)

ax_top.set_title("Smile Clusters (Landmark Features + PCA)", fontsize=16)
ax_top.set_xlabel("Principal Component 1")
ax_top.set_ylabel("Principal Component 2")
ax_top.legend(title="Cluster")

# Display 1 Representative Image per Cluster at the bottom
ax_bottom = plt.subplot(gs[1])
ax_bottom.axis("off")
img_grid = gridspec.GridSpecFromSubplotSpec(1, k, subplot_spec=gs[1], hspace=0.5)

# Sample 1 image per cluster
sampled = (
    df.groupby("landmark_cluster")
    .apply(lambda g: g.sample(1))
    .reset_index(drop=True)
)

for i, row in sampled.iterrows():
    ax_img = plt.subplot(img_grid[i])
    img_path = os.path.join("../data/smile_frames", row["filename"])
    try:
        img = Image.open(img_path)
        ax_img.imshow(np.array(img))
        ax_img.set_title(f"Cluster {int(row['landmark_cluster'])}", fontsize=10, loc='left')
    except Exception as e:
        print(f"Error loading image: {img_path} — {e}")
        ax_img.text(0.5, 0.5, f"Cluster {int(row['landmark_cluster'])}", ha="center", va="center")
    ax_img.axis("off")

plt.tight_layout()
plt.savefig("../figures/landmark_clusters.png", dpi=300)
print("Saved image grid to ../figures/landmark_clusters.png")
