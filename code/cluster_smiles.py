import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from PIL import Image
import glob

# Load AU features from openface_output dataset
df = pd.read_csv("../data/openface_output/smile_frames.csv")

# Select AU columns (keep only *_r columns)
au_features = [col for col in df.columns if "_r" in col]
X = df[au_features]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Get list of image filenames, sorted alphabetically
image_files = sorted(glob.glob("../data/smile_frames/*.jpg"))

# Sanity check
assert len(image_files) == len(df), "Mismatch between number of frames and AU rows"

# Associate image filenames to the data
df["image"] = [os.path.basename(path) for path in image_files]
# Save updated file for later use
df.to_csv("../data/openface_output/smile_frames_with_image.csv", index=False)
print("Added image filenames to AU data")

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
plt.savefig("../figures/elbow_silhouette.png", dpi=300)
print("Saved elbow/silhouette plot to ../figures/elbow_silhouette.png")

# K-Means clustering (try k=4,6 and7)
k=7
kmeans = KMeans(n_clusters=k, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pc1'] = X_pca[:, 0]
df['pc2'] = X_pca[:, 1]

# # Map numeric clusters to meaningful labels
# cluster_interpretations = {
#     0: "Genuine/Charming",
#     1: "Polite/Social",
#     2: "Nervous/Awkward",
#     3: "Sarcastic",
#     4: "Dubious/Evil"
# }

# df['smile_type'] = df['kmeans_cluster'].map(cluster_interpretations)

# Create a new figure with grid layout
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# Top section: the cluster plot
ax_top = plt.subplot(gs[0])
sns.scatterplot(
    data=df, 
    x='pc1', 
    y='pc2', 
    hue='kmeans_cluster',
    palette='tab10',
    s=80,
    alpha=0.7,
    ax=ax_top
)
# ax_top.set_title("Smile Types Identified by Clustering", fontsize=18)
ax_top.set_title("Smile Clusters with k = 7 (AUs features + PCA)", fontsize=18)
ax_top.set_xlabel("Principal Component 1", fontsize=14)
ax_top.set_ylabel("Principal Component 2", fontsize=14)
ax_top.legend(title="Cluster", fontsize=10)

# === Example Image per Cluster ===
ax_bottom = plt.subplot(gs[1])
ax_bottom.axis("off")
img_grid = gridspec.GridSpecFromSubplotSpec(k, 3, subplot_spec=gs[1], wspace=0.1)

# Sample 3 images per cluster
example_images = (
    df.groupby("kmeans_cluster")
    .apply(lambda g: g.sample(3, random_state=42))
    .reset_index(drop=True)
)

for i, row in example_images.iterrows():
    cluster_id = int(row["kmeans_cluster"])
    col = i % 3  # 0, 1, 2
    ax_img = plt.subplot(img_grid[cluster_id, col])
    img_path = os.path.join("../data/smile_frames", row["image"])
    try:
        img = Image.open(img_path)
        ax_img.imshow(np.array(img))
        if col == 0:
            ax_img.set_title(f"Cluster {cluster_id}", fontsize=10, loc='left')
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        ax_img.text(0.5, 0.5, f"Cluster {cluster_id}", ha="center", va="center")
    ax_img.axis("off")

# Save visualizations in figures/
plt.tight_layout()
plt.savefig("../figures/clusters_k7_visualization.png", dpi=300)
print("Visualization saved to ../figures/clusters_k7_visualization.png")

# # Create a legend that maps clusters to interpretations
# legend_elements = []
# for i, (cluster_id, smile_type) in enumerate(cluster_interpretations.items()):
#     color = plt.cm.tab10(i)
#     legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
#                           label=f"Cluster {cluster_id}: {smile_type}",
#                           markerfacecolor=color, markersize=10))

# # Add the interpretation legend
# ax_top.legend(handles=legend_elements, title="Interpreted Clusters", 
#               loc="upper right", fontsize=10)

# # Bottom section: example images from each cluster
# ax_bottom = plt.subplot(gs[1])
# ax_bottom.axis('off')  # Hide axes for the image section

# # Create a grid for the images
# img_grid = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1], wspace=0.1)

# # Get all image files from smile_frames directory
# image_files = glob.glob("../data/smile_frames/*.jpg") + glob.glob("../data/smile_frames/*.png")

# # Manually select one example image for each cluster type
# example_images = {
#     0: image_files[0] if len(image_files) > 0 else None,    # Genuine/Charming
#     1: image_files[1] if len(image_files) > 1 else None,    # Polite/Social
#     2: image_files[2] if len(image_files) > 2 else None,    # Nervous/Awkward
#     3: image_files[3] if len(image_files) > 3 else None,    # Sarcastic
#     4: image_files[4] if len(image_files) > 4 else None,    # Dubious/Evil
# }

# # Display images or labels
# for i, cluster_id in enumerate(range(5)):
#     ax_img = plt.subplot(img_grid[i])
    
#     if cluster_id in example_images and example_images[cluster_id] is not None:
#         try:
#             img = Image.open(example_images[cluster_id])
#             ax_img.imshow(np.array(img))
#             ax_img.set_title(f"{cluster_interpretations[cluster_id]}", fontsize=12)
#         except Exception as e:
#             print(f"Error loading image for cluster {cluster_id}: {e}")
#             ax_img.text(0.5, 0.5, cluster_interpretations[cluster_id], 
#                         ha='center', va='center', fontsize=12)
#     else:
#         ax_img.text(0.5, 0.5, cluster_interpretations[cluster_id], 
#                    ha='center', va='center', fontsize=12)
#     ax_img.axis('off')

# plt.tight_layout()
# plt.savefig("../figures/clusters_with_examples.png", dpi=300, bbox_inches='tight')
# print("Visualization saved to ../figures/clusters_with_examples.png")
# print(f"Found {len(image_files)} image files in smile_frames directory")