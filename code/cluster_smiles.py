import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from PIL import Image
import glob

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

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pc1'] = X_pca[:, 0]
df['pc2'] = X_pca[:, 1]

# Map numeric clusters to meaningful labels
cluster_interpretations = {
    0: "Genuine/Charming",
    1: "Polite/Social",
    2: "Nervous/Awkward",
    3: "Sarcastic",
    4: "Dubious/Evil"
}

df['smile_type'] = df['kmeans_cluster'].map(cluster_interpretations)

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
ax_top.set_title("Smile Types Identified by Clustering", fontsize=18)
ax_top.set_xlabel("Principal Component 1", fontsize=14)
ax_top.set_ylabel("Principal Component 2", fontsize=14)

# Create a legend that maps clusters to interpretations
legend_elements = []
for i, (cluster_id, smile_type) in enumerate(cluster_interpretations.items()):
    color = plt.cm.tab10(i)
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                          label=f"Cluster {cluster_id}: {smile_type}",
                          markerfacecolor=color, markersize=10))

# Add the interpretation legend
ax_top.legend(handles=legend_elements, title="Interpreted Clusters", 
              loc="upper right", fontsize=10)

# Bottom section: example images from each cluster
ax_bottom = plt.subplot(gs[1])
ax_bottom.axis('off')  # Hide axes for the image section

# Create a grid for the images
img_grid = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1], wspace=0.1)

# Get all image files from smile_frames directory
image_files = glob.glob("../data/smile_frames/*.jpg") + glob.glob("../data/smile_frames/*.png")

# Manually select one example image for each cluster type
example_images = {
    0: image_files[0] if len(image_files) > 0 else None,    # Genuine/Charming
    1: image_files[1] if len(image_files) > 1 else None,    # Polite/Social
    2: image_files[2] if len(image_files) > 2 else None,    # Nervous/Awkward
    3: image_files[3] if len(image_files) > 3 else None,    # Sarcastic
    4: image_files[4] if len(image_files) > 4 else None,    # Dubious/Evil
}

# Display images or labels
for i, cluster_id in enumerate(range(5)):
    ax_img = plt.subplot(img_grid[i])
    
    if cluster_id in example_images and example_images[cluster_id] is not None:
        try:
            img = Image.open(example_images[cluster_id])
            ax_img.imshow(np.array(img))
            ax_img.set_title(f"{cluster_interpretations[cluster_id]}", fontsize=12)
        except Exception as e:
            print(f"Error loading image for cluster {cluster_id}: {e}")
            ax_img.text(0.5, 0.5, cluster_interpretations[cluster_id], 
                        ha='center', va='center', fontsize=12)
    else:
        ax_img.text(0.5, 0.5, cluster_interpretations[cluster_id], 
                   ha='center', va='center', fontsize=12)
    ax_img.axis('off')

plt.tight_layout()
plt.savefig("../figures/clusters_with_examples.png", dpi=300, bbox_inches='tight')
print("Visualization saved to ../figures/clusters_with_examples.png")
print(f"Found {len(image_files)} image files in smile_frames directory")