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

def select_representative_images(cluster_interpretations):
    """
    Manually select representative images for each cluster.
    
    Returns a dictionary with cluster IDs as keys and lists of image filenames as values.
    """
    selected_images = {
        0: [  # Duchenne Smile
            "anil_clip_026_1.jpg",
            "asmita_clip_029_1.jpg",
            "sana_clip_053_2.jpg"
        ],
        1: [  # Polite Smile
            "anil_clip_005_3.jpg",
            "sana_clip_029_3.jpg",
            "sana_clip_054_2.jpg"
        ],
        2: [  # Nervous Smile
            "anil_clip_010_3.jpg",
            "anil_clip_021_1.jpg",
            "anil_clip_023_1.jpg"
        ],
        3: [  # Dubious Smile
            "asmita_clip_028_1.jpg",
            "asmita_clip_006_2.jpg",
            "sana_clip_041_1.jpg"
        ],
        4: [  # Mixed Expressions
            "anil_clip_031_2.jpg",
            "asmita_clip_003_2.jpg",
            "anil_clip_033_2.jpg"
        ]
    }
    return selected_images

def get_specific_images(df, selected_images):
    specific_images = []
    for cluster_id, image_names in selected_images.items():
        for image_name in image_names:
            image_row = df[df['image'] == image_name].copy()
            image_row['kmeans_cluster'] = cluster_id
            specific_images.append(image_row)
    return pd.concat(specific_images)

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

# Map numeric clusters to meaningful labels - based on experimental analysis. 
cluster_interpretations = {
    0: "Duchenne Smile",
    1: "Polite Smile",
    2: "Nervous Smile", 
    3: "Dubious Smile",
    4: "Mixed Expressions"
}

df['smile_type'] = df['kmeans_cluster'].map(cluster_interpretations)

# Create a new figure with grid layout
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# Top section: the clustering visualization plot
ax_top = plt.subplot(gs[0])

sns.scatterplot(
    data=df, 
    x='pc1', 
    y='pc2', 
    hue='smile_type',
    palette='tab10',
    s=80,
    alpha=0.7,
    ax=ax_top
)

ax_top.set_title("Smile Clusters with k=7 (Action Units + PCA)", fontsize=18)
ax_top.set_xlabel("Principal Component 1", fontsize=14)
ax_top.set_ylabel("Principal Component 2", fontsize=14)
ax_top.legend(title="Smile Type", fontsize=10)

# Example Image per Cluster
plt.subplot(gs[1])
plt.axis("off")
img_grid = gridspec.GridSpecFromSubplotSpec(len(cluster_interpretations), 3, subplot_spec=gs[1], wspace=0.1, hspace=0.2)

# Sample 3 images per cluster
selected_images = select_representative_images(cluster_interpretations)
example_images = get_specific_images(df, selected_images)

for cluster_id, smile_type in cluster_interpretations.items():
    # Filter images for this specific cluster
    cluster_images = example_images[example_images['kmeans_cluster'] == cluster_id]
    
    for col in range(3):
        ax_img = plt.subplot(img_grid[cluster_id, col])
        
        if col < len(cluster_images):
            row = cluster_images.iloc[col]
            img_path = os.path.join("../data/smile_frames", row["image"])
            try:
                img = Image.open(img_path)
                ax_img.imshow(np.array(img))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                ax_img.text(0.5, 0.5, "Image Not Found", ha="center", va="center")
        
        ax_img.axis("off")
        
        # Add cluster label
        if col == 0:
            ax_img.text(-0.5, 0.5, smile_type, transform=ax_img.transAxes, 
                        ha='right', va='center', fontsize=10)

# Save visualizations in figures/
plt.tight_layout()
plt.savefig("../figures/clusters_k7_visualization.png", dpi=300)
print("Visualization saved to ../figures/clusters_k7_visualization.png")