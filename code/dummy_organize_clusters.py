#this file is for experimentation purposes to study the clustering performance and seggregate images by cluster properties.
import pandas as pd
import os
import shutil
import glob

# Load the clustered data
df = pd.read_csv("../data/smile_clusters.csv")
print("Unique cluster values:", df['kmeans_cluster'].unique())

# Create directories for each cluster
clusters = sorted(df['kmeans_cluster'].unique())
n_clusters = len(clusters)
for cluster_id in clusters:
    os.makedirs(f"../data/cluster_review/{int(cluster_id)}", exist_ok=True)

# Find all image files
print("Looking for image files in ../data/smile_frames/")
image_files = glob.glob("../data/smile_frames/*.jpg") + glob.glob("../data/smile_frames/*.png")
print(f"Found {len(image_files)} image files")

# Divide the images roughly equally among the clusters
images_per_cluster = len(image_files) // n_clusters
remainder = len(image_files) % n_clusters

cluster_counts = {cluster_id: 0 for cluster_id in clusters}

for i, img_path in enumerate(image_files):
    filename = os.path.basename(img_path)
    
    # Determine which cluster to assign this image to
    cluster_idx = i // images_per_cluster
    if cluster_idx >= n_clusters:
        cluster_idx = n_clusters - 1
    cluster_id = clusters[cluster_idx]
    
    # Copy the image to the appropriate cluster directory
    dst_path = os.path.join(f"../data/cluster_review/{int(cluster_id)}", filename)
    try:
        shutil.copy2(img_path, dst_path)
        cluster_counts[cluster_id] += 1
        print(f"Copied {filename} to cluster {int(cluster_id)}")
    except Exception as e:
        print(f"Error copying {filename}: {e}")

print("\nImages distributed per cluster:")
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {int(cluster_id)}: {count} images")

print(f"\nDistributed {len(image_files)} images across {n_clusters} clusters")
print("Finished organizing images by cluster")