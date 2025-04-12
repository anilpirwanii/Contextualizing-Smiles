# Contextualizing Smiles: Clustering Facial Expressions using AU and Landmark Features
CMPT 419/724 Affective Computing Spring 2025 - Final Project (Due April 11, 2025)

This project explores unsupervised clustering of human smiles using two types of facial features:
1. Facial Landmarks extracted using MediaPipe FaceMesh
2. Action Units (AUs) extracted using OpenFace

The goal is to discover and visualize different types of smiles (e.g., nervous, charming, polite) using K-Means clustering and evaluate clusters using both quantitative metrics (elbow and silhouette scores) and qualitative inspection of visual frames to study subtleties in the facial expressions and how different smiles relate to each other in terms of facial activity.

---

## Project Structure

All components of the project are organized into the following folders:

- `code/` – contains all Python scripts used for data processing, feature extraction, and clustering:
  - `extract_frames.py` – extracts top 3 frames per video from smile clips
  - `extract_landmarks.py` – uses MediaPipe to extract facial landmarks
  - `cluster_smiles.py` – performs clustering using Action Units
  - `cluster_landmarks.py` – performs clustering using landmark coordinates
  - `process_smiles.sh` – Bash script to automatically download and trim videos from YouTube

- `data/` – all dataset-related files and outputs:
  - `smile_dataset/` – contains downloaded and trimmed smile video clips
  - `smile_frames/` – stores extracted top-3 image frames from each video
  - `openface_output/smile_frames.csv` – AU feature vectors extracted using OpenFace
  - `features_landmarks.csv` – landmark coordinates extracted via MediaPipe
  - `smile_videos.csv` – list of YouTube links to the original smile videos

- `figures/` – all output plots and visualizations:
  - `elbow_silhouette.png`, `landmark_clusters.png`, etc.
  - Combined cluster visualizations with example frames

- `requirements.txt` – Python dependencies (Note: OpenFace must be installed separately for AU feature extraction)

- `README.md` – this file

---

## How to Run the Project

Make sure you have Python 3.8+ installed. Then follow these steps:

1. **Clone the project**

   ```git clone https://github.com/your-username/contextualizing-smiles.git```

2. **Install dependencies**
   ```pip3 install -r requirements.txt```
3. Navigate to the code directory, using ```cd code``` and run command ```./process_smiles.sh``` This will download YouTube videos listed in `data/smile_videos.csv` and save the clips to `data/smile_dataset/`.
4. **Extract top-3 frames from each video**
   ``` python3 extract_frames.py```
5. **Extract facial landmarks**
   ```python3 extract_landmarks.py```
6. **Extract Action Units using OpenFace**
(requires OpenFace installed separately)
``` cd Openface ./build/bin/FeatureExtraction -fdir ../data/smile_frames -aus -out_dir ../data/openface_output```
7. Perform Clustering
   ```python3 cluster_smiles.py # For AU-based clustering```
   ```python3 cluster_landmarks.py # For Landmark-based clustering```
---
## Self-Evaluation

This project meets the original objectives outlined in the proposal:

- Manually collected and extracted **141 smile video samples** from Youtube videos and extract **403 smile frames** from all clips in total, meeting the dataset samples requirement for this course project.
- Extracted feature vectors for each smile frame based on Facial Landmarks and Facial Action Units (AUs).
- Implemented AU-based and landmark-based feature clustering of different types of smile expressions.
- Used elbow and silhouette scores to evaluate optimal number of clusters.
- Visualized representative images for each cluster to evaluate clustering performance.
- Compared clustering performance between raw geometric (landmark) and abstracted (AU) features
- Labelled clusters based on representative images and used Inter-Rater Agreement to agree on CLuster-Label assignments. 

### Changes from proposal:
- Dropped DBSCAN in favor of KMeans due to clearer cluster control and visualization
- Extract top_3 smile frames using MediaPipe FaceMesh instead of Haar Cascades for accurate and representative smile frames for each video
- Used MediaPipe instead of OpenFace for landmarks due to ease of setup
- Focused on deeper visual analysis over real-time inference

---
## Notes for TA

- All key outputs used in the final report/poster are saved in `figures/`
- The clustering process can be reproduced using the two scripts: `cluster_smiles.py` and `cluster_landmarks.py`

Thank you for reviewing our project!
