import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

def extract_landmarks_from_image(image_path):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        return [coord for pt in landmarks for coord in (pt.x, pt.y, pt.z)]
    return None

def process_all_images(input_folder, output_csv):
    data = []
    filenames = []

    for file in tqdm(sorted(os.listdir(input_folder))):
        if file.endswith(".jpg"):
            path = os.path.join(input_folder, file)
            landmarks = extract_landmarks_from_image(path)

            if landmarks:
                data.append(landmarks)
                filenames.append(file)

    df = pd.DataFrame(data)
    df['filename'] = filenames
    df.to_csv(output_csv, index=False)
    print(f"Saved landmark features to {output_csv}")

if __name__ == "__main__":
    process_all_images("../data/smile_frames", "../data/features_landmarks.csv")
