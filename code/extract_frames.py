import cv2
import os
import argparse
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

# Landmarks for mouth corners and lips
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
UPPER_LIP = 13
LOWER_LIP = 14

def compute_smile_score(landmarks, img_width, img_height):
    def to_pixel(pt): return np.array([pt.x * img_width, pt.y * img_height])

    left = to_pixel(landmarks[LEFT_MOUTH])
    right = to_pixel(landmarks[RIGHT_MOUTH])
    upper = to_pixel(landmarks[UPPER_LIP])
    lower = to_pixel(landmarks[LOWER_LIP])

    smile_width = np.linalg.norm(right - left)
    mouth_open = np.linalg.norm(upper - lower)

    return smile_width + 0.5 * mouth_open

def extract_top_n_frames(video_path, top_n=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open: {video_path}")
        return []

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    scored_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            score = compute_smile_score(landmarks, w, h)
            scored_frames.append((score, frame.copy(), frame_idx))

        frame_idx += 1

    cap.release()

    # Sort by smile score, descending
    scored_frames.sort(key=lambda x: x[0], reverse=True)
    return scored_frames[:top_n]

def process_folder(input_dir, output_dir, top_n=3):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".mp4"):
            video_path = os.path.join(input_dir, file)
            base_name = os.path.splitext(file)[0]

            print(f"Processing: {file}")
            top_frames = extract_top_n_frames(video_path, top_n=top_n)

            if not top_frames:
                print(f"No valid smile frames in: {file}")
                continue

            for i, (_, frame, frame_idx) in enumerate(top_frames, 1):
                out_path = os.path.join(output_dir, f"{base_name}_{i}.jpg")
                cv2.imwrite(out_path, frame)
                print(f"Saved frame {i} from {file} (score rank {i}) to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract top-N smile frames using MediaPipe.")
    parser.add_argument("--input", type=str, default="../data/smile_dataset", help="Input video folder")
    parser.add_argument("--output", type=str, default="../data/smile_frames", help="Output image folder")
    parser.add_argument("--top_n", type=int, default=3, help="Number of top smile frames to extract per video")
    args = parser.parse_args()

    process_folder(args.input, args.output, top_n=args.top_n)

if __name__ == "__main__":
    main()
