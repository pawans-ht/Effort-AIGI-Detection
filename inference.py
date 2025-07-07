#!/usr/bin/env python3
"""
Enhanced inference script that replicates inference.sh functionality
but with detailed CSV output for analysis.

This script:
1. Extracts frames from videos at 1FPS (like inference.sh)
2. Runs deepfake detection on each frame
3. Saves detailed results to CSV files for each video
4. Includes frame number, model score, and Real/Fake classification

Usage:
    python inference.py <video_path_or_directory> [weights_name]
    python inference.py /path/to/video.mp4 effort_ckpt
    python inference.py /path/to/videos/ effort_ckpt
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import csv
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
import yaml
import dlib
import torch
import torch.nn.functional as F
from PIL import Image as pil_image
import torchvision.transforms as T
from tqdm import tqdm

# Add the DeepfakeBench training directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'DeepfakeBench', 'training'))

from trainer.trainer import Trainer
from detectors import DETECTOR
from imutils import face_utils
from skimage import transform as trans

# Configuration
WEIGHTS_PATH = "/Users/uddeshyaagrawal/Documents/Notes/DeepFaceDetection/Effort-AIGI-Detection/effort_clip_L14_trainOn_FaceForensic.pth"
LANDMARK_MODEL = "DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat"
DETECTOR_CONFIG = "DeepfakeBench/training/config/detector/effort.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VideoInferenceProcessor:
    def __init__(self, weights_path: str, landmark_model_path: str, detector_config: str):
        self.weights_path = weights_path
        self.landmark_model_path = landmark_model_path
        self.detector_config = detector_config
        self.model = None
        self.face_detector = None
        self.shape_predictor = None
        
        # Verify files exist
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        if not os.path.exists(landmark_model_path):
            raise FileNotFoundError(f"Landmark model not found: {landmark_model_path}")
        if not os.path.exists(detector_config):
            raise FileNotFoundError(f"Detector config not found: {detector_config}")
    
    def load_model(self):
        """Load the deepfake detection model and face detection components"""
        print("Loading deepfake detection model...")
        
        # Load detector model
        with open(self.detector_config, "r") as f:
            cfg = yaml.safe_load(f)
        
        model_cls = DETECTOR[cfg["model_name"]]
        self.model = model_cls(cfg).to(device)
        
        ckpt = torch.load(self.weights_path, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        
        # Load face detection components
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(self.landmark_model_path)
        
        print("[✓] Model and face detection components loaded successfully")
    
    def extract_frames(self, video_path: str, output_dir: str) -> str:
        """Extract frames from video at 1FPS using ffmpeg"""
        video_name = Path(video_path).stem
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        print(f"Extracting frames from: {video_path}")
        
        # Use ffmpeg to extract frames at 1FPS
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', 'fps=1',
            '-q:v', '2',
            os.path.join(video_output_dir, 'frame_%04d.jpg'),
            '-y'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Successfully extracted frames to: {video_output_dir}")
            return video_output_dir
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frames from {video_path}: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            return None
    
    def get_keypts(self, image, face):
        """Extract facial keypoints for alignment"""
        shape = self.shape_predictor(image, face)
        
        # Select key points for eyes, nose, and mouth
        leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
        reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
        nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
        lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
        rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
        
        pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)
        return pts
    
    def img_align_crop(self, img, landmark, outsize=(224, 224), scale=1.3):
        """Align and crop face according to landmarks"""
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        
        if target_size[1] == 112:
            dst[:, 0] += 8.0
        
        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]
        
        target_size = outsize
        
        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.
        
        # Move and resize
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)
        
        src = landmark.astype(np.float32)
        
        # Use skimage transformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]
        
        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))
        
        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        return img
    
    def extract_aligned_face(self, image, res=224):
        """Extract and align face from image"""
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_detector(rgb, 1)
        if len(faces):
            # Take the biggest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Get landmarks
            landmarks = self.get_keypts(rgb, face)
            
            # Align and crop face
            cropped_face = self.img_align_crop(rgb, landmarks, outsize=(res, res))
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
            
            return cropped_face, face
        
        return None, None
    
    def preprocess_face(self, img_bgr: np.ndarray):
        """Preprocess face image for model input"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073],
                        [0.26862954, 0.26130258, 0.27577711]),
        ])
        
        return transform(pil_image.fromarray(img_rgb)).unsqueeze(0)
    
    @torch.inference_mode()
    def inference(self, data_dict):
        """Run model inference"""
        data, label = data_dict['image'], data_dict['label']
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        predictions = self.model(data_dict, inference=True)
        return predictions
    
    @torch.inference_mode()
    def infer_single_image(self, img_bgr: np.ndarray) -> Tuple[int, float]:
        """
        Infer single image and return classification and probability
        Returns:
            cls_out: 0 for real, 1 for fake, -1 for no face detected, -2 for processing failed
            prob: probability score (0-1)
        """
        try:
            face_aligned, _ = self.extract_aligned_face(img_bgr, res=224)

            if face_aligned is None:
                # No face detected, return default values
                return -1, 0.0

            face_tensor = self.preprocess_face(face_aligned).to(device)
            data = {"image": face_tensor, "label": torch.tensor([0]).to(device)}
            preds = self.inference(data)

            cls_out = preds["cls"].squeeze().cpu().numpy()  # 0/1
            prob = preds["prob"].squeeze().cpu().numpy()    # probability

            # Handle both scalar and array outputs
            if cls_out.ndim == 0:
                # Scalar value
                cls_result = int(cls_out.item())
            else:
                # Array value - take the first element or argmax
                cls_result = int(cls_out.argmax() if cls_out.size > 1 else cls_out[0])

            if prob.ndim == 0:
                # Scalar value
                prob_result = float(prob.item())
            else:
                # Array value - take max probability or first element
                prob_result = float(prob.max() if prob.size > 1 else prob[0])

            return cls_result, prob_result

        except Exception as e:
            print(f"Warning: Processing failed for frame: {str(e)}")
            return -2, 0.0  # Processing failed
    
    def process_video_frames(self, frames_dir: str, video_name: str, output_dir: str):
        """Process all frames from a video and save results to CSV"""
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        
        if not frame_files:
            print(f"Warning: No frame files found in {frames_dir}")
            return
        
        print(f"Processing {len(frame_files)} frames for video: {video_name}")
        
        # Prepare CSV output
        csv_path = os.path.join(output_dir, f"{video_name}_analysis.csv")
        
        results = []
        
        for i, frame_path in enumerate(tqdm(frame_files, desc=f"Processing {video_name}")):
            frame_number = i + 1
            frame_filename = os.path.basename(frame_path)
            
            # Load and process frame
            img = cv2.imread(frame_path)
            if img is None:
                print(f"Warning: Could not load frame {frame_path}")
                continue
            
            # Run inference
            cls_out, prob = self.infer_single_image(img)

            # Determine classification
            if cls_out == -1:
                classification = "No Face Detected"
                fake_prob = 0.0
            elif cls_out == -2:
                classification = "Processing Failed"
                fake_prob = 0.0
            else:
                classification = "Fake" if prob >= 0.5 else "Real"
                fake_prob = prob
            
            # Store result
            result = {
                'frame_number': frame_number,
                'frame_filename': frame_filename,
                'model_score': fake_prob,
                'classification': classification,
                'prediction_label': cls_out,
                'timestamp': frame_number  # Assuming 1FPS, frame number = seconds
            }
            results.append(result)
        
        # Save to CSV
        if results:
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['frame_number', 'frame_filename', 'model_score', 'classification', 'prediction_label', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
            
            print(f"Results saved to: {csv_path}")
            
            # Print summary
            total_frames = len(results)
            fake_frames = sum(1 for r in results if r['classification'] == 'Fake')
            real_frames = sum(1 for r in results if r['classification'] == 'Real')
            no_face_frames = sum(1 for r in results if r['classification'] == 'No Face Detected')
            failed_frames = sum(1 for r in results if r['classification'] == 'Processing Failed')

            # Calculate percentages based on successfully processed frames (excluding failed frames)
            processed_frames = total_frames - failed_frames

            print(f"Summary for {video_name}:")
            print(f"  Total frames: {total_frames}")
            print(f"  Real frames: {real_frames}")
            print(f"  Fake frames: {fake_frames}")
            print(f"  No face detected: {no_face_frames}")
            print(f"  Processing failed: {failed_frames}")

            if processed_frames > 0:
                print(f"  Fake percentage (of processed): {(fake_frames/processed_frames)*100:.1f}%")
                print(f"  Real percentage (of processed): {(real_frames/processed_frames)*100:.1f}%")
                print(f"  No face percentage (of processed): {(no_face_frames/processed_frames)*100:.1f}%")

            if failed_frames > 0:
                print(f"  Failed percentage (of total): {(failed_frames/total_frames)*100:.1f}%")

            print("----------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Enhanced video deepfake inference with CSV output")
    parser.add_argument("input_path", help="Video file or directory containing videos")
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="Path to model weights")
    parser.add_argument("--landmark_model", default=LANDMARK_MODEL, help="Path to landmark model")
    parser.add_argument("--detector_config", default=DETECTOR_CONFIG, help="Path to detector config")
    parser.add_argument("--output_dir", default="./inference_results", help="Output directory for CSV files")
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.input_path):
        print(f"Error: Input path does not exist: {args.input_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processor
    try:
        processor = VideoInferenceProcessor(args.weights, args.landmark_model, args.detector_config)
        processor.load_model()
    except Exception as e:
        print(f"Error initializing processor: {e}")
        sys.exit(1)
    
    # Create temporary directory for frames
    temp_dir = f"./temp_frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        
        if os.path.isfile(args.input_path):
            # Single video file
            if Path(args.input_path).suffix.lower() not in video_extensions:
                print("Error: File does not appear to be a video file")
                sys.exit(1)
            
            print(f"Processing single video: {args.input_path}")
            frames_dir = processor.extract_frames(args.input_path, temp_dir)
            if frames_dir:
                video_name = Path(args.input_path).stem
                processor.process_video_frames(frames_dir, video_name, args.output_dir)
        
        elif os.path.isdir(args.input_path):
            # Directory containing videos
            print(f"Processing directory: {args.input_path}")
            
            video_files = []
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(args.input_path, f"*{ext}")))
            
            if not video_files:
                print(f"Error: No video files found in directory {args.input_path}")
                sys.exit(1)
            
            print(f"Found {len(video_files)} video files")
            
            for video_file in video_files:
                print(f"\nProcessing: {video_file}")
                frames_dir = processor.extract_frames(video_file, temp_dir)
                if frames_dir:
                    video_name = Path(video_file).stem
                    processor.process_video_frames(frames_dir, video_name, args.output_dir)
        
        else:
            print("Error: Input path is neither a file nor a directory")
            sys.exit(1)
    
    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
    
    print("Inference completed!")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
