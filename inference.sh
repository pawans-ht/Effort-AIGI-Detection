#!/bin/bash

# Take in a video path / Directory as an argument.
if [ $# -eq 0 ]; then
    echo "Usage: $0 <video_path_or_directory> [weights_name]"
    echo "Example: $0 /path/to/video.mp4 effort_ckpt"
    echo "Example: $0 /path/to/videos/ effort_ckpt"
    exit 1
fi

INPUT_PATH="$1"

# Define Weights path as variable
WEIGHTS_PATH="/Users/uddeshyaagrawal/Documents/Notes/DeepFaceDetection/Effort-AIGI-Detection/effort_clip_L14_trainOn_FaceForensic.pth"
LANDMARK_MODEL="/Users/uddeshyaagrawal/Documents/Notes/DeepFaceDetection/Effort-AIGI-Detection/DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat"

# Check if weights file exists
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Error: Weights file not found at $WEIGHTS_PATH"
    echo "Please ensure the weights file exists or provide the correct weights name."
    echo "Available weights should be placed in ./DeepfakeBench/training/weights/"
    exit 1
fi

# Create a temporary directory for extracted frames in the current working directory
TEMP_DIR="./temp_frames_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEMP_DIR"
echo "Created temporary directory: $TEMP_DIR"

# Function to extract frames from a single video at 1FPS
extract_frames() {
    local video_file="$1"
    local output_dir="$2"

    echo "Extracting frames from: $video_file"

    # Create output directory for this video
    video_name=$(basename "$video_file" | sed 's/\.[^.]*$//')
    video_output_dir="$output_dir/$video_name"
    mkdir -p "$video_output_dir"

    # Extract frames at 1FPS using ffmpeg
    ffmpeg -i "$video_file" -vf fps=1 -q:v 2 "$video_output_dir/frame_%04d.jpg" -y 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "Successfully extracted frames to: $video_output_dir"
        return 0
    else
        echo "Error: Failed to extract frames from $video_file"
        return 1
    fi
}

# Function to run inference on extracted frames
run_inference() {
    local frames_dir="$1"

    echo "Running inference on frames in: $frames_dir"

    # Find all subdirectories (each containing frames from one video)
    for video_frames_dir in "$frames_dir"/*; do
        if [ -d "$video_frames_dir" ]; then
            video_name=$(basename "$video_frames_dir")
            echo "Processing frames for video: $video_name"

            # Check if there are any image files in this directory
            image_count=$(find "$video_frames_dir" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
            if [ $image_count -eq 0 ]; then
                echo "Warning: No image files found in $video_frames_dir, skipping..."
                continue
            fi

            echo "Found $image_count frames, running inference..."

            # Run inference on this specific video's frames
            uv run DeepfakeBench/training/demo.py \
                --detector_config DeepfakeBench/training/config/detector/effort.yaml \
                --weights "$WEIGHTS_PATH" \
                --image "$video_frames_dir" \
                --landmark_model "$LANDMARK_MODEL"

            echo "Completed inference for video: $video_name"
            echo "----------------------------------------"
        fi
    done
}

# Check if input is a file or directory
if [ -f "$INPUT_PATH" ]; then
    # Single video file
    echo "Processing single video file: $INPUT_PATH"

    # Check if it's a video file (basic check by extension)
    if [[ "$INPUT_PATH" =~ \.(mp4|avi|mov|mkv|flv|wmv|webm)$ ]]; then
        extract_frames "$INPUT_PATH" "$TEMP_DIR"
        if [ $? -eq 0 ]; then
            run_inference "$TEMP_DIR"
        fi
    else
        echo "Error: File does not appear to be a video file (unsupported extension)"
        exit 1
    fi

elif [ -d "$INPUT_PATH" ]; then
    # Directory containing multiple videos
    echo "Processing directory: $INPUT_PATH"

    # Find all video files in the directory
    video_count=0
    for video_file in "$INPUT_PATH"/*.{mp4,avi,mov,mkv,flv,wmv,webm}; do
        # Check if the file actually exists (in case no files match the pattern)
        if [ -f "$video_file" ]; then
            extract_frames "$video_file" "$TEMP_DIR"
            if [ $? -eq 0 ]; then
                ((video_count++))
            fi
        fi
    done

    if [ $video_count -eq 0 ]; then
        echo "Error: No video files found in directory $INPUT_PATH"
        exit 1
    fi

    echo "Processed $video_count video files"

    # Run inference on all extracted frames
    run_inference "$TEMP_DIR"

else
    echo "Error: Input path does not exist or is not a file/directory"
    exit 1
fi

# Cleanup: Remove temporary directory
echo "Cleaning up temporary directory: $TEMP_DIR"
rm -rf "$TEMP_DIR"

echo "Inference completed!"