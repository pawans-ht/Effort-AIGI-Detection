# Enhanced Inference Script

This `inference.py` script replicates the functionality of `inference.sh` but provides detailed analysis and CSV output for each video processed.

## Features

- **Frame Extraction**: Extracts frames from videos at 1FPS using ffmpeg (same as inference.sh)
- **Deepfake Detection**: Uses the same EFFORT model for classification
- **Detailed Analysis**: Saves comprehensive results to CSV files
- **Batch Processing**: Handles single videos or directories of videos
- **Face Detection**: Includes face detection status in results

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- dlib
- PIL/Pillow
- scikit-image
- tqdm
- numpy
- PyYAML

## Usage

### Single Video
```bash
python inference.py /path/to/video.mp4
```

### Directory of Videos
```bash
python inference.py /path/to/videos/
```

### Custom Parameters
```bash
python inference.py /path/to/video.mp4 \
    --weights /path/to/custom/weights.pth \
    --landmark_model /path/to/landmarks.dat \
    --output_dir ./custom_results
```

## Output

For each video processed, the script creates a CSV file with the following columns:

- **frame_number**: Sequential frame number (1, 2, 3, ...)
- **frame_filename**: Original frame filename (frame_0001.jpg, etc.)
- **model_score**: Raw probability score from the model (0.0 to 1.0)
- **classification**: Human-readable classification ("Real", "Fake", "No Face Detected", "Processing Failed")
- **prediction_label**: Model prediction (0=Real, 1=Fake, -1=No Face, -2=Processing Failed)
- **timestamp**: Time in seconds (assuming 1FPS extraction)

### Classification Categories

- **Real**: Frame classified as authentic (model_score < 0.5)
- **Fake**: Frame classified as deepfake (model_score ≥ 0.5)
- **No Face Detected**: No face found in the frame for analysis
- **Processing Failed**: Frame processing encountered an error (separate from fake detection)

## Example Output

```csv
frame_number,frame_filename,model_score,classification,prediction_label,timestamp
1,frame_0001.jpg,0.1234,Real,0,1
2,frame_0002.jpg,0.8765,Fake,1,2
3,frame_0003.jpg,0.0000,No Face Detected,-1,3
```

## Summary Statistics

After processing each video, the script prints summary statistics:
- Total frames processed
- Number of Real frames
- Number of Fake frames
- Number of frames with no face detected
- Percentage of fake frames

## Differences from inference.sh

1. **CSV Output**: Detailed frame-by-frame analysis saved to CSV files
2. **Face Detection Status**: Tracks frames where no face was detected
3. **Structured Data**: Machine-readable output for further analysis
4. **Progress Tracking**: Shows progress bars during processing
5. **Summary Statistics**: Provides overview of detection results

## File Structure

```
./inference_results/
├── video1_analysis.csv
├── video2_analysis.csv
└── ...
```

## Error Handling

- Validates input paths and required files
- Handles videos with no detectable faces
- Graceful handling of corrupted frames
- Automatic cleanup of temporary directories

## Performance Notes

- Processing time depends on video length and resolution
- GPU acceleration used when available
- Temporary frame storage requires disk space
- Memory usage scales with batch size (currently single frame processing)
