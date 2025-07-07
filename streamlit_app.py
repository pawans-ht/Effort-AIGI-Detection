#!/usr/bin/env python3
"""
Streamlit UI for Deepfake Detection Analysis

This application provides a beautiful interface for:
1. Running deepfake inference on videos
2. Visualizing results with comprehensive metrics
3. Comparing multiple videos
4. Exporting analysis reports
"""

import streamlit as st
import pandas as pd
import time

# Configure Streamlit for better file upload handling
st.set_page_config(
    page_title="Deepfake Detection Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import subprocess
import glob
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import json
import base64
from typing import Tuple
import cv2
from PIL import Image

# Page configuration moved to top of file

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
    }

    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .metric-card h2 {
        margin: 0 0 0.5rem 0;
        font-size: 1.8rem;
        font-weight: bold;
    }

    .metric-card p {
        margin: 0;
        font-size: 0.8rem;
        opacity: 0.9;
    }

    /* Metrics container for proper alignment */
    .metrics-container {
        margin: 1rem 0;
    }

    /* Ensure consistent column alignment */
    .stColumn {
        display: flex;
        align-items: stretch;
    }

    .stColumn > div {
        width: 100%;
    }
    
    .status-real {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    
    .status-fake {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    
    .status-no-face {
        background: linear-gradient(135deg, #bdc3c7 0%, #2c3e50 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DeepfakeAnalyzer:
    def __init__(self):
        self.results_dir = "./streamlit_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_inference(self, video_path: str, progress_callback=None) -> Tuple[bool, str]:
        """Run inference on a video file"""
        try:
            output_dir = os.path.join(self.results_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            cmd = [
                "python", "inference.py", 
                video_path,
                "--output_dir", output_dir
            ]
            
            if progress_callback:
                progress_callback("Starting inference...")
            
            # Run inference
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                return True, output_dir
            else:
                error_msg = result.stderr.strip()
                if "TypeError: only length-1 arrays can be converted to Python scalars" in error_msg:
                    st.error("🔧 Model output format error detected. This has been fixed - please try again.")
                elif "FileNotFoundError" in error_msg and "ffmpeg" in error_msg:
                    st.error("❌ FFmpeg not found. Please install FFmpeg to extract video frames.")
                    st.info("Install FFmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Ubuntu)")
                elif "No such file or directory" in error_msg:
                    st.error("❌ Required model files not found. Please check model weights and configuration files.")
                else:
                    st.error(f"Inference failed: {error_msg}")
                return False, ""

        except subprocess.TimeoutExpired:
            st.error("⏱️ Inference timed out after 30 minutes. Try with a shorter video.")
            return False, ""
        except Exception as e:
            st.error(f"Error running inference: {e}")
            return False, ""
    
    def load_analysis_data(self, csv_path: str) -> pd.DataFrame | None:
        """Load and validate CSV analysis data"""
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_cols = ['frame_number', 'frame_filename', 'model_score', 'classification', 'prediction_label', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV missing required columns. Expected: {required_cols}")
                return None
            
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return None
    
    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate comprehensive metrics from analysis data"""
        total_frames = len(df)
        
        # Basic counts
        real_frames = len(df[df['classification'] == 'Real'])
        fake_frames = len(df[df['classification'] == 'Fake'])
        no_face_frames = len(df[df['classification'] == 'No Face Detected'])
        failed_frames = len(df[df['classification'] == 'Processing Failed'])

        # Calculate processed frames (excluding failed frames)
        processed_frames = total_frames - failed_frames

        # Percentages based on processed frames
        real_percentage = (real_frames / processed_frames) * 100 if processed_frames > 0 else 0
        fake_percentage = (fake_frames / processed_frames) * 100 if processed_frames > 0 else 0
        no_face_percentage = (no_face_frames / processed_frames) * 100 if processed_frames > 0 else 0
        failed_percentage = (failed_frames / total_frames) * 100 if total_frames > 0 else 0

        # Face detection rate (of successfully processed frames)
        face_detection_rate = ((processed_frames - no_face_frames) / processed_frames) * 100 if processed_frames > 0 else 0
        
        # Model confidence statistics (for frames with faces, excluding failed frames)
        face_frames = df[(df['classification'] != 'No Face Detected') &
                        (df['classification'] != 'Processing Failed')]
        if len(face_frames) > 0:
            avg_confidence = face_frames['model_score'].mean()
            confidence_std = face_frames['model_score'].std()
            max_confidence = face_frames['model_score'].max()
            min_confidence = face_frames['model_score'].min()
        else:
            avg_confidence = confidence_std = max_confidence = min_confidence = 0
        
        # Video health score (higher = more real content)
        health_score = real_percentage
        
        # Fake segments analysis
        fake_segments = self.find_consecutive_segments(df, 'Fake')
        longest_fake_segment = max([seg[1] - seg[0] + 1 for seg in fake_segments]) if fake_segments else 0
        
        return {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'real_frames': real_frames,
            'fake_frames': fake_frames,
            'no_face_frames': no_face_frames,
            'failed_frames': failed_frames,
            'real_percentage': real_percentage,
            'fake_percentage': fake_percentage,
            'no_face_percentage': no_face_percentage,
            'failed_percentage': failed_percentage,
            'face_detection_rate': face_detection_rate,
            'health_score': health_score,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence,
            'fake_segments': fake_segments,
            'longest_fake_segment': longest_fake_segment,
            'total_fake_segments': len(fake_segments)
        }
    
    def find_consecutive_segments(self, df: pd.DataFrame, classification: str) -> list[tuple[int, int]]:
        """Find consecutive segments of a specific classification"""
        target_frames = df[df['classification'] == classification]['timestamp'].tolist()
        
        if not target_frames:
            return []
        
        segments = []
        start = target_frames[0]
        end = target_frames[0]
        
        for i in range(1, len(target_frames)):
            if target_frames[i] == target_frames[i-1] + 1:
                end = target_frames[i]
            else:
                if end > start:  # Only include segments longer than 1 frame
                    segments.append((start, end))
                start = end = target_frames[i]
        
        # Add the last segment
        if end > start:
            segments.append((start, end))
        
        return segments

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Deepfake Detection Studio</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = DeepfakeAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🎯 Navigation")
        
        page = st.selectbox(
            "Choose Analysis Mode",
            ["🏠 Home", "📹 Single Video Analysis", "📊 Batch Analysis", "🔍 Compare Videos", "📈 Advanced Analytics"]
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick stats if results exist
        results_files = glob.glob(os.path.join(analyzer.results_dir, "*", "*_analysis.csv"))
        if results_files:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.header("📊 Quick Stats")
            st.metric("Total Analyses", len(results_files))
            
            # Load latest analysis for quick preview
            latest_file = max(results_files, key=os.path.getctime)
            latest_df = analyzer.load_analysis_data(latest_file)
            if latest_df is not None:
                metrics = analyzer.calculate_metrics(latest_df)
                st.metric("Latest Health Score", f"{metrics['health_score']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📹 Single Video Analysis":
        show_single_video_analysis(analyzer)
    elif page == "📊 Batch Analysis":
        show_batch_analysis(analyzer)
    elif page == "🔍 Compare Videos":
        show_compare_videos(analyzer)
    elif page == "📈 Advanced Analytics":
        show_advanced_analytics(analyzer)

def show_home_page():
    """Display the home page with overview and instructions"""
    _, col2, _ = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        ## Welcome to Deepfake Detection Studio! 🎭
        
        This powerful tool helps you analyze videos for deepfake content using state-of-the-art AI models.
        
        ### 🚀 Features:
        - **Single Video Analysis**: Upload and analyze individual videos
        - **Batch Processing**: Analyze multiple videos at once
        - **Comparative Analysis**: Compare results across videos
        - **Advanced Metrics**: Detailed confidence scores and temporal analysis
        - **Beautiful Visualizations**: Interactive charts and graphs
        - **Export Reports**: Download analysis results
        
        ### 📊 Key Metrics:
        - **Health Score**: Overall authenticity percentage
        - **Temporal Analysis**: Frame-by-frame timeline
        - **Confidence Distribution**: Model certainty analysis
        - **Face Detection Rate**: Processing success rate
        - **Fake Segments**: Consecutive manipulation periods
        
        ### 🎯 Get Started:
        1. Choose "Single Video Analysis" from the sidebar
        2. Upload your video file
        3. Run the analysis
        4. Explore the results!
        """)
        
        st.markdown("### 🎯 Ready to Get Started?")
        st.markdown("""
        Choose an analysis mode from the sidebar to begin:
        - **Single Video Analysis** for individual video processing
        - **Batch Analysis** for processing multiple videos from a directory
        - **Compare Videos** to analyze differences between processed videos
        - **Advanced Analytics** for cross-video insights and patterns
        """)

def show_single_video_analysis(analyzer):
    """Display single video analysis interface"""
    st.header("📹 Single Video Analysis")

    # Initialize session state for upload error tracking
    if 'upload_errors' not in st.session_state:
        st.session_state.upload_errors = 0
    if 'last_upload_error' not in st.session_state:
        st.session_state.last_upload_error = None

    # Add file size warning
    st.info("📁 **File Upload Tips:**\n"
            "- Maximum file size: 500MB (configured in .streamlit/config.toml)\n"
            "- Supported formats: MP4, AVI, MOV, MKV, FLV, WMV, WebM\n"
            "- For larger files, use the file path option below\n"
            "- If you get upload errors, try using the file path option instead")

    # File upload with enhanced error handling
    uploaded_file = None
    upload_error = None

    try:
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'],
            help="Upload a video file to analyze for deepfake content",
            key="video_uploader",
            accept_multiple_files=False
        )
    except Exception as e:
        upload_error = str(e)
        st.session_state.upload_errors += 1
        st.session_state.last_upload_error = upload_error

        st.error(f"❌ File upload widget error: {upload_error}")
        st.error("**This is likely due to:**")
        st.error("- File size exceeding 500MB limit")
        st.error("- Network connectivity problems")
        st.error("- Browser compatibility issues")
        st.error("- Streamlit server configuration")
        st.error("**Solution:** Use the file path option below instead.")
        uploaded_file = None

    # Show upload status and recovery suggestions
    if uploaded_file is not None:
        # Log file size for debugging
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"ℹ️ Uploaded file size: {file_size_mb:.2f} MB")

        # Check against the 500MB limit
        if file_size_mb > 500:
            st.error("❌ **File size exceeds 500MB limit!**")
            st.error("Please upload a smaller file or use the file path option below.")
            uploaded_file = None  # Prevent further processing
        else:
            st.success(f"📁 File selected: {uploaded_file.name}")
            # Reset error counter on successful upload
            st.session_state.upload_errors = 0
    elif upload_error:
        st.warning("⚠️ Upload widget failed - use file path option below")

        # Show additional help for repeated errors
        if st.session_state.upload_errors > 1:
            st.error("🔄 **Repeated upload errors detected!**")
            with st.expander("🆘 Advanced Troubleshooting", expanded=True):
                st.markdown("""
                **Try these solutions in order:**

                1. **Refresh the page** (Ctrl+F5 or Cmd+Shift+R)
                2. **Clear browser cache and cookies**
                3. **Try a different browser** (Chrome, Firefox, Safari)
                4. **Check browser console** (F12 → Console tab) for detailed errors
                5. **Restart Streamlit server** and try again
                6. **Use file path option** instead of upload (most reliable)

                **If errors persist:**
                - Check your internet connection stability
                - Try uploading a smaller test video first
                - Contact support with browser console error details
                """)

                if st.button("🔄 Reset Error Counter", key="reset_errors"):
                    st.session_state.upload_errors = 0
                    st.session_state.last_upload_error = None
                    st.rerun()

    # Alternative: Direct file path input
    st.markdown("**Or enter video file path directly:**")
    video_path_input = st.text_input(
        "Video file path",
        placeholder="/path/to/your/video.mp4",
        help="Enter the full path to your video file"
    )

    # Troubleshooting section
    with st.expander("🔧 Troubleshooting Upload Issues"):
        st.markdown("""
        **Common solutions for upload errors:**

        1. **AxiosError 400**: Usually indicates file size or format issues
           - Try a smaller video file (< 500MB)
           - Use the file path option instead of upload
           - Check that the file format is supported

        2. **Network timeout**: For large files
           - Use the file path option for files > 100MB
           - Ensure stable internet connection

        3. **File format issues**:
           - Supported: MP4, AVI, MOV, MKV, FLV, WMV, WebM
           - Convert unsupported formats using FFmpeg

        4. **Browser issues**:
           - Try refreshing the page
           - Clear browser cache
           - Try a different browser
        """)

    temp_video_path = None
    video_name = None

    if uploaded_file is not None:
        try:
            # Debug info
            st.info(f"📁 Processing file: {uploaded_file.name}")

            # Check if file has content
            if not hasattr(uploaded_file, 'getvalue'):
                st.error("❌ Invalid file object. Please try uploading again.")
                st.error("**Troubleshooting:** Try refreshing the page and uploading again, or use the file path option.")
                return

            # Get file content safely
            try:
                file_content = uploaded_file.getvalue()
                file_size = len(file_content)

                # Additional validation
                if file_content is None:
                    st.error("❌ File content is empty. Please check the file and try again.")
                    return

            except Exception as e:
                st.error(f"❌ Failed to read file content: {str(e)}")
                st.error("**This might be due to:**")
                st.error("- Network timeout during upload")
                st.error("- File corruption during transfer")
                st.error("- Browser memory limitations")
                st.error("**Solution:** Try uploading again or use the file path option.")
                return

            # Check file size (configured limit is 500MB)
            if file_size == 0:
                st.error("❌ File appears to be empty. Please check the file and try again.")
                return

            if file_size > 500 * 1024 * 1024:  # 500MB
                st.error("❌ File too large! Please use a file smaller than 500MB or use the file path option.")
                return

            st.success(f"✅ File uploaded successfully: {uploaded_file.name} ({file_size / (1024*1024):.1f} MB)")

            # Save uploaded file temporarily with better error handling
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()

                # Create temporary file with proper permissions
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}", mode='wb') as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file.flush()  # Ensure data is written
                    os.fsync(tmp_file.fileno())  # Force write to disk
                    temp_video_path = tmp_file.name
                    video_name = uploaded_file.name

                # Verify the file was written correctly
                if not os.path.exists(temp_video_path):
                    st.error("❌ Failed to save temporary file. Please try again.")
                    st.error("**This might be due to:**")
                    st.error("- Insufficient disk space")
                    st.error("- Permission issues")
                    st.error("- System temporary directory issues")
                    return

                actual_size = os.path.getsize(temp_video_path)
                if actual_size != file_size:
                    st.error(f"❌ File size mismatch. Expected {file_size}, got {actual_size}. Please try again.")
                    st.error("**This indicates a file transfer error. Try uploading again or use the file path option.**")
                    # Clean up the corrupted file
                    try:
                        os.unlink(temp_video_path)
                    except OSError:
                        pass
                    return

                st.success(f"✅ File saved temporarily: {os.path.basename(temp_video_path)}")

            except PermissionError as e:
                st.error(f"❌ Permission error saving file: {str(e)}")
                st.error("**Solution:** Try using the file path option instead.")
                return
            except OSError as e:
                st.error(f"❌ System error saving file: {str(e)}")
                st.error("**This might be due to:**")
                st.error("- Insufficient disk space")
                st.error("- System temporary directory issues")
                st.error("**Solution:** Try using the file path option instead.")
                return
            except Exception as e:
                st.error(f"❌ Failed to save temporary file: {str(e)}")
                st.error("**Solution:** Try using the file path option instead.")
                return

        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {str(e)}")
            st.error("**Troubleshooting tips:**")
            st.error("- Try refreshing the page and uploading again")
            st.error("- Use the file path option instead")
            st.error("- Check your internet connection")
            st.error("- Try a different browser")
            return

    elif video_path_input.strip():
        # Use direct file path
        if os.path.exists(video_path_input.strip()):
            temp_video_path = video_path_input.strip()
            video_name = os.path.basename(temp_video_path)
        else:
            st.error("❌ File not found. Please check the path and try again.")
            return

    if temp_video_path is not None:
        
        # Display video info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display video preview
            if uploaded_file is not None:
                st.video(uploaded_file)
            else:
                # For direct file path, show the video file
                try:
                    st.video(temp_video_path)
                except Exception as e:
                    st.error(f"Cannot display video preview: {str(e)}")
                    st.info("Video file loaded successfully but preview unavailable.")
        
        with col2:
            st.markdown("### 📋 Video Information")
            st.write(f"**Filename:** {video_name}")
            # Get file size from the actual file
            try:
                file_size = os.path.getsize(temp_video_path) / (1024*1024)
                st.write(f"**Size:** {file_size:.1f} MB")
            except Exception:
                st.write("**Size:** Unknown")
            
            # Get video properties
            try:
                cap = cv2.VideoCapture(temp_video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                st.write(f"**Duration:** {duration:.1f} seconds")
                st.write(f"**Resolution:** {width}x{height}")
                st.write(f"**FPS:** {fps:.1f}")
                st.write(f"**Estimated frames to analyze:** {int(duration)}")  # 1 FPS extraction
            except Exception:
                st.write("Could not read video properties")
        
        # Analysis button
        if st.button("🚀 Run Deepfake Analysis", type="primary", use_container_width=True, key="run_single_analysis"):
            with st.spinner("Running deepfake analysis... This may take several minutes."):
                # Wait and verify file is ready
                time.sleep(0.5) # Initial wait
                retries = 5
                for i in range(retries):
                    if os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 0:
                        break
                    time.sleep(0.5)
                else:
                    st.error("❌ File verification failed. The temporary file is not ready.")
                    return

                # Run inference
                success, output_dir = analyzer.run_inference(temp_video_path)
                
                if success:
                    # Find the CSV file
                    csv_files = glob.glob(os.path.join(output_dir, "*_analysis.csv"))
                    if csv_files:
                        csv_path = csv_files[0]
                        df = analyzer.load_analysis_data(csv_path)
                        
                        if df is not None:
                            st.success("✅ Analysis completed successfully!")
                            
                            # Store results in session state for persistence
                            st.session_state['current_analysis'] = {
                                'df': df,
                                'video_name': video_name,
                                'csv_path': csv_path
                            }

                            # Display results
                            display_video_results(analyzer, df, video_name or "Unknown", "_current")
                        else:
                            st.error("Failed to load analysis results")
                    else:
                        st.error("No analysis results found")
                else:
                    st.error("Analysis failed")
        
        # Clean up temporary file (only if it was uploaded, not direct path)
        if uploaded_file is not None:
            try:
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
            except OSError:
                pass
    
    # Display previous results if available
    if 'current_analysis' in st.session_state:
        st.markdown("---")
        st.header("📊 Previous Analysis Results")
        analysis = st.session_state['current_analysis']
        display_video_results(analyzer, analysis['df'], analysis['video_name'], "_previous")

def display_video_results(analyzer, df: pd.DataFrame, video_name: str, key_suffix: str = ""):
    """Display comprehensive analysis results for a video"""
    # Calculate metrics
    metrics = analyzer.calculate_metrics(df)
    
    # Header with video name
    st.markdown(f"## 📊 Analysis Results: {video_name}")
    
    # Key metrics cards with equal spacing
    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        health_color = "🟢" if metrics['health_score'] > 70 else "🟡" if metrics['health_score'] > 30 else "🔴"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{health_color} Health Score</h3>
            <h2>{metrics['health_score']:.1f}%</h2>
            <p>Authenticity Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Face Detection</h3>
            <h2>{metrics['face_detection_rate']:.1f}%</h2>
            <p>Successful Detection Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Avg Confidence</h3>
            <h2>{metrics['avg_confidence']:.3f}</h2>
            <p>Model Certainty</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ Fake Segments</h3>
            <h2>{metrics['total_fake_segments']}</h2>
            <p>Suspicious Periods</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        failed_color = "🔴" if metrics['failed_frames'] > 0 else "🟢"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{failed_color} Failed Frames</h3>
            <h2>{metrics['failed_frames']}</h2>
            <p>{metrics['failed_percentage']:.1f}% of Total</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # Close metrics-container

    # Detailed breakdown
    st.markdown("### 📈 Detailed Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        # Classification pie chart
        classification_counts = df['classification'].value_counts()

        # Define colors for each classification type
        color_map = {
            'Real': '#38ef7d',           # Green
            'Fake': '#ff4b2b',           # Red
            'No Face Detected': '#bdc3c7',  # Gray
            'Processing Failed': '#f39c12'   # Orange
        }
        colors = [color_map.get(label, '#95a5a6') for label in classification_counts.index]

        fig_pie = go.Figure(data=[go.Pie(
            labels=classification_counts.index,
            values=classification_counts.values,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=12
        )])
        fig_pie.update_layout(
            title="Frame Classification Distribution",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True, key=f"classification_pie_chart{key_suffix}")

    with col2:
        # Confidence distribution histogram (exclude no face and failed frames)
        face_frames = df[(df['classification'] != 'No Face Detected') &
                        (df['classification'] != 'Processing Failed')]
        if len(face_frames) > 0:
            fig_hist = px.histogram(
                face_frames,
                x='model_score',
                nbins=20,
                title="Model Confidence Distribution",
                color_discrete_sequence=['#667eea']
            )
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red",
                              annotation_text="Decision Threshold")
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True, key=f"confidence_histogram{key_suffix}")
        else:
            st.info("No frames with detected faces for confidence analysis")

    # Timeline analysis
    st.markdown("### ⏱️ Timeline Analysis")

    # Create timeline plot
    fig_timeline = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Model Confidence Over Time", "Classification Timeline"),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )

    # Confidence timeline
    face_frames = df[df['classification'] != 'No Face Detected']
    if len(face_frames) > 0:
        fig_timeline.add_trace(
            go.Scatter(
                x=face_frames['timestamp'],
                y=face_frames['model_score'],
                mode='lines+markers',
                name='Confidence Score',
                line=dict(color='#667eea', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )

        # Add threshold line to first subplot
        fig_timeline.add_shape(
            type="line",
            x0=face_frames['timestamp'].min(),
            x1=face_frames['timestamp'].max(),
            y0=0.5,
            y1=0.5,
            line=dict(color="red", dash="dash"),
            row=1, col=1
        )

    # Classification timeline (color-coded bars)
    colors_map = {'Real': '#38ef7d', 'Fake': '#ff4b2b', 'No Face Detected': '#bdc3c7'}
    for classification in df['classification'].unique():
        class_data = df[df['classification'] == classification]
        fig_timeline.add_trace(
            go.Bar(
                x=class_data['timestamp'],
                y=[1] * len(class_data),
                name=classification,
                marker_color=colors_map[classification],
                showlegend=True
            ),
            row=2, col=1
        )

    fig_timeline.update_layout(
        height=600,
        title_text="Temporal Analysis Dashboard",
        showlegend=True
    )
    fig_timeline.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig_timeline.update_yaxes(title_text="Confidence Score", row=1, col=1)
    fig_timeline.update_yaxes(title_text="Classification", row=2, col=1)

    st.plotly_chart(fig_timeline, use_container_width=True, key=f"timeline_chart{key_suffix}")

    # Fake segments analysis
    if metrics['fake_segments']:
        st.markdown("### ⚠️ Suspicious Segments Analysis")

        segments_data = []
        for i, (start, end) in enumerate(metrics['fake_segments']):
            duration = end - start + 1
            segments_data.append({
                'Segment': f"Segment {i+1}",
                'Start Time (s)': start,
                'End Time (s)': end,
                'Duration (s)': duration,
                'Frames': duration
            })

        segments_df = pd.DataFrame(segments_data)
        st.dataframe(segments_df, use_container_width=True, key=f"fake_segments_dataframe{key_suffix}")

        # Segments visualization
        fig_segments = px.timeline(
            segments_df,
            x_start='Start Time (s)',
            x_end='End Time (s)',
            y='Segment',
            title="Fake Content Timeline",
            color_discrete_sequence=['#ff4b2b']
        )
        fig_segments.update_layout(height=300)
        st.plotly_chart(fig_segments, use_container_width=True, key=f"fake_segments_chart{key_suffix}")

    # Statistical summary
    st.markdown("### 📊 Statistical Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Frame Counts")
        st.write(f"**Total Frames:** {metrics['total_frames']}")
        st.write(f"**Processed Frames:** {metrics['processed_frames']}")
        st.write(f"**Real Frames:** {metrics['real_frames']} ({metrics['real_percentage']:.1f}%)")
        st.write(f"**Fake Frames:** {metrics['fake_frames']} ({metrics['fake_percentage']:.1f}%)")
        st.write(f"**No Face:** {metrics['no_face_frames']} ({metrics['no_face_percentage']:.1f}%)")
        if metrics['failed_frames'] > 0:
            st.write(f"**Failed Frames:** {metrics['failed_frames']} ({metrics['failed_percentage']:.1f}%)")

    with col2:
        st.markdown("#### Confidence Statistics")
        if metrics['avg_confidence'] > 0:
            st.write(f"**Mean Score:** {metrics['avg_confidence']:.4f}")
            st.write(f"**Std Deviation:** {metrics['confidence_std']:.4f}")
            st.write(f"**Min Score:** {metrics['min_confidence']:.4f}")
            st.write(f"**Max Score:** {metrics['max_confidence']:.4f}")
        else:
            st.write("No confidence data available")

    with col3:
        st.markdown("#### Segment Analysis")
        st.write(f"**Fake Segments:** {metrics['total_fake_segments']}")
        st.write(f"**Longest Fake:** {metrics['longest_fake_segment']} seconds")
        st.write(f"**Face Detection:** {metrics['face_detection_rate']:.1f}%")

        # Overall assessment
        if metrics['health_score'] > 80:
            st.success("🟢 **Assessment:** Likely Authentic")
        elif metrics['health_score'] > 50:
            st.warning("🟡 **Assessment:** Mixed Content")
        else:
            st.error("🔴 **Assessment:** Likely Manipulated")

    # Export options
    st.markdown("### 💾 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Download CSV
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV Data",
            data=csv_data,
            file_name=f"{video_name}_analysis.csv",
            mime="text/csv"
        )

    with col2:
        # Download JSON report
        report_data = {
            'video_name': video_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'summary': {
                'health_score': metrics['health_score'],
                'recommendation': 'Authentic' if metrics['health_score'] > 80 else 'Mixed' if metrics['health_score'] > 50 else 'Suspicious'
            }
        }

        json_data = json.dumps(report_data, indent=2, default=str)
        st.download_button(
            label="📋 Download JSON Report",
            data=json_data,
            file_name=f"{video_name}_report.json",
            mime="application/json"
        )

    with col3:
        # View raw data
        if st.button("👁️ View Raw Data", key=f"view_raw_data_single{key_suffix}"):
            st.dataframe(df, use_container_width=True, key=f"raw_data_dataframe{key_suffix}")

def show_batch_analysis(analyzer):
    """Display batch analysis interface"""
    st.header("📊 Batch Analysis")
    st.info("Provide a directory path containing video files for batch processing")

    # Directory path input
    directory_path = st.text_input(
        "Directory Path",
        placeholder="/path/to/video/directory",
        help="Enter the full path to a directory containing video files"
    )

    if directory_path:
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            # Find video files in directory
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.webm']
            video_files = []

            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(directory_path, ext)))
                video_files.extend(glob.glob(os.path.join(directory_path, ext.upper())))

            if video_files:
                st.success(f"Found {len(video_files)} video files in directory")

                # Display file list
                with st.expander("📁 Video Files Found", expanded=False):
                    for i, file_path in enumerate(video_files):
                        file_name = os.path.basename(file_path)
                        file_size = os.path.getsize(file_path) / (1024*1024)
                        st.write(f"{i+1}. {file_name} ({file_size:.1f} MB)")

                if st.button("🚀 Run Batch Analysis", type="primary", key="run_batch_analysis"):
                    batch_results = []

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, video_path in enumerate(video_files):
                        video_name = os.path.basename(video_path)
                        status_text.text(f"Processing {video_name}...")
                        progress_bar.progress((i + 1) / len(video_files))

                        # Run analysis
                        success, output_dir = analyzer.run_inference(video_path)

                        if success:
                            csv_files = glob.glob(os.path.join(output_dir, "*_analysis.csv"))
                            if csv_files:
                                df = analyzer.load_analysis_data(csv_files[0])
                                if df is not None:
                                    metrics = analyzer.calculate_metrics(df)
                                    batch_results.append({
                                        'video_name': video_name,
                                        'metrics': metrics,
                                        'df': df
                                    })

                    # Display batch results
                    if batch_results:
                        st.success(f"✅ Completed analysis of {len(batch_results)} videos")
                        display_batch_results(batch_results)
                    else:
                        st.error("No successful analyses completed")
            else:
                st.warning("No video files found in the specified directory")
        else:
            st.error("Invalid directory path. Please provide a valid directory path.")

def display_batch_results(batch_results: list[dict]):
    """Display comprehensive batch analysis results"""
    st.markdown("## 📊 Batch Analysis Results")

    # Summary table
    summary_data = []
    for result in batch_results:
        metrics = result['metrics']
        summary_data.append({
            'Video': result['video_name'],
            'Health Score (%)': f"{metrics['health_score']:.1f}",
            'Total Frames': metrics['total_frames'],
            'Processed Frames': metrics['processed_frames'],
            'Fake Frames': metrics['fake_frames'],
            'Failed Frames': metrics['failed_frames'],
            'Face Detection (%)': f"{metrics['face_detection_rate']:.1f}",
            'Avg Confidence': f"{metrics['avg_confidence']:.3f}",
            'Fake Segments': metrics['total_fake_segments']
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, key="batch_summary_dataframe")

    # Comparative visualizations
    st.markdown("### 📈 Comparative Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Health scores comparison
        health_scores = [result['metrics']['health_score'] for result in batch_results]
        video_names = [result['video_name'] for result in batch_results]

        fig_health = px.bar(
            x=video_names,
            y=health_scores,
            title="Health Score Comparison",
            color=health_scores,
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig_health.update_layout(height=400)
        fig_health.update_xaxes(tickangle=45)
        st.plotly_chart(fig_health, use_container_width=True, key="batch_health_scores")

    with col2:
        # Face detection rates
        face_rates = [result['metrics']['face_detection_rate'] for result in batch_results]

        fig_face = px.bar(
            x=video_names,
            y=face_rates,
            title="Face Detection Rate Comparison",
            color_discrete_sequence=['#667eea']
        )
        fig_face.update_layout(height=400)
        fig_face.update_xaxes(tickangle=45)
        st.plotly_chart(fig_face, use_container_width=True, key="batch_face_detection")

def show_compare_videos(analyzer):
    """Display video comparison interface"""
    st.header("🔍 Compare Videos")
    st.info("Select existing analysis results to compare")

    # Find available analyses
    results_files = glob.glob(os.path.join(analyzer.results_dir, "*", "*_analysis.csv"))

    if not results_files:
        st.warning("No analysis results found. Please run some analyses first.")
        return

    # Extract video names from file paths
    available_analyses = {}
    for file_path in results_files:
        video_name = os.path.basename(file_path).replace('_analysis.csv', '')
        available_analyses[video_name] = file_path

    # Multi-select for comparison
    selected_videos = st.multiselect(
        "Select videos to compare",
        options=list(available_analyses.keys()),
        help="Choose 2 or more videos for comparison"
    )

    if len(selected_videos) >= 2:
        # Load selected analyses
        comparison_data = []
        for video_name in selected_videos:
            df = analyzer.load_analysis_data(available_analyses[video_name])
            if df is not None:
                metrics = analyzer.calculate_metrics(df)
                comparison_data.append({
                    'video_name': video_name,
                    'metrics': metrics,
                    'df': df
                })

        if comparison_data:
            display_comparison_results(comparison_data)

def display_comparison_results(comparison_data: list[dict]):
    """Display detailed comparison between videos"""
    st.markdown("## 🔍 Video Comparison Results")

    # Side-by-side metrics
    cols = st.columns(len(comparison_data))

    for i, data in enumerate(comparison_data):
        with cols[i]:
            metrics = data['metrics']
            st.markdown(f"### {data['video_name']}")

            # Health score with color coding
            st.metric("Health Score", f"{metrics['health_score']:.1f}%", delta=None)
            st.metric("Face Detection", f"{metrics['face_detection_rate']:.1f}%")
            st.metric("Avg Confidence", f"{metrics['avg_confidence']:.3f}")
            st.metric("Fake Segments", metrics['total_fake_segments'])

def show_advanced_analytics(analyzer):
    """Display advanced analytics and insights"""
    st.header("📈 Advanced Analytics")
    st.info("Deep dive into analysis patterns and trends")

    # Find all available analyses
    results_files = glob.glob(os.path.join(analyzer.results_dir, "*", "*_analysis.csv"))

    if not results_files:
        st.warning("No analysis results found. Please run some analyses first.")
        return

    # Load all data for advanced analytics
    all_data = []
    for file_path in results_files:
        video_name = os.path.basename(file_path).replace('_analysis.csv', '')
        df = analyzer.load_analysis_data(file_path)
        if df is not None:
            df['video_name'] = video_name
            all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        display_advanced_insights(combined_df)

def display_advanced_insights(combined_df: pd.DataFrame):
    """Display advanced insights from all analyses"""
    st.markdown("## 🧠 Advanced Insights")

    # Overall statistics
    total_videos = combined_df['video_name'].nunique()
    total_frames = len(combined_df)
    overall_fake_rate = (len(combined_df[combined_df['classification'] == 'Fake']) / total_frames) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Videos Analyzed", total_videos)
    with col2:
        st.metric("Total Frames Processed", total_frames)
    with col3:
        st.metric("Overall Fake Rate", f"{overall_fake_rate:.1f}%")

    # Advanced visualizations
    st.markdown("### 📊 Cross-Video Analysis")

    # Confidence distribution across all videos
    face_frames = combined_df[combined_df['classification'] != 'No Face Detected']
    if len(face_frames) > 0:
        fig_violin = px.violin(
            face_frames,
            y='model_score',
            x='video_name',
            title="Confidence Score Distribution by Video",
            box=True
        )
        fig_violin.update_layout(height=500)
        fig_violin.update_xaxes(tickangle=45)
        st.plotly_chart(fig_violin, use_container_width=True, key="advanced_confidence_violin")

if __name__ == "__main__":
    main()
