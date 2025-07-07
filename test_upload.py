#!/usr/bin/env python3
"""
Simple test script to debug Streamlit upload issues
"""

import streamlit as st
import tempfile
import os

st.set_page_config(
    page_title="Upload Test",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 File Upload Test")
st.info("This is a simple test to debug file upload issues")

# Test basic file upload
st.header("Basic File Upload Test")

uploaded_file = st.file_uploader(
    "Choose any file",
    help="Upload any file to test basic functionality"
)

if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Get file info
    try:
        file_content = uploaded_file.getvalue()
        file_size = len(file_content)
        st.write(f"**File size:** {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        st.write(f"**File type:** {uploaded_file.type}")
        
        # Test saving to temp file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
                tmp_file.write(file_content)
                tmp_file.flush()
                temp_path = tmp_file.name
            
            # Verify file was saved
            if os.path.exists(temp_path):
                actual_size = os.path.getsize(temp_path)
                st.success(f"✅ Temporary file created: {temp_path}")
                st.write(f"**Saved size:** {actual_size} bytes")
                
                if actual_size == file_size:
                    st.success("✅ File sizes match - upload successful!")
                else:
                    st.error(f"❌ File size mismatch: expected {file_size}, got {actual_size}")
                
                # Clean up
                os.unlink(temp_path)
                st.info("🧹 Temporary file cleaned up")
            else:
                st.error("❌ Failed to create temporary file")
                
        except Exception as e:
            st.error(f"❌ Error saving file: {str(e)}")
            
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

# Test video-specific upload
st.header("Video File Upload Test")

video_file = st.file_uploader(
    "Choose a video file",
    type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'],
    help="Upload a video file to test video-specific functionality"
)

if video_file is not None:
    st.success(f"✅ Video uploaded: {video_file.name}")
    
    try:
        file_content = video_file.getvalue()
        file_size = len(file_content)
        st.write(f"**Video size:** {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        
        # Test video preview
        try:
            st.video(video_file)
            st.success("✅ Video preview works")
        except Exception as e:
            st.error(f"❌ Video preview failed: {str(e)}")
            
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")

# System info
st.header("System Information")
st.write(f"**Streamlit version:** {st.__version__}")
st.write(f"**Python version:** {os.sys.version}")
st.write(f"**Current working directory:** {os.getcwd()}")
st.write(f"**Temp directory:** {tempfile.gettempdir()}")

# Test temp directory access
try:
    test_file = tempfile.NamedTemporaryFile(delete=True)
    test_file.close()
    st.success("✅ Temporary directory is accessible")
except Exception as e:
    st.error(f"❌ Temporary directory issue: {str(e)}")
