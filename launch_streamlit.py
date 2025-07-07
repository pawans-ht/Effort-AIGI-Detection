#!/usr/bin/env python3
"""
Simple launcher for the Streamlit Deepfake Detection UI

This script launches the Streamlit application with proper configuration.
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        print("✅ Streamlit is available")
        return True
    except ImportError:
        print("❌ Streamlit not found. Please install it:")
        print("   pip install streamlit plotly")
        return False

def launch_app():
    """Launch the Streamlit application"""
    if not os.path.exists("streamlit_app.py"):
        print("❌ Error: streamlit_app.py not found in current directory")
        print("Please run this script from the project root directory.")
        return False
    
    if not check_requirements():
        return False
    
    print("🚀 Launching Streamlit Deepfake Detection Studio...")
    print("📱 The app will open in your default browser")
    print("🌐 URL: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = launch_app()
    if not success:
        sys.exit(1)
