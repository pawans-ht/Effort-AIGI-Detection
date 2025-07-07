# 🎭 Deepfake Detection Studio - Streamlit UI

A beautiful, comprehensive web interface for deepfake video analysis with advanced visualizations and insights.

## 🌟 Features

### 📹 Single Video Analysis
- **Upload & Analyze**: Drag-and-drop video upload with real-time processing
- **Comprehensive Dashboard**: Multi-metric analysis with interactive visualizations
- **Timeline Analysis**: Frame-by-frame confidence tracking over time
- **Export Options**: Download CSV data and JSON reports

### 📊 Batch Processing
- **Directory Input**: Process all videos in a specified directory
- **Automatic File Discovery**: Finds all supported video formats
- **Comparative Analysis**: Side-by-side performance metrics
- **Batch Statistics**: Aggregate insights across video collections
- **Health Score Rankings**: Sort videos by authenticity ratings

### 🔍 Video Comparison
- **Side-by-Side Analysis**: Compare multiple video analyses
- **Relative Metrics**: Understand performance differences
- **Trend Identification**: Spot patterns across video sets

### 📈 Advanced Analytics
- **Cross-Video Insights**: Statistical analysis across all processed videos
- **Distribution Analysis**: Confidence score patterns and trends
- **Pattern Recognition**: Identify common characteristics in fake content

## 🎯 Key Metrics & Visualizations

### 📊 Primary Metrics
- **Health Score (0-100%)**: Overall video authenticity rating
- **Face Detection Rate**: Percentage of frames with detectable faces
- **Average Confidence**: Mean model certainty score
- **Fake Segments**: Number of consecutive suspicious periods
- **Longest Fake Segment**: Duration of most suspicious content

### 📈 Visualizations
1. **Timeline Plot**: Confidence scores over time with threshold indicators
2. **Classification Timeline**: Color-coded frame-by-frame results
3. **Confidence Distribution**: Histogram of model scores
4. **Pie Chart**: Frame classification breakdown
5. **Segment Analysis**: Fake content timeline visualization
6. **Comparative Charts**: Multi-video performance comparison
7. **Violin Plots**: Cross-video confidence distributions

## 🚀 Quick Start

### 1. Installation
```bash
# Install Streamlit-specific requirements
pip install -r streamlit_requirements.txt

# Or install manually
pip install streamlit plotly pandas numpy opencv-python Pillow
```

### 2. Launch the App
```bash
# Simple launcher
python launch_streamlit.py

# Or launch directly
streamlit run streamlit_app.py
```

### 3. Access the Interface
- Open your browser to `http://localhost:8501`
- Navigate using the sidebar menu
- Start with "Single Video Analysis" for your first analysis

## 📋 Usage Guide

### Single Video Analysis Workflow
1. **Upload Video**: Choose "Single Video Analysis" from sidebar
2. **Select File**: Upload MP4, AVI, MOV, or other supported formats
3. **Review Info**: Check video properties (duration, resolution, estimated frames)
4. **Run Analysis**: Click "Run Deepfake Analysis" and wait for processing
5. **Explore Results**: Navigate through comprehensive analysis dashboard
6. **Export Data**: Download CSV or JSON reports for further analysis

### Batch Analysis Workflow
1. **Select Batch Mode**: Choose "Batch Analysis" from sidebar
2. **Enter Directory Path**: Provide path to directory containing video files
3. **Review Files**: Check the list of discovered video files
4. **Start Batch**: Click "Run Batch Analysis" for simultaneous processing
5. **Compare Results**: View comparative metrics and rankings
6. **Analyze Patterns**: Identify trends across video collection

### Comparison Workflow
1. **Access Previous Results**: Choose "Compare Videos" from sidebar
2. **Select Videos**: Pick 2+ previously analyzed videos
3. **View Comparisons**: Examine side-by-side metrics and visualizations
4. **Identify Differences**: Understand relative performance characteristics

## 📊 Understanding the Results

### Health Score Interpretation
- **🟢 80-100%**: Likely authentic content
- **🟡 50-79%**: Mixed content, requires review
- **🔴 0-49%**: Likely manipulated content

### Confidence Score Analysis
- **0.0-0.3**: Strong indication of real content
- **0.3-0.7**: Uncertain region, requires careful analysis
- **0.7-1.0**: Strong indication of fake content
- **Threshold**: 0.5 (scores ≥0.5 classified as fake)

### Timeline Patterns
- **Consistent Low Scores**: Likely authentic video
- **Sudden Spikes**: Potential manipulation points
- **Sustained High Scores**: Likely fake content segments
- **Oscillating Patterns**: Mixed or partially manipulated content

## 🎨 UI Components

### Navigation Sidebar
- **Mode Selection**: Choose analysis type
- **Quick Stats**: Overview of processed videos
- **Settings**: Configuration options

### Main Dashboard
- **Metric Cards**: Key performance indicators with color coding
- **Interactive Charts**: Plotly-powered visualizations with zoom/pan
- **Data Tables**: Sortable, filterable result displays
- **Export Buttons**: Download options for reports and data

### Color Coding System
- **🟢 Green**: Real/Authentic content
- **🔴 Red**: Fake/Manipulated content  
- **⚪ Gray**: No face detected
- **🔵 Blue**: Neutral/Information elements

## 🔧 Technical Details

### Performance Considerations
- **Processing Time**: ~1-2 seconds per frame (depends on hardware)
- **Memory Usage**: Scales with video length and batch size
- **Storage**: Results cached for comparison features
- **GPU Acceleration**: Automatically used when available

### File Support
- **Video Formats**: MP4, AVI, MOV, MKV, FLV, WMV, WEBM
- **Frame Extraction**: 1 FPS (configurable in inference.py)
- **Face Detection**: dlib-based facial landmark detection
- **Model**: EFFORT deepfake detection model

### Data Export
- **CSV Format**: Frame-level analysis data
- **JSON Reports**: Comprehensive analysis summaries
- **Visualization Export**: Chart images (via Plotly)

## 🛠️ Customization

### Adding New Metrics
1. Modify `calculate_metrics()` in `streamlit_app.py`
2. Add visualization components in display functions
3. Update export formats to include new metrics

### Custom Visualizations
1. Use Plotly for interactive charts
2. Follow existing color scheme for consistency
3. Add to appropriate dashboard sections

### UI Theming
1. Modify CSS in the `st.markdown()` sections
2. Update color schemes in visualization functions
3. Customize metric card styling

## 🐛 Troubleshooting

### Common Issues

#### Model/Inference Issues
- **TypeError: only length-1 arrays can be converted to Python scalars**: This has been fixed in the latest version
- **FFmpeg not found**: Install FFmpeg (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Ubuntu)
- **Model files not found**: Ensure model weights and configuration files are in the correct locations
- **Processing Failed frames**: These are now tracked separately from fake detections for more accurate analysis

#### Application Issues
- **Analysis Fails**: Check model file paths and dependencies
- **Slow Processing**: Normal for large videos, consider shorter clips for testing
- **Memory Errors**: Reduce batch size or process videos individually
- **UI Not Loading**: Ensure all Streamlit requirements are installed

#### Directory Path Issues
- **No videos found**: Ensure the directory path is correct and contains supported video formats
- **Permission errors**: Check that the application has read access to the specified directory

### Debug Mode
```bash
# Run with debug logging
streamlit run streamlit_app.py --logger.level=debug
```

## 📈 Future Enhancements

### Planned Features
- **Real-time Video Stream Analysis**: Live camera feed processing
- **Advanced Filtering**: Custom threshold and confidence settings
- **Model Comparison**: Multiple detection models side-by-side
- **Annotation Tools**: Manual frame labeling and correction
- **API Integration**: REST API for programmatic access
- **Cloud Deployment**: Scalable cloud-based processing

### Visualization Improvements
- **3D Confidence Surfaces**: Multi-dimensional analysis views
- **Heatmaps**: Spatial analysis of manipulation regions
- **Animation**: Temporal progression visualization
- **Interactive Filtering**: Dynamic threshold adjustment

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-visualization`
3. Commit changes: `git commit -am 'Add new visualization'`
4. Push to branch: `git push origin feature/new-visualization`
5. Submit pull request

## 📄 License

This project is part of the Effort-AIGI-Detection framework. Please refer to the main project license.
