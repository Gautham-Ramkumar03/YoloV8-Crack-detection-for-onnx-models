# YOLOv8 Crack Detection Inference

This repository provides tools for running inference with YOLOv8 crack detection segmentation models in both native PyTorch and ONNX formats.

## Project Overview

This project enables efficient crack detection in images and videos using YOLOv8 segmentation models specifically trained for crack detection. The models are hosted by OpenSistemas on Hugging Face and can be used directly or converted to ONNX format for optimized inference.

## Features

- Run inference with YOLOv8 models for crack detection/segmentation
- Convert YOLOv8 models to ONNX format for optimized inference
- Run inference with converted ONNX models
- Generate detailed performance reports and visualizations
- Support for both CPU and CUDA acceleration
- Process videos or camera feeds
- Real-time visualization with optional output video saving
- Comprehensive evaluation metrics (inference time, CPU/GPU usage, detection confidence)

## Setup and Installation

### Prerequisites
- Python 3.8+
- Git LFS (to handle large model files)
- (Optional) CUDA toolkit for GPU acceleration

### Installing Git LFS
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs
git lfs install
```

### Cloning the Repository and Models
```bash
git clone https://github.com/your-username/yolov8-crack-detection.git
cd yolov8-crack-detection
git lfs pull
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Inference with YOLOv8 Models
```bash
python scripts/predict.py --model models/yolov8_models/yolov8n-seg.pt --source data/images/test.jpg
```
Use `--source 0` for webcam or a video path for video inference.

### Converting to ONNX
```bash
python scripts/convert_to_onnx.py --model models/yolov8_models/yolov8n-seg.pt --output models/onnx_models/yolov8n-seg.onnx
```

### Inference with ONNX Models
```bash
python scripts/onnx_predict.py --model models/onnx_models/yolov8n-seg.onnx --source data/images/test.jpg
```

## Model Deployment

Deploy ONNX models to edge devices or use ONNX Runtime Server:
```bash
onnxruntime-server --model_path models/onnx_models/yolov8n-seg.onnx --http_port 8001
```

## Troubleshooting

1. **Corrupted or incomplete model files**  
   Verify Git LFS is installed and run `git lfs pull`.
2. **CUDA out of memory**  
   Use a smaller model or reduce batch sizes.
3. **ONNX runtime issues**  
   Reinstall ONNX Runtime or check compatible versions.
4. **Import errors**  
   Confirm dependencies are installed and Python environment is active.

## Directory Structure
# YoloV8-Crack-detection-for-onnx-models
