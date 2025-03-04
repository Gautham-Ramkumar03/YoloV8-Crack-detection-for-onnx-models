#!/bin/bash

# Setup script for YOLOv8 Crack Detection models
# This script installs Git LFS, clones the models, and validates the files

echo "Setting up YOLOv8 Crack Detection models..."

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Installing Git LFS..."
    sudo apt update
    sudo apt install -y git-lfs
fi

# Initialize Git LFS
echo "Initializing Git LFS..."
git lfs install

# Create models directory if it doesn't exist
mkdir -p models

# Clone the model repository
echo "Cloning YOLOv8 model repository..."
if [ -d "models/YOLOv8-crack-seg" ]; then
    echo "Model directory already exists. Pulling latest updates..."
    cd models/YOLOv8-crack-seg
    git pull
    git lfs pull
    cd ../..
else
    git clone https://huggingface.co/OpenSistemas/YOLOv8-crack-seg models/YOLOv8-crack-seg
fi

# Validate model files
echo "Validating model files..."
MODEL_FILES=$(ls -l models/YOLOv8-crack-seg/*.pt 2>/dev/null | wc -l)

if [ $MODEL_FILES -eq 0 ]; then
    echo "No model files found. Pulling LFS files..."
    cd models/YOLOv8-crack-seg
    git lfs pull
    cd ../..
fi

# Check file sizes to ensure they're not just LFS pointers
SMALL_FILES=$(find models/YOLOv8-crack-seg -name "*.pt" -size -100k 2>/dev/null | wc -l)
if [ $SMALL_FILES -gt 0 ]; then
    echo "Found small model files (likely LFS pointers). Pulling actual model files..."
    cd models/YOLOv8-crack-seg
    git lfs pull
    cd ../..
fi

# List model files
echo "Available model files:"
ls -lh models/YOLOv8-crack-seg/*.pt 2>/dev/null || echo "No model files found."

echo "Setup complete!"
echo "To convert models to ONNX format, run:"
echo "python -c \"from ultralytics import YOLO; model = YOLO('models/YOLOv8-crack-seg/yolov8n-seg-crack.pt'); model.export(format='onnx')\""
echo "To run inference with an ONNX model, run:"
echo "python onnx_inference.py --model models/YOLOv8-crack-seg/yolov8n-seg-crack.onnx --source path/to/video.mp4"

exit 0
