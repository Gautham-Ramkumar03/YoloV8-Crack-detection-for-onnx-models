import argparse
import os
import logging
import time
from pathlib import Path

from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_conversion.log')
    ]
)

def convert_to_onnx(
    pt_model_path: str,
    output_dir: str = "onnx_models",
    imgsz: tuple = (640, 640),
    dynamic: bool = True,
    simplify: bool = True,
    opset: int = 12
) -> str:
    """
    Convert PyTorch model to ONNX format
    
    Args:
        pt_model_path: Path to PyTorch model (.pt)
        output_dir: Directory to save ONNX model
        imgsz: Input image size (width, height)
        dynamic: Enable dynamic axes
        simplify: Simplify ONNX model
        opset: ONNX opset version
        
    Returns:
        Path to saved ONNX model
    """
    try:
        start_time = time.time()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load PyTorch model
        logging.info(f"Loading PyTorch model from {pt_model_path}")
        model = YOLO(pt_model_path)
        
        # Get model name for output file
        model_name = Path(pt_model_path).stem
        onnx_path = str(output_path / f"{model_name}.onnx")
        
        # Export to ONNX
        logging.info(f"Converting model to ONNX (imgsz={imgsz}, dynamic={dynamic}, simplify={simplify}, opset={opset})")
        model.export(format="onnx", imgsz=imgsz, dynamic=dynamic, simplify=simplify, opset=opset)
        
        # Move the exported model to our desired location if needed
        default_export_path = f"{Path(pt_model_path).parent}/{model_name}.onnx"
        if os.path.exists(default_export_path) and default_export_path != onnx_path:
            os.rename(default_export_path, onnx_path)
            logging.info(f"Moved ONNX model to {onnx_path}")
        
        # Log model info
        conversion_time = time.time() - start_time
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # Size in MB
        
        logging.info(f"ONNX model saved to: {onnx_path}")
        logging.info(f"Model size: {onnx_size:.2f} MB")
        logging.info(f"Conversion completed in {conversion_time:.2f} seconds")
        
        # Print model comparison
        pt_size = os.path.getsize(pt_model_path) / (1024 * 1024)
        logging.info(f"Model size comparison:")
        logging.info(f"  - PyTorch: {pt_size:.2f} MB")
        logging.info(f"  - ONNX: {onnx_size:.2f} MB")
        logging.info(f"  - Size ratio: {onnx_size/pt_size:.2f}x")
        
        return onnx_path
        
    except Exception as e:
        logging.error(f"Error converting model to ONNX: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert YOLOv8 PyTorch model to ONNX format")
    parser.add_argument("--model", required=True, type=str, 
                       help="Path to PyTorch model file (.pt)")
    parser.add_argument("--output-dir", type=str, default="onnx_models",
                       help="Directory to save ONNX model")
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 640],
                       help="Image size (width height) for the model")
    parser.add_argument("--dynamic", action="store_true", 
                       help="Use dynamic input/output shapes")
    parser.add_argument("--no-simplify", action="store_false", dest="simplify",
                       help="Don't simplify ONNX model")
    parser.add_argument("--opset", type=int, default=12,
                       help="ONNX opset version")
    parser.add_argument("--half", action="store_true",
                       help="Export in half precision (FP16)")

    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Convert model to ONNX
    try:
        onnx_path = convert_to_onnx(
            pt_model_path=args.model,
            output_dir=args.output_dir,
            imgsz=tuple(args.imgsz),
            dynamic=args.dynamic,
            simplify=args.simplify,
            opset=args.opset
        )
        
        logging.info(f"Successfully converted model to ONNX: {onnx_path}")
    except Exception as e:
        logging.error(f"Conversion failed: {str(e)}")

if __name__ == "__main__":
    main()
