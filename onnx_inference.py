import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, List
from collections import deque
from datetime import datetime
import time
import psutil
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import onnxruntime as ort
import torch
import pynvml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('onnx_inference.log')
    ]
)

class EvaluationMetrics:
    """Handles evaluation metrics and report generation."""
    def __init__(self):
        self.frame_data = []
        self.start_time = datetime.now()
        self.performance_metrics = {
            'inference_times': [],
            'processing_times': [],
            'cpu_usage': [],
            'ram_usage': [],
            'gpu_usage': [],
            'gpu_memory': []
        }
        
    def add_frame_data(self, frame_number: int, has_detection: bool, confidence: float) -> None:
        """Add frame data to report."""
        self.frame_data.append({
            'frame_number': frame_number,
            'has_detection': has_detection,
            'confidence': confidence
        })
    
    def add_performance_data(self, inference_time: float, processing_time: float, 
                           cpu_usage: float, ram_usage: float,
                           gpu_usage: float = None, gpu_memory: float = None) -> None:
        """Add performance metrics data."""
        self.performance_metrics['inference_times'].append(inference_time)
        self.performance_metrics['processing_times'].append(processing_time)
        self.performance_metrics['cpu_usage'].append(cpu_usage)
        self.performance_metrics['ram_usage'].append(ram_usage)
        
        if gpu_usage is not None:
            self.performance_metrics['gpu_usage'].append(gpu_usage)
        if gpu_memory is not None:
            self.performance_metrics['gpu_memory'].append(gpu_memory)
    
    def generate_visualizations(self, output_dir: Path) -> None:
        """Generate visualization plots."""
        if not self.frame_data:
            logging.warning("No frame data available for visualization")
            return
            
        df = pd.DataFrame(self.frame_data)
        df = df.rename(columns={'frame_number': 'frame'})
        
        # Detection Timeline
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['confidence'], 'b-')
        plt.title('Detection Confidence Timeline')
        plt.xlabel('Frame Number')
        plt.ylabel('Confidence Score')
        plt.savefig(output_dir / 'detection_timeline.png')
        plt.close()
        
        # Confidence Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='confidence', bins=30)
        plt.title('Confidence Score Distribution')
        plt.savefig(output_dir / 'confidence_distribution.png')
        plt.close()
        
        # Performance Metrics
        if self.performance_metrics['inference_times']:
            # Inference Time Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(self.performance_metrics['inference_times'], bins=30)
            plt.title('Inference Time Distribution (s)')
            plt.savefig(output_dir / 'inference_time_distribution.png')
            plt.close()
            
            # CPU and RAM Usage
            plt.figure(figsize=(12, 6))
            plt.plot(self.performance_metrics['cpu_usage'], 'r-', label='CPU Usage (%)')
            plt.plot(np.array(self.performance_metrics['ram_usage']) / 100, 'b-', 
                    label='RAM Usage (GB / 100)')
            plt.title('CPU and RAM Usage')
            plt.xlabel('Frame')
            plt.ylabel('Usage')
            plt.legend()
            plt.savefig(output_dir / 'cpu_ram_usage.png')
            plt.close()
            
            # GPU metrics if available
            if self.performance_metrics['gpu_usage'] and self.performance_metrics['gpu_memory']:
                plt.figure(figsize=(12, 6))
                plt.plot(self.performance_metrics['gpu_usage'], 'g-', label='GPU Usage (%)')
                plt.plot(np.array(self.performance_metrics['gpu_memory']), 'y-', 
                        label='GPU Memory (GB)')
                plt.title('GPU Usage and Memory')
                plt.xlabel('Frame')
                plt.ylabel('Usage')
                plt.legend()
                plt.savefig(output_dir / 'gpu_usage.png')
                plt.close()
    
    def generate_report(self, output_dir: Path, model_info: dict) -> None:
        """Generate comprehensive markdown report."""
        df = pd.DataFrame(self.frame_data)
        total_frames = len(df)
        
        if total_frames == 0:
            logging.warning("No frames processed, cannot generate report")
            return
            
        frames_with_detections = df['has_detection'].sum()
        false_negatives = total_frames - frames_with_detections
        
        high_conf = df[df['confidence'] > 0.7].shape[0]
        med_conf = df[(df['confidence'] >= 0.5) & (df['confidence'] <= 0.7)].shape[0]
        low_conf = df[df['confidence'] < 0.5].shape[0]
        
        # Performance metrics
        avg_inference_time = np.mean(self.performance_metrics['inference_times'])
        avg_processing_time = np.mean(self.performance_metrics['processing_times'])
        avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        avg_cpu_usage = np.mean(self.performance_metrics['cpu_usage'])
        avg_ram_usage = np.mean(self.performance_metrics['ram_usage'])
        
        gpu_metrics = ""
        if self.performance_metrics['gpu_usage'] and self.performance_metrics['gpu_memory']:
            avg_gpu_usage = np.mean(self.performance_metrics['gpu_usage'])
            avg_gpu_memory = np.mean(self.performance_metrics['gpu_memory'])
            
            gpu_metrics = f"""
## GPU Performance
- Average GPU Usage: {avg_gpu_usage:.2f}%
- Average GPU Memory Usage: {avg_gpu_memory:.2f} GB
- GPU Usage Range: {min(self.performance_metrics['gpu_usage']):.2f}% - {max(self.performance_metrics['gpu_usage']):.2f}%
- GPU Memory Range: {min(self.performance_metrics['gpu_memory']):.2f} GB - {max(self.performance_metrics['gpu_memory']):.2f} GB

![GPU Usage and Memory](gpu_usage.png)
"""
        
        report_content = f"""# ONNX Model Evaluation Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Model Type: ONNX
- Model File: {model_info.get('model_path', 'N/A')}
- Model Size: {model_info.get('model_size', 'N/A')} MB
- Device: {model_info.get('device', 'N/A')}
- Input Shape: {model_info.get('input_shape', 'N/A')}
- Precision: {model_info.get('precision', 'N/A')}

## Summary
- Total Frames Analyzed: {total_frames}
- Frames with Detections: {frames_with_detections}
- False Negatives: {false_negatives}
- Raw Detection Accuracy: {(frames_with_detections/total_frames)*100:.2f}%
- Confidence-Weighted Accuracy: {df['confidence'].mean()*100:.2f}%
- Average Confidence Score: {df['confidence'].mean():.2f}

## Performance Metrics
- Average Inference Time: {avg_inference_time*1000:.2f} ms/frame
- Average Total Processing Time: {avg_processing_time*1000:.2f} ms/frame
- Average FPS: {avg_fps:.2f}
- Average CPU Usage: {avg_cpu_usage:.2f}%
- Average RAM Usage: {avg_ram_usage:.2f} MB
- Latency Range: {min(self.performance_metrics['inference_times'])*1000:.2f} - {max(self.performance_metrics['inference_times'])*1000:.2f} ms
- FPS Range: {1.0/max(self.performance_metrics['processing_times']):.2f} - {1.0/min(self.performance_metrics['processing_times']):.2f}

![Inference Time Distribution](inference_time_distribution.png)

![CPU and RAM Usage](cpu_ram_usage.png)

{gpu_metrics}

## Confidence Analysis
- High Confidence Detections (>0.7): {high_conf} ({high_conf/total_frames*100:.1f}%)
- Medium Confidence Detections (0.5-0.7): {med_conf} ({med_conf/total_frames*100:.1f}%)
- Low Confidence Detections (<0.5): {low_conf} ({low_conf/total_frames*100:.1f}%)

## Detection Timeline
![Detection Timeline](detection_timeline.png)

## Confidence Distribution
![Confidence Distribution](confidence_distribution.png)

## Detailed Metrics
| Metric | Value |
|--------|-------|
| Total Frames | {total_frames} |
| Successful Detections | {frames_with_detections} |
| Missed Detections | {false_negatives} |
| Raw Accuracy | {(frames_with_detections/total_frames)*100:.2f}% |
| Confidence-Weighted Accuracy | {df['confidence'].mean()*100:.2f}% |
| Avg. Confidence | {df['confidence'].mean():.2f} |
| Avg. Inference Time | {avg_inference_time*1000:.2f} ms |
| Avg. FPS | {avg_fps:.2f} |
| Avg. CPU Usage | {avg_cpu_usage:.2f}% |
| Avg. RAM Usage | {avg_ram_usage/1024:.2f} GB |
"""
        with open(output_dir / 'evaluation_report.md', 'w') as f:
            f.write(report_content)
        
        # Also save raw metrics as JSON for potential later analysis
        with open(output_dir / 'raw_metrics.json', 'w') as f:
            json.dump({
                'frame_data': self.frame_data,
                'performance_metrics': {
                    k: list(map(float, v)) for k, v in self.performance_metrics.items()
                },
                'model_info': model_info
            }, f, indent=2)

class ONNXInference:
    """Class for handling ONNX model inference."""
    def __init__(self, model_path: str, imgsz: Tuple[int, int] = (640, 640), 
                 conf_thres: float = 0.25, provider: str = "CUDAExecutionProvider"):
        """
        Initialize ONNX inference.
        
        Args:
            model_path: Path to ONNX model file
            imgsz: Input image size (width, height)
            conf_thres: Confidence threshold for detections
            provider: Execution provider (CUDAExecutionProvider or CPUExecutionProvider)
        """
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.provider = provider
        
        # Initialize ONNX Runtime
        available_providers = ort.get_available_providers()
        logging.info(f"Available providers: {available_providers}")
        
        if provider not in available_providers:
            fallback_provider = "CPUExecutionProvider"
            logging.warning(f"{provider} not available. Falling back to {fallback_provider}")
            self.provider = fallback_provider
        
        # Initialize PYNVML for GPU monitoring if using CUDA
        self.use_gpu = "CUDA" in self.provider
        if self.use_gpu:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Using first GPU
                logging.info("GPU monitoring initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize GPU monitoring: {e}")
                self.use_gpu = False
        
        # Create session
        logging.info(f"Creating ONNX Runtime session with provider {self.provider}")
        self.session = ort.InferenceSession(model_path, providers=[self.provider])
        
        # Get model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Store model info
        self.model_info = {
            'model_path': model_path,
            'model_size': os.path.getsize(model_path) / (1024 * 1024),  # Size in MB
            'device': self.provider,
            'input_shape': self.session.get_inputs()[0].shape,
            'precision': 'FP32'  # Assume FP32 by default
        }
        
        logging.info(f"Model loaded: {model_path}")
        logging.info(f"Input name: {self.input_name}")
        logging.info(f"Output names: {self.output_names}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ONNX model inference.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image tensor
        """
        # Resize and convert to RGB
        img = cv2.resize(image, self.imgsz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize and transpose
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def infer(self, image: np.ndarray) -> dict:
        """
        Run inference on an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with inference results
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Preprocess image
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Process outputs based on model type (detection or segmentation)
        if len(outputs) >= 3:  # Segmentation model outputs
            boxes, scores, class_ids = outputs[:3]
            masks = outputs[3] if len(outputs) > 3 else None
        else:  # Detection model outputs
            output = outputs[0]
            boxes, scores, class_ids = self.process_output(output)
            masks = None
        
        # Filter by confidence threshold
        mask = scores >= self.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        if masks is not None and len(masks) > 0:
            masks = masks[mask]
        
        # Scale boxes to original image size
        if len(boxes) > 0:
            # YOLO outputs normalized coordinates
            boxes[:, 0] *= width
            boxes[:, 1] *= height
            boxes[:, 2] *= width
            boxes[:, 3] *= height
        
        return {
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids,
            'masks': masks
        }
    
    def process_output(self, output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process YOLOv8 model output.
        
        Args:
            output: Model output tensor
            
        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        predictions = np.squeeze(output)
        
        # Check if output is empty
        if predictions.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])
            
        # Get boxes
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        
        # Get scores and class ids
        scores = predictions[:, 4:].max(axis=1)
        class_ids = predictions[:, 4:].argmax(axis=1)
        
        return boxes, scores, class_ids
    
    def get_gpu_metrics(self) -> Tuple[float, float]:
        """Get GPU utilization and memory usage."""
        if not self.use_gpu:
            return None, None
            
        try:
            # Get GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            gpu_utilization = utilization.gpu
            
            # Get GPU memory usage
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            gpu_memory_used = memory_info.used / (1024**3)  # Convert to GB
            
            return gpu_utilization, gpu_memory_used
        except Exception as e:
            logging.warning(f"Failed to get GPU metrics: {e}")
            return None, None

    def draw_predictions(self, image: np.ndarray, results: dict, fps: float = None) -> np.ndarray:
        """
        Draw predictions on image.
        
        Args:
            image: Input image
            results: Inference results
            fps: FPS to display
            
        Returns:
            Image with predictions drawn
        """
        boxes = results['boxes']
        scores = results['scores']
        class_ids = results['class_ids']
        masks = results.get('masks', None)
        
        # Create copy of image
        vis_img = image.copy()
        
        # Draw masks if available (for segmentation models)
        if masks is not None and len(masks) > 0:
            # Convert masks to correct format and scale them to original image size
            height, width = image.shape[:2]
            resized_masks = []
            for mask in masks:
                # Assuming mask is [1, H, W]
                m = cv2.resize(mask[0], (width, height))
                resized_masks.append(m > 0.5)  # Threshold
                
            # Combine all masks
            combined_mask = np.zeros((height, width), dtype=bool)
            for m in resized_masks:
                combined_mask = np.logical_or(combined_mask, m)
                
            # Create colored mask overlay
            color = [0, 255, 0]  # Green for cracks
            colored_mask = np.zeros_like(vis_img)
            colored_mask[combined_mask] = color
            
            # Blend with image
            alpha = 0.5
            vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, alpha, 0)

        # Draw boxes
        for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Crack: {score:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            y1 = max(y1, label_height)
            cv2.rectangle(
                vis_img,
                (x1, y1 - label_height - baseline),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                vis_img,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
        
        # Draw FPS
        if fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                vis_img, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
        
        return vis_img

def run_inference(
    model_path: str,
    source: str,
    output_dir: str = "output_onnx",
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    imgsz: Tuple[int, int] = (640, 640),
    save_video: bool = False,
    show_fps: bool = True,
    provider: str = "CUDAExecutionProvider",
    display: bool = True
) -> None:
    """
    Run ONNX model inference on video source.
    
    Args:
        model_path: Path to ONNX model
        source: Path to video file or camera index
        output_dir: Directory to save outputs
        conf_thres: Confidence threshold for detections
        iou_thres: IoU threshold for NMS
        imgsz: Input image size (width, height)
        save_video: Whether to save output video
        show_fps: Whether to display FPS on video
        provider: ONNX Runtime provider (CUDAExecutionProvider or CPUExecutionProvider)
        display: Whether to display frames in window (will be auto-disabled if GUI not available)
    """
    # Check if OpenCV has GUI support
    has_gui_support = True
    try:
        # Try to create and destroy a small window to test GUI support
        if display:
            cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("Test")
    except cv2.error:
        has_gui_support = False
        display = False
        logging.warning("OpenCV was built without GUI support. Display will be disabled.")
        logging.warning("Install opencv-python instead of opencv-python-headless for GUI support.")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = ONNXInference(
        model_path=model_path,
        imgsz=imgsz,
        conf_thres=conf_thres,
        provider=provider
    )
    
    # Initialize evaluation metrics
    metrics = EvaluationMetrics()
    
    # Open video source
    try:
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
            
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
    except Exception as e:
        logging.error(f"Error opening video source: {e}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer if needed
    video_writer = None
    if save_video:
        source_name = Path(source).stem if isinstance(source, str) and not source.isdigit() else f"camera_{source}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = str(output_path / f"{source_name}_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        logging.info(f"Saving output video to {output_video_path}")
    
    # Display video info
    logging.info(f"Video: {source}, Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")
    
    # Process frames
    frame_idx = 0
    fps_deque = deque(maxlen=30)  # Store last 30 FPS values
    processing_fps = 0
    
    try:
        while True:
            # Get CPU usage before inference
            cpu_percent = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
            
            # Read frame
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference
            inference_start = time.time()
            results = model.infer(frame)
            inference_time = time.time() - inference_start
            
            # Get GPU metrics
            gpu_usage, gpu_memory = model.get_gpu_metrics()
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps_deque.append(1 / elapsed if elapsed > 0 else 0)
            processing_fps = sum(fps_deque) / len(fps_deque)
            
            # Add metrics
            has_detection = len(results['boxes']) > 0
            confidence = float(np.max(results['scores'])) if has_detection else 0.0
            metrics.add_frame_data(frame_idx, has_detection, confidence)
            metrics.add_performance_data(
                inference_time=inference_time, 
                processing_time=elapsed,
                cpu_usage=cpu_percent,
                ram_usage=ram_usage,
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory
            )
            
            # Draw predictions
            vis_frame = model.draw_predictions(frame, results, processing_fps if show_fps else None)
            
            # Save frame to video if requested
            if video_writer is not None:
                video_writer.write(vis_frame)
            
            # Display frame if GUI support is available and display is enabled
            if display and has_gui_support:
                cv2.imshow("YOLOv8 ONNX Inference", vis_frame)
                if cv2.waitKey(1) == 27:  # ESC key
                    break
                
            frame_idx += 1
            
            # Progress logging
            if total_frames > 0 and frame_idx % 100 == 0:
                progress = frame_idx / total_frames * 100
                logging.info(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames}), FPS: {processing_fps:.1f}")
    
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Error during inference: {e}")
    finally:
        # Clean up
        cap.release()
        if video_writer is not None:
            video_writer.release()
            
        # Only destroy windows if OpenCV has GUI support
        if has_gui_support:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        
        # Generate reports
        logging.info(f"Processed {frame_idx} frames")
        metrics.generate_visualizations(output_path)
        metrics.generate_report(output_path, model.model_info)
        
        # Log final metrics
        if metrics.performance_metrics['inference_times']:
            avg_inference = np.mean(metrics.performance_metrics['inference_times'])
            avg_fps = 1.0 / np.mean(metrics.performance_metrics['processing_times'])
            logging.info(f"Average inference time: {avg_inference*1000:.2f} ms")
            logging.info(f"Average FPS: {avg_fps:.2f}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 ONNX Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=str, required=True, help="Path to video file or camera index")
    parser.add_argument("--output-dir", type=str, default="output_onnx", help="Directory to save outputs")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[640, 640], help="Input image size (width height)")
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    parser.add_argument("--show-fps", action="store_true", help="Display FPS on video")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run inference on")
    parser.add_argument("--no-display", action="store_false", dest="display", 
                      help="Disable display window (auto-disabled if GUI support is missing)")
    
    args = parser.parse_args()
    
    # Check ONNX model file
    if not os.path.exists(args.model):
        logging.error(f"Model file not found: {args.model}")
        return
        
    # Determine provider based on device
    provider = "CUDAExecutionProvider" if args.device == "cuda" else "CPUExecutionProvider"
    
    # Run inference
    run_inference(
        model_path=args.model,
        source=args.source,
        output_dir=args.output_dir,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        imgsz=tuple(args.imgsz),
        save_video=args.save_video,
        show_fps=args.show_fps,
        provider=provider,
        display=args.display
    )

if __name__ == "__main__":
    main()
