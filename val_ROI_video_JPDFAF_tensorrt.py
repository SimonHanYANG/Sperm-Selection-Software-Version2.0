"""
TensorRT ROI Video Processing with Detection, Tracking and Segmentation

This script combines YOLOv8 detection+tracking with ROI segmentation using TensorRT.
The pipeline:
1. Detect objects using YOLOv8 TensorRT
2. Track objects using JPDAF tracker
3. Extract ROIs around tracked objects
4. Run segmentation on these ROIs
5. Overlay segmentation results on the original video
6. Visualize gating ellipses for tracking

Usage:
python val_ROI_video_JPDFAF_tensorrt.py \
    --video input_video.mp4 \
    --output results/output_video.mp4 \
    --det_engine yolo_weights/best.engine \
    --seg_model ADSCNet_sperm_ROINAHead_250707 \
    --seg_engine model_fp32.engine \
    --conf 0.25 \
    --preview

# 启用门限椭圆可视化（默认）
python val_ROI_video_tensorrt.py --video input.mp4 --seg_model model_name

# 禁用门限椭圆可视化
python val_ROI_video_tensorrt.py --video input.mp4 --seg_model model_name --no_gating_ellipses

# Han test
python val_ROI_video_JPDFAF_tensorrt.py --video test-videoes/20241105_115426118.mp4 --output output-videoes/20241105_115426118_ROI_seg_tracking_JPDAFDraw.mp4 --det_engine yolo_weights/best.engine --seg_model ADSCNet_sperm_ROINAHead_250707 --seg_engine model_fp32.engine --conf 0.25 

python val_ROI_video_JPDFAF_tensorrt.py --video E:\\MMRL-LAB-NAS\\Han-Workspace\\19201200-savedImg\\capture_20250808-225344.mp4 --output E:\\MMRL-LAB-NAS\\Han-Workspace\\19201200-savedImg\\results\\capture_20250808-225344_ROI_seg_tracking_JPDAFDraw.mp4 --det_engine yolo_weights/best.engine --seg_model ADSCNet_sperm_ROINAHead_250707 --seg_engine model_fp32.engine --conf 0.25 

python val_ROI_video_JPDFAF_tensorrt.py --video E:\\MMRL-LAB-NAS\\Han-Workspace\\19201200-savedImg\\capture_20250808-225705.mp4 --output E:\\MMRL-LAB-NAS\\Han-Workspace\\19201200-savedImg\\results\\capture_20250808-225705_ROI_seg_tracking_JPDAFDraw.mp4 --det_engine yolo_weights/best.engine --seg_model ADSCNet_sperm_ROINAHead_250707 --seg_engine model_fp32.engine --conf 0.25 

python val_ROI_video_JPDFAF_tensorrt.py --video E:\\MMRL-LAB-NAS\\Han-Workspace\\19201200-savedImg\\capture_20250808-230336.mp4 --output E:\\MMRL-LAB-NAS\\Han-Workspace\\19201200-savedImg\\results\\capture_20250808-230336_ROI_seg_tracking_JPDAFDraw.mp4 --det_engine yolo_weights/best.engine --seg_model ADSCNet_sperm_ROINAHead_250707 --seg_engine model_fp32.engine --conf 0.25 

# xiangya test
python val_ROI_video_JPDFAF_tensorrt.py --video E:\\MMRL-LAB-NAS\\Han-Workspace\\0819\\2022_8_19_13_47_9.avi --output E:\\MMRL-LAB-NAS\\Han-Workspace\\0819\\results\\2022_8_19_13_47_9_ROI_seg_tracking_JPDAFDraw.mp4 --det_engine yolo_weights/best.engine --seg_model ADSCNet_sperm_ROINAHead_250707 --seg_engine model_fp32.engine --conf 0.25 

python val_ROI_video_JPDFAF_tensorrt.py --video E:\\MMRL-LAB-NAS\\Han-Workspace\\0819\\2022_8_19_13_51_8.avi --output E:\\MMRL-LAB-NAS\\Han-Workspace\\0819\\results\\2022_8_19_13_51_8_ROI_seg_tracking_JPDAFDraw.mp4 --det_engine yolo_weights/best.engine --seg_model ADSCNet_sperm_ROINAHead_250707 --seg_engine model_fp32.engine --conf 0.25 

# 20250813 Swim Up Using
python val_ROI_video_JPDFAF_tensorrt.py --video E:\\MMRL-LAB-NAS\\Han-Workspace\\20250815-SwimUpUsingLatest\\using2.mp4 --output E:\\MMRL-LAB-NAS\\Han-Workspace\\20250815-SwimUpUsingLatest\\results\\using2_5min_ROI_seg_tracking_JPDAFDraw.mp4 --det_engine yolo_weights/best.engine --seg_model ADSCNet_sperm_ROINAHead_250707 --seg_engine model_fp32.engine --conf 0.25 


"""

import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from pathlib import Path
import json
import argparse
from collections import defaultdict
import colorsys
import yaml
import shutil

# Import JPDAF tracker
from jpdaf_tracker import JPDAFilter

# Import Albumentations for segmentation preprocessing
import albumentations as A
from albumentations.core.composition import Compose

# Check environment
print(f"Using GPU: {cuda.Device(0).name()}")
print(f"CUDA version: {cuda.get_version()}")
print(f"TensorRT version: {trt.__version__}")

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def dims_to_tuple(dims):
    """将TensorRT Dims对象转换为Python元组 (适用于TensorRT 10.x)"""
    try:
        # 尝试直接转换为列表
        dims_list = list(dims)
        return tuple(dims_list)
    except:
        # 备用方法: 打印为字符串并解析
        dims_str = str(dims)
        # 移除可能的括号并分割
        dims_str = dims_str.strip('()[]{}')
        # 解析数字
        try:
            return tuple(int(x) for x in dims_str.split(',') if x.strip())
        except:
            # 最后的尝试：使用固定的形状
            print(f"警告: 无法解析维度 {dims}，使用默认形状 (1, 3, 640, 640)")
            return (1, 3, 640, 640)

class TensorRTDetector:
    def __init__(self, engine_path):
        """Initialize TensorRT detection engine (适用于TensorRT 10.x)"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        print("Loading TensorRT detection engine...")
        
        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output information (TensorRT 10.x API)
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        # 打印张量数量
        print(f"Engine tensor count: {self.engine.num_io_tensors}")
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            
            print(f"Processing tensor {i}: {tensor_name}, mode: {tensor_mode}")
            
            try:
                tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                tensor_shape_dims = self.engine.get_tensor_shape(tensor_name)
                tensor_shape = dims_to_tuple(tensor_shape_dims)
                
                print(f"Tensor shape: {tensor_shape}, data type: {tensor_dtype}")
                
                # 计算内存大小
                size = int(np.prod(tensor_shape))
                
                # 分配设备内存
                device_mem = cuda.mem_alloc(size * np.dtype(tensor_dtype).itemsize)
                self.bindings.append(int(device_mem))
                
                # 分配主机内存 - 使用扁平化内存
                host_mem = cuda.pagelocked_empty(size, tensor_dtype)
                
                if tensor_mode == trt.TensorIOMode.INPUT:
                    self.inputs.append({
                        'name': tensor_name, 
                        'shape': tensor_shape, 
                        'dtype': tensor_dtype, 
                        'host': host_mem, 
                        'device': device_mem,
                        'size': size
                    })
                else:
                    self.outputs.append({
                        'name': tensor_name, 
                        'shape': tensor_shape, 
                        'dtype': tensor_dtype, 
                        'host': host_mem, 
                        'device': device_mem,
                        'size': size
                    })
            except Exception as e:
                print(f"Error processing tensor {tensor_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print("Detection engine loaded successfully")
        if self.inputs:
            print(f"Input shape: {self.inputs[0]['shape']}")
            print(f"Input data type: {self.inputs[0]['dtype']}")
        print(f"Output count: {len(self.outputs)}")
    
    def preprocess(self, img):
        """Preprocess image (accepts OpenCV image)"""
        if img is None:
            raise ValueError("Cannot process empty image")
        
        self.original_shape = img.shape[:2]  # Save original dimensions
        
        # Convert color space BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 640x640
        img_resized = cv2.resize(img, (640, 640))
        
        # Normalize to [0,1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to CHW format
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_chw, axis=0)
        
        # Ensure continuous memory and convert to correct data type
        img_batch = np.ascontiguousarray(img_batch)
        
        # Convert to model's required data type
        if self.inputs and self.inputs[0]['dtype'] != img_batch.dtype:
            img_batch = img_batch.astype(self.inputs[0]['dtype'])
            
        return img_batch
    
    def infer(self, input_data):
        """Run inference (适用于TensorRT 10.x)"""
        try:
            if not self.inputs or not self.outputs:
                raise RuntimeError("Uninitialized input/output buffers")
                
            # Ensure input data matches the input shape
            expected_shape = self.inputs[0]['shape']
            
            # If shapes don't match, try to reshape
            if input_data.shape != expected_shape:
                print(f"Warning: Input shape mismatch. Expected: {expected_shape}, Got: {input_data.shape}")
                if np.prod(input_data.shape) == np.prod(expected_shape):
                    input_data = input_data.reshape(expected_shape)
                    print(f"Reshaped input data to: {expected_shape}")
                else:
                    raise ValueError(f"Input data shape cannot be adjusted: {input_data.shape} -> {expected_shape}")
            
            # Ensure data type is correct
            expected_dtype = self.inputs[0]['dtype']
            if input_data.dtype != expected_dtype:
                print(f"Input data type mismatch: expected {expected_dtype}, got {input_data.dtype}")
                input_data = input_data.astype(expected_dtype)
            
            # Flatten input data and copy to host memory
            flat_input = input_data.flatten()
            
            # Check if flattened size matches
            if flat_input.size != self.inputs[0]['size']:
                raise ValueError(f"Flattened input size mismatch: got {flat_input.size}, expected {self.inputs[0]['size']}")
            
            # Copy to host memory
            np.copyto(self.inputs[0]['host'], flat_input)
            
            # Copy to device memory
            cuda.memcpy_htod_async(
                self.inputs[0]['device'], 
                self.inputs[0]['host'], 
                self.stream
            )
            
            # Set input and output buffers (TensorRT 10.x API)
            for inp in self.inputs:
                self.context.set_tensor_address(inp['name'], int(inp['device']))
                
            for out in self.outputs:
                self.context.set_tensor_address(out['name'], int(out['device']))
            
            # Execute inference (TensorRT 10.x API)
            status = self.context.execute_async_v3(stream_handle=self.stream.handle)
            
            if not status:
                raise RuntimeError("Inference execution failed")
            
            # Copy output data to host
            for output in self.outputs:
                cuda.memcpy_dtoh_async(
                    output['host'], 
                    output['device'], 
                    self.stream
                )
            
            # Synchronize
            self.stream.synchronize()
            
            # Reshape output
            output_data = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
            return output_data
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def postprocess(self, output, conf_threshold=0.25, iou_threshold=0.45):
        """Post-process - Adapt to YOLOv8 output format"""
        # YOLOv8 output format: [1, 84, 8400] or [1, num_classes+4, num_boxes]
        # Where first 4 are bbox coordinates, followed by class confidences
        
        # Remove batch dimension
        if output.ndim == 3:
            output = output[0]  # [84, 8400] or [num_classes+4, num_boxes]
        
        # Transpose to get [num_boxes, num_classes+4] format
        predictions = output.T
        
        # Extract bounding boxes and scores
        boxes = predictions[:, :4]  # x, y, w, h
        scores = predictions[:, 4:]  # all class scores
        
        # Get highest score and corresponding class for each box
        class_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        # Filter low confidence detections
        mask = class_scores > conf_threshold
        if not mask.any():
            return [], [], []
        
        boxes = boxes[mask]
        class_scores = class_scores[mask]
        class_ids = class_ids[mask]
        
        # Convert bounding box format: YOLO format(cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes_xyxy = self.xywh2xyxy(boxes)
        
        # Scale to original image size
        scale_x = self.original_shape[1] / 640
        scale_y = self.original_shape[0] / 640
        
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y
        
        # NMS
        indices = self.nms(boxes_xyxy, class_scores, iou_threshold)
        
        return boxes_xyxy[indices], class_scores[indices], class_ids[indices]
    
    def xywh2xyxy(self, boxes):
        """Convert bounding box format"""
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return boxes_xyxy
    
    def nms(self, boxes, scores, iou_threshold):
        """Non-maximum suppression"""
        if len(boxes) == 0:
            return []
        
        # Calculate area
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Calculate IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            # Keep boxes with IoU less than threshold
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep

class TensorRTSegmenter:
    """TensorRT segmentation inference class (适用于TensorRT 10.x)"""
    def __init__(self, engine_path, num_classes):
        self.num_classes = num_classes
        
        # Load engine
        print(f"Loading TensorRT segmentation engine from {engine_path}")
        with open(engine_path, 'rb') as f:
            self.runtime = trt.Runtime(TRT_LOGGER)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT segmentation engine")
        
        self.context = self.engine.create_execution_context()
        
        # Get input tensor name and index (TensorRT 10.x API)
        self.input_tensor_names = []
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_tensor_names.append(tensor_name)
                
        if not self.input_tensor_names:
            raise RuntimeError("Engine has no input tensors")
            
        # Use the first input tensor
        self.input_tensor_name = self.input_tensor_names[0]
        
        # Get engine's actual input shape
        engine_input_dims = self.engine.get_tensor_shape(self.input_tensor_name)
        self.engine_input_shape = dims_to_tuple(engine_input_dims)
        
        # To ensure using batch_size=1, create a new input shape
        self.input_shape = (1, *self.engine_input_shape[1:])
        
        # Set execution context's input shape (TensorRT 10.x API)
        self.context.set_input_shape(self.input_tensor_name, self.input_shape)
        
        # Allocate GPU memory
        self.allocate_buffers()
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        print(f"Segmentation engine loaded successfully")
        print(f"Engine input shape: {self.engine_input_shape}")
        print(f"Using input shape: {self.input_shape}")
        print(f"Number of IO tensors: {self.engine.num_io_tensors}")
        
        # Set up preprocessing
        self.transform = Compose([
            A.Resize(self.input_shape[2], self.input_shape[3]),
            A.Normalize(),
        ])
        
    def allocate_buffers(self):
        """Allocate GPU memory buffers (适用于TensorRT 10.x)"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.output_shapes = []
        self.output_tensor_names = []
        
        # 分配内存 (TensorRT 10.x API)
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            
            print(f"Processing tensor {i}: {tensor_name}, mode: {tensor_mode}")
            
            try:
                tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                
                if tensor_mode == trt.TensorIOMode.INPUT:
                    # Use custom batch_size=1 input shape
                    shape = self.input_shape
                    # Set input shape
                    self.context.set_input_shape(tensor_name, shape)
                else:
                    # Get output shape
                    try:
                        output_dims = self.context.get_tensor_shape(tensor_name)
                        shape = dims_to_tuple(output_dims)
                        print(f"Output tensor original shape: {shape}")
                        
                        # If output shape's first dimension (batch_size) doesn't match input, adjust it
                        if len(shape) >= 4 and shape[0] != self.input_shape[0] and shape[0] != -1:
                            shape = (self.input_shape[0], *shape[1:])
                            print(f"Adjusted output shape: {shape}")
                    except Exception as e:
                        print(f"Failed to get output shape: {e}, using default shape")
                        # Use a possible default output shape
                        shape = (1, self.num_classes, self.input_shape[2], self.input_shape[3])
                        print(f"Using default output shape: {shape}")
                    
                    self.output_tensor_names.append(tensor_name)
                
                # Calculate memory size
                size = int(np.prod(shape))
                
                # Allocate device memory
                device_mem = cuda.mem_alloc(size * np.dtype(tensor_dtype).itemsize)
                
                # Allocate host memory - use flat memory
                host_mem = cuda.pagelocked_empty(size, tensor_dtype)
                
                if tensor_mode == trt.TensorIOMode.INPUT:
                    self.inputs.append({
                        'host': host_mem, 
                        'device': device_mem, 
                        'shape': shape,
                        'name': tensor_name,
                        'size': size
                    })
                else:
                    self.outputs.append({
                        'host': host_mem, 
                        'device': device_mem, 
                        'shape': shape,
                        'name': tensor_name,
                        'size': size
                    })
                    self.output_shapes.append(shape)
            except Exception as e:
                print(f"Error processing tensor {tensor_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Memory allocation complete. Inputs: {len(self.inputs)}, Outputs: {len(self.outputs)}")
    
    def preprocess(self, img):
        """Preprocess ROI for segmentation"""
        if img is None:
            raise ValueError("Cannot process empty image")
        
        # Apply same transformations as val_folder_seg
        dummy_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        augmented = self.transform(image=img, mask=dummy_mask)
        img_transformed = augmented['image']
        
        # Convert to float32 and normalize to [0,1]
        img_normalized = img_transformed.astype('float32') / 255
        
        # Convert to CHW format (PyTorch compatible)
        img_chw = img_normalized.transpose(2, 0, 1)
        
        # Add batch dimension
        img_batch = np.expand_dims(img_chw, axis=0)
        
        # Ensure continuous memory
        img_batch = np.ascontiguousarray(img_batch)
        
        return img_batch
    
    def infer(self, input_data):
        """Run segmentation inference (适用于TensorRT 10.x)"""
        try:
            # Check input shape
            if input_data.shape != self.input_shape:
                print(f"Warning: Input shape mismatch: got {input_data.shape}, expected {self.input_shape}")
                # Try to adjust shape
                if np.prod(input_data.shape) == np.prod(self.input_shape):
                    input_data = input_data.reshape(self.input_shape)
                    print(f"Reshaped input to: {self.input_shape}")
                else:
                    raise ValueError(f"Input shape mismatch and cannot be adjusted: {input_data.shape} vs {self.input_shape}")
            
            # Flatten and copy input data to host memory
            flat_input = input_data.flatten()
            
            # Check if flattened size matches
            if flat_input.size != self.inputs[0]['size']:
                raise ValueError(f"Flattened input size mismatch: got {flat_input.size}, expected {self.inputs[0]['size']}")
            
            # Copy to host memory
            np.copyto(self.inputs[0]['host'], flat_input)
            
            # Transfer input data to GPU
            cuda.memcpy_htod_async(
                self.inputs[0]['device'],
                self.inputs[0]['host'],
                self.stream
            )
            
            # Set input and output buffers (TensorRT 10.x API)
            for inp in self.inputs:
                self.context.set_tensor_address(inp['name'], int(inp['device']))
                
            for out in self.outputs:
                self.context.set_tensor_address(out['name'], int(out['device']))
            
            # Run inference (TensorRT 10.x API)
            status = self.context.execute_async_v3(stream_handle=self.stream.handle)
            
            if not status:
                raise RuntimeError("Inference execution failed")
            
            # Transfer output data to host
            for output in self.outputs:
                cuda.memcpy_dtoh_async(
                    output['host'],
                    output['device'],
                    self.stream
                )
            
            # Synchronize
            self.stream.synchronize()
            
            # Get output
            outputs = []
            for i, output in enumerate(self.outputs):
                # Reshape the flattened output to the expected shape
                output_data = output['host'].reshape(output['shape'])
                outputs.append(output_data)
            
            return outputs[0] if len(outputs) == 1 else outputs
            
        except Exception as e:
            print(f"Error during segmentation inference: {e}")
            import traceback
            traceback.print_exc()
            # Return a dummy output as a fallback
            dummy_output = np.zeros((1, self.num_classes, self.input_shape[2], self.input_shape[3]), dtype=np.float32)
            return dummy_output
    
    def postprocess(self, output, roi_size=(64, 64)):
        """Post-process segmentation output"""
        # Apply sigmoid
        output_prob = 1 / (1 + np.exp(-output))  # sigmoid
        
        # Binarize
        output_binary = output_prob.copy()
        output_binary[output_binary >= 0.5] = 1
        output_binary[output_binary < 0.5] = 0
        
        # Resize to ROI size if needed
        segmentation_masks = []
        for c in range(self.num_classes):
            # Resize to specified ROI size
            mask_resized = cv2.resize(
                output_binary[0, c],
                (roi_size[1], roi_size[0])  # (width, height)
            )
            
            segmentation_masks.append(mask_resized)
        
        return segmentation_masks, output_prob[0]  # Return both binary masks and probability maps

def get_track_colors(num_tracks):
    """Generate colors for tracked objects"""
    colors = []
    for i in range(num_tracks):
        # Use HSV color space to generate evenly distributed colors
        h = i / num_tracks
        s = 0.8
        v = 0.8
        rgb = colorsys.hsv_to_rgb(h, s, v)
        rgb = tuple(int(x * 255) for x in rgb)
        colors.append(rgb)
    return colors

def get_segmentation_colors(num_classes):
    """Generate colors for segmentation classes"""
    seg_colors = []
    for i in range(num_classes):
        # Use different hue values for good separation
        h = (i * 0.618033988749895) % 1.0  # Golden ratio provides good distribution
        s = 0.7
        v = 0.95
        rgb = colorsys.hsv_to_rgb(h, s, v)
        rgb = tuple(int(x * 255) for x in rgb)
        seg_colors.append(rgb)
    return seg_colors

def extract_roi(frame, center_x, center_y, roi_size=64):
    """Extract ROI around center point with padding if needed"""
    h, w = frame.shape[:2]
    
    # Calculate ROI bounds
    half_size = roi_size // 2
    x1 = int(max(0, center_x - half_size))
    y1 = int(max(0, center_y - half_size))
    x2 = int(min(w, center_x + half_size))
    y2 = int(min(h, center_y + half_size))
    
    # Extract ROI
    roi = frame[y1:y2, x1:x2]
    
    # Handle boundary cases with padding
    if roi.shape[0] != roi_size or roi.shape[1] != roi_size:
        # Create a black canvas of required size
        padded_roi = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
        
        # Place the actual ROI on the canvas
        roi_h, roi_w = roi.shape[:2]
        padded_roi[:roi_h, :roi_w, :] = roi
        roi = padded_roi
    
    return roi, (x1, y1, x2, y2)

def overlay_segmentation(frame, masks, roi_box, seg_colors, alpha=0.5):
    """Overlay segmentation results on the frame without adding shadows to non-segmented areas"""
    x1, y1, x2, y2 = roi_box
    roi_h, roi_w = y2 - y1, x2 - x1
    
    # Create a copy of the frame to avoid modifying the original
    overlay = frame.copy()
    
    # Get the ROI from the original frame
    roi = overlay[y1:y2, x1:x2]
    
    # For each mask, apply color only where the mask is active
    for i, mask in enumerate(masks):
        # Resize mask to match actual ROI size if needed
        if mask.shape[0] != roi_h or mask.shape[1] != roi_w:
            mask = cv2.resize(mask, (roi_w, roi_h))
        
        # Create binary mask (ensure it's boolean for indexing)
        binary_mask = mask > 0.5
        
        # Only modify pixels where the mask is active
        if np.any(binary_mask):
            # Apply the segmentation color with alpha blending only to masked areas
            color_array = np.array(seg_colors[i], dtype=np.uint8)
            for c in range(3):  # RGB channels
                roi[:, :, c] = np.where(
                    binary_mask,
                    roi[:, :, c] * (1 - alpha) + color_array[c] * alpha,
                    roi[:, :, c]
                )
    
    return overlay

def draw_gating_ellipse(img, track):
    """Draw the gating ellipse for a track"""
    try:
        # Get gating ellipse parameters
        ellipse_params = track.gatingEllipse
        
        if ellipse_params is None:
            return img
        
        # Extract ellipse parameters: ((center_x, center_y), (width, height), angle)
        center, axes, angle = ellipse_params
        center_x, center_y = center
        width, height = axes
        
        # Ensure the ellipse is within image bounds
        img_h, img_w = img.shape[:2]
        if (center_x < 0 or center_x >= img_w or 
            center_y < 0 or center_y >= img_h):
            return img
        
        # Convert to integers
        center_int = (int(center_x), int(center_y))
        axes_int = (int(width/2), int(height/2))
        angle_int = int(angle)
        
        # Generate a unique color for each track (using track ID)
        colors = [(255, 255, 0),   # Yellow
                  (255, 0, 255),   # Magenta  
                  (0, 255, 255),   # Cyan
                  (255, 128, 0),   # Orange
                  (128, 255, 0),   # Lime
                  (255, 0, 128),   # Pink
                  (0, 128, 255),   # Light Blue
                  (128, 0, 255)]   # Purple
        
        ellipse_color = colors[track.id % len(colors)]
        
        # Draw ellipse outline only (thickness=2, no fill)
        cv2.ellipse(img, center_int, axes_int, angle_int, 0, 360, ellipse_color, 2)
        
        # Optional: Draw a small circle at the predicted center
        cv2.circle(img, center_int, 2, ellipse_color, -1)
        
    except Exception as e:
        print(f"Error drawing gating ellipse for track {track.id}: {e}")
    
    return img

def draw_boxes_tracks_and_segments(img, boxes, scores, class_ids, class_names, 
                                   segmentation_results, tracker=None, show_gating_ellipses=True):
    """Draw detection boxes, tracking trajectories, segmentation overlays, and gating ellipses"""
    # Copy image to avoid modifying the original
    result_img = img.copy()
    
    # Generate colors
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                   (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)]
    
    # First draw gating ellipses (if enabled and tracker is available)
    if show_gating_ellipses and tracker is not None:
        # Draw ellipses for all tracks (including those being predicted)
        for track in tracker.tracks:
            if track.age >= 1:  # Show ellipses for tracks that have been initialized
                result_img = draw_gating_ellipse(result_img, track)
    
    # Then draw tracking trajectories
    if tracker is not None:
        active_tracks = tracker.get_active_tracks()
        track_colors = get_track_colors(100)  # Pre-generate 100 colors
        
        for track in active_tracks:
            # Only draw targets with trajectories long enough
            if len(track.trajectory) >= 2:
                color = track_colors[track.id % len(track_colors)]
                
                # Draw trajectory line
                for i in range(1, len(track.trajectory)):
                    pt1 = (int(track.trajectory[i-1][0]), int(track.trajectory[i-1][1]))
                    pt2 = (int(track.trajectory[i][0]), int(track.trajectory[i][1]))
                    cv2.line(result_img, pt1, pt2, color, 2)
                
                # Draw trajectory ID
                last_pt = track.trajectory[-1]
                cv2.putText(result_img, f"ID:{track.id}", 
                           (int(last_pt[0]), int(last_pt[1]) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Then draw detection boxes
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(result_img.shape[1], x2)
        y2 = min(result_img.shape[0], y2)
        
        color = base_colors[int(class_id) % len(base_colors)]
        
        # Draw bounding box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = f"{class_names[int(class_id)]}: {score:.2f}"
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw text background
        cv2.rectangle(result_img, (x1, y1 - text_height - baseline), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(result_img, label, (x1, y1 - baseline), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Apply segmentation overlays (if available)
    if segmentation_results:
        for track_id, (masks, roi_box) in segmentation_results.items():
            # Get segmentation colors
            seg_colors = get_segmentation_colors(len(masks))
            
            # Overlay segmentation on the image
            result_img = overlay_segmentation(result_img, masks, roi_box, seg_colors)
    
    return result_img

def process_video(det_engine, seg_engine, video_path, output_path, det_class_names, 
                  conf_threshold=0.25, iou_threshold=0.45, show_preview=False,
                  roi_size=64, show_gating_ellipses=True):
    """Process video with detection, tracking and segmentation"""
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames total")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize tracker
    tracker = JPDAFilter(process_noise=20.0, measure_noise=2.0, detect_prob=0.7, gate_prob=0.95)
    
    # Performance statistics
    detection_times = []
    segmentation_times = []
    tracking_times = []
    total_times = []
    frame_count = 0
    total_detections = 0
    total_segmentations = 0
    
    # Process video frames
    print("Starting video processing...")
    start_time = time.time()
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Show progress
        if frame_count % 10 == 0 or frame_count == 1:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        # Start timer
        frame_start_time = time.time()
        
        # 1. Detection
        det_start_time = time.time()
        
        # Preprocess
        input_data = det_engine.preprocess(frame)
        
        # Inference
        output = det_engine.infer(input_data)
        
        # Postprocess
        boxes, scores, class_ids = det_engine.postprocess(output, conf_threshold, iou_threshold)
        
        det_time = (time.time() - det_start_time) * 1000  # to milliseconds
        detection_times.append(det_time)
        total_detections += len(boxes)
        
        # 2. Tracking
        track_start_time = time.time()
        
        # Convert bounding boxes to tracking points (use center points)
        detection_points = []
        for box in boxes:
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            detection_points.append((x_center, y_center))
        
        # Update tracker
        tracker.predict()
        tracker.correct(detection_points)
        
        track_time = (time.time() - track_start_time) * 1000  # to milliseconds
        tracking_times.append(track_time)
        
        # 3. Segmentation on ROIs
        seg_start_time = time.time()
        
        # Get active tracks
        active_tracks = tracker.get_active_tracks()
        
        # Store segmentation results
        segmentation_results = {}  # {track_id: (masks, roi_box)}
        
        # Process each tracked object
        for track in active_tracks:
            if len(track.trajectory) > 0:
                # Get latest position
                center_x, center_y = track.trajectory[-1]
                
                # Extract ROI
                roi, roi_box = extract_roi(frame, center_x, center_y, roi_size=roi_size)
                
                # Only process ROI if it's valid
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    try:
                        # Preprocess ROI
                        roi_tensor = seg_engine.preprocess(roi)
                        
                        # Run segmentation
                        seg_output = seg_engine.infer(roi_tensor)
                        
                        # Postprocess
                        masks, probs = seg_engine.postprocess(seg_output, roi_size=(roi_size, roi_size))
                        
                        # Store results
                        segmentation_results[track.id] = (masks, roi_box)
                        total_segmentations += 1
                    except Exception as e:
                        print(f"Error processing segmentation for track {track.id}: {e}")
        
        seg_time = (time.time() - seg_start_time) * 1000  # to milliseconds
        segmentation_times.append(seg_time)
        
        # 4. Visualize results
        result_frame = draw_boxes_tracks_and_segments(
            frame, boxes, scores, class_ids, det_class_names, 
            segmentation_results, tracker, show_gating_ellipses
        )
        
        # Add frame processing info
        frame_time = (time.time() - frame_start_time) * 1000  # to milliseconds
        total_times.append(frame_time)
        
        info_text = (f"Frame: {frame_count}/{total_frames} | "
                    f"Det: {det_time:.1f}ms | "
                    f"Track: {track_time:.1f}ms | "
                    f"Seg: {seg_time:.1f}ms | "
                    f"Total: {frame_time:.1f}ms")
        
        cv2.putText(result_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Write to output video
        out.write(result_frame)
        
        # Show preview (optional)
        if show_preview:
            # Resize preview window if original frame is too large
            preview_frame = result_frame
            if width > 1280 or height > 720:
                scale = min(1280 / width, 720 / height)
                preview_frame = cv2.resize(result_frame, (int(width * scale), int(height * scale)))
            
            cv2.imshow('ROI Detection, Tracking & Segmentation', preview_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Calculate total processing time
    total_time = time.time() - start_time
    
    # Release resources
    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    # Calculate statistics
    if detection_times:
        avg_detection_time = np.mean(detection_times)
        avg_tracking_time = np.mean(tracking_times)
        avg_segmentation_time = np.mean(segmentation_times)
        avg_total_time = np.mean(total_times)
        fps_achieved = 1000 / avg_total_time
        
        # Generate report
        report = f"""
==================================================
TensorRT ROI Video Processing Performance Report:
==================================================
Video file: {video_path}
Output file: {output_path}
Total frames: {frame_count}
Total detections: {total_detections}
Total segmentations: {total_segmentations}
Average detections per frame: {total_detections/frame_count:.2f}
Average segmentations per frame: {total_segmentations/frame_count:.2f}

Time statistics:
- Total processing time: {total_time:.2f} seconds
- Average detection time: {avg_detection_time:.2f} ms/frame
- Average tracking time: {avg_tracking_time:.2f} ms/frame
- Average segmentation time: {avg_segmentation_time:.2f} ms/frame
- Average total processing time: {avg_total_time:.2f} ms/frame
- Achieved processing rate: {fps_achieved:.2f} FPS

Original video info:
- Resolution: {width}x{height}
- Frame rate: {fps} FPS

Features enabled:
- Gating ellipses visualization: {show_gating_ellipses}
"""
        
        print(report)
        
        # Save report
        report_path = Path(output_path).with_suffix('.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Video processing complete! Results saved to: {output_path}")
        print(f"Performance report saved to: {report_path}")
        
        return {
            'frames_processed': frame_count,
            'total_detections': total_detections,
            'total_segmentations': total_segmentations,
            'avg_detection_time': avg_detection_time,
            'avg_tracking_time': avg_tracking_time,
            'avg_segmentation_time': avg_segmentation_time,
            'avg_total_time': avg_total_time,
            'achieved_fps': fps_achieved
        }
    
    return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLOv8 TensorRT ROI Video Processing with Segmentation and Gating Ellipses')
    parser.add_argument('--video', type=str, required=True, help='Input video file path')
    parser.add_argument('--output', type=str, help='Output video file path (default adds _processed suffix)')
    parser.add_argument('--det_engine', type=str, default="yolo_weights/best.engine", help='Detection TensorRT engine file path')
    parser.add_argument('--seg_model', type=str, required=True, help='Segmentation model name folder')
    parser.add_argument('--seg_engine', type=str, default="model_fp32.engine", help='Segmentation engine file name')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--roi_size', type=int, default=64, help='ROI size for segmentation')
    parser.add_argument('--preview', action='store_true', help='Show real-time processing preview window')
    parser.add_argument('--no_gating_ellipses', action='store_true', help='Disable gating ellipses visualization')
    args = parser.parse_args()
    
    # Set default output path (if not specified)
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.with_stem(f"{video_path.stem}_processed").with_suffix('.mp4'))
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detection class names (adjust based on your dataset)
    det_class_names = ['sperm']  # Add more classes if needed
    
    # Load segmentation configuration
    seg_config_path = os.path.join('seg_weights', args.seg_model, 'config.yml')
    with open(seg_config_path, 'r') as f:
        seg_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Build segmentation engine path
    seg_engine_path = os.path.join('seg_weights', args.seg_model, args.seg_engine)
    
    # Initialize detection engine
    det_engine = TensorRTDetector(args.det_engine)
    
    # Initialize segmentation engine
    seg_engine = TensorRTSegmenter(seg_engine_path, seg_config['num_classes'])
    
    # Warmup both engines
    print("Performing warmup runs...")
    try:
        # Warmup detection engine
        if det_engine.inputs:
            dummy_shape = det_engine.inputs[0]['shape']
            dummy_dtype = det_engine.inputs[0]['dtype']
            print(f"Creating warmup input for detection, shape: {dummy_shape}, type: {dummy_dtype}")
            dummy_input_det = np.random.randn(*dummy_shape).astype(dummy_dtype)
            for _ in range(5):
                det_engine.infer(dummy_input_det)
        
        # Warmup segmentation engine
        dummy_input_seg = np.random.randn(*seg_engine.input_shape).astype(np.float32)
        for _ in range(5):
            seg_engine.infer(dummy_input_seg)
        
        print("Warmup complete!\n")
    except Exception as e:
        print(f"Error during warmup: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing execution, but performance may be affected")
    
    # Process video
    process_video(
        det_engine, seg_engine, args.video, args.output, det_class_names,
        conf_threshold=args.conf, iou_threshold=args.iou, 
        show_preview=args.preview, roi_size=args.roi_size,
        show_gating_ellipses=not args.no_gating_ellipses
    )

if __name__ == "__main__":
    main()