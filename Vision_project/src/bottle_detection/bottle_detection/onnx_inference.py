import numpy as np
import onnxruntime as ort
import cv2
import time
from typing import List, Tuple, Dict, Any
import rclpy

class ONNXBottleDetector:
    def __init__(self, model_path: str = "yolov8m.onnx", confidence_threshold: float = 0.5):
        """
        Initialize ONNX-based bottle detector
        
        Args:
            model_path: Path to ONNX model file
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.bottle_class_id = 39  # COCO class ID for bottles
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = (640, 640)  # Default to 640x640 if not determined
        
        # Initialize ONNX Runtime session
        self._initialize_session(model_path)
        
    def _initialize_session(self, model_path: str):
        """Initialize ONNX Runtime session with the model"""
        try:
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider']  # Use 'CUDAExecutionProvider' for GPU
            )
            
            # Get input details
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            
            # Handle dynamic axes - if shape contains strings, use default values
            if len(input_shape) >= 4:
                # Check if dimensions are strings (dynamic axes)
                height = input_shape[2]
                width = input_shape[3]
                
                # If they're strings, use default size
                if isinstance(height, str) or height is None:
                    height = 640
                if isinstance(width, str) or width is None:
                    width = 640
                    
                # Convert to integers if they're floats
                self.input_shape = (int(height), int(width))
            else:
                # Fallback to default size
                self.input_shape = (640, 640)
            
            # Get output details
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            rclpy.logging.get_logger('onnx_inference').info(
                f"ONNX model loaded successfully from {model_path}")
            rclpy.logging.get_logger('onnx_inference').info(
                f"Input shape: {self.input_shape}")
            rclpy.logging.get_logger('onnx_inference').info(
                f"Output names: {self.output_names}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {str(e)}")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """
        Preprocess image for ONNX model
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            preprocessed: Normalized and resized image
            ratio: Scaling ratio (width_ratio, height_ratio)
            pad: Padding added (width, height)
        """
        # Get original dimensions
        h, w = image.shape[:2]
        
        # Ensure we have numeric values for input dimensions
        input_h, input_w = self.input_shape
        
        # Calculate scaling ratio using numeric values
        r = min(input_h / h, input_w / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = input_w - new_unpad[0], input_h - new_unpad[1]
        
        # Scale and pad image
        if (w, h) != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # HWC to CHW format
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Return preprocessed image, ratio, and pad
        return image, (r, r), (dw // 2, dh // 2)
    
    def postprocess(
        self, 
        outputs: List[np.ndarray], 
        ratio: Tuple[float, float], 
        pad: Tuple[int, int],
        original_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Postprocess ONNX model outputs to get detections
        
        Args:
            outputs: Model outputs from ONNX Runtime
            ratio: Scaling ratio used in preprocessing
            pad: Padding added during preprocessing
            original_shape: Original image shape (height, width)
            
        Returns:
            List of detection dictionaries with keys:
                - 'class_id': Class ID
                - 'confidence': Detection confidence
                - 'bbox': Bounding box [x1, y1, x2, y2] in original image coordinates
        """
        # Get the actual output format
        output = outputs[0]
        
        # YOLOv8 ONNX output format can vary - handle different cases
        if output.shape[1] == 84:  # [batch, 84, boxes] format
            # Convert to [batch, boxes, 84]
            output = np.transpose(output, (0, 2, 1))
        elif len(output.shape) == 2:  # [boxes, 84] format
            output = np.expand_dims(output, axis=0)  # Add batch dimension
        # Otherwise assume [batch, boxes, 84] format
        
        # Extract boxes and class scores
        boxes = output[:, :, :4]  # x, y, w, h
        scores = output[:, :, 4:]  # Class scores
        
        # Process each detection
        detections = []
        for i in range(boxes.shape[1]):
            # Get max confidence and corresponding class
            max_conf = np.max(scores[0, i])
            class_id = np.argmax(scores[0, i])
            
            # Only consider bottles with sufficient confidence
            if class_id == self.bottle_class_id and max_conf >= self.confidence_threshold:
                # Convert box format from [x, y, w, h] to [x1, y1, x2, y2]
                x, y, w, h = boxes[0, i]
                
                # Convert to original image coordinates
                # Ensure all values are numeric before division
                x1 = float((x - w/2 - pad[0])) / float(ratio[0])
                y1 = float((y - h/2 - pad[1])) / float(ratio[1])
                x2 = float((x + w/2 - pad[0])) / float(ratio[0])
                y2 = float((y + h/2 - pad[1])) / float(ratio[1])
                
                # Clip to image boundaries
                x1 = max(0, min(original_shape[1], x1))
                y1 = max(0, min(original_shape[0], y1))
                x2 = max(0, min(original_shape[1], x2))
                y2 = max(0, min(original_shape[0], y2))
                
                detections.append({
                    'class_id': int(class_id),
                    'confidence': float(max_conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detection dictionaries
        """
        # Record start time
        start_time = time.time()
        
        # Get original shape
        original_shape = (image.shape[0], image.shape[1])
        
        # Preprocess image
        input_tensor, ratio, pad = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess results
        detections = self.postprocess(outputs, ratio, pad, original_shape)
        
        # Record inference time
        inference_time = time.time() - start_time
        
        return detections, inference_time