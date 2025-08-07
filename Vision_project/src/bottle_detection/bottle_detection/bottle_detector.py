# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
# import os
# import time
# from .database import DetectionDatabase  # Import our database module

# class BottleDetector(Node):
#     def __init__(self):
#         super().__init__('bottle_detector')
        
#         # Initialize CvBridge for image conversion
#         self.bridge = CvBridge()
        
#         # Setup directories
#         self.base_dir = os.path.expanduser('~/vision_project_ws/src/bottle_detection/bottle_detection')
#         self.image_dir = os.path.join(self.base_dir, 'data/images')
#         os.makedirs(self.image_dir, exist_ok=True)
        
#         # Initialize database
#         self.db = DetectionDatabase(os.path.join(self.base_dir, 'detections.db'))
#         self.get_logger().info(f'Database initialized at {self.db.db_path}')
        
#         # Load YOLOv8 model (using large version for accuracy)
#         try:
#             self.model = YOLO('yolov8m.pt')
#             self.get_logger().info('YOLOv8 model loaded successfully')
#         except Exception as e:
#             self.get_logger().error(f'Failed to load YOLO model: {str(e)}')
#             raise
        
#         # Bottle class ID in COCO dataset is 39
#         self.bottle_class_id = 39
        
#         # Detection parameters
#         self.confidence_threshold = 0.5  # Only process detections above this confidence
#         self.min_detections_interval = 1.0  # Minimum time between detection logs (seconds)
#         self.last_detection_time = 0
        
#         # Create subscription to camera image with appropriate QoS
#         qos_profile = QoSProfile(
#             history=QoSHistoryPolicy.KEEP_LAST,
#             depth=5,
#             reliability=QoSReliabilityPolicy.BEST_EFFORT
#         )
#         self.image_sub = self.create_subscription(
#             Image,
#             '/camera/image_raw',
#             self.image_callback,
#             qos_profile
#         )
        
#         # Create publisher for annotated image
#         self.annotated_pub = self.create_publisher(Image, '/detection/bottle_detection', 10)
        
#         # Create publisher for debug image (with confidence scores)
#         self.debug_pub = self.create_publisher(Image, '/detection/bottle_debug', 10)
        
#         # Create timer for periodic database status checks
#         self.db_timer = self.create_timer(30.0, self.check_db_status)
        
#         self.get_logger().info('Bottle detection node initialized and ready')
#         self.get_logger().info('Subscribed to: /camera/image_raw')
#         self.get_logger().info('Publishing to: /detection/bottle_detection')
#         self.get_logger().info('Publishing debug to: /detection/bottle_debug')
#         self.get_logger().info(f'Database file: {self.db.db_path}')

#     def image_callback(self, msg):
#         try:
#             # Rate limit database logging
#             current_time = time.time()
#             if current_time - self.last_detection_time < self.min_detections_interval:
#                 return
                
#             # Convert ROS Image message to OpenCV image
#             cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
#             # Run YOLO inference
#             results = self.model(cv_image, conf=self.confidence_threshold, stream=True)
            
#             # Create copies for different visualizations
#             display_img = cv_image.copy()
#             debug_img = cv_image.copy()
            
#             bottle_count = 0
#             bottle_confidences = []
#             bottle_boxes = []
            
#             # Process each detection result
#             for result in results:
#                 boxes = result.boxes.cpu().numpy()
                
#                 for box in boxes:
#                     # Check if this is a bottle (COCO class ID 39)
#                     if int(box.cls[0]) == self.bottle_class_id:
#                         bottle_count += 1
                        
#                         # Get box coordinates and confidence
#                         x1, y1, x2, y2 = box.xyxy[0]
#                         conf = float(box.conf[0])
                        
#                         # Store for database
#                         bottle_confidences.append(conf)
#                         bottle_boxes.append((x1, y1, x2, y2))
                        
#                         # Draw bounding box (green for bottles)
#                         cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
#                         # Add label with confidence
#                         label = f'Bottle: {conf:.2f}'
#                         cv2.putText(display_img, label, (int(x1), int(y1)-10), 
#                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
#                         # For debug image, add more details
#                         cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#                         details = f'Bottle ID:{self.bottle_class_id} Conf:{conf:.2f}'
#                         cv2.putText(debug_img, details, (int(x1), int(y1)-10), 
#                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
#             # Only log if we detected bottles or if it's been a while
#             if bottle_count > 0 or (current_time - self.last_detection_time > 5.0):
#                 self.last_detection_time = current_time
                
#                 # Save image for reference
#                 timestamp = int(current_time)
#                 image_filename = f"bottle_detection_{timestamp}.jpg"
#                 image_path = os.path.join(self.image_dir, image_filename)
#                 cv2.imwrite(image_path, cv_image)
                
#                 # Calculate confidence statistics
#                 min_conf = min(bottle_confidences) if bottle_confidences else 0.0
#                 max_conf = max(bottle_confidences) if bottle_confidences else 0.0
#                 avg_conf = sum(bottle_confidences)/len(bottle_confidences) if bottle_confidences else 0.0
                
#                 # Log to database
#                 detection_id = self.db.log_detection(
#                     bottle_count=bottle_count,
#                     image_path=image_filename,
#                     min_conf=min_conf,
#                     max_conf=max_conf,
#                     avg_conf=avg_conf
#                 )
                
#                 # Log individual bottle details
#                 for i, (conf, box) in enumerate(zip(bottle_confidences, bottle_boxes)):
#                     x1, y1, x2, y2 = box
#                     self.db.log_bottle_details(
#                         detection_id=detection_id,
#                         confidence=conf,
#                         x1=float(x1),
#                         y1=float(y1),
#                         x2=float(x2),
#                         y2=float(y2)
#                     )
                
#                 # Log results
#                 if bottle_count > 0:
#                     self.get_logger().info(
#                         f'Detected {bottle_count} bottle(s). '
#                         f'Conf: min={min_conf:.2f}, max={max_conf:.2f}, avg={avg_conf:.2f}. '
#                         f'Saved to DB (ID: {detection_id})'
#                     )
#                 else:
#                     self.get_logger().info('No bottles detected, but logged status to DB')
            
#             # Convert back to ROS Image messages
#             annotated_msg = self.bridge.cv2_to_imgmsg(display_img, "bgr8")
#             debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            
#             # Preserve the original header
#             annotated_msg.header = msg.header
#             debug_msg.header = msg.header
            
#             # Publish annotated images
#             self.annotated_pub.publish(annotated_msg)
#             self.debug_pub.publish(debug_msg)
                
#         except Exception as e:
#             self.get_logger().error(f'Error processing image: {str(e)}')

#     def check_db_status(self):
#         """Periodically check database status and log recent entries"""
#         try:
#             recent = self.db.get_recent_detections(3)
#             if recent:
#                 self.get_logger().info(f'Recent detections (last 3):')
#                 for record in recent:
#                     self.get_logger().info(
#                         f'ID: {record[0]}, Time: {record[1]}, Bottles: {record[2]}, '
#                         f'Conf: min={record[4]:.2f}, max={record[5]:.2f}'
#                     )
#         except Exception as e:
#             self.get_logger().error(f'Database status check failed: {str(e)}')

#     def destroy_node(self):
#         """Clean up when node is destroyed"""
#         self.db.close()
#         super().destroy_node()

# def main(args=None):
#     rclpy.init(args=args)
#     bottle_detector = BottleDetector()
    
#     try:
#         rclpy.spin(bottle_detector)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         bottle_detector.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import time
import os
from .database import DetectionDatabase

class BottleDetector(Node):
    def __init__(self):
        super().__init__('bottle_detector')
        
        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()
        
        # Setup directories
        self.base_dir = os.path.expanduser('~/Desktop/Tasks/Vision_project/src/bottle_detection/bottle_detection')
        self.image_dir = os.path.join(self.base_dir, 'data/images')
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize database
        self.db = DetectionDatabase(os.path.join(self.base_dir, 'detections.db'))
        self.get_logger().info(f'Database initialized at {self.db.db_path}')
        
        # Path to ONNX model
        self.onnx_model_path = os.path.join(self.base_dir, 'yolov8n.onnx')
        
        # Check if model exists, if not try common locations
        if not os.path.exists(self.onnx_model_path):
            # Try workspace root
            self.onnx_model_path = os.path.expanduser('~/Desktop/Tasks/Vision_project/yolov8n.onnx')
            if not os.path.exists(self.onnx_model_path):
                self.get_logger().fatal('ONNX model not found. Please run the export command first.')
                raise FileNotFoundError('ONNX model not found')
        
        # Initialize ONNX Runtime session
        self.get_logger().info(f'Loading ONNX model from: {self.onnx_model_path}')
        self.session = ort.InferenceSession(
            self.onnx_model_path,
            providers=['CPUExecutionProvider']  # Use 'CUDAExecutionProvider' if you have GPU
        )
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape
        self.get_logger().info(f'Model input shape: {self.input_shape}')
        
        # Bottle class ID in COCO dataset is 39
        self.bottle_class_id = 39
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.min_detections_interval = 1.0
        self.last_detection_time = 0
        
        # Create subscription to camera image
        from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile
        )
        
        # Create publishers
        self.annotated_pub = self.create_publisher(Image, '/detection/bottle_detection', 10)
        self.debug_pub = self.create_publisher(Image, '/detection/bottle_debug', 10)
        
        # Create timer for periodic database status checks
        self.db_timer = self.create_timer(30.0, self.check_db_status)
        
        self.get_logger().info('Bottle detection node initialized with ONNX backend')
        self.get_logger().info(f'Model: {os.path.basename(self.onnx_model_path)}')
        self.get_logger().info('Subscribed to: /camera/image_raw')

    def preprocess(self, image):
        """Preprocess image for ONNX model"""
        # Resize while maintaining aspect ratio
        h, w = image.shape[:2]
        input_h, input_w = self.input_shape[2], self.input_shape[3]
        
        # Calculate ratio and new dimensions
        r = min(input_h / h, input_w / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = input_w - new_unpad[0], input_h - new_unpad[1]
        
        # Resize
        if (h, w) != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add padding
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # HWC to CHW and normalize
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image, (h, w), (top, left, r)

    def postprocess(self, output, orig_shape, pad_info):
        """Process ONNX model output to get detections"""
        # Extract dimensions
        h, w = orig_shape
        top, left, r = pad_info
        
        # Process output (YOLOv8 output format)
        # Output shape: [batch, num_boxes, 84] where 84 = 4 (bbox) + 80 (classes)
        predictions = output[0]  # First output tensor
        
        # Filter by confidence
        scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Get bounding boxes
        boxes = predictions[:, :4]
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        x_c = boxes[:, 0] / self.input_shape[3]
        y_c = boxes[:, 1] / self.input_shape[2]
        w_box = boxes[:, 2] / self.input_shape[3]
        h_box = boxes[:, 3] / self.input_shape[2]
        
        x1 = (x_c - w_box/2) * self.input_shape[3]
        y1 = (y_c - h_box/2) * self.input_shape[2]
        x2 = (x_c + w_box/2) * self.input_shape[3]
        y2 = (y_c + h_box/2) * self.input_shape[2]
        
        # Adjust for padding and resize
        x1 = (x1 - left) / r
        y1 = (y1 - top) / r
        x2 = (x2 - left) / r
        y2 = (y2 - top) / r
        
        # Clip to image boundaries
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        x2 = np.clip(x2, 0, h)
        
        # Filter by confidence threshold
        mask = confidences > self.confidence_threshold
        boxes = np.column_stack((x1, y1, x2, y2))[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Apply NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            confidences.tolist(), 
            self.confidence_threshold, 
            0.45
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = boxes[indices]
            confidences = confidences[indices]
            class_ids = class_ids[indices]
        
        return boxes, confidences, class_ids

    def image_callback(self, msg):
        try:
            # Rate limit database logging
            current_time = time.time()
            if current_time - self.last_detection_time < self.min_detections_interval:
                return
                
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            orig_h, orig_w = cv_image.shape[:2]
            
            # Preprocess image
            input_tensor, orig_shape, pad_info = self.preprocess(cv_image)
            
            # Run inference
            start_time = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            inference_time = time.time() - start_time
            
            # Postprocess results
            boxes, confidences, class_ids = self.postprocess(outputs, orig_shape, pad_info)
            
            # Create copies for visualization
            display_img = cv_image.copy()
            debug_img = cv_image.copy()
            
            bottle_count = 0
            bottle_confidences = []
            bottle_boxes = []
            
            # Process detections
            for i in range(len(boxes)):
                if class_ids[i] == self.bottle_class_id:
                    bottle_count += 1
                    
                    x1, y1, x2, y2 = boxes[i]
                    conf = confidences[i]
                    
                    # Store for database
                    bottle_confidences.append(conf)
                    bottle_boxes.append((x1, y1, x2, y2))
                    
                    # Draw bounding box
                    cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f'Bottle: {conf:.2f}'
                    cv2.putText(display_img, label, (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # For debug image
                    cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    details = f'Bottle ID:{self.bottle_class_id} Conf:{conf:.2f} ({inference_time*1000:.1f}ms)'
                    cv2.putText(debug_img, details, (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Log to database if needed
            if bottle_count > 0 or (current_time - self.last_detection_time > 5.0):
                self.last_detection_time = current_time
                
                # Save image for reference
                timestamp = int(current_time)
                image_filename = f"bottle_detection_{timestamp}.jpg"
                image_path = os.path.join(self.image_dir, image_filename)
                cv2.imwrite(image_path, cv_image)
                
                # Calculate confidence statistics
                min_conf = min(bottle_confidences) if bottle_confidences else 0.0
                max_conf = max(bottle_confidences) if bottle_confidences else 0.0
                avg_conf = sum(bottle_confidences)/len(bottle_confidences) if bottle_confidences else 0.0
                
                # Log to database
                detection_id = self.db.log_detection(
                    bottle_count=bottle_count,
                    image_path=image_filename,
                    min_conf=min_conf,
                    max_conf=max_conf,
                    avg_conf=avg_conf
                )
                
                # Log individual bottle details
                for i, (conf, box) in enumerate(zip(bottle_confidences, bottle_boxes)):
                    x1, y1, x2, y2 = box
                    self.db.log_bottle_details(
                        detection_id=detection_id,
                        confidence=conf,
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2)
                    )
                
                # Log results
                if bottle_count > 0:
                    self.get_logger().info(
                        f'Detected {bottle_count} bottle(s) in {inference_time*1000:.1f}ms. '
                        f'Conf: min={min_conf:.2f}, max={max_conf:.2f}, avg={avg_conf:.2f}. '
                        f'Saved to DB (ID: {detection_id})'
                    )
                else:
                    self.get_logger().info('No bottles detected, but logged status to DB')
            
            # Convert back to ROS Image messages
            annotated_msg = self.bridge.cv2_to_imgmsg(display_img, "bgr8")
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            
            # Preserve the original header
            annotated_msg.header = msg.header
            debug_msg.header = msg.header
            
            # Publish annotated images
            self.annotated_pub.publish(annotated_msg)
            self.debug_pub.publish(debug_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def check_db_status(self):
        """Periodically check database status"""
        try:
            recent = self.db.get_recent_detections(3)
            if recent:
                self.get_logger().info(f'Recent detections (last 3):')
                for record in recent:
                    self.get_logger().info(
                        f'ID: {record[0]}, Time: {record[1]}, Bottles: {record[2]}, '
                        f'Conf: min={record[4]:.2f}, max={record[5]:.2f}'
                    )
        except Exception as e:
            self.get_logger().error(f'Database status check failed: {str(e)}')

    def destroy_node(self):
        """Clean up when node is destroyed"""
        self.db.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    bottle_detector = BottleDetector()
    
    try:
        rclpy.spin(bottle_detector)
    except KeyboardInterrupt:
        pass
    finally:
        bottle_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()