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
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
import os
import time
import sys
from .database import DetectionDatabase
from .onnx_inference import ONNXBottleDetector

class BottleDetector(Node):
    def __init__(self):
        super().__init__('bottle_detector')
        
        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()
        
        # Find the package directory reliably
        package_dir = self.get_package_directory()
        self.get_logger().info(f'Package directory: {package_dir}')
        
        # Setup directories
        self.base_dir = package_dir
        self.image_dir = os.path.join(self.base_dir, 'data/images')
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize database
        self.db = DetectionDatabase(os.path.join(self.base_dir, 'detections.db'))
        self.get_logger().info(f'Database initialized at {self.db.db_path}')
        
        # Try to find the ONNX model - check multiple locations
        model_candidates = [
            os.path.join(self.base_dir, 'yolov8n.onnx'),
            os.path.join(self.base_dir, 'yolov8m.onnx'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8n.onnx'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8m.onnx')
        ]
        
        model_path = None
        for candidate in model_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break
                
        if not model_path:
            self.get_logger().fatal(f'ONNX model not found in any expected location. Checked: {model_candidates}')
            raise FileNotFoundError('ONNX model not found')
        
        self.get_logger().info(f'Using ONNX model: {model_path}')
        
        # Initialize ONNX detector
        try:
            self.detector = ONNXBottleDetector(
                model_path=model_path,
                confidence_threshold=0.5
            )
            self.get_logger().info('ONNX bottle detector initialized successfully')
        except Exception as e:
            self.get_logger().fatal(f'Failed to initialize ONNX detector: {str(e)}')
            raise
        
        # Detection parameters
        self.min_detections_interval = 1.0  # Minimum time between detection logs (seconds)
        self.last_detection_time = 0
        self.fps_update_interval = 2.0  # Seconds between FPS updates
        self.last_fps_update = 0
        self.frame_count = 0
        self.inference_times = []
        
        # Create subscription to camera image with appropriate QoS
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
        
        # Create publisher for annotated image
        self.annotated_pub = self.create_publisher(Image, '/detection/bottle_detection', 10)
        
        # Create publisher for debug image (with confidence scores)
        self.debug_pub = self.create_publisher(Image, '/detection/bottle_debug', 10)
        
        # Create timer for periodic database status checks
        self.db_timer = self.create_timer(30.0, self.check_db_status)
        
        self.get_logger().info('Bottle detection node initialized and ready')
        self.get_logger().info('Subscribed to: /camera/image_raw')
        self.get_logger().info('Publishing to: /detection/bottle_detection')
        self.get_logger().info('Publishing debug to: /detection/bottle_debug')
        self.get_logger().info(f'Database file: {self.db.db_path}')

    def get_package_directory(self):
        """Find the package directory reliably regardless of installation method"""
        # Method 1: Use __file__ (works when installed)
        try:
            return os.path.dirname(os.path.abspath(__file__))
        except:
            pass
            
        # Method 2: Use ament_index (ROS 2 standard way)
        try:
            from ament_index_python.packages import get_package_share_directory
            return get_package_share_directory('bottle_detection')
        except:
            pass
            
        # Method 3: Fallback to current directory (for development)
        return os.path.dirname(os.path.abspath(sys.argv[0]))

    def image_callback(self, msg):
        try:
            # Rate limit database logging
            current_time = time.time()
            if current_time - self.last_detection_time < self.min_detections_interval:
                return
                
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run ONNX detection
            detections, inference_time = self.detector.detect(cv_image)
            
            # Update FPS calculation
            self.frame_count += 1
            self.inference_times.append(inference_time)
            if current_time - self.last_fps_update > self.fps_update_interval:
                avg_inference = sum(self.inference_times) / len(self.inference_times)
                fps = self.frame_count / (current_time - self.last_fps_update)
                self.get_logger().info(
                    f'Performance: {fps:.1f} FPS, '
                    f'Inference: {avg_inference*1000:.1f} ms'
                )
                self.frame_count = 0
                self.inference_times = []
                self.last_fps_update = current_time
            
            # Create copies for different visualizations
            display_img = cv_image.copy()
            debug_img = cv_image.copy()
            
            bottle_count = len(detections)
            bottle_confidences = []
            bottle_boxes = []
            
            # Process detections
            for detection in detections:
                confidence = detection['confidence']
                x1, y1, x2, y2 = detection['bbox']
                
                # Store for database
                bottle_confidences.append(confidence)
                bottle_boxes.append((x1, y1, x2, y2))
                
                # Draw bounding box (green for bottles)
                cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add label with confidence
                label = f'Bottle: {confidence:.2f}'
                cv2.putText(display_img, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # For debug image, add more details
                cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                details = f'Bottle ID:{detection["class_id"]} Conf:{confidence:.2f}'
                cv2.putText(debug_img, details, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Only log if we detected bottles or if it's been a while
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
                        f'Detected {bottle_count} bottle(s). '
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
        """Periodically check database status and log recent entries"""
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