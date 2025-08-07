#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
import os
import time
from .database import DetectionDatabase  # Import our database module

class BottleDetector(Node):
    def __init__(self):
        super().__init__('bottle_detector')
        
        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()
        
        # Setup directories
        self.base_dir = os.path.expanduser('~/vision_project_ws/src/bottle_detection/bottle_detection')
        self.image_dir = os.path.join(self.base_dir, 'data/images')
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize database
        self.db = DetectionDatabase(os.path.join(self.base_dir, 'detections.db'))
        self.get_logger().info(f'Database initialized at {self.db.db_path}')
        
        # Load YOLOv8 model (using nano version for speed)
        try:
            self.model = YOLO('yolov8n.pt')
            self.get_logger().info('YOLOv8 model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {str(e)}')
            raise
        
        # Bottle class ID in COCO dataset is 39
        self.bottle_class_id = 39
        
        # Detection parameters
        self.confidence_threshold = 0.5  # Only process detections above this confidence
        self.min_detections_interval = 1.0  # Minimum time between detection logs (seconds)
        self.last_detection_time = 0
        
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

    def image_callback(self, msg):
        try:
            # Rate limit database logging
            current_time = time.time()
            if current_time - self.last_detection_time < self.min_detections_interval:
                return
                
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run YOLO inference
            results = self.model(cv_image, conf=self.confidence_threshold, stream=True)
            
            # Create copies for different visualizations
            display_img = cv_image.copy()
            debug_img = cv_image.copy()
            
            bottle_count = 0
            bottle_confidences = []
            bottle_boxes = []
            
            # Process each detection result
            for result in results:
                boxes = result.boxes.cpu().numpy()
                
                for box in boxes:
                    # Check if this is a bottle (COCO class ID 39)
                    if int(box.cls[0]) == self.bottle_class_id:
                        bottle_count += 1
                        
                        # Get box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = float(box.conf[0])
                        
                        # Store for database
                        bottle_confidences.append(conf)
                        bottle_boxes.append((x1, y1, x2, y2))
                        
                        # Draw bounding box (green for bottles)
                        cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Add label with confidence
                        label = f'Bottle: {conf:.2f}'
                        cv2.putText(display_img, label, (int(x1), int(y1)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # For debug image, add more details
                        cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        details = f'Bottle ID:{self.bottle_class_id} Conf:{conf:.2f}'
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