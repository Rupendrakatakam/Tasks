#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

class BottleDetector(Node):
    def __init__(self):
        super().__init__('bottle_detector')
        
        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()
        
        # Load YOLOv8 model (using medium version for speed and accuracy)
        try:
            self.model = YOLO('yolov8m.pt')  # You can also use yolov8s.pt for better accuracy
            self.get_logger().info('YOLOv8 model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {str(e)}')
            raise
        
        # Bottle class ID in COCO dataset is 39
        self.bottle_class_id = 39
        
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
        
        self.get_logger().info('Bottle detection node initialized and ready')
        self.get_logger().info('Subscribed to: /camera/image_raw')
        self.get_logger().info('Publishing to: /detection/bottle_detection')
        self.get_logger().info('Publishing debug to: /detection/bottle_debug')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run YOLO inference with stream=True for memory efficiency
            results = self.model(cv_image, stream=True)
            
            # Create copies for different visualizations
            display_img = cv_image.copy()
            debug_img = cv_image.copy()
            
            bottle_count = 0
            
            # Process each detection result
            for result in results:
                boxes = result.boxes.cpu().numpy()
                
                for box in boxes:
                    # Check if this is a bottle (COCO class ID 39)
                    if int(box.cls[0]) == self.bottle_class_id:
                        bottle_count += 1
                        
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = box.conf[0]
                        
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
            
            # Convert back to ROS Image messages
            annotated_msg = self.bridge.cv2_to_imgmsg(display_img, "bgr8")
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            
            # Preserve the original header
            annotated_msg.header = msg.header
            debug_msg.header = msg.header
            
            # Publish annotated images
            self.annotated_pub.publish(annotated_msg)
            self.debug_pub.publish(debug_msg)
            
            # Log detection results
            if bottle_count > 0:
                self.get_logger().info(f'Detected {bottle_count} bottle(s)', throttle_duration_sec=1)
            else:
                self.get_logger().debug('No bottles detected', throttle_duration_sec=5)
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

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