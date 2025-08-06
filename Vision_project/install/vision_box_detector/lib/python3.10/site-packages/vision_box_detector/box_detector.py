#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import os


class BoxDetector(Node):
    def __init__(self):
        super().__init__('box_detector')

        # Load YOLOv8 model
        pkg_path = get_package_share_directory('vision_box_detector')
        model_path = os.path.join(pkg_path, 'yolov8n.pt')
        self.model = YOLO(model_path)

        # Bridge to convert ROS Image to OpenCV
        self.bridge = CvBridge()

        # Subscriber to camera
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publisher for annotated image
        self.image_pub = self.create_publisher(Image, '/detected_boxes_image', 10)

        # Timer for logging
        self.timer = self.create_timer(2.0, self.timer_callback)
        self.frame_count = 0
        self.get_logger().info("Box detector node started. Waiting for images...")

    def image_callback(self, msg):
        self.frame_count += 1

        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Run YOLOv8 inference
        results = self.model(cv_image, conf=0.5, classes=39)  # 39 = "box" in COCO

        # Draw bounding boxes
        annotated_frame = cv_image.copy()
        detections = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Get confidence and class
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id]

                # Only draw if it's a "box" or "crate" (COCO: 39=box, 41=crate)
                if cls_id in [39, 41]:
                    # Draw rectangle
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Label
                    label = f'box: {conf:.2f}'
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    detections += 1

        # Log detections
        if detections > 0:
            self.get_logger().info(f"Detected {detections} box(es)")

        # Convert back to ROS Image and publish
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, 'bgr8')
        self.image_pub.publish(annotated_msg)

    def timer_callback(self):
        self.get_logger().info(f"Processed {self.frame_count} frames")


def main(args=None):
    rclpy.init(args=args)
    node = BoxDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()