#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import random
import math
import time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

class ItemSpawner(Node):
    def __init__(self):
        super().__init__('item_spawner')
        
        # Create client for spawn service
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Spawn service not available, waiting...')
        
        # Item types with defect probabilities
        self.item_types = [
            {'name': 'box', 'model': 'models/box/model.sdf', 'defect_prob': 0.1},
            {'name': 'cylinder', 'model': 'models/cylinder/model.sdf', 'defect_prob': 0.15},
            {'name': 'sphere', 'model': 'models/sphere/model.sdf', 'defect_prob': 0.05}
        ]
        
        # Conveyor dimensions (match your world file)
        self.conveyor_width = 0.8  # meters
        self.spawn_position = 1.5  # meters from start
        
        # Start spawning timer
        self.spawn_timer = self.create_timer(2.0, self.spawn_random_item)
        self.get_logger().info('Item spawner initialized')

    def spawn_random_item(self):
        """Spawn a random item with possible defects"""
        item = random.choice(self.item_types)
        item_name = f"{item['name']}_{int(time.time())}"
        
        # Random position across conveyor width
        x = random.uniform(-self.conveyor_width/2, self.conveyor_width/2)
        y = self.spawn_position
        z = 0.1  # Slightly above belt
        
        # Create pose
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        
        # Random orientation (for defect simulation)
        roll = 0.0
        pitch = 0.0
        yaw = random.uniform(0, math.pi/4) if random.random() < item['defect_prob'] else 0.0
        
        pose.orientation.x = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        pose.orientation.y = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        pose.orientation.z = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        pose.orientation.w = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        
        # Load SDF model
        sdf_path = f"{self.get_package_share_directory('item_spawner')}/models/{item['name']}/model.sdf"
        try:
            with open(sdf_path, 'r') as f:
                sdf_content = f.read()
        except Exception as e:
            self.get_logger().error(f'Could not read model: {str(e)}')
            return
        
        # Create request
        request = SpawnEntity.Request()
        request.name = item_name
        request.xml = sdf_content
        request.robot_namespace = ""
        request.reference_frame = "world"
        request.initial_pose = pose
        
        # Send request
        future = self.spawn_client.call_async(request)
        future.add_done_callback(self.spawn_response)
        
        defect_status = "DEFECTIVE" if yaw != 0 else "GOOD"
        self.get_logger().info(f'Spawning {item_name} ({defect_status}) at ({x:.2f}, {y:.2f}, {z:.2f})')

    def spawn_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Successfully spawned: {response.status_message}')
            else:
                self.get_logger().error(f'Failed to spawn: {response.status_message}')
        except Exception as e:
            self.get_logger().error(f'Spawn service call failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    spawner = ItemSpawner()
    rclpy.spin(spawner)
    spawner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()