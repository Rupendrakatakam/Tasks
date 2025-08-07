import sqlite3
import os
from datetime import datetime

class DetectionDatabase:
    def __init__(self, db_path='detections.db'):
        """
        Initialize database connection and create tables if needed
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_tables()
        
    def connect(self):
        """Establish connection to the database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = 1")
            return True
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return False
            
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.conn:
            if not self.connect():
                return
                
        cursor = self.conn.cursor()
        
        # Create detections table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            bottle_count INTEGER NOT NULL,
            image_path TEXT,
            min_confidence REAL,
            max_confidence REAL,
            avg_confidence REAL
        )
        ''')
        
        # Create bottle details table (for individual bottle data)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bottle_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            confidence REAL NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            FOREIGN KEY(detection_id) REFERENCES detections(id) ON DELETE CASCADE
        )
        ''')
        
        self.conn.commit()
        
    def log_detection(self, bottle_count, image_path=None, 
                     min_conf=0.0, max_conf=0.0, avg_conf=0.0):
        """
        Log a detection event
        
        Args:
            bottle_count (int): Number of bottles detected
            image_path (str, optional): Path to saved image
            min_conf (float): Minimum confidence of detected bottles
            max_conf (float): Maximum confidence of detected bottles
            avg_conf (float): Average confidence of detected bottles
            
        Returns:
            int: ID of the new detection record
        """
        if not self.conn:
            if not self.connect():
                return None
                
        cursor = self.conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute('''
        INSERT INTO detections (timestamp, bottle_count, image_path, 
                              min_confidence, max_confidence, avg_confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, bottle_count, image_path, min_conf, max_conf, avg_conf))
        
        detection_id = cursor.lastrowid
        self.conn.commit()
        return detection_id
        
    def log_bottle_details(self, detection_id, confidence, x1, y1, x2, y2):
        """
        Log details of an individual bottle
        
        Args:
            detection_id (int): ID from detections table
            confidence (float): Detection confidence
            x1, y1, x2, y2 (float): Bounding box coordinates
            
        Returns:
            bool: Success status
        """
        if not self.conn:
            if not self.connect():
                return False
                
        cursor = self.conn.cursor()
        
        cursor.execute('''
        INSERT INTO bottle_details (detection_id, confidence, x1, y1, x2, y2)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (detection_id, confidence, x1, y1, x2, y2))
        
        self.conn.commit()
        return True
        
    def get_recent_detections(self, limit=10):
        """
        Get recent detection records
        
        Args:
            limit (int): Number of records to return
            
        Returns:
            list: Detection records
        """
        if not self.conn:
            if not self.connect():
                return []
                
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT id, timestamp, bottle_count, image_path, 
               min_confidence, max_confidence, avg_confidence
        FROM detections
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (limit,))
        
        return cursor.fetchall()
        
    def get_bottles_for_detection(self, detection_id):
        """
        Get all bottles for a specific detection
        
        Args:
            detection_id (int): ID of detection record
            
        Returns:
            list: Bottle detail records
        """
        if not self.conn:
            if not self.connect():
                return []
                
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT confidence, x1, y1, x2, y2
        FROM bottle_details
        WHERE detection_id = ?
        ORDER BY confidence DESC
        ''', (detection_id,))
        
        return cursor.fetchall()
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None