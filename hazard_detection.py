import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import base64
import json
import threading
import queue

# Global queue for alerts to be picked up by another route
hazard_alerts_queue = queue.Queue(maxsize=100)

class HazardDetector:
    def __init__(self):
        # Using specialized model for fire (Abonia1/YOLOv8-Fire-and-Smoke-Detection)
        # Root yolov8n.pt in that repo is fine-tuned for Fire (Class 0) and Smoke (Class 1)
        self.fire_model = YOLO('models/fire_detection.pt')
        
        self.last_alert_time = 0
        self.alert_cooldown = 1.0 # Reduced cooldown for more responsive fire alerts
        
    def detect_hazards(self, frame):
        hazards = []
        
        # 1. Fire and Smoke Detection ONLY (Specialized Model)
        fire_results = self.fire_model(frame, conf=0.4, verbose=False)[0]
        
        processed_frame = frame.copy()
        
        for box in fire_results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Abonia1 model classes: 0: fire, 1: smoke
            if cls_id == 0:
                label = "Fire Detected"
                color = (0, 0, 255) # Red for Fire
                
                hazards.append({
                    'type': label,
                    'conf': conf
                })
                
                # Draw Fire Bbox
                b = box.xyxy[0].cpu().numpy()
                cv2.rectangle(processed_frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 3)
                cv2.putText(processed_frame, f"FIRE {conf:.2f}", (int(b[0]), int(b[1])-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # We skip smoke and other classes as the user requested ONLY fire for now
        
        # Push to global queue for the status endpoint
        current_time = time.time()
        if hazards and (current_time - self.last_alert_time > self.alert_cooldown):
            self.last_alert_time = current_time
            try:
                hazard_alerts_queue.put_nowait(hazards)
            except queue.Full:
                try:
                    hazard_alerts_queue.get_nowait()
                    hazard_alerts_queue.put_nowait(hazards)
                except:
                    pass

        return hazards, processed_frame, 0 # Person count not needed for pure fire detection

def generate_hazard_frames(rtsp_url):
    detector = HazardDetector()
    
    # Robust connection options for RTSP/Mobile Cameras
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce latency
    
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream at {rtsp_url}")
        return

    print(f"Connected to RTSP stream: {rtsp_url}")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Stream interrupted. Attempting to reconnect...")
            cap.release()
            time.sleep(2) # Wait 2 seconds before reconnecting
            cap = cv2.VideoCapture(rtsp_url)
            continue
            
        hazards, processed_frame, p_count = detector.detect_hazards(frame)
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # We can't easily send JSON meta-data through the same multipart stream 
        # without complex client-side parsing. For now, we'll focus on the visual.
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
    cap.release()
