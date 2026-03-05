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

def generate_usb_frames(camera_index=0):
    """Generates raw frames from USB camera or DroidCam MJPEG stream."""
    # Check if camera_index is a DroidCam IP:Port
    if isinstance(camera_index, str) and ('.' in camera_index or ':' in camera_index):
        # DroidCam MJPEG URL format
        if not camera_index.startswith('http'):
            source = f"http://{camera_index}/video"
        else:
            source = camera_index
    else:
        try:
            source = int(camera_index)
        except:
            source = camera_index

    print(f"--- Attempting to open source: {source} ---")
    if isinstance(source, int):
        # Hardware USB
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        # MJPEG or RTSP
        # Try default first, then FFMPEG for DroidCam
        cap = cv2.VideoCapture(source)
        if not cap.isOpened() and isinstance(source, str) and source.startswith('http'):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    
    # Increase timeout for network streams
    if isinstance(source, str) and (source.startswith('http') or source.startswith('rtsp')):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    
    if isinstance(source, int): # Only set for hardware USB
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        # Fallback for DroidCam MJPEG if /video endpoint fails
        if isinstance(source, str) and '/video' in source:
            alt_source = source.replace('/video', '/mjpegfeed')
            print(f"--- Retrying with alternate DroidCam endpoint: {alt_source} ---")
            cap = cv2.VideoCapture(alt_source)
            if not cap.isOpened():
                cap = cv2.VideoCapture(alt_source, cv2.CAP_FFMPEG)
            
            if cap.isOpened():
                source = alt_source
            else:
                # Direct base URL check
                base_url = source.split('/video')[0]
                print(f"--- Final retry with base URL: {base_url} ---")
                cap = cv2.VideoCapture(base_url)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(base_url, cv2.CAP_FFMPEG)
                
                if not cap.isOpened():
                    print(f"--- Error: All stream attempts failed for {source} ---")
                    return
                source = base_url
        else:
            print(f"--- Error: Could not open source {source} ---")
            return

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()
def generate_raw_rtsp_frames(rtsp_url):
    """Generates raw frames from RTSP stream without processing."""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        return

    while True:
        success, frame = cap.read()
        if not success:
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url)
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()
