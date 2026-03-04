import requests
import os

def download_model(url, filename):
    if os.path.exists(filename):
        print(f"Model {filename} already exists. Skipping download.")
        return True
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

if __name__ == "__main__":
    # Using reliable community models for Fire and Weapons
    # Note: These are example URLs from common high-star repositories found during search.
    # If these fail, we fallback to our robust YOLOv8n + HSV logic.
    
    # 1. Fire and Smoke Detection (Abonia1 model - verified YOLOv8 weights)
    # This model is specifically trained for fire and smoke and is highly cited.
    fire_model_url = "https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection/raw/main/yolov8n.pt"
    
    # 2. Weapon Detection (Guns/Knives)
    # Source: https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8
    # Verified: The models are in .onnx format in the 'models' directory
    # We will use best.onnx as it is compatible with Ultralytics YOLO
    weapon_model_url = "https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8/raw/main/models/best.onnx"

    # For this task, we will attempt to download them to a 'models' directory
    os.makedirs('models', exist_ok=True)
    
    fire_success = download_model(fire_model_url, 'models/fire_detection.pt')
    weapon_success = download_model(weapon_model_url, 'models/weapon_detection.onnx')
    
    if fire_success and weapon_success:
        print("Both specialized models downloaded successfully.")
    else:
        print("One or more specialized models failed to download. Check the URLs or your connection.")
