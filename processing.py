import cv2
import numpy as np
import os
import tempfile
import shutil
from ultralytics import YOLO
from tracker import CentroidTracker
import similarity
import face_recognition
from decision_engine import DecisionEngine

import base64

def process_video(video_path, reference_image_path, progress_queue):
    print("--- Starting video processing ---")
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt") # Using the nano version for speed
    print("YOLOv8 model loaded.")
    ct = CentroidTracker()
    engine = DecisionEngine()
    reference_image = face_recognition.load_image_file(reference_image_path)
    reference_cv2_image = cv2.imread(reference_image_path)
    # Resize image to speed up color analysis
    resized_ref_image = cv2.resize(reference_cv2_image, (100, 100))
    print("Analyzing reference image colors...")
    reference_color = similarity.get_dominant_color(resized_ref_image)
    print("Reference image analysis complete.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        progress_queue.put({'error': 'Could not open video.'})
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_filename = os.path.basename(video_path)
    # Create a temporary file for the output video
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_filename = temp_file.name
    temp_file.close()

    # Get the dimensions for the output video from a resized sample frame
    ret, frame = cap.read()
    if not ret:
        return None, "Error: Could not read the first frame."
    resized_frame = cv2.resize(frame, (800, int(frame.shape[0] * 800 / frame.shape[1])))
    output_height, output_width, _ = resized_frame.shape
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video to the beginning

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_filename, fourcc, 20.0, (output_width, output_height))

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 != 0:
                # Still send progress on skipped frames to make the bar smoother
                progress = int((frame_count / total_frames) * 100)
                progress_queue.put({'progress': progress})
                continue

            # Resize frame for faster processing
            frame = cv2.resize(frame, (800, int(frame.shape[0] * 800 / frame.shape[1])))

            results = model(frame, stream=True)
            rects = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Check if the detected object is a person (class 0 in COCO dataset)
                    if box.cls[0] == 0 and box.conf[0] > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        rects.append((x1, y1, x2, y2))

            objects = ct.update(rects)
            frame_match_id = None

            persons_in_frame = []
            for (objectID, centroid) in objects.items():
                person_data = {'id': objectID, 'centroid': centroid, 'rect': None, 'decision': 'Uncertain'}
                for r in rects:
                    (startX, startY, endX, endY) = r
                    cX = int((startX + endX) / 2.0)
                    cY = int((startY + endY) / 2.0)
                    if cX == centroid[0] and cY == centroid[1]:
                        person_data['rect'] = r
                        cropped_image = frame[startY:endY, startX:endX]
                        if cropped_image.size > 0:
                            rgb_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                            face_match = similarity.get_facial_similarity(reference_image, rgb_cropped_image)
                            unknown_color = similarity.get_dominant_color(cropped_image)
                            color_sim = similarity.get_color_similarity(reference_color, unknown_color)
                            texture_sim = similarity.get_texture_similarity(reference_cv2_image, cropped_image)
                            engine.update(objectID, face_match, color_sim, texture_sim)
                        break
                person_data['decision'] = engine.get_decision(objectID)
                if person_data['decision'] == "Match Confirmed":
                    frame_match_id = objectID
                persons_in_frame.append(person_data)

            # Drawing logic
            if frame_match_id is not None:
                # If a match is confirmed, only draw the matched person
                for person in persons_in_frame:
                    if person['id'] == frame_match_id and person['rect'] is not None:
                        (startX, startY, endX, endY) = person['rect']
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                        text = f"ID {person['id']}: Match Confirmed"
                        cv2.putText(frame, text, (person['centroid'][0] - 10, person['centroid'][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.circle(frame, (person['centroid'][0], person['centroid'][1]), 4, (0, 0, 255), -1)
                        break
            else:
                # Otherwise, draw all tracked persons with their status
                for person in persons_in_frame:
                    if person['rect'] is not None:
                        (startX, startY, endX, endY) = person['rect']
                        color = (0, 255, 0)  # Default green for Uncertain
                        if person['decision'] == "Match Rejected":
                            color = (255, 0, 0) # Blue for rejected
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        text = f"ID {person['id']}: {person['decision']}"
                        cv2.putText(frame, text, (person['centroid'][0] - 10, person['centroid'][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.circle(frame, (person['centroid'][0], person['centroid'][1]), 4, color, -1)

            # Send progress update
            progress = int((frame_count / total_frames) * 100)
            _, buffer = cv2.imencode('.jpg', frame)
            thumbnail = base64.b64encode(buffer).decode('utf-8')
            progress_queue.put({'progress': progress, 'thumbnail': thumbnail})

            out.write(frame)

            # Stop processing if a match is confirmed
            if frame_match_id is not None:
                print(f"--- Match confirmed for ID {frame_match_id}. Stopping processing. ---")
                break
    except Exception as e:
        print(f"--- An error occurred during video processing: {e} ---")
        progress_queue.put({'error': 'An error occurred while processing the video. The file may be corrupt or in an unsupported format.'})
        return
    finally:
        cap.release()
        out.release()

    # Move the temporary file to the final destination
    final_output_path = os.path.join('static', 'processed', video_filename)
    shutil.move(temp_filename, final_output_path)

    final_decision_id = None
    for oid in engine.tracked_persons:
        if engine.get_decision(oid) == "Match Confirmed":
            final_decision_id = oid
            break

    if final_decision_id is not None:
        scores = engine.get_latest_scores(final_decision_id)
        explanation = (
            f"Match Confirmed: The lost person is identified as ID {final_decision_id}.\n"
            f"Decision was made based on consistent high similarity scores over {engine.consecutive_frames} consecutive frames.\n"
            f"Final Scores - Face Match: {scores[0]}, Clothing Color Similarity: {scores[1]:.2f}, Clothing Texture Similarity: {scores[2]:.2f}."
        )
    else:
        explanation = "No definitive match found. While some individuals showed partial similarities, none met all the required criteria for a confident match."
    print("--- Video processing complete ---")
    final_data = {
        'progress': 100,
        'video_path': os.path.join('processed', video_filename),
        'explanation': explanation
    }
    progress_queue.put(final_data)
