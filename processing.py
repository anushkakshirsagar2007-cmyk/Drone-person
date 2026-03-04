import cv2
import numpy as np
import os
import tempfile
import shutil
from tracker import CentroidTracker
import similarity
from decision_engine import DecisionEngine
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import base64
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

def process_video(video_path, reference_image_path, progress_queue):
    print("--- Starting video processing with SAHI and InsightFace (Buffalo_L) ---")
    
    # Initialize SAHI Detection Model (YOLOv8 for person detection)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path="yolov8n.pt",
        confidence_threshold=0.3,
        device="cuda:0",
    )
    
    ct = CentroidTracker()
    # Threshold 0.5 as requested for ArcFace
    engine = DecisionEngine(face_threshold=0.5)
    
    reference_cv2_image = cv2.imread(reference_image_path)
    # Extract 512-dim embedding from reference image
    ref_embedding = similarity.get_face_embedding(reference_cv2_image)
    
    if ref_embedding is None:
        print("Error: Could not extract embedding from reference image.")
        progress_queue.put({'error': 'Reference face not detected. Please use a clearer profile picture.'})
        return

    # Resize image to speed up color analysis
    resized_ref_image = cv2.resize(reference_cv2_image, (100, 100))
    print("Analyzing reference image colors...")
    reference_color = similarity.get_dominant_color(resized_ref_image)
    print("Reference image analysis complete.")
    # layer_names and output_layers are obsolete for YOLOv8
    # output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

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

    # Track best match
    best_match_score = -1.0
    best_match_image_path = None
    all_raw_data = [] # List of dicts for each detected face
    
    # Pre-calculate reference embedding once to avoid repetitive extraction
    print(f"--- Loaded reference embedding. Dimension: {ref_embedding.shape} ---")

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
    # Process every 9th frame as requested
    skip_frames = 9 
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                # Update progress even on skipped frames
                if frame_count % 10 == 0:
                    progress = int((frame_count / total_frames) * 100)
                    progress_queue.put({'progress': progress})
                continue

            # REMOVED: frame = cv2.resize(frame, (800, ...))
            # Keeping full resolution for detecting small faces in crowds

            # Determine if we should use SAHI based on crowd density
            # First, do a quick standard detection to count people
            # We use a very fast nano model for this pre-check
            base_detect = detection_model.model(frame, classes=[0], conf=0.3, verbose=False)[0]
            person_count_pre = len(base_detect.boxes)
            
            use_sahi = person_count_pre >= 10
            
            if use_sahi:
                # Crowd Mode: Use SAHI Sliced Prediction for tiny objects
                print(f"--- Crowd Detected ({person_count_pre} persons). Using SAHI Sliced Prediction. ---")
                result = get_sliced_prediction(
                    frame,
                    detection_model,
                    slice_height=640,
                    slice_width=640,
                    overlap_height_ratio=0.15,
                    overlap_width_ratio=0.15,
                    postprocess_type="NMM",
                    postprocess_match_threshold=0.5
                )
                
                # Convert SAHI results to rect format
                person_rects = []
                for object_prediction in result.object_prediction_list:
                    if object_prediction.category.name == "person" and object_prediction.score.value > 0.35:
                        bbox = object_prediction.bbox.to_xyxy()
                        person_rects.append((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            else:
                # Small Group Mode: Use standard detection for tighter, more accurate bounding boxes
                print(f"--- Small Group Detected ({person_count_pre} persons). Using standard detection. ---")
                person_rects = []
                for box in base_detect.boxes:
                    b = box.xyxy[0].cpu().numpy()
                    person_rects.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))

            # Detect faces in full frame using Buffalo_L
            faces = similarity.get_faces(frame)

            # Update tracker with high-resolution person bounding boxes
            objects = ct.update(person_rects)
            frame_match_id = None

            persons_in_frame = []
            
            # Helper function for parallel similarity computation
            def process_face_parallel(person_id, person_bbox, current_frame):
                # 1. Biometric Check (Face)
                # Find the face that falls within this person's bounding box
                f_sim = 0.0
                face_obj = None
                
                for face in faces:
                    f_bbox = face.bbox.astype(int)
                    
                    # Skip low resolution faces (must be at least 40x40 for reliable recognition)
                    f_w = f_bbox[2] - f_bbox[0]
                    f_h = f_bbox[3] - f_bbox[1]
                    if f_w < 40 or f_h < 40:
                        continue
                        
                    # Check if face center is inside person bbox
                    f_cX = int((f_bbox[0] + f_bbox[2]) / 2.0)
                    f_cY = int((f_bbox[1] + f_bbox[3]) / 2.0)
                    
                    if (person_bbox[0] <= f_cX <= person_bbox[2] and 
                        person_bbox[1] <= f_cY <= person_bbox[3]):
                        face_obj = face
                        f_sim = similarity.get_cosine_similarity(ref_embedding, face.embedding)
                        break
                
                # 2. Clothing/Body Check (Using the full person bbox)
                h, w = current_frame.shape[:2]
                startX, startY, endX, endY = person_bbox
                
                # Ensure coordinates are within frame
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                
                cropped_person = current_frame[startY:endY, startX:endX]
                
                if cropped_person.size > 0:
                    u_color = similarity.get_dominant_color(cropped_person)
                    c_sim = similarity.get_color_similarity(reference_color, u_color)
                    t_sim = similarity.get_texture_similarity(reference_cv2_image, cropped_person)
                else:
                    c_sim, t_sim = 1000.0, 0.0
                    
                return person_id, f_sim, c_sim, t_sim, face_obj

            # Map tracked IDs to detected objects
            person_to_process = []
            for (objectID, centroid) in objects.items():
                person_data = {'id': objectID, 'centroid': centroid, 'rect': None, 'decision': 'Uncertain'}
                # Find the SAHI person rect that matches this tracked ID
                for rect in person_rects:
                    startX, startY, endX, endY = rect
                    cX = int((startX + endX) / 2.0)
                    cY = int((startY + endY) / 2.0)
                    if abs(cX - centroid[0]) < 50 and abs(cY - centroid[1]) < 50:
                        person_data['rect'] = rect
                        person_to_process.append((objectID, rect))
                        break
                persons_in_frame.append(person_data)

            # Process similarity in parallel
            if person_to_process:
                with ThreadPoolExecutor(max_workers=min(len(person_to_process), 4)) as executor:
                    results = list(executor.map(lambda p: process_face_parallel(p[0], p[1], frame), person_to_process))
                    
                        # Update engine with results AND collect raw data
                    for p_id, f_sim, c_sim, t_sim, face_obj in results:
                        # Find the person rect from person_to_process
                        p_rect = next(p[1] for p in person_to_process if p[0] == p_id)
                        
                        # Store raw data for each detected person in the global list
                        raw_entry = {
                            'id': int(p_id),
                            'frame': int(frame_count),
                            'face_similarity': float(round(float(f_sim), 4)),
                            'clothing_similarity': float(round(float(c_sim), 4)),
                            'texture_similarity': float(round(float(t_sim), 4)),
                            'bbox': [int(x) for x in p_rect]
                        }
                        
                        # Face crop saving disabled as requested
                        if face_obj:
                            raw_entry['embedding_sample'] = face_obj.embedding[:10].astype(float).tolist()
                        else:
                            raw_entry['embedding_sample'] = [0.0] * 10
                            
                        all_raw_data.append(raw_entry)
                        
                        # Track best match image (prioritize face similarity)
                        if float(f_sim) > best_match_score:
                            best_match_score = float(f_sim)
                            best_match_filename = f"best_match_{video_filename.split('.')[0]}.jpg"
                            best_match_path = os.path.join('static', 'processed', best_match_filename)
                            
                            # Use the full person bbox for the result image if it's the best match
                            startX, startY, endX, endY = p_rect
                            best_match_crop = frame[max(0, startY):min(frame.shape[0], endY), 
                                                   max(0, startX):min(frame.shape[1], endX)]
                            if best_match_crop.size > 0:
                                cv2.imwrite(best_match_path, best_match_crop)
                                best_match_image_path = best_match_filename

                        engine.update(p_id, f_sim, c_sim, t_sim)

            for person in persons_in_frame:
                person['decision'] = engine.get_decision(person['id'])
                if person['decision'] == "Match Confirmed":
                    frame_match_id = person['id']

            # Resize only for output visualization and writing to video
            display_frame = cv2.resize(frame, (output_width, output_height))
            # Scale factor for drawing bounding boxes on resized display frame
            scale_x = output_width / frame.shape[1]
            scale_y = output_height / frame.shape[0]

            # Drawing logic
            if frame_match_id is not None:
                # If a match is confirmed, only draw the matched person in GREEN
                for person in persons_in_frame:
                    if person['id'] == frame_match_id and person['rect'] is not None:
                        # Extract similarity score for this specific person
                        person_f_sim = next((r[1] for r in results if r[0] == person['id']), 0.0)
                        
                        (startX, startY, endX, endY) = person['rect']
                        # Scale coordinates for display
                        dsX, dsY = int(startX * scale_x), int(startY * scale_y)
                        deX, deY = int(endX * scale_x), int(endY * scale_y)
                        dCX, dCY = int(person['centroid'][0] * scale_x), int(person['centroid'][1] * scale_y)
                        
                        # Only GREEN if face similarity meets threshold
                        if person_f_sim >= engine.face_threshold:
                            color = (0, 255, 0) # GREEN
                            text = f"TARGET FOUND: SBT_{person['id']} (S:{person_f_sim:.4f})"
                        else:
                            color = (0, 0, 255) # RED (Tracking but not target face)
                            text = f"ID {person['id']}: Tracking"

                        cv2.rectangle(display_frame, (dsX, dsY), (deX, deY), color, 2)
                        cv2.putText(display_frame, text, (dsX, dsY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.circle(display_frame, (dCX, dCY), 4, color, -1)
            else:
                # Otherwise, draw all tracked persons with color based on face match only
                for person in persons_in_frame:
                    if person['rect'] is not None:
                        # Extract similarity score for this specific person from results
                        person_f_sim = next((r[1] for r in results if r[0] == person['id']), 0.0)
                        
                        (startX, startY, endX, endY) = person['rect']
                        # Scale coordinates for display
                        dsX, dsY = int(startX * scale_x), int(startY * scale_y)
                        deX, deY = int(endX * scale_x), int(endY * scale_y)
                        dCX, dCY = int(person['centroid'][0] * scale_x), int(person['centroid'][1] * scale_y)
                        
                        # RED by default, GREEN only if face match threshold is met
                        if person_f_sim >= engine.face_threshold:
                            color = (0, 255, 0) # GREEN
                            text = f"MATCH: SBT_{person['id']} (S:{person_f_sim:.4f})"
                        else:
                            color = (0, 0, 255) # RED
                            text = f"ID {person['id']}: Tracking"

                        cv2.rectangle(display_frame, (dsX, dsY), (deX, deY), color, 2)
                        cv2.putText(display_frame, text, (dsX, dsY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.circle(display_frame, (dCX, dCY), 4, color, -1)

            # Send progress update
            progress = int((frame_count / total_frames) * 100)
            _, buffer = cv2.imencode('.jpg', display_frame)
            thumbnail = base64.b64encode(buffer).decode('utf-8')
            progress_queue.put({'progress': progress, 'thumbnail': thumbnail})

            out.write(display_frame)

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
        face_score = f"{scores[0]:.4f}" if scores[0] is not None else "N/A"
        color_score = f"{scores[1]:.4f}" if scores[1] is not None else "N/A"
        texture_score = f"{scores[2]:.4f}" if scores[2] is not None else "N/A"
        
        # Calculate how many frames this person was detected in
        detection_frames = [d for d in all_raw_data if d['id'] == final_decision_id]
        frame_list = sorted(list(set([d['frame'] for d in detection_frames])))
        
        explanation = (
            f"TARGET IDENTIFIED: Person detected in frame sequence {frame_list[0]} to {frame_list[-1]}.\n"
            f"ID {final_decision_id} successfully matched against reference profile.\n"
            f"Verification criteria met: Consistent biometric and clothing similarity across {len(frame_list)} analyzed frames.\n"
            f"Final Analysis Metrics - Neural Face Match: {face_score}, Clothing Color Similarity: {color_score}, Surface Texture: {texture_score}."
        )
    else:
        # If no target match is confirmed, we explicitly clear the best_match_image
        best_match_image_path = None
        
        # Check if ANY persons were detected at all
        if not all_raw_data:
            explanation = "NO TARGETS DETECTED: Surveillance scan complete. No human subjects were identified within the processed video frames."
        else:
            unique_ids = set([d['id'] for d in all_raw_data])
            explanation = (
                f"SCAN COMPLETE: {len(unique_ids)} subjects tracked, but NO TARGET MATCH found.\n"
                f"Surveillance subjects identified in frames do not meet the 512-dim ArcFace similarity threshold (>{engine.face_threshold}) or clothing similarity for a confirmed match."
            )
    print("--- Video processing complete ---")
    final_data = {
        'progress': 100,
        'video_path': os.path.join('processed', video_filename),
        'best_match_image': best_match_image_path,
        'best_match_score': round(float(best_match_score), 4),
        'raw_data': all_raw_data[:50], # Limit to first 50 detections for UI stability
        'explanation': explanation
    }
    progress_queue.put(final_data)
