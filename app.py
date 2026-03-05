from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
import os
import processing
import threading
import queue
import json
from hazard_detection import generate_raw_rtsp_frames, generate_usb_frames, hazard_alerts_queue

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'

# A queue to hold progress updates
progress_queue = queue.Queue()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'reference_image' not in request.files or 'video' not in request.files:
            return redirect(request.url)
        ref_image = request.files['reference_image']
        video = request.files['video']
        if ref_image.filename == '' or video.filename == '':
            return redirect(request.url)
        if ref_image and video:
            # Clear the queue for the new session
            while not progress_queue.empty():
                try:
                    progress_queue.get_nowait()
                except queue.Empty:
                    continue

            ref_image_filename = secure_filename(ref_image.filename)
            video_filename = secure_filename(video.filename)
            ref_image_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_image_filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            ref_image.save(ref_image_path)
            video.save(video_path)

            # Start processing in a background thread
            thread = threading.Thread(target=processing.process_video, args=(video_path, ref_image_path, progress_queue, False))
            thread.start()
            
            return render_template('results.html')
    return render_template('index.html')

@app.route('/analyze_rtsp', methods=['POST'])
def analyze_rtsp():
    rtsp_url = request.form.get('rtsp_url')
    ref_image = request.files.get('reference_image')
    is_usb = request.form.get('is_usb') == 'true'

    if not rtsp_url and not is_usb:
        return json.dumps({'error': 'Source required'}), 400
    
    if not ref_image:
        return json.dumps({'error': 'Reference image required'}), 400

    # Save reference image
    ref_image_filename = secure_filename(ref_image.filename)
    ref_image_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_image_filename)
    ref_image.save(ref_image_path)

    # Clear progress queue
    while not progress_queue.empty():
        try: progress_queue.get_nowait()
        except: break

    # Start analysis thread
    source = rtsp_url
    if is_usb:
        if rtsp_url and ('.' in rtsp_url or ':' in rtsp_url):
            # Ensure it has http:// and the endpoint for DroidCam MJPEG
            if not rtsp_url.startswith('http'):
                source = f"http://{rtsp_url}"
            else:
                source = rtsp_url
            
            # If no endpoint is specified, default to /video
            if not ('/video' in source or '/mjpegfeed' in source):
                source = f"{source.rstrip('/')}/video"
        else:
            source = int(rtsp_url) if rtsp_url and rtsp_url.isdigit() else 0

    thread = threading.Thread(target=processing.process_video, 
                              args=(source, ref_image_path, progress_queue, not is_usb, is_usb))
    thread.start()
    
    return redirect(url_for('results_page'))

@app.route('/hazard')
def hazard():
    return render_template('hazard.html')

@app.route('/hazard_feed')
def hazard_feed():
    rtsp_url = request.args.get('rtsp_url')
    is_usb = request.args.get('is_usb') == 'true'
    
    if is_usb:
        # If rtsp_url contains a dot or colon, it's likely a DroidCam IP:Port string
        if rtsp_url and ('.' in rtsp_url or ':' in rtsp_url):
            camera_source = rtsp_url
        else:
            camera_source = int(rtsp_url) if rtsp_url and rtsp_url.isdigit() else 0
            
        return Response(generate_usb_frames(camera_source),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    if not rtsp_url:
        return "RTSP URL required", 400
    return Response(generate_raw_rtsp_frames(rtsp_url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hazard_alerts')
def hazard_alerts():
    def event_stream():
        while True:
            try:
                # Wait for alerts from the queue
                alerts = hazard_alerts_queue.get(timeout=1.0)
                yield f"data: {json.dumps(alerts)}\n\n"
            except queue.Empty:
                continue
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/results_page')
def results_page():
    return render_template('results.html')

@app.route('/progress')
def progress():
    def generate():
        while True:
            try:
                data = progress_queue.get(timeout=120) # Increased timeout for long processing steps
                yield f"data: {json.dumps(data)}\n\n"
                if data.get('progress') == 100:
                    break
            except queue.Empty:
                # If the queue is empty for too long, assume the process is done or stalled
                break
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True, threaded=True, use_reloader=False)
