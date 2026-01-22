from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
import os
import processing
import threading
import queue
import json

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
            thread = threading.Thread(target=processing.process_video, args=(video_path, ref_image_path, progress_queue))
            thread.start()
            
            return render_template('results.html')
    return render_template('index.html')

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
