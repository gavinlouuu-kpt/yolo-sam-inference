from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import json
import webbrowser
import threading
import time
from typing import Dict, Tuple, List
import mimetypes

app = Flask(__name__)

# Add MIME types for common image formats
mimetypes.add_type('image/tiff', '.tiff')
mimetypes.add_type('image/tiff', '.tif')

# Global variables to store state
conditions: List[str] = []
current_condition_idx = 0
roi_coordinates: Dict[str, Dict[str, int]] = {}
first_images: Dict[str, Path] = {}
output_dir: Path = None

@app.route('/')
def index():
    if current_condition_idx >= len(conditions):
        return "All conditions processed"
    return render_template('roi_selection.html')

@app.route('/select_roi')
def select_roi():
    condition = request.args.get('condition')
    if not condition or condition not in first_images:
        return "Invalid condition", 400
    return render_template('roi_selection.html')

@app.route('/image')
def get_image():
    condition = request.args.get('condition')
    if not condition or condition not in first_images:
        return "Invalid condition", 400
    
    image_path = first_images[condition]
    try:
        # Get the MIME type based on file extension
        mime_type = mimetypes.guess_type(image_path)[0]
        if mime_type is None:
            mime_type = 'application/octet-stream'
        
        return send_file(
            str(image_path),
            mimetype=mime_type,
            as_attachment=False,
            download_name=image_path.name
        )
    except Exception as e:
        print(f"Error serving image: {str(e)}")
        return f"Error serving image: {str(e)}", 500

@app.route('/confirm_roi', methods=['POST'])
def confirm_roi():
    global current_condition_idx
    
    data = request.json
    condition = data.get('condition')
    x_min = data.get('x_min')
    x_max = data.get('x_max')
    y_min = data.get('y_min')
    y_max = data.get('y_max')
    
    if not all([condition, x_min is not None, x_max is not None, y_min is not None, y_max is not None]):
        return jsonify({"error": "Missing required data"}), 400
    
    roi_coordinates[condition] = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max
    }
    current_condition_idx += 1
    
    # Save ROI coordinates after each confirmation
    if output_dir:
        with open(output_dir / "roi_coordinates.json", 'w') as f:
            json.dump(roi_coordinates, f, indent=2)
    
    next_condition = conditions[current_condition_idx] if current_condition_idx < len(conditions) else None
    return jsonify({"next_condition": next_condition})

def run_server(host='127.0.0.1', port=5000):
    app.run(host=host, port=port)

def get_roi_coordinates_web(
    condition_dirs: List[Path],
    run_output_dir: Path
) -> Dict[str, Dict[str, int]]:
    """Get ROI coordinates for each condition using web interface."""
    global conditions, output_dir, first_images
    
    # Reset global state
    conditions = []
    roi_coordinates.clear()
    first_images.clear()
    output_dir = run_output_dir
    
    # Initialize conditions and first images
    for condition_dir in condition_dirs:
        batch_dirs = [d for d in condition_dir.iterdir() if d.is_dir()]
        if not batch_dirs:
            continue
            
        image_files = list(batch_dirs[0].glob("*.png")) + \
                     list(batch_dirs[0].glob("*.jpg")) + \
                     list(batch_dirs[0].glob("*.tiff"))
        if not image_files:
            continue
            
        conditions.append(condition_dir.name)
        first_images[condition_dir.name] = image_files[0]
    
    if not conditions:
        raise ValueError("No valid conditions found")
    
    # Start Flask server in a separate thread
    server_thread = threading.Thread(
        target=run_server,
        kwargs={'host': '127.0.0.1', 'port': 5000}
    )
    server_thread.daemon = True
    server_thread.start()
    
    # Give the server a moment to start
    time.sleep(1)
    
    # Open web browser
    webbrowser.open(f'http://127.0.0.1:5000/select_roi?condition={conditions[0]}')
    
    # Wait for all conditions to be processed
    while current_condition_idx < len(conditions):
        time.sleep(0.5)
    
    return roi_coordinates 