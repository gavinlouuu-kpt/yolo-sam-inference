from flask import Flask, render_template, request, jsonify, send_file, redirect
from pathlib import Path
import json
import webbrowser
import threading
import time
from typing import Dict, Tuple, List
import mimetypes
import io
from PIL import Image
import numpy as np
import os

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
    """Redirect to the test page or first condition if available."""
    print("DEBUG: Index route called")
    if not conditions:
        print("DEBUG: No conditions available, redirecting to test page")
        return redirect('/test')
    
    if current_condition_idx >= len(conditions):
        print("DEBUG: All conditions processed")
        return "All conditions processed"
    
    print(f"DEBUG: Redirecting to first condition: {conditions[0]}")
    return redirect(f'/select_roi?condition={conditions[0]}')

@app.route('/select_roi')
def select_roi():
    condition = request.args.get('condition')
    print(f"DEBUG: select_roi called for condition: {condition}")
    if not condition or condition not in first_images:
        print(f"DEBUG: Invalid condition in select_roi. Available conditions: {list(first_images.keys())}")
        return "Invalid condition", 400
    
    print(f"DEBUG: Rendering template for condition: {condition}")
    return render_template('roi_selection.html', condition=condition)

@app.route('/image')
def get_image():
    condition = request.args.get('condition')
    print(f"DEBUG: Requested image for condition: {condition}")
    if not condition or condition not in first_images:
        print(f"DEBUG: Invalid condition. Available conditions: {list(first_images.keys())}")
        return "Invalid condition", 400
    
    image_path = first_images[condition]
    print(f"DEBUG: Image path: {image_path}, exists: {image_path.exists()}")
    try:
        # Check if the image is a TIFF file
        if image_path.suffix.lower() in ['.tiff', '.tif']:
            print(f"DEBUG: Converting TIFF image to JPEG")
            # Convert TIFF to JPEG for browser compatibility
            img = Image.open(image_path)
            print(f"DEBUG: Image opened successfully. Mode: {img.mode}, Size: {img.size}")
            # Convert to RGB if it's not already (handles grayscale or RGBA)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print(f"DEBUG: Converted image to RGB mode")
            
            # Save as JPEG to a BytesIO object
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=90)
            img_byte_arr.seek(0)
            print(f"DEBUG: Image converted to JPEG successfully")
            
            return send_file(
                img_byte_arr,
                mimetype='image/jpeg',
                as_attachment=False,
                download_name=f"{image_path.stem}.jpg"
            )
        else:
            # For non-TIFF images, serve directly
            print(f"DEBUG: Serving non-TIFF image directly")
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
        print(f"DEBUG: Error serving image: {str(e)}")
        import traceback
        traceback.print_exc()
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

@app.route('/test')
def test():
    """Simple test route to verify the server is working."""
    return """
    <html>
    <head>
        <title>Server Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            .info { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .debug { background-color: #f2dede; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            ul { list-style-type: none; padding-left: 10px; }
            li { margin-bottom: 10px; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>Server is working!</h1>
        
        <div class="info">
            <h2>Server Information</h2>
            <p>Flask server is running on port 9487</p>
            <p>Available conditions: {}</p>
        </div>
        
        <div class="debug">
            <h2>Debug Information</h2>
            <p>Current working directory: {}</p>
            <p>First images dictionary has {} entries</p>
        </div>
        
        <h2>Available Conditions:</h2>
        <ul>
            {}
        </ul>
        
        <h2>Direct Image Links:</h2>
        <ul>
            {}
        </ul>
    </body>
    </html>
    """.format(
        len(conditions),
        os.getcwd(),
        len(first_images),
        ''.join(f'<li><a href="/select_roi?condition={c}">{c}</a></li>' for c in conditions),
        ''.join(f'<li><a href="/image?condition={c}" target="_blank">{c}</a> - Path: {first_images.get(c)}</li>' for c in conditions)
    )

def run_server(host='0.0.0.0', port=9487):
    app.run(host=host, port=port, debug=False)

def get_roi_coordinates_web(
    condition_dirs: List[Path],
    run_output_dir: Path
) -> Dict[str, Dict[str, int]]:
    """Get ROI coordinates for each condition using web interface."""
    global conditions, current_condition_idx, roi_coordinates, first_images
    
    print(f"DEBUG: Starting get_roi_coordinates_web with {len(condition_dirs)} condition directories")
    for i, d in enumerate(condition_dirs):
        print(f"DEBUG: Condition dir {i+1}: {d}")
    
    conditions = []
    roi_coordinates.clear()
    first_images.clear()
    output_dir = run_output_dir
    
    # Initialize conditions and first images
    for condition_dir in condition_dirs:
        print(f"DEBUG: Processing condition directory: {condition_dir}")
        # Look for images recursively in condition directory and its subdirectories
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.tiff']:
            found_files = list(condition_dir.glob(f"**/*{ext}"))
            print(f"DEBUG:   Found {len(found_files)} {ext} files")
            image_files.extend(found_files)
        
        # Filter out background image
        original_count = len(image_files)
        image_files = [f for f in image_files if 'background' not in f.name.lower()]
        print(f"DEBUG:   After filtering out background images: {len(image_files)} of {original_count} files remain")
        
        if not image_files:
            print(f"DEBUG:   No valid images found in {condition_dir}, skipping")
            continue
            
        conditions.append(condition_dir.name)
        first_images[condition_dir.name] = image_files[0]
        print(f"DEBUG:   Added condition '{condition_dir.name}' with first image: {image_files[0]}")
    
    print(f"DEBUG: Found {len(conditions)} valid conditions with images")
    if not conditions:
        raise ValueError("No valid conditions found")
    
    # Start Flask server in a separate thread
    server_thread = threading.Thread(
        target=run_server,
        kwargs={'host': '0.0.0.0', 'port': 9487}
    )
    server_thread.daemon = True
    server_thread.start()
    
    # Give the server a moment to start
    time.sleep(1)
    
    # Instead of automatically opening the browser, print the URL for manual opening
    url_localhost = f'http://localhost:9487/select_roi?condition={conditions[0]}'
    test_url = f'http://localhost:9487/test'
    print("\n" + "="*80)
    print(f"WSL BROWSER ACCESS: Please manually open one of these URLs in your Windows browser:")
    print(f"1. Test page (try this first): {test_url}")
    print(f"2. Using localhost: {url_localhost}")
    print(f"3. Or find your WSL IP address with 'ip addr show eth0' and use: http://YOUR_WSL_IP:9487/test")
    print("="*80 + "\n")
    
    # Wait for all conditions to be processed
    while current_condition_idx < len(conditions):
        time.sleep(0.5)
    
    return roi_coordinates 