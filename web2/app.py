from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.linear_model import LinearRegression
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app) #Enable cors for cross-domain requests

points = []
image_data = None
pixel_um = 1.0
depth_um = 10.0
roi_lines = []


# app.py changes
@app.route('/upload', methods=['POST'])
def upload_image():
    global image_data, points, roi_lines
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400
    
    try:
        # Read the image using PIL to verify it's valid
        img = Image.open(file)
        
        # Convert to RGB if necessary
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        
        # Save to bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Convert to base64
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        points = []
        roi_lines = []
        return jsonify({'message': 'Image uploaded'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_image', methods=['GET'])
def get_image():
    global image_data
    if image_data:
        return jsonify({'image': image_data})
    else:
        return jsonify({'image':None}), 204

@app.route('/click', methods=['POST'])
def handle_click():
    global points
    data = request.get_json()
    if data and 'x' in data and 'y' in data:
        x = data['x']
        y = data['y']
        points.append({'x': x, 'y': y})
        return jsonify({'message': 'Point added'}), 200
    else:
        return jsonify({'error': 'Invalid data'}), 400


@app.route('/remove_point', methods=['POST'])
def remove_point():
    global points
    data = request.get_json()
    if data and 'x' in data and 'y' in data:
         x = data['x']
         y = data['y']
         points = [p for p in points if (abs(p['x'] - x) > 10 or abs(p['y'] - y) > 10)]
         return jsonify({'message': 'Point removed'}), 200
    else:
        return jsonify({'error': 'Invalid data'}), 400

@app.route('/get_data', methods=['GET'])
def get_data():
    global points, roi_lines
    spline_points = []
    if len(points) >= 2:
      try:
            spline_points = create_bezier_spline(points)
      except Exception as e:
            print(f"Error creating spline: {e}")
            spline_points = [] # dont return if an error occured
    
    return jsonify({
        'points': points,
        'spline': spline_points,
         'roi_lines': roi_lines
    })

@app.route('/update_params', methods=['POST'])
def update_params():
    global pixel_um, depth_um
    data = request.get_json()
    if data and 'pixel_um' in data and 'depth_um' in data:
         try:
              pixel_um = float(data['pixel_um'])
              depth_um = float(data['depth_um'])
              return jsonify({'message': 'Parameters updated'}), 200
         except ValueError:
              return jsonify({'error': 'Invalid pixel/um or depth [um]'}), 400
    else:
         return jsonify({'error': 'Invalid data'}), 400
    
@app.route('/create_roi', methods=['POST'])
def create_roi():
    global points, roi_lines, pixel_um, depth_um
    if len(points) < 2:
        return jsonify({'error': 'Not enough points to create ROI'}), 400

    try:
        depth_pixels = depth_um * pixel_um
        
        # 1. Linear Regression
        x = np.array([p['x'] for p in points]).reshape(-1, 1)
        y = np.array([p['y'] for p in points])
        model = LinearRegression()
        model.fit(x, y)
        slope = model.coef_[0]
    
        # 2. Calculate Unit Normal
        normal_x = -slope
        normal_y = 1
        norm = np.sqrt(normal_x**2 + normal_y**2)
        unit_normal = {'x': normal_x / norm, 'y': normal_y / norm}
        
        # 3. Shift First and Last Points
        first_point = points[0]
        last_point = points[-1]
        
        shifted_first_point = {
            'x': first_point['x'] + (unit_normal['x'] * depth_pixels),
            'y': first_point['y'] + (unit_normal['y'] * depth_pixels)
        }
        shifted_last_point = {
            'x': last_point['x'] + (unit_normal['x'] * depth_pixels),
            'y': last_point['y'] + (unit_normal['y'] * depth_pixels)
        }
        
        # Create the line segments for the ROI
        roi_lines = [
            [first_point, shifted_first_point],
            [shifted_first_point, shifted_last_point],
            [shifted_last_point, last_point]
        ]
        return jsonify({'message': 'ROI created'}), 200
    
    except Exception as e:
         print(f"Error creating ROI: {e}")
         return jsonify({'error': str(e)}), 500
    
@app.route('/reset_roi', methods=['POST'])
def reset_roi():
    global roi_lines
    roi_lines = []
    return jsonify({'message': 'ROI Reset'}), 200

def create_bezier_spline(points):
    if len(points) < 2:
        return []

    x = [p['x'] for p in points]
    y = [p['y'] for p in points]

    try:
        tck, _ = splprep([x, y], s=0)
        u = np.linspace(0, 1, num=500)
        spline_x, spline_y = splev(u, tck)
        return [{'x': px, 'y': py} for px, py in zip(spline_x, spline_y)]
    except Exception as e:
        print(f"Spline creation error: {e}")
        return []
        
if __name__ == '__main__':
    app.run(debug=True)
