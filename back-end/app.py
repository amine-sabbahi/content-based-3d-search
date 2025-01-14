from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import bcrypt
import os
import uuid
from flask import send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
import traceback
import json
import pandas as pd
from werkzeug.utils import secure_filename
import shutil
import csv  # Added for CSV handling

app = Flask(__name__)
CORS(app)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["image_db"]
users_collection = db["users"]
images_collection = db["images"]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
# Configure upload folders
UPLOAD_FOLDER = 'Uploads'
MODELS_FOLDER = os.path.join(UPLOAD_FOLDER, 'models')
THUMBNAILS_FOLDER = os.path.join(UPLOAD_FOLDER, 'thumbnails')
DATASET_THUMBNAILS_FOLDER = '3d-dataset/Thumbnails'

os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(THUMBNAILS_FOLDER, exist_ok=True)

CATEGORIES = ['Abstract', 'Alabastron', 'All Models', 'Amphora', 'Aryballos', 'Bowl', 'Dinos', 'Hydria', 'Kalathos', 'Kantharos', 'Krater', 'Kyathos', 'Kylix', 'Lagynos', 'Lebes', 'Lekythos', 'Lydion', 'Mastos', 'Modern-Bottle', 'Modern-Glass', 'Modern-Mug', 'Modern-Vase', 'Mug', 'Native American - Bottle', 'Native American - Bowl', 'Native American - Effigy', 'Native American - Jar', 'Nestoris', 'Oinochoe', 'Other', 'Pelike', 'Picher Shaped', 'Pithoeidi', 'Pithos', 'Psykter', 'Pyxis', 'Skyphos']

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CSV file to track uploaded models
CSV_FILE_PATH = 'uploaded_models.csv'
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model Name', 'Category', 'Model Path', 'Thumbnail Path'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Register and login routes remain unchanged
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    full_name = data.get('fullName')
    email = data.get('email')
    password = data.get('password')

    if not full_name or not email or not password:
        return jsonify({"message": "All fields are required"}), 400

    if users_collection.find_one({'email': email}):
        return jsonify({"message": "Email already registered"}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({
        'full_name': full_name,
        'email': email,
        'password': hashed_password,
    })
    return jsonify({"message": "User registered successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data['email']
    password = data['password']
    
    user = users_collection.find_one({'email': email})
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return jsonify({"message": "Invalid username or password"}), 401
    
    return jsonify({"message": "Login successful"}), 200

@app.route('/Uploads/thumbnails/<filename>', methods=['GET'])
def serve_thumbnail(filename):
    try:
        return send_from_directory(THUMBNAILS_FOLDER, filename)
    except Exception as e:
        return jsonify({"error": f"Could not load image: {str(e)}"}), 404

@app.route('/Uploads/thumbnails/<filename>', methods=['GET'])
def serve_rsscn_image(filename):
    return send_from_directory('Uploads', filename)


# Modify your upload_model route to ensure proper file paths
@app.route('/upload-model', methods=['POST'])
def upload_model():
    if 'file' not in request.files or 'category' not in request.form:
        return jsonify({"error": "No file or category provided"}), 400

    file = request.files['file']
    category = request.form['category']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if category not in CATEGORIES:
        return jsonify({"error": "Invalid category"}), 400

    # Save the 3D model
    filename = secure_filename(file.filename)
    model_path = os.path.join(MODELS_FOLDER, filename)
    file.save(model_path)

    # Look for the corresponding thumbnail
    thumbnail_name = os.path.splitext(filename)[0] + '.jpg'
    thumbnail_path = os.path.join(DATASET_THUMBNAILS_FOLDER, category, thumbnail_name)

    if os.path.exists(thumbnail_path):
        # Copy the thumbnail to the Uploads/thumbnails folder
        destination_thumbnail_path = os.path.join(THUMBNAILS_FOLDER, thumbnail_name)
        shutil.copy(thumbnail_path, destination_thumbnail_path)
        # Store the relative URL path that matches our new route
        thumbnail_url = f"/Uploads/thumbnails/{thumbnail_name}"
    else:
        thumbnail_url = None

    # Update the CSV file
    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, category, model_path, thumbnail_url])

    return jsonify({
        "message": "Model uploaded successfully",
        "model_path": model_path,
        "thumbnailUrl": thumbnail_url
    }), 200


@app.route('/get-uploaded-models', methods=['GET'])
def get_uploaded_models():
    try:
        models = []
        with open(CSV_FILE_PATH, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                # Return the relative URL path
                models.append({
                    "filename": row[0],
                    "category": row[1],
                    "model_path": row[2],
                    "thumbnailUrl": row[3]  # This should be the relative path starting with /uploads/thumbnails/
                })
        return jsonify({"models": models}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Make sure CORS is properly configured to allow image requests

@app.route('/delete-model', methods=['DELETE'])
def delete_model():
    data = request.json
    filename = data.get('filename')

    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    try:
        # Delete the model file
        model_path = os.path.join(MODELS_FOLDER, filename)
        if os.path.exists(model_path):
            os.remove(model_path)
        else:
            return jsonify({"error": "Model file not found"}), 404

        # Delete the corresponding thumbnail
        thumbnail_name = os.path.splitext(filename)[0] + '.jpg'
        thumbnail_path = os.path.join(THUMBNAILS_FOLDER, thumbnail_name)
        if os.path.exists(thumbnail_path):
            os.remove(thumbnail_path)

        # Remove the entry from the CSV file
        rows = []
        with open(CSV_FILE_PATH, mode='r') as file:
            reader = csv.reader(file)
            rows = [row for row in reader if row[0] != filename]  # Skip the row with the deleted model

        # Write the updated rows back to the CSV
        with open(CSV_FILE_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        return jsonify({"message": "Model and thumbnail deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


CORS(app, resources={r"/*": {"origins": "*"}})

if __name__ == '__main__':
    app.run(debug=True)