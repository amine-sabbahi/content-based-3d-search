# app.py
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
import numpy as np


app = Flask(__name__)
CORS(app)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["image_db"]
users_collection = db["users"]
images_collection = db["images"]

# Configure upload folder
UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_csv_database(csv_path):
    """Load descriptors from a CSV file into a dictionary."""
    df = pd.read_csv(csv_path)
    database = {}
    
    for _, row in df.iterrows():
        # Get the image name
        image_name = row['Image Name']
        
        # Initialize list to store all features
        all_features = []
        
        # Process each feature column
        for col in df.columns[1:]:  # Skip the Image Name column
            value = row[col]
            if isinstance(value, str):
                if value.startswith('[') and value.endswith(']'):
                    # Convert string representation of array to numpy array
                    try:
                        # Remove any newlines and extra spaces
                        value = value.replace('\n', '').strip()
                        # Convert string to list of floats
                        value = np.fromstring(value.strip('[]'), sep=' ')
                    except ValueError:
                        print(f"Error parsing column {col} for image {image_name}")
                        continue
            
            # Convert to numpy array if it isn't already
            if not isinstance(value, np.ndarray):
                value = np.array([value])
            
            value = value / np.linalg.norm(value) if np.linalg.norm(value) > 0 else value
                
            # Flatten array and append to features
            all_features.extend(value.flatten())
        
        # Store the concatenated feature vector
        database[image_name] = np.array(all_features)
    
    image_ids = list(database.keys())  # List of image paths
    return database, image_ids



@app.route('/upload-images', methods=['POST'])
def upload_images():
    if 'images' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    files = request.files.getlist('images')
    categories = request.form.getlist('categories')
    
    if not files or len(files) == 0:
        return jsonify({"error": "No files uploaded"}), 400
    
    uploaded_files = []
    
    for i, file in enumerate(files):
        if file and allowed_file(file.filename):
            # Generate a unique filename
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            category = categories[i] if i < len(categories) else 'uncategorized'
            
            # Create category subdirectory if it doesn't exist
            category_path = os.path.join(UPLOAD_FOLDER, category)
            os.makedirs(category_path, exist_ok=True)
            
            # Save the file
            filepath = os.path.join(category_path, filename)
            file.save(filepath)
            
            # Store file metadata in MongoDB
            file_metadata = {
                'filename': filename,
                'original_name': file.filename,
                'category': category,
                'filepath': filepath
            }
            images_collection.insert_one(file_metadata)
            
            uploaded_files.append({
                'filename': filename,
                'originalName': file.filename,
                'category': category
            })
        else:
            return jsonify({"error": f"Invalid file: {file.filename}"}), 400
    
    return jsonify({
        "message": "Files uploaded successfully", 
        "files": uploaded_files
    }), 200

@app.route('/delete-image', methods=['DELETE'])
def delete_image():
    data = request.json
    filename = data.get('filename')
    category = data.get('category')
    
    if not filename or not category:
        return jsonify({"error": "Filename and category are required"}), 400
    
    try:
        # Remove file from filesystem
        filepath = os.path.join(UPLOAD_FOLDER, category, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Remove metadata from MongoDB
        images_collection.delete_one({
            'filename': filename, 
            'category': category
        })
        
        return jsonify({"message": "Image deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Previous authentication routes remain the same
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    full_name = data.get('fullName')
    email = data.get('email')
    password = data.get('password')

    # Check if all fields are present
    if not full_name or not email or not password:
        return jsonify({"message": "All fields are required"}), 400

    # Check if user already exists
    if users_collection.find_one({'email': email}):
        return jsonify({"message": "Email already registered"}), 400

    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Insert user into the database
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

# New route to fetch existing images
@app.route('/get-existing-images', methods=['GET'])
def get_existing_images():
    try:
        # Retrieve all image metadata from MongoDB
        existing_images = list(images_collection.find({}, {'_id': 0}))
        
        # Verify that the files actually exist in the filesystem
        verified_images = []
        for image in existing_images:
            filepath = os.path.join(UPLOAD_FOLDER, image['category'], image['filename'])
            if os.path.exists(filepath):
                verified_images.append(image)
        
        return jsonify({"images": verified_images}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/transform-image', methods=['POST'])
def transform_image():
    try:
        # Check if image details are in the form data
        if 'image' not in request.form:
            return jsonify({"error": "No image filename provided"}), 400
        
        filename = request.form['image']
        category = request.form['category']
        transformations = json.loads(request.form['transformations'])
        
        # Find image metadata in MongoDB
        image_metadata = images_collection.find_one({
            'filename': filename, 
            'category': category
        })
        
        if not image_metadata:
            return jsonify({"error": "Image not found in database"}), 404
        
        filepath = image_metadata['filepath']
        
        # Open the image
        img = PILImage.open(filepath)
        
        # Apply transformations in order
        for transform in transformations:
            transform_type = transform['type']
            params = transform['params']
            
            if transform_type == 'crop':
                img = img.crop((
                    params['x'], 
                    params['y'], 
                    min(params['x'] + params['width'], img.width), 
                    min(params['y'] + params['height'], img.height)
                ))
            
            elif transform_type == 'resize':
                img = img.resize((
                    params['width'], 
                    params['height']
                ))
            
            elif transform_type == 'rotate':
                img = img.rotate(params)
            
            elif transform_type == 'scale':
                scale_params = float(json.loads(request.form.get('scale', '1')))
                new_size = (
                int(img.width * scale_params), 
                int(img.height * scale_params))
                img = img.resize(new_size)
            elif transform_type == 'translate':
                # Create a new blank image with expanded dimensions
                new_width = img.width + abs(params['x'])
                new_height = img.height + abs(params['y'])
                new_img = PILImage.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
                
                # Paste the original image with offset
                new_img.paste(img, (max(0, params['x']), max(0, params['y'])))
                img = new_img
            
            elif transform_type == 'flip':
                if params == 'horizontal':
                    img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
                elif params == 'vertical':
                    img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
        
        # Generate a new unique filename
        file_ext = os.path.splitext(filename)[1]
        new_filename = f'transformed_{uuid.uuid4()}{file_ext}'
        new_filepath = os.path.join(UPLOAD_FOLDER, category, new_filename)
        
        # Save the transformed image
        img.save(new_filepath)
        
        # Store new image metadata
        new_image_metadata = {
            'filename': new_filename,
            'original_name': f'Transformed_{image_metadata["original_name"]}',
            'category': category,
            'filepath': new_filepath,
            'original_image': filename
        }
        images_collection.insert_one(new_image_metadata)
        
        return jsonify({
            "message": "Image transformed successfully", 
            "newImage": {
                "filename": new_filename,
                "category": category
            }
        }), 200
    
    except Exception as e:
        # Log the full error traceback
        print("Transformation Error:", str(e))
        print(traceback.format_exc())
        return jsonify({
            "error": "Failed to transform image", 
            "details": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)