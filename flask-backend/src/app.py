from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import bcrypt
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
import keras
import cv2
from keras.layers import TFSMLayer
from datetime import datetime
from flask import jsonify
from bson import ObjectId  # To handle ObjectId conversion
import traceback


app = Flask(__name__)
CORS(app)

model = TFSMLayer('Model/model_enhanced_1', call_endpoint='serving_default')
#UPLOAD_FOLDER = 'C:\Users\Lawanya\Documents'  # Adjust the path as needed


# MongoDB setup
client = MongoClient('mongodb+srv://root:root@clustor0.kcp2uk6.mongodb.net/?retryWrites=true&w=majority&appName=clustor0')
db = client['user_db']
users_collection = db['users']
doctors_collection = db['doctors']
results_collection = db['results']


def preprocess_image(image):
    # Gamma Correction
    gamma = 1.2  # Adjust gamma value as needed
    gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(gamma_corrected)
    clahe_channels = [clahe.apply(channel) for channel in channels]
    enhanced_image = cv2.merge(clahe_channels)

    # Unsharp Masking for Edge Enhancement
    gaussian_blurred = cv2.GaussianBlur(enhanced_image, (9, 9), 10.0)
    enhanced_image = cv2.addWeighted(enhanced_image, 1.5, gaussian_blurred, -0.5, 0)
    
    return enhanced_image



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Step 2: Apply preprocessing (Gamma correction, CLAHE, Unsharp Masking)
    img = preprocess_image(img)
    
    # Step 3: Resize to the expected input size
    img = cv2.resize(img, (224, 224))
    
    # Step 4: Normalize and expand dimensions
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')  # Ensuring it’s in float32 for the model
    
    # Step 5: Convert to tensor
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    # Extract patient details from the request
    patient_name = request.form.get('name')
    patient_age = request.form.get('age')
    patient_nic = request.form.get('nic')

    # Log the patient details for debugging
    print(f"Received patient details - Name: {patient_name}, Age: {patient_age}, NIC: {patient_nic}")
    
    # Make prediction
    try:
        predictions = model(img_tensor)

        # Access the predictions tensor from the dictionary
        predictions_tensor = predictions.get('dense_3', None)

        # Check if predictions_tensor is None
        if predictions_tensor is None:
            return jsonify({'error': 'No predictions returned'}), 500

        # Convert the tensor to a NumPy array
        predictions_array = predictions_tensor.numpy()  # Convert to NumPy array

        # Print the prediction probabilities
        print(f"Prediction probabilities: {predictions_array}")

        # Get the predicted class
        predicted_class = np.argmax(predictions_array[0])

        # Map the predicted class index back to class names
        class_names = ['Normal', 'Pneumonia', 'Tuberculosis']
        predicted_label = class_names[predicted_class]
        

        # Save the result to MongoDB with patient details
        insert_result = results_collection.insert_one({
            'patient_name': patient_name,
            'patient_age': patient_age,
            'patient_nic': patient_nic,
            'predicted_class': predicted_label,
            'timestamp': datetime.utcnow()  # Save the current UTC time
        })
        print(f"Inserted document id: {insert_result.inserted_id}")

        return jsonify({'predicted_class': predicted_label})

    except Exception as e:
        print('Error during prediction:', str(e))  # Log the error
        return jsonify({'error': str(e)}), 500

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    user_type = data.get('userType', 'user')  

    if user_type == 'doctor':
        collection = doctors_collection
        required_fields = ["firstName", "lastName", "email", "password"]
    else:
        collection = users_collection
        required_fields = ["fullName", "email", "password"]

    for field in required_fields:
        if field not in data:
            return jsonify({"message": f"{field} is required"}), 400

    email = data['email']
    password = data['password']

    # Check if the email already exists
    user = collection.find_one({"email": email})
    if user:
        return jsonify({"message": "email already exists"}), 400

    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Insert the new user into the database
    data['password'] = hashed_password
    collection.insert_one(data)

    return jsonify({"message": "User registered successfully"})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data['email']
    password = data['password']
    user_type = data.get('userType', 'user')  # Default to 'user' if not provided

    if user_type == 'doctor':
        collection = doctors_collection
    else:
        collection = users_collection

    # Find the user by email
    user = collection.find_one({"email": email})
    if not user:
        return jsonify({"message": "Invalid email or password"}), 400

    # Check if the password matches
    if not bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return jsonify({"message": "Invalid email or password"}), 400

    user_details = {
        "fullName": user.get('fullName', f"{user.get('firstName', '')} {user.get('lastName', '')}"),
        "email": user['email'],
        "userType": user_type,
        "nic": user.get('nic'),
        "medicalLicenseNumber": user.get('medicalLicenseNumber'),
        "specialization": user.get('specialization'),
        "yearsOfExperience": user.get('yearsOfExperience'),
        "hospitalName": user.get('hospitalName'),
        "department": user.get('department')
    }

    return jsonify({"message": "Login successful", "user": user_details})

@app.route('/results', methods=['GET'])
def get_all_patient_results():
    try:
        # Fetch all patient records from MongoDB
        patients = list(results_collection.find({}))

        # Convert the MongoDB ObjectId to string for JSON serialization
        for patient in patients:
            patient['_id'] = str(patient['_id'])

        return jsonify(patients), 200

    except Exception as e:
        print(f"Error fetching patient records: {str(e)}")
        return jsonify({"error": "Could not fetch patient records"}), 500
    
def serialize_doctor(doctor):
    doctor['_id'] = str(doctor['_id'])
    doctor.pop('password', None)  # Remove 'password' field
    doctor.pop('confirmPassword', None)  # Remove 'confirmPassword' field
    return doctor

@app.route('/doctors')
def get_doctors():
    try:
        # Fetch doctors from the database
        doctors = list(doctors_collection.find({}))
        
        # Process and serialize the doctors list
        doctors = [serialize_doctor(doctor) for doctor in doctors]

        return jsonify(doctors), 200
    except Exception as e:
        print(f"Error fetching doctor records: {e}")
        return jsonify({"error": "Internal Server Error"}), 500
    
@app.route("/doctors/<doctor_id>", methods=["PUT"])
def update_doctor(doctor_id):
    data = request.json
    data.pop('_id', None)  # Remove '_id' if it exists

    # Check if the doctor_id is a valid ObjectId
    if not ObjectId.is_valid(doctor_id):
        return jsonify({"error": "Invalid doctor ID"}), 400

    try:
        # Perform the update operation
        result = collection.update_one({"_id": ObjectId(doctor_id)}, {"$set": data})

        if result.modified_count == 1:
            return jsonify({"message": "Doctor updated successfully"}), 200
        else:
            return jsonify({"message": "No changes made to the document"}), 200

    except Exception as e:
        print(f"Error updating doctor: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/doctors/<doctor_id>', methods=['DELETE'])
def delete_doctor(doctor_id):
    try:
        delete_result = doctors_collection.delete_one({"_id": ObjectId(doctor_id)})
        if delete_result.deleted_count == 0:
            return jsonify({"error": "Doctor not found"}), 404
        return jsonify({"message": "Doctor deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/users', methods=['GET'])
def get_all_users():
    try:
        users = list(users_collection.find({}))
        for user in users:
            user['_id'] = str(user['_id'])  # Convert ObjectId to string for JSON serialization
            user.pop('password', None)  # Remove password field for security
        return jsonify(users), 200
    except Exception as e:
        print(f"Error fetching user records: {str(e)}")
        return jsonify({"error": "Could not fetch user records"}), 500

@app.route('/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    try:
        updated_data = request.json
        # Validate incoming data if necessary
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": updated_data}
        )
        if result.modified_count > 0:
            return jsonify({"message": "User updated successfully"}), 200
        else:
            return jsonify({"error": "User not found or no changes made"}), 404
    except Exception as e:
        print(f"Error updating user: {str(e)}")
        return jsonify({"error": "Could not update user"}), 500

@app.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        result = users_collection.delete_one({"_id": ObjectId(user_id)})
        if result.deleted_count > 0:
            return jsonify({"message": "User deleted successfully"}), 200
        else:
            return jsonify({"error": "User not found"}), 404
    except Exception as e:
        print(f"Error deleting user: {str(e)}")
        return jsonify({"error": "Could not delete user"}), 500

@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.json
    print("Received data:", data)  # Log received data
    email = data.get('email')
    current_password = data.get('currentPassword')
    new_password = data.get('newPassword')
    user_type = data.get('userType')

    if not email or not current_password or not new_password or not user_type:
        return jsonify({"message": "All fields are required"}), 400

    # Select the appropriate collection based on userType
    collection = users_collection if user_type == 'user' else doctors_collection

    # Find the user by email
    user = collection.find_one({"email": email})
    if not user:
        return jsonify({"message": "User not found"}), 404

    # Check if the current password is correct
    if not bcrypt.checkpw(current_password.encode('utf-8'), user['password']):
        return jsonify({"message": "Current password is incorrect"}), 400

    # Hash the new password
    hashed_new_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

    # Update the user's password in the database
    collection.update_one({"email": email}, {"$set": {"password": hashed_new_password}})

    return jsonify({"message": "Password reset successful"}), 200


if __name__ == '__main__':
    app.run(debug=True)

