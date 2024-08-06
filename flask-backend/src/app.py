
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import bcrypt

app = Flask(__name__)
CORS(app)

# MongoDB setup
client = MongoClient('mongodb+srv://root:root@clustor0.kcp2uk6.mongodb.net/?retryWrites=true&w=majority&appName=clustor0')
db = client['user_db']
users_collection = db['users']
doctors_collection = db['doctors']

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    user_type = data.get('userType', 'user')  # Default to 'user' if not provided

    if user_type == 'doctor':
        collection = doctors_collection
        required_fields = ["firstName", "lastName", "username", "password"]
    else:
        collection = users_collection
        required_fields = ["fullName", "username", "password"]

    for field in required_fields:
        if field not in data:
            return jsonify({"message": f"{field} is required"}), 400

    username = data['username']
    password = data['password']

    # Check if the username already exists
    user = collection.find_one({"username": username})
    if user:
        return jsonify({"message": "Username already exists"}), 400

    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Insert the new user into the database
    data['password'] = hashed_password
    collection.insert_one(data)

    return jsonify({"message": "User registered successfully"})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data['username']
    password = data['password']
    user_type = data.get('userType', 'user')  # Default to 'user' if not provided

    if user_type == 'doctor':
        collection = doctors_collection
    else:
        collection = users_collection

    # Find the user by username
    user = collection.find_one({"username": username})
    if not user:
        return jsonify({"message": "Invalid username or password"}), 400

    # Check if the password matches
    if not bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return jsonify({"message": "Invalid username or password"}), 400

    user_details = {
        "fullName": user.get('fullName', f"{user.get('firstName', '')} {user.get('lastName', '')}"),
        "username": user['username'],
        "userType": user_type,
        "nic": user.get('nic'),
        "medicalLicenseNumber": user.get('medicalLicenseNumber'),
        "specialization": user.get('specialization'),
        "yearsOfExperience": user.get('yearsOfExperience'),
        "hospitalName": user.get('hospitalName'),
        "department": user.get('department')
    }

    return jsonify({"message": "Login successful", "user": user_details})

if __name__ == '__main__':
    app.run(debug=True)
