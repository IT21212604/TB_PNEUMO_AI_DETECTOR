# TB Pneumo AI Detector - Backend

The backend of the **TB Pneumo AI Detector** project, built using Flask, powers the diagnostic capabilities and data processing for detecting tuberculosis and pneumonia from chest X-ray images. This component handles API requests, interacts with the AI model, manages the database, and serves as the interface between the frontend and the diagnostic model.

## Project Overview

The backend component processes chest X-ray images uploaded via the frontend, leverages a trained AI model to diagnose tuberculosis and pneumonia, and returns diagnostic results. This project aims to support healthcare professionals by providing accurate and timely diagnoses, improving patient outcomes.

## Features

- **Image Processing**: Receives and preprocesses X-ray images for AI model prediction.
- **Diagnostic Prediction**: Uses the AI model to detect tuberculosis and pneumonia from images.
- **User Management**: Manages user authentication, including login, registration, and password resets.
- **Data Storage**: Stores user data, images, and diagnostic results in a secure MongoDB database.
- **API Endpoints**: Provides RESTful endpoints for interacting with the frontend.
- **Logging & Error Handling**: Comprehensive logging for monitoring backend processes and error handling.

## Technologies Used

- **Flask**: For creating the REST API and handling server-side logic.
- **Python**: Primary language for backend logic and AI integration.
- **TensorFlow**: For loading and running the trained AI model.
- **MongoDB**: Database for securely storing user and diagnostic data.
- **OpenCV**: For image processing tasks and format handling.

## Setup Instructions

1. **Clone the Repository**:
   git clone https://github.com/IT21212604/TB_PNEUMO_AI_DETECTOR_BACKEND

2. **Navigate to the Backend Directory**:
  cd TB_Pneumo_AI_Detector_Backend

3. **Create and Activate Virtual Environment**:
   python3 -m venv venv
   source venv/bin/activate

5. **Install Dependencies**:
  pip install -r requirements.txt

6. **Run the Server:**:
   python -m flask run


## Collaborators

- **A.P.J.K.V. Gunawardana**
- **H.T.D.G. Lawanya**
- **K.A.M.M.U. Kuruppu**
- **K.A.I. Oshada**
- **Y.V.H.G. Ranathunga**
   


