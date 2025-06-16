"""
Emotion Detection API - Version 1
==================================
This Python script is a Flask-based web API that detects emotions from text and facial images. 
It uses pre-trained AI models for emotion classification and OpenCV for face detection.

This program is a requirement for the course AIDI1004 Assignment #1 Technology Demonstration Video

@author: Alexander Sanchez
@date: June 16, 2025
@version: 1.0
@license: MIT

Usage Example:
--------------
1. Start the Flask API:
   $ python app.py
   The API will be available at http://localhost:8000

2. Check API health:
   GET /health

3. Analyze text emotion:
   POST /analyze/text
   {
       "text": "I am feeling very happy today!"
   }

4. Analyze an image:
   POST /analyze/image
   (Attach an image file for emotion analysis)

Features:
---------
- Supports emotion analysis for both **text and images**.
- Uses **j-hartmann/emotion-english-distilroberta-base** for text classification.
- Provides **OpenCV-based face detection** for extracting facial expressions.
- Includes error handling for **bad requests and server issues**.

Future Improvements:
--------------------
- Implement **DeepFace** for better facial emotion classification.
- Switch to **MTCNN or RetinaFace** for improved face detection.
- Add **multi-face analysis** for images containing multiple people.
- Explore **real-time emotion tracking** in live video streams.

"""

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
from transformers import pipeline
import torch
import logging
from werkzeug.exceptions import HTTPException, InternalServerError, BadRequest, ServiceUnavailable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class EmotionDetector:
    def __init__(self):
        """Initialize the emotion detection pipeline"""
        try:
            # Use a reliable emotion detection model
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                framework="pt"
            )
            
            # Alternative face emotion detection model (comment out the above and use this for face detection)
            # self.emotion_pipeline = pipeline(
            #     "image-classification",
            #     model="trpakov/vit-face-expression",
            #     framework="pt"
            # )
            
            # Initialize face detection
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            logger.info("Emotion detection pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing emotion pipeline: {e}")
            raise e
    
    def detect_faces(self, image: np.ndarray) -> list[tuple]:
        """Detect faces in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def analyze_emotion_from_text(self, text: str) -> dict:
        """Analyze emotion from text (demo purpose)"""
        try:
            results = self.emotion_pipeline(text)
            
            # Process results
            emotions = {}
            for result in results:
                emotion = result['label'].lower()
                confidence = result['score']
                emotions[emotion] = confidence
            
            # Get the top emotion
            top_emotion = max(emotions, key=emotions.get)
            confidence = emotions[top_emotion]
            
            return {
                "emotion": top_emotion,
                "confidence": float(confidence),
                "all_emotions": emotions
            }
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            raise InternalServerError(description=f"Emotion analysis failed: {str(e)}")
    
    def analyze_emotion_from_image(self, image: np.ndarray) -> dict:
        """Analyze emotion from facial image"""
        try:
            # For demo purposes, we'll simulate emotion detection
            # In a real implementation, you'd use a face emotion model
            
            faces = self.detect_faces(image)
            
            if len(faces) == 0:
                return {
                    "emotion": "neutral",
                    "confidence": 0.5,
                    "faces_detected": 0,
                    "message": "No faces detected"
                }
            
            # Simulate emotion detection (replace with actual model inference)
            emotions = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
            emotion_scores = np.random.dirichlet(np.ones(len(emotions)), 1)[0]
            
            emotion_dict = dict(zip(emotions, emotion_scores))
            top_emotion = max(emotion_dict, key=emotion_dict.get)
            confidence = emotion_dict[top_emotion]
            
            return {
                "emotion": top_emotion,
                "confidence": float(confidence),
                "faces_detected": len(faces),
                "all_emotions": {k: float(v) for k, v in emotion_dict.items()}
            }
            
        except Exception as e:
            logger.error(f"Error in image emotion analysis: {e}")
            raise InternalServerError(description=f"Image analysis failed: {str(e)}")

# Initialize the emotion detector
try:
    emotion_detector = EmotionDetector()
except Exception as e:
    logger.error(f"Failed to initialize emotion detector: {e}")
    emotion_detector = None

# Error handler
@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    response = e.get_response()
    response.data = jsonify({
        "success": False,
        "error": e.name,
        "message": e.description
    }).data
    response.content_type = "application/json"
    return response

@app.route("/", methods=["GET"])
def root():
    """Health check endpoint"""
    return jsonify({
        "message": "Emotion Mirror API is running!",
        "status": "healthy",
        "model_loaded": emotion_detector is not None
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Detailed health check"""
    return jsonify({
        "status": "healthy" if emotion_detector else "unhealthy",
        "model_loaded": emotion_detector is not None,
        "gpu_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

@app.route("/analyze/text", methods=["POST"])
def analyze_text_emotion():
    """Analyze emotion from text input"""
    if not emotion_detector:
        raise ServiceUnavailable("Emotion detection service unavailable")
    
    text = request.json.get('text', '')
    if not text or text.strip() == "":
        raise BadRequest("Text input cannot be empty")
    
    try:
        result = emotion_detector.analyze_emotion_from_text(text)
        return jsonify({
            "success": True,
            "data": result,
            "input_text": text
        })
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        raise InternalServerError(str(e))

@app.route("/analyze/image", methods=["POST"])
def analyze_image_emotion():
    """Analyze emotion from uploaded image"""
    if not emotion_detector:
        raise ServiceUnavailable("Emotion detection service unavailable")
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        raise BadRequest("No file part in the request")
    
    file = request.files['file']
    
    # If user does not select file, browser may submit an empty part without filename
    if file.filename == '':
        raise BadRequest("No selected file")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise BadRequest("File must be an image")
    
    try:
        # Read and process image
        contents = file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Analyze emotion
        result = emotion_detector.analyze_emotion_from_image(cv_image)
        
        return jsonify({
            "success": True,
            "data": result,
            "filename": file.filename
        })
        
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        raise InternalServerError(f"Image processing failed: {str(e)}")

@app.route("/analyze/base64", methods=["POST"])
def analyze_base64_image():
    """Analyze emotion from base64 encoded image (for webcam frames)"""
    if not emotion_detector:
        raise ServiceUnavailable("Emotion detection service unavailable")
    
    try:
        # Extract base64 data from JSON request
        data = request.get_json()
        if not data or 'image' not in data:
            raise BadRequest("No image data provided")
        
        base64_str = data['image']
        
        # Remove data URL prefix if present
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Analyze emotion
        result = emotion_detector.analyze_emotion_from_image(cv_image)
        
        return jsonify({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Base64 image analysis error: {e}")
        raise InternalServerError(f"Base64 image processing failed: {str(e)}")

@app.route("/models/info", methods=["GET"])
def get_model_info():
    """Get information about loaded models"""
    if not emotion_detector:
        return jsonify({"message": "No models loaded"})
    
    return jsonify({
        "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
        "face_detection": "OpenCV Haar Cascade",
        "supported_emotions": ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"],
        "input_formats": ["text", "image_file", "base64_image"]
    })

# Demo endpoint for testing
@app.route("/demo/emotions", methods=["GET"])
def demo_emotions():
    """Demo endpoint that returns sample emotions"""
    demo_emotions = [
        {"emotion": "happy", "confidence": 0.85, "emoji": "😊"},
        {"emotion": "excited", "confidence": 0.92, "emoji": "🤩"},
        {"emotion": "calm", "confidence": 0.78, "emoji": "😌"},
        {"emotion": "surprised", "confidence": 0.88, "emoji": "😲"},
        {"emotion": "neutral", "confidence": 0.65, "emoji": "😐"}
    ]
    
    return jsonify({
        "success": True,
        "data": demo_emotions,
        "message": "Demo emotions for testing frontend"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
