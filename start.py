# start.py - Flask runner script
from flask import Flask
import sys

if __name__ == "__main__":
    print("🚀 Starting Emotion Mirror API (Flask version)...")
    print("📝 API Endpoint: http://localhost:8000/")
    print("❤️  Health Check: http://localhost:8000/health")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        # Import your Flask app (assuming it's in a file called app.py)
        from app import app
        
        # Run the Flask development server
        app.run(
            host="0.0.0.0",
            port=8000,
            debug=True
        )
    except KeyboardInterrupt:
        print("\n👋 API stopped!")
        sys.exit(0)