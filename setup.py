#!/usr/bin/env python3
"""
Setup script for Study Time Predictor
Initializes the database and generates mock data for development
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app, db
from data.generate_mock_data import create_mock_data
from ml_models.predictor import StudyTimePredictor

def setup_application():
    """Setup the complete application"""
    print("=== Study Time Predictor Setup ===\n")
    
    # Step 1: Create and initialize the Flask app
    print("1. Creating Flask application...")
    app = create_app()
    
    with app.app_context():
        # Step 2: Create database tables
        print("2. Creating database tables...")
        db.create_all()
        print("   ✓ Database tables created")
        
        # Check if we already have data
        from app.models import Student, Topic, StudySession
        
        student_count = Student.query.count()
        topic_count = Topic.query.count()
        session_count = StudySession.query.count()
        
        if student_count == 0 or topic_count == 0 or session_count == 0:
            print(f"   Current data: {student_count} students, {topic_count} topics, {session_count} sessions")
            
            # Step 3: Generate mock data
            print("3. Generating mock data...")
            create_mock_data()
            print("   ✓ Mock data generated successfully")
            
            # Verify data creation
            student_count = Student.query.count()
            topic_count = Topic.query.count() 
            session_count = StudySession.query.count()
        
        print(f"   Final data: {student_count} students, {topic_count} topics, {session_count} sessions")
    
    # Step 4: Train the ML model
    print("4. Training machine learning model...")
    try:
        predictor = StudyTimePredictor()
        
        # Check if model already exists
        model_path = os.path.join('ml_models', 'study_time_model.h5')
        if os.path.exists(model_path):
            print("   ✓ Model already exists, loading...")
            predictor.load_model()
        else:
            print("   Training new model (this may take a few minutes)...")
            history, mae, mse = predictor.train(epochs=100, verbose=0)
            print(f"   ✓ Model trained successfully! MAE: {mae:.2f} hours")
    except Exception as e:
        print(f"   ⚠ Warning: Could not train model - {str(e)}")
        print("   You can train the model later using the web interface")
    
    print("\n=== Setup Complete! ===")
    print("\nTo start the application:")
    print("  python run.py")
    print("\nThen visit: http://localhost:5000")
    print("\nFeatures available:")
    print("  • Study time predictions")
    print("  • Student and topic management")
    print("  • Analytics dashboard")
    print("  • Model retraining")

def install_dependencies():
    """Install required dependencies"""
    print("Installing required Python packages...")
    
    import subprocess
    
    try:
        # Install requirements
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ All dependencies installed successfully")
            return True
        else:
            print(f"Error installing dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error installing dependencies: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Study Time Predictor')
    parser.add_argument('--install-deps', action='store_true', 
                       help='Install Python dependencies first')
    parser.add_argument('--skip-model', action='store_true',
                       help='Skip ML model training')
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            print("Failed to install dependencies. Please install manually:")
            print("pip install -r requirements.txt")
            sys.exit(1)
    
    # Set up the application
    try:
        setup_application()
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you're in a virtual environment")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check if all directories exist")
        sys.exit(1)