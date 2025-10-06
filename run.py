"""
Main application entry point
Run this file to start the Flask development server
"""

from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run in debug mode for development
    app.run(debug=True, host='0.0.0.0', port=5000)