# Study Time Predictor - Complete Setup Guide

## 🚀 Quick Start (Current Status)

✅ **Good News**: Your application is already set up and running!

- **Application URL**: http://localhost:5000
- **Status**: ✅ Running with 30 students, 25 topics, 600 sessions
- **Model**: ✅ Trained (MAE: 0.99 hours)
- **Database**: ✅ SQLite with synthetic data

## 🔄 Fresh Installation Guide

If you need to set up the application on a new system:

### Prerequisites

1. **Python 3.12+** installed
2. **Git** for version control
3. **Virtual environment** (recommended)

### Step-by-Step Installation

#### 1. Clone Repository
```bash
git clone https://github.com/tfasanya79/study-time-predictor.git
cd study-time-predictor
```

#### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Initialize Application
```bash
python setup.py
```
This will:
- ✅ Create SQLite database
- ✅ Generate 30 students with diverse profiles  
- ✅ Create 25 topics across 5 subjects
- ✅ Generate 600 realistic study sessions
- ✅ Train neural network model (2-3 minutes)

#### 5. Start Application
```bash
python run.py
```

#### 6. Access Application
Open your browser and navigate to: **http://localhost:5000**

## 🎯 Features Overview

### 🤖 AI-Powered Predictions
- **Neural Network Engine**: TensorFlow/Keras model with 0.99-hour accuracy
- **Personalized Estimates**: Considers 15+ student and context factors
- **Real-Time Interface**: Instant predictions through web form
- **Confidence Indicators**: High/Medium/Low confidence ratings

### 📊 Analytics Dashboard
- **Interactive Charts**: Performance by subject, learning style, time of day
- **Performance Insights**: Comprehension rates, completion statistics
- **Trend Analysis**: Historical performance tracking
- **Export Capabilities**: Data visualization with Chart.js

### 👥 Student Management
- **Profile System**: Age, education level, learning style, focus duration
- **Performance Tracking**: Individual session history and metrics  
- **Learning Analytics**: Personal comprehension and time trends
- **Study Patterns**: Optimal study times and methods per student

### 📚 Topic Catalog
- **Multi-Subject Coverage**: Math, Computer Science, Physics, Chemistry, History
- **Difficulty Ratings**: 1-10 scale with performance statistics
- **Content Types**: Reading, video, practice, mixed format support
- **Prerequisites Tracking**: Dependency management for learning paths

### 🔄 Model Management
- **Automated Training**: One-click model retraining with new data
- **Performance Monitoring**: Track model accuracy over time
- **Version Control**: Model versioning and rollback capabilities
- **Continuous Learning**: Improves with more study session data

## Project Structure

```
study-time-predictor/
├── app/                    # Flask application
│   ├── models/            # Database models
│   ├── routes/            # Web routes
│   ├── static/            # CSS, JS, images
│   └── templates/         # HTML templates
├── data/                  # Data generation scripts
├── ml_models/             # Machine learning models
├── run.py                 # Application entry point
├── setup.py              # Setup and initialization
└── requirements.txt       # Python dependencies
```

## Technical Details

### Architecture
- **Backend**: Flask (Python web framework)
- **Database**: SQLite (easily upgradeable to PostgreSQL)
- **ML Framework**: TensorFlow/Keras
- **Frontend**: Bootstrap 5 + Chart.js
- **Data Processing**: Pandas, NumPy, Scikit-learn

### Neural Network Model
- Multi-layer feedforward network
- Input features: student characteristics, topic metadata, session context
- Output: Predicted study time in hours
- Trained with early stopping and learning rate reduction

### Data Features
The model considers:
- **Student features**: Age, education level, learning style, focus duration
- **Topic features**: Subject, difficulty, prerequisites, content type  
- **Context features**: Time of day, study method, environment, energy level

## Customization

### Adding Your Own Data
Replace the mock data by modifying `data/generate_mock_data.py` or create your own data import scripts.

### Retraining the Model
- Use the "Train Model" button in the web interface
- Or run: `python -c "from ml_models.predictor import train_model; train_model()"`

### Scaling to Production
1. Switch to PostgreSQL database
2. Use Gunicorn WSGI server  
3. Set up proper environment variables
4. Configure reverse proxy (nginx)

## 🎮 Application Usage Guide

### Making Predictions
1. **Navigate to Predict**: Click "Predict" in the navigation menu
2. **Select Student**: Choose from 30 available student profiles
3. **Choose Topic**: Pick from 25+ topics across 5 subjects
4. **Set Context**: Configure time, method, environment, energy levels
5. **Get Results**: Receive AI-powered time estimate with confidence rating

### Exploring Analytics
1. **Dashboard Overview**: View performance by subject and learning style
2. **Interactive Charts**: Radar charts for time-of-day analysis
3. **Statistical Insights**: Comprehension rates and completion statistics
4. **Performance Trends**: Historical data visualization

### Managing Data
1. **Browse Students**: View all student profiles and performance metrics
2. **Explore Topics**: Filter topics by subject and difficulty
3. **Session History**: Detailed study session records with outcomes
4. **Model Training**: Use "Train Model" to retrain with current data

## 🛠 Troubleshooting Guide

### Installation Issues

#### "No module named 'app'" Error
```bash
# Ensure you're in the correct directory
pwd  # Should show: /path/to/study-time-predictor

# Verify virtual environment is active
which python  # Should show: .venv/bin/python

# Reinstall if necessary
pip install -r requirements.txt
```

#### "No such table: students" Error
```bash
# Reinitialize database
rm study_predictor.db  # Remove existing database
python setup.py        # Recreate with fresh data
```

#### "Model not found" Error
```bash
# Check if model files exist
ls -la ml_models/

# Retrain model if missing
python -c "from ml_models.predictor import train_model; train_model()"

# Or use web interface "Train Model" button
```

#### "Port 5000 already in use" Error
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process (replace PID with actual process ID)
kill -9 <PID>

# Or change port in run.py
# app.run(debug=True, host='0.0.0.0', port=8080)
```

### Runtime Issues

#### "TensorFlow warnings" (Normal)
The CUDA-related warnings are normal and don't affect functionality:
```
E0000 00:00:1759746740.323974 Unable to register cuDNN factory...
```
These can be ignored - the model runs on CPU successfully.

#### "Memory issues"
```bash
# Check available memory
free -h

# Monitor Python memory usage
ps aux | grep python

# Reduce batch size if needed (in ml_models/predictor.py)
# history = model.fit(..., batch_size=16)  # Reduce from 32
```

#### "Database locked" Error
```bash
# Close any database connections
pkill -f "python.*run.py"

# Wait a moment and restart
python run.py
```

### Performance Optimization

#### Faster Model Training
```bash
# Use fewer epochs for testing
python -c "
from ml_models.predictor import StudyTimePredictor
predictor = StudyTimePredictor()
predictor.train(epochs=50, verbose=1)  # Reduce from 150
"
```

#### Improved Prediction Accuracy
- **More Data**: The model improves with additional study sessions
- **Diverse Data**: Ensure variety in students, topics, and contexts  
- **Regular Retraining**: Use "Train Model" button monthly
- **Quality Data**: Remove outliers or incorrect entries

### Development Tips

#### Adding Your Own Data
```python
# Add students programmatically
from app import create_app, db
from app.models import Student

app = create_app()
with app.app_context():
    new_student = Student(
        name="Your Name",
        email="your.email@domain.com",
        age=25,
        education_level="graduate",
        learning_style="visual",
        avg_focus_duration=90.0
    )
    db.session.add(new_student)
    db.session.commit()
```

#### Custom Topics
```python
# Add topics programmatically
from app.models import Topic

app = create_app()
with app.app_context():
    new_topic = Topic(
        name="Your Topic",
        subject="Your Subject", 
        difficulty_level=7,
        estimated_base_time=3.0,
        prerequisites_count=2,
        content_type="mixed"
    )
    db.session.add(new_topic)
    db.session.commit()
```

### Getting Help

#### Check Logs
```bash
# View application logs (if logging is enabled)
tail -f logs/app.log

# Check Python errors
python run.py 2>&1 | tee error.log
```

#### Debug Mode
```bash
# Run with enhanced debugging
FLASK_DEBUG=1 python run.py
```

#### Health Check
```bash
# Test if application is responding
curl http://localhost:5000/
curl http://localhost:5000/health  # If health endpoint exists
```

### System Requirements

#### Minimum Requirements
- **Python**: 3.8+
- **Memory**: 2GB RAM  
- **Storage**: 500MB free space
- **CPU**: Any modern processor

#### Recommended Requirements
- **Python**: 3.12+
- **Memory**: 4GB RAM
- **Storage**: 2GB free space  
- **CPU**: Multi-core processor for faster training

### Environment Variables

#### Optional Configuration (.env file)
```env
SECRET_KEY=your-secure-secret-key-change-in-production
FLASK_ENV=development
DATABASE_URL=sqlite:///study_predictor.db
TF_ENABLE_ONEDNN_OPTS=0  # Disable TensorFlow warnings
```

## Contributing

This is an educational project. Feel free to:
- Add new features
- Improve the UI/UX  
- Enhance the ML model
- Add more data sources
- Create mobile app version

## License

MIT License - See LICENSE file for details.