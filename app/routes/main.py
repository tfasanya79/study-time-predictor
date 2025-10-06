"""
Main routes for the Study Time Predictor web application
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from app import db
from app.models import Student, Topic, StudySession
from ml_models.predictor import StudyTimePredictor
import os
from datetime import datetime

bp = Blueprint('main', __name__)

# Initialize predictor (will load model if available)
predictor = StudyTimePredictor()

@bp.route('/')
def index():
    """Home page with overview statistics"""
    students_count = Student.query.count()
    topics_count = Topic.query.count()
    sessions_count = StudySession.query.count()
    
    # Get recent sessions for activity feed
    recent_sessions = StudySession.query.order_by(StudySession.session_date.desc()).limit(5).all()
    
    # Calculate average metrics
    avg_duration = db.session.query(db.func.avg(StudySession.actual_duration)).scalar()
    avg_comprehension = db.session.query(db.func.avg(StudySession.comprehension_score)).scalar()
    
    stats = {
        'students': students_count,
        'topics': topics_count,
        'sessions': sessions_count,
        'avg_duration': round(avg_duration, 2) if avg_duration else 0,
        'avg_comprehension': round(avg_comprehension, 1) if avg_comprehension else 0
    }
    
    return render_template('index.html', stats=stats, recent_sessions=recent_sessions)

@bp.route('/predict')
def predict_form():
    """Show the prediction form"""
    students = Student.query.all()
    topics = Topic.query.all()
    
    return render_template('predict.html', students=students, topics=topics)

@bp.route('/api/predict', methods=['POST'])
def make_prediction():
    """API endpoint for making study time predictions"""
    try:
        data = request.json
        
        # Get student and topic data
        student = Student.query.get(data['student_id'])
        topic = Topic.query.get(data['topic_id'])
        
        if not student or not topic:
            return jsonify({'error': 'Student or topic not found'}), 404
        
        # Prepare input data
        student_data = {
            'age': student.age,
            'education_level': student.education_level,
            'learning_style': student.learning_style,
            'avg_focus_duration': student.avg_focus_duration
        }
        
        topic_data = {
            'subject': topic.subject,
            'difficulty_level': topic.difficulty_level,
            'estimated_base_time': topic.estimated_base_time,
            'prerequisites_count': topic.prerequisites_count,
            'content_type': topic.content_type
        }
        
        session_context = {
            'time_of_day': data.get('time_of_day', 'morning'),
            'study_method': data.get('study_method', 'reading'),
            'environment': data.get('environment', 'home'),
            'energy_level': int(data.get('energy_level', 3)),
            'distraction_level': int(data.get('distraction_level', 3)),
            'breaks_taken': int(data.get('breaks_taken', 1))
        }
        
        # Make prediction
        predicted_time = predictor.predict(student_data, topic_data, session_context)
        
        # Convert to hours and minutes for display
        hours = int(predicted_time)
        minutes = int((predicted_time - hours) * 60)
        
        return jsonify({
            'predicted_hours': predicted_time,
            'display_time': f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m",
            'student_name': student.name,
            'topic_name': topic.name,
            'confidence': 'High' if topic.estimated_base_time > 0 else 'Medium'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/students')
def students():
    """Show all students"""
    students_list = Student.query.all()
    return render_template('students.html', students=students_list)

@bp.route('/student/<int:student_id>')
def student_detail(student_id):
    """Show detailed view of a specific student"""
    student = Student.query.get_or_404(student_id)
    
    # Get student's study sessions with topic information
    sessions = db.session.query(StudySession, Topic).join(Topic)\
        .filter(StudySession.student_id == student_id)\
        .order_by(StudySession.session_date.desc()).all()
    
    # Calculate student statistics
    total_sessions = len(sessions)
    total_time = sum(session.StudySession.actual_duration for session in sessions)
    avg_comprehension = sum(session.StudySession.comprehension_score for session in sessions) / total_sessions if total_sessions > 0 else 0
    
    student_stats = {
        'total_sessions': total_sessions,
        'total_time': round(total_time, 2),
        'avg_comprehension': round(avg_comprehension, 1),
        'avg_session_time': round(total_time / total_sessions, 2) if total_sessions > 0 else 0
    }
    
    return render_template('student_detail.html', student=student, sessions=sessions, stats=student_stats)

@bp.route('/topics')
def topics():
    """Show all topics"""
    topics_list = Topic.query.all()
    return render_template('topics.html', topics=topics_list)

@bp.route('/topic/<int:topic_id>')
def topic_detail(topic_id):
    """Show detailed view of a specific topic"""
    topic = Topic.query.get_or_404(topic_id)
    
    # Get sessions for this topic with student information
    sessions = db.session.query(StudySession, Student).join(Student)\
        .filter(StudySession.topic_id == topic_id)\
        .order_by(StudySession.session_date.desc()).all()
    
    # Calculate topic statistics
    total_sessions = len(sessions)
    if total_sessions > 0:
        avg_time = sum(session.StudySession.actual_duration for session in sessions) / total_sessions
        avg_comprehension = sum(session.StudySession.comprehension_score for session in sessions) / total_sessions
        completion_rate = sum(session.StudySession.completion_percentage for session in sessions) / total_sessions
    else:
        avg_time = avg_comprehension = completion_rate = 0
    
    topic_stats = {
        'total_sessions': total_sessions,
        'avg_time': round(avg_time, 2),
        'avg_comprehension': round(avg_comprehension, 1),
        'completion_rate': round(completion_rate, 1)
    }
    
    return render_template('topic_detail.html', topic=topic, sessions=sessions, stats=topic_stats)

@bp.route('/analytics')
def analytics():
    """Show analytics dashboard"""
    # Get data for various charts
    
    # Subject performance
    subject_stats = db.session.query(
        Topic.subject,
        db.func.avg(StudySession.actual_duration).label('avg_duration'),
        db.func.avg(StudySession.comprehension_score).label('avg_comprehension'),
        db.func.count(StudySession.id).label('session_count')
    ).join(StudySession).group_by(Topic.subject).all()
    
    # Learning style performance
    learning_style_stats = db.session.query(
        Student.learning_style,
        db.func.avg(StudySession.actual_duration).label('avg_duration'),
        db.func.avg(StudySession.comprehension_score).label('avg_comprehension')
    ).join(StudySession).group_by(Student.learning_style).all()
    
    # Time of day performance
    time_stats = db.session.query(
        StudySession.time_of_day,
        db.func.avg(StudySession.comprehension_score).label('avg_comprehension'),
        db.func.count(StudySession.id).label('session_count')
    ).group_by(StudySession.time_of_day).all()
    
    return render_template('analytics.html', 
                         subject_stats=subject_stats,
                         learning_style_stats=learning_style_stats,
                         time_stats=time_stats)

@bp.route('/train-model')
def train_model():
    """Train or retrain the ML model"""
    try:
        # Check if we have enough data
        session_count = StudySession.query.count()
        if session_count < 50:
            flash(f'Need at least 50 study sessions to train model. Currently have {session_count}.', 'warning')
            return redirect(url_for('main.index'))
        
        # Train the model
        global predictor
        predictor = StudyTimePredictor()
        history, mae, mse = predictor.train(epochs=100, verbose=0)
        
        flash(f'Model trained successfully! MAE: {mae:.2f} hours', 'success')
        
    except Exception as e:
        flash(f'Error training model: {str(e)}', 'error')
    
    return redirect(url_for('main.index'))

@bp.route('/api/student/<int:student_id>')
def api_student_data(student_id):
    """API endpoint to get student data"""
    student = Student.query.get_or_404(student_id)
    return jsonify(student.to_dict())

@bp.route('/api/topic/<int:topic_id>')
def api_topic_data(topic_id):
    """API endpoint to get topic data"""
    topic = Topic.query.get_or_404(topic_id)
    return jsonify(topic.to_dict())