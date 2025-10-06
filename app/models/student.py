"""
Student model for the Study Time Predictor
Represents a student with their learning characteristics
"""

from app import db
from datetime import datetime

class Student(db.Model):
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    age = db.Column(db.Integer)
    education_level = db.Column(db.String(50))  # high_school, undergraduate, graduate
    learning_style = db.Column(db.String(50))   # visual, auditory, kinesthetic, reading
    avg_focus_duration = db.Column(db.Float)    # in minutes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    study_sessions = db.relationship('StudySession', backref='student', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Student {self.name}>'
    
    def to_dict(self):
        """Convert student to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'age': self.age,
            'education_level': self.education_level,
            'learning_style': self.learning_style,
            'avg_focus_duration': self.avg_focus_duration,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }