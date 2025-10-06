"""
Study Session model for the Study Time Predictor
Represents individual study sessions with outcomes
"""

from app import db
from datetime import datetime

class StudySession(db.Model):
    __tablename__ = 'study_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), nullable=False)
    
    # Session timing
    planned_duration = db.Column(db.Float)  # in hours
    actual_duration = db.Column(db.Float)   # in hours
    session_date = db.Column(db.DateTime, default=datetime.utcnow)
    time_of_day = db.Column(db.String(20))  # morning, afternoon, evening, night
    
    # Session context
    breaks_taken = db.Column(db.Integer, default=0)
    distraction_level = db.Column(db.Integer)  # 1-5 scale
    energy_level = db.Column(db.Integer)       # 1-5 scale
    
    # Outcomes
    comprehension_score = db.Column(db.Float)  # 0-100 percentage
    completion_percentage = db.Column(db.Float)  # 0-100 percentage
    satisfaction_rating = db.Column(db.Integer)  # 1-5 scale
    
    # Additional context
    study_method = db.Column(db.String(50))  # reading, practice, video, discussion
    environment = db.Column(db.String(50))   # library, home, cafe, etc.
    
    def __repr__(self):
        return f'<StudySession {self.id}: Student {self.student_id} - Topic {self.topic_id}>'
    
    def to_dict(self):
        """Convert study session to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'topic_id': self.topic_id,
            'planned_duration': self.planned_duration,
            'actual_duration': self.actual_duration,
            'session_date': self.session_date.isoformat() if self.session_date else None,
            'time_of_day': self.time_of_day,
            'breaks_taken': self.breaks_taken,
            'distraction_level': self.distraction_level,
            'energy_level': self.energy_level,
            'comprehension_score': self.comprehension_score,
            'completion_percentage': self.completion_percentage,
            'satisfaction_rating': self.satisfaction_rating,
            'study_method': self.study_method,
            'environment': self.environment
        }