"""
Topic model for the Study Time Predictor
Represents subjects and topics that can be studied
"""

from app import db

class Topic(db.Model):
    __tablename__ = 'topics'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(50), nullable=False)  # Math, Science, History, etc.
    difficulty_level = db.Column(db.Integer, nullable=False)  # 1-10 scale
    estimated_base_time = db.Column(db.Float)  # Base time in hours for average student
    prerequisites_count = db.Column(db.Integer, default=0)
    content_type = db.Column(db.String(50))  # reading, video, practice, mixed
    
    # Relationships
    study_sessions = db.relationship('StudySession', backref='topic', lazy=True)
    
    def __repr__(self):
        return f'<Topic {self.name}>'
    
    def to_dict(self):
        """Convert topic to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'subject': self.subject,
            'difficulty_level': self.difficulty_level,
            'estimated_base_time': self.estimated_base_time,
            'prerequisites_count': self.prerequisites_count,
            'content_type': self.content_type
        }