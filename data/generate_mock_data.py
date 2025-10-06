"""
Generate synthetic data for the Study Time Predictor
Creates realistic mock data for students, topics, and study sessions
"""

import random
import pandas as pd
from datetime import datetime, timedelta
from app import create_app, db
from app.models import Student, Topic, StudySession

# Sample data for generation
STUDENT_NAMES = [
    "Alex Johnson", "Sarah Chen", "Michael Rodriguez", "Emily Davis", 
    "James Wilson", "Jessica Brown", "David Kim", "Ashley Garcia",
    "Christopher Lee", "Amanda Thompson", "Daniel Martinez", "Rachel White",
    "Matthew Taylor", "Lauren Anderson", "Joshua Harris", "Stephanie Clark"
]

EDUCATION_LEVELS = ["high_school", "undergraduate", "graduate"]
LEARNING_STYLES = ["visual", "auditory", "kinesthetic", "reading"]
TIME_OF_DAY = ["morning", "afternoon", "evening", "night"]
STUDY_METHODS = ["reading", "practice", "video", "discussion", "flashcards"]
ENVIRONMENTS = ["library", "home", "cafe", "classroom", "outdoor"]

# Subject and topic data
TOPICS_DATA = {
    "Mathematics": [
        {"name": "Linear Algebra Basics", "difficulty": 6, "base_time": 3.0, "prereq": 2, "type": "mixed"},
        {"name": "Calculus Integration", "difficulty": 8, "base_time": 4.0, "prereq": 4, "type": "practice"},
        {"name": "Statistics Fundamentals", "difficulty": 4, "base_time": 2.5, "prereq": 1, "type": "reading"},
        {"name": "Probability Theory", "difficulty": 7, "base_time": 3.5, "prereq": 3, "type": "mixed"},
        {"name": "Differential Equations", "difficulty": 9, "base_time": 5.0, "prereq": 5, "type": "practice"}
    ],
    "Computer Science": [
        {"name": "Python Programming Basics", "difficulty": 3, "base_time": 2.0, "prereq": 0, "type": "practice"},
        {"name": "Data Structures", "difficulty": 6, "base_time": 4.0, "prereq": 2, "type": "mixed"},
        {"name": "Algorithms Analysis", "difficulty": 8, "base_time": 4.5, "prereq": 3, "type": "practice"},
        {"name": "Machine Learning Intro", "difficulty": 7, "base_time": 3.5, "prereq": 4, "type": "mixed"},
        {"name": "Database Design", "difficulty": 5, "base_time": 3.0, "prereq": 2, "type": "reading"}
    ],
    "Physics": [
        {"name": "Classical Mechanics", "difficulty": 7, "base_time": 3.5, "prereq": 3, "type": "mixed"},
        {"name": "Electromagnetism", "difficulty": 8, "base_time": 4.0, "prereq": 4, "type": "practice"},
        {"name": "Thermodynamics", "difficulty": 6, "base_time": 3.0, "prereq": 2, "type": "reading"},
        {"name": "Quantum Physics Basics", "difficulty": 9, "base_time": 5.0, "prereq": 5, "type": "mixed"},
        {"name": "Optics and Waves", "difficulty": 5, "base_time": 2.5, "prereq": 2, "type": "video"}
    ],
    "Chemistry": [
        {"name": "Organic Chemistry", "difficulty": 8, "base_time": 4.0, "prereq": 3, "type": "mixed"},
        {"name": "Chemical Bonding", "difficulty": 4, "base_time": 2.0, "prereq": 1, "type": "reading"},
        {"name": "Reaction Mechanisms", "difficulty": 7, "base_time": 3.5, "prereq": 4, "type": "practice"},
        {"name": "Analytical Chemistry", "difficulty": 6, "base_time": 3.0, "prereq": 2, "type": "mixed"},
        {"name": "Physical Chemistry", "difficulty": 9, "base_time": 4.5, "prereq": 5, "type": "practice"}
    ],
    "History": [
        {"name": "World War II", "difficulty": 3, "base_time": 2.0, "prereq": 0, "type": "reading"},
        {"name": "Ancient Civilizations", "difficulty": 4, "base_time": 2.5, "prereq": 1, "type": "video"},
        {"name": "Industrial Revolution", "difficulty": 5, "base_time": 3.0, "prereq": 2, "type": "reading"},
        {"name": "Cold War Politics", "difficulty": 6, "base_time": 3.5, "prereq": 2, "type": "mixed"},
        {"name": "Renaissance Art & Culture", "difficulty": 4, "base_time": 2.5, "prereq": 1, "type": "video"}
    ]
}

def generate_students(n=50):
    """Generate n synthetic students"""
    students = []
    for i in range(n):
        student = Student(
            name=random.choice(STUDENT_NAMES) + f" {i+1}",
            email=f"student{i+1}@university.edu",
            age=random.randint(18, 35),
            education_level=random.choice(EDUCATION_LEVELS),
            learning_style=random.choice(LEARNING_STYLES),
            avg_focus_duration=random.uniform(30, 120)  # 30-120 minutes
        )
        students.append(student)
    return students

def generate_topics():
    """Generate all topics from the TOPICS_DATA"""
    topics = []
    for subject, topic_list in TOPICS_DATA.items():
        for topic_data in topic_list:
            topic = Topic(
                name=topic_data["name"],
                subject=subject,
                difficulty_level=topic_data["difficulty"],
                estimated_base_time=topic_data["base_time"],
                prerequisites_count=topic_data["prereq"],
                content_type=topic_data["type"]
            )
            topics.append(topic)
    return topics

def generate_study_sessions(students, topics, sessions_per_student=15):
    """Generate realistic study sessions for each student"""
    sessions = []
    
    for student in students:
        # Each student studies different topics with varying patterns
        student_topics = random.sample(topics, k=min(8, len(topics)))
        
        for _ in range(sessions_per_student):
            topic = random.choice(student_topics)
            
            # Generate realistic session timing based on student and topic characteristics
            base_duration = topic.estimated_base_time
            
            # Adjust for student characteristics
            focus_multiplier = student.avg_focus_duration / 60.0  # Convert to hours
            difficulty_multiplier = 1 + (topic.difficulty_level - 5) * 0.1
            
            # Add some randomness
            planned_duration = base_duration * focus_multiplier * difficulty_multiplier * random.uniform(0.7, 1.3)
            actual_duration = planned_duration * random.uniform(0.8, 1.4)  # Sometimes over/under
            
            # Generate session date (last 3 months)
            days_ago = random.randint(0, 90)
            session_date = datetime.now() - timedelta(days=days_ago)
            
            # Context variables with realistic correlations
            energy_level = random.randint(1, 5)
            distraction_level = random.randint(1, 5)
            
            # Better performance with higher energy, lower distraction
            base_score = 50 + (energy_level - 3) * 10 - (distraction_level - 3) * 8
            comprehension_score = max(0, min(100, base_score + random.uniform(-15, 15)))
            
            # Completion correlates with comprehension and planned vs actual time
            time_ratio = actual_duration / planned_duration if planned_duration > 0 else 1
            completion_base = comprehension_score * (2 - time_ratio) * 0.8
            completion_percentage = max(0, min(100, completion_base + random.uniform(-10, 10)))
            
            # Satisfaction correlates with comprehension and completion
            satisfaction_base = (comprehension_score + completion_percentage) / 40
            satisfaction_rating = max(1, min(5, int(satisfaction_base + random.uniform(-0.5, 0.5))))
            
            session = StudySession(
                student_id=student.id,
                topic_id=topic.id,
                planned_duration=round(planned_duration, 2),
                actual_duration=round(actual_duration, 2),
                session_date=session_date,
                time_of_day=random.choice(TIME_OF_DAY),
                breaks_taken=random.randint(0, max(1, int(actual_duration * 2))),
                distraction_level=distraction_level,
                energy_level=energy_level,
                comprehension_score=round(comprehension_score, 1),
                completion_percentage=round(completion_percentage, 1),
                satisfaction_rating=satisfaction_rating,
                study_method=random.choice(STUDY_METHODS),
                environment=random.choice(ENVIRONMENTS)
            )
            sessions.append(session)
    
    return sessions

def create_mock_data():
    """Create and populate the database with mock data"""
    app = create_app()
    
    with app.app_context():
        # Clear existing data
        db.drop_all()
        db.create_all()
        
        print("Generating students...")
        students = generate_students(30)  # 30 students
        for student in students:
            db.session.add(student)
        db.session.commit()
        
        print("Generating topics...")
        topics = generate_topics()
        for topic in topics:
            db.session.add(topic)
        db.session.commit()
        
        print("Generating study sessions...")
        sessions = generate_study_sessions(students, topics, 20)  # 20 sessions per student
        for session in sessions:
            db.session.add(session)
        db.session.commit()
        
        print(f"Created {len(students)} students, {len(topics)} topics, and {len(sessions)} study sessions")
        
        # Create summary statistics
        create_data_summary()

def create_data_summary():
    """Create a summary of the generated data"""
    app = create_app()
    
    with app.app_context():
        students_count = Student.query.count()
        topics_count = Topic.query.count()
        sessions_count = StudySession.query.count()
        
        # Calculate some interesting statistics
        avg_session_duration = db.session.query(db.func.avg(StudySession.actual_duration)).scalar()
        avg_comprehension = db.session.query(db.func.avg(StudySession.comprehension_score)).scalar()
        
        summary = {
            "students": students_count,
            "topics": topics_count,
            "study_sessions": sessions_count,
            "avg_session_duration_hours": round(avg_session_duration, 2) if avg_session_duration else 0,
            "avg_comprehension_score": round(avg_comprehension, 1) if avg_comprehension else 0
        }
        
        print("\nData Summary:")
        print(f"Students: {summary['students']}")
        print(f"Topics: {summary['topics']}")
        print(f"Study Sessions: {summary['study_sessions']}")
        print(f"Average Session Duration: {summary['avg_session_duration_hours']} hours")
        print(f"Average Comprehension Score: {summary['avg_comprehension_score']}%")
        
        return summary

if __name__ == "__main__":
    create_mock_data()