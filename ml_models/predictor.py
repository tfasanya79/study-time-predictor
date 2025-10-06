"""
Neural Network Model for Study Time Prediction
Uses TensorFlow/Keras to predict study time based on student and topic features
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from app import create_app, db
from app.models import Student, Topic, StudySession

class StudyTimePredictor:
    def __init__(self, model_dir='ml_models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
    def prepare_features(self, df):
        """Prepare features from the dataframe for ML model"""
        
        # Student features
        features = pd.DataFrame()
        
        # Numerical student features
        features['age'] = df['age']
        features['avg_focus_duration'] = df['avg_focus_duration']
        
        # Topic features
        features['difficulty_level'] = df['difficulty_level']
        features['estimated_base_time'] = df['estimated_base_time']
        features['prerequisites_count'] = df['prerequisites_count']
        
        # Session context features
        features['breaks_taken'] = df['breaks_taken']
        features['distraction_level'] = df['distraction_level']
        features['energy_level'] = df['energy_level']
        
        # Categorical features (need encoding)
        categorical_features = ['education_level', 'learning_style', 'subject', 
                              'content_type', 'time_of_day', 'study_method', 'environment']
        
        for cat_feature in categorical_features:
            if cat_feature not in self.label_encoders:
                self.label_encoders[cat_feature] = LabelEncoder()
                features[cat_feature] = self.label_encoders[cat_feature].fit_transform(df[cat_feature].fillna('unknown'))
            else:
                # Handle unseen categories during prediction
                le = self.label_encoders[cat_feature]
                encoded_values = []
                for val in df[cat_feature].fillna('unknown'):
                    if val in le.classes_:
                        encoded_values.append(le.transform([val])[0])
                    else:
                        encoded_values.append(0)  # Default to first class for unseen values
                features[cat_feature] = encoded_values
        
        # Derived features
        features['difficulty_focus_ratio'] = features['difficulty_level'] / (features['avg_focus_duration'] / 60)
        features['energy_distraction_diff'] = features['energy_level'] - features['distraction_level']
        
        # Hour of day encoding (cyclical)
        hour_mapping = {'morning': 9, 'afternoon': 14, 'evening': 19, 'night': 23}
        features['time_of_day_numeric'] = df['time_of_day'].map(hour_mapping)
        features['hour_sin'] = np.sin(2 * np.pi * features['time_of_day_numeric'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['time_of_day_numeric'] / 24)
        features.drop('time_of_day_numeric', axis=1, inplace=True)
        
        self.feature_names = list(features.columns)
        return features
    
    def load_training_data(self):
        """Load and prepare training data from database"""
        app = create_app()
        
        with app.app_context():
            # Query all data with joins
            query = db.session.query(
                StudySession.actual_duration,
                StudySession.planned_duration,
                StudySession.breaks_taken,
                StudySession.distraction_level,
                StudySession.energy_level,
                StudySession.time_of_day,
                StudySession.study_method,
                StudySession.environment,
                StudySession.comprehension_score,
                Student.age,
                Student.education_level,
                Student.learning_style,
                Student.avg_focus_duration,
                Topic.subject,
                Topic.difficulty_level,
                Topic.estimated_base_time,
                Topic.prerequisites_count,
                Topic.content_type
            ).join(Student, StudySession.student_id == Student.id)\
             .join(Topic, StudySession.topic_id == Topic.id)
            
            # Convert to DataFrame
            df = pd.read_sql(query.statement, db.engine)
            
        return df
    
    def build_model(self, input_dim):
        """Build the neural network architecture"""
        model = keras.Sequential([
            # Input layer
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            
            # Output layer - single neuron for regression
            layers.Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, epochs=100, validation_split=0.2, verbose=1):
        """Train the neural network model"""
        print("Loading training data...")
        df = self.load_training_data()
        
        if df.empty:
            raise ValueError("No training data available. Please generate mock data first.")
        
        print(f"Loaded {len(df)} training samples")
        
        # Prepare features and target
        X = self.prepare_features(df)
        y = df['actual_duration'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build and train model
        self.model = self.build_model(X_train_scaled.shape[1])
        
        print("Training model...")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=10, min_lr=0.001
        )
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        # Evaluate model
        test_loss, test_mae, test_mse = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        print(f"\nModel Performance:")
        print(f"Test MAE: {test_mae:.3f} hours")
        print(f"Test RMSE: {np.sqrt(test_mse):.3f} hours")
        
        # Save model and preprocessors
        self.save_model()
        
        return history, test_mae, test_mse
    
    def predict(self, student_data, topic_data, session_context):
        """Make a prediction for a specific student, topic, and context"""
        if self.model is None:
            self.load_model()
        
        # Create DataFrame from input data
        input_data = {**student_data, **topic_data, **session_context}
        df = pd.DataFrame([input_data])
        
        # Prepare features
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled, verbose=0)[0][0]
        
        return max(0.1, prediction)  # Ensure minimum positive prediction
    
    def save_model(self):
        """Save the trained model and preprocessors"""
        model_path = os.path.join(self.model_dir, 'study_time_model.h5')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
        features_path = os.path.join(self.model_dir, 'feature_names.pkl')
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoders, encoders_path)
        joblib.dump(self.feature_names, features_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load the trained model and preprocessors"""
        model_path = os.path.join(self.model_dir, 'study_time_model.h5')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
        features_path = os.path.join(self.model_dir, 'feature_names.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError("No trained model found. Please train the model first.")
        
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(encoders_path)
        self.feature_names = joblib.load(features_path)
        
        print("Model loaded successfully")
    
    def get_feature_importance(self):
        """Analyze feature importance (simplified version for neural networks)"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # This is a simplified approach - in practice, you might use SHAP or similar
        weights = self.model.get_weights()
        first_layer_weights = np.abs(weights[0]).mean(axis=1)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': first_layer_weights
        }).sort_values('importance', ascending=False)
        
        return importance_df

def train_model():
    """Train the study time prediction model"""
    predictor = StudyTimePredictor()
    history, mae, mse = predictor.train(epochs=150, verbose=1)
    
    print("\nTraining completed!")
    print("Model saved and ready for predictions.")
    
    return predictor

if __name__ == "__main__":
    train_model()