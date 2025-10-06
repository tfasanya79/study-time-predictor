# Model Training Guide - Study Time Predictor

## Overview

This guide explains the machine learning model architecture, training process, and optimization techniques used in the Study Time Predictor.

---

## Model Architecture

### Neural Network Design

The prediction model uses a **feedforward neural network** implemented with TensorFlow/Keras:

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_features,)),
    Dropout(0.3),                    # Prevent overfitting
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')    # Regression output
])
```

### Architecture Rationale

- **Input Layer (128 neurons)**: Large enough to capture feature interactions
- **Hidden Layers**: Progressively smaller (64→32→16) for feature compression
- **Dropout Layers**: Prevent overfitting with rates 0.3→0.2→0.2
- **Linear Output**: Single neuron for continuous time prediction
- **ReLU Activation**: Effective for non-linear pattern recognition

---

## Feature Engineering

### Input Features (15 total)

#### Student Characteristics (4 features)
```python
student_features = [
    'age',                    # Numerical: 18-35
    'education_level',        # Categorical: encoded 0-2
    'learning_style',         # Categorical: encoded 0-3
    'avg_focus_duration'      # Numerical: minutes
]
```

#### Topic Metadata (5 features)
```python
topic_features = [
    'subject',                # Categorical: encoded by subject
    'difficulty_level',       # Numerical: 1-10 scale
    'estimated_base_time',    # Numerical: hours
    'prerequisites_count',    # Numerical: count
    'content_type'           # Categorical: encoded by type
]
```

#### Session Context (6 features)
```python
session_features = [
    'time_of_day',           # Categorical: morning/afternoon/evening/night
    'study_method',          # Categorical: reading/practice/video/etc
    'environment',           # Categorical: home/library/cafe/etc
    'energy_level',          # Numerical: 1-5 scale
    'distraction_level',     # Numerical: 1-5 scale
    'breaks_taken'           # Numerical: count
]
```

### Derived Features (2 additional)
```python
derived_features = [
    'difficulty_focus_ratio',     # difficulty_level / (avg_focus_duration / 60)
    'energy_distraction_diff'     # energy_level - distraction_level
]
```

### Cyclical Encoding
Time of day is encoded cyclically to capture temporal patterns:
```python
hour_mapping = {'morning': 9, 'afternoon': 14, 'evening': 19, 'night': 23}
features['hour_sin'] = np.sin(2 * π * hour / 24)
features['hour_cos'] = np.cos(2 * π * hour / 24)
```

---

## Training Process

### Data Preparation

#### 1. Data Loading
```python
def load_training_data():
    query = db.session.query(
        StudySession.actual_duration,  # Target variable
        # ... all feature columns
    ).join(Student).join(Topic)
    
    return pd.read_sql(query.statement, db.engine)
```

#### 2. Feature Preprocessing
```python
# Categorical encoding
for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 3. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

### Model Compilation

```python
model.compile(
    optimizer='adam',           # Adaptive learning rate
    loss='mse',                # Mean Squared Error for regression
    metrics=['mae', 'mse']     # Track both MAE and MSE
)
```

### Training Configuration

#### Callbacks
```python
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,           # Stop if no improvement for 15 epochs
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,            # Reduce LR by 5x when plateau
        patience=10,
        min_lr=0.001
    )
]
```

#### Training Parameters
```python
history = model.fit(
    X_train, y_train,
    epochs=150,                # Maximum epochs
    validation_split=0.2,      # 20% for validation
    callbacks=callbacks,
    batch_size=32,            # Default batch size
    verbose=1                 # Show progress
)
```

---

## Performance Evaluation

### Current Model Metrics

**Training Results (600 samples):**
- **MAE (Mean Absolute Error)**: 0.99 hours
- **RMSE (Root Mean Square Error)**: 1.39 hours
- **Training Time**: ~2-3 minutes
- **Model Size**: ~50KB

### Evaluation Methods

#### 1. Cross-Validation
```python
from sklearn.model_selection import cross_val_score

# K-fold cross validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
print(f"CV MAE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

#### 2. Prediction Analysis
```python
# Prediction vs actual scatter plot
predictions = model.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Actual Time (hours)')
plt.ylabel('Predicted Time (hours)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
```

#### 3. Residual Analysis
```python
residuals = y_test - predictions.flatten()
plt.hist(residuals, bins=20)
plt.xlabel('Residuals (hours)')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
```

---

## Model Optimization

### Hyperparameter Tuning

#### Grid Search Example
```python
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor

def create_model(neurons=64, dropout=0.2, learning_rate=0.001):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout),
        Dense(neurons//2, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='linear')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Grid search
param_grid = {
    'neurons': [32, 64, 128],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(
    KerasRegressor(model=create_model),
    param_grid,
    cv=3,
    scoring='neg_mean_absolute_error'
)
```

### Advanced Architectures

#### 1. Ensemble Model
```python
class EnsemblePredictor:
    def __init__(self):
        self.models = []
        for i in range(5):  # 5 models in ensemble
            model = create_model()
            self.models.append(model)
    
    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0)
```

#### 2. Recurrent Neural Network (LSTM)
```python
# For sequential data (student's historical sessions)
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, features)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])
```

#### 3. Attention Mechanism
```python
# Focus on important features
from tensorflow.keras.layers import MultiHeadAttention

class AttentionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=4, key_dim=16)
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.output_layer = Dense(1)
    
    def call(self, inputs):
        attention_output = self.attention(inputs, inputs)
        x = self.dense1(attention_output)
        x = self.dense2(x)
        return self.output_layer(x)
```

---

## Feature Importance Analysis

### Method 1: Permutation Importance
```python
from sklearn.inspection import permutation_importance

def calculate_feature_importance(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)
    
    return importance_df
```

### Method 2: SHAP Values
```python
import shap

# Calculate SHAP values for model interpretability
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Visualize feature importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### Method 3: Weight Analysis (Simple)
```python
def get_feature_importance_weights(model):
    # Get first layer weights
    first_layer_weights = model.get_weights()[0]
    importance = np.abs(first_layer_weights).mean(axis=1)
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
```

---

## Data Augmentation

### Synthetic Data Generation
```python
def augment_training_data(df, augmentation_factor=0.2):
    """Add noise to existing data to increase training samples"""
    
    augmented_data = []
    
    for _, row in df.iterrows():
        # Create variations with small noise
        for _ in range(int(len(df) * augmentation_factor)):
            new_row = row.copy()
            
            # Add noise to numerical features
            new_row['actual_duration'] *= np.random.normal(1.0, 0.1)
            new_row['energy_level'] = np.clip(
                new_row['energy_level'] + np.random.normal(0, 0.5), 1, 5
            )
            new_row['distraction_level'] = np.clip(
                new_row['distraction_level'] + np.random.normal(0, 0.5), 1, 5
            )
            
            augmented_data.append(new_row)
    
    return pd.concat([df, pd.DataFrame(augmented_data)], ignore_index=True)
```

---

## Model Deployment and Updates

### Version Control for Models
```python
import datetime
import joblib

def save_model_with_version(model, scaler, encoders, metrics):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = f"ml_models/v_{timestamp}"
    os.makedirs(version_dir, exist_ok=True)
    
    # Save model components
    model.save(f"{version_dir}/model.h5")
    joblib.dump(scaler, f"{version_dir}/scaler.pkl")
    joblib.dump(encoders, f"{version_dir}/encoders.pkl")
    
    # Save metadata
    metadata = {
        'version': timestamp,
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'training_samples': metrics['samples'],
        'features': feature_names
    }
    
    with open(f"{version_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
```

### A/B Testing Framework
```python
class ModelManager:
    def __init__(self):
        self.models = {
            'current': load_current_model(),
            'challenger': None
        }
        self.traffic_split = 0.9  # 90% current, 10% challenger
    
    def predict(self, features, user_id=None):
        # Route prediction based on user_id hash
        if self.models['challenger'] and hash(user_id) % 10 == 0:
            return self.models['challenger'].predict(features)
        else:
            return self.models['current'].predict(features)
    
    def update_challenger(self, new_model):
        self.models['challenger'] = new_model
    
    def promote_challenger(self):
        if self.models['challenger']:
            self.models['current'] = self.models['challenger']
            self.models['challenger'] = None
```

---

## Monitoring and Maintenance

### Performance Monitoring
```python
def monitor_model_performance():
    """Track model performance over time"""
    
    # Get recent predictions vs actual (if available)
    recent_sessions = StudySession.query.filter(
        StudySession.session_date >= datetime.now() - timedelta(days=30)
    ).all()
    
    if len(recent_sessions) < 10:
        return None
    
    # Calculate current performance metrics
    predictions = [predict_session(session) for session in recent_sessions]
    actuals = [session.actual_duration for session in recent_sessions]
    
    current_mae = mean_absolute_error(actuals, predictions)
    current_rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    # Compare with baseline metrics
    baseline_mae = 0.99  # From initial training
    
    if current_mae > baseline_mae * 1.2:
        # Performance degraded, trigger retraining
        send_alert("Model performance degraded", {
            'current_mae': current_mae,
            'baseline_mae': baseline_mae
        })
        
        return trigger_retraining()
    
    return {
        'status': 'healthy',
        'mae': current_mae,
        'rmse': current_rmse
    }
```

### Automated Retraining
```python
def schedule_retraining():
    """Retrain model when new data is available"""
    
    # Check if enough new data is available
    last_training = get_last_training_date()
    new_sessions = StudySession.query.filter(
        StudySession.session_date > last_training
    ).count()
    
    if new_sessions >= 50:  # Retrain with 50+ new samples
        print(f"Retraining with {new_sessions} new samples")
        
        # Retrain model
        new_model = train_model()
        
        # Validate performance
        if validate_new_model(new_model):
            deploy_model(new_model)
            update_last_training_date()
        else:
            print("New model performance insufficient, keeping current model")
```

---

## Troubleshooting

### Common Training Issues

#### 1. Overfitting
**Symptoms**: Training loss decreases but validation loss increases
**Solutions**:
```python
# Increase dropout
model.add(Dropout(0.4))

# Add L2 regularization
Dense(64, activation='relu', kernel_regularizer=l2(0.01))

# Reduce model complexity
# Use fewer layers or neurons

# Early stopping
EarlyStopping(patience=10, restore_best_weights=True)
```

#### 2. Underfitting
**Symptoms**: Both training and validation loss plateau at high values
**Solutions**:
```python
# Increase model capacity
Dense(256, activation='relu')  # More neurons

# Add more layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# Reduce regularization
Dropout(0.1)  # Lower dropout rate

# More training epochs
model.fit(epochs=300)
```

#### 3. Poor Convergence
**Symptoms**: Loss oscillates or doesn't decrease
**Solutions**:
```python
# Reduce learning rate
Adam(learning_rate=0.0001)

# Learning rate scheduling
ReduceLROnPlateau(factor=0.5, patience=5)

# Gradient clipping
model.compile(optimizer=Adam(clipnorm=1.0))

# Better initialization
Dense(64, kernel_initializer='he_normal')
```

### Data Quality Issues

#### 1. Insufficient Data
```python
# Check data distribution
print(f"Training samples: {len(X_train)}")
print(f"Features per sample: {X_train.shape[1]}")
print(f"Target distribution: {y_train.describe()}")

# Generate more synthetic data if needed
if len(X_train) < 100:
    augmented_data = generate_synthetic_data(X_train, y_train)
```

#### 2. Feature Scaling Issues
```python
# Check feature scales
print("Feature statistics:")
print(pd.DataFrame(X_train, columns=feature_names).describe())

# Ensure proper scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

#### 3. Target Variable Issues
```python
# Check target distribution
plt.hist(y_train, bins=20)
plt.title("Target Variable Distribution")

# Remove outliers if necessary
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
outliers = (y_train < Q1 - 1.5*IQR) | (y_train > Q3 + 1.5*IQR)
print(f"Outliers detected: {outliers.sum()}")
```

This comprehensive guide covers all aspects of the machine learning model from architecture to deployment and maintenance. The model can be continuously improved as more data becomes available.