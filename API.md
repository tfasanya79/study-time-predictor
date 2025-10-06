# API Documentation - Study Time Predictor

## Overview

The Study Time Predictor provides both web interface routes and API endpoints for programmatic access to predictions and data.

## Authentication

Currently, no authentication is required. This is suitable for development and demo purposes. For production deployment, consider implementing authentication mechanisms.

## Base URL

```
http://localhost:5000
```

---

## Web Interface Routes

### Home Page
```
GET /
```
**Description**: Main dashboard with statistics and recent activity

**Response**: HTML page with overview statistics

---

### Prediction Interface
```
GET /predict
```
**Description**: Interactive form for making study time predictions

**Response**: HTML form with student/topic dropdowns and context inputs

---

### Students Management
```
GET /students
```
**Description**: List all students with their profiles

**Response**: HTML page with student cards

```
GET /student/<int:student_id>
```
**Description**: Detailed view of specific student with session history

**Parameters**: 
- `student_id` (integer): Student ID

**Response**: HTML page with student details and performance metrics

---

### Topics Management
```
GET /topics
```
**Description**: Browse all available study topics

**Response**: HTML page with filterable topic cards

```
GET /topic/<int:topic_id>
```
**Description**: Detailed view of specific topic with student performance data

**Parameters**:
- `topic_id` (integer): Topic ID

**Response**: HTML page with topic details and analytics

---

### Analytics Dashboard
```
GET /analytics
```
**Description**: Comprehensive analytics with charts and insights

**Response**: HTML page with interactive Chart.js visualizations

---

## API Endpoints

### Make Prediction
```
POST /api/predict
```

**Description**: Generate study time prediction using the neural network model

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "student_id": 1,
  "topic_id": 5,
  "time_of_day": "morning",
  "study_method": "reading",
  "environment": "library",
  "energy_level": 4,
  "distraction_level": 2,
  "breaks_taken": 1
}
```

**Request Parameters**:
- `student_id` (integer, required): ID of the student
- `topic_id` (integer, required): ID of the topic
- `time_of_day` (string, required): "morning", "afternoon", "evening", "night"
- `study_method` (string, required): "reading", "practice", "video", "discussion", "flashcards"
- `environment` (string, required): "home", "library", "cafe", "classroom", "outdoor"
- `energy_level` (integer, required): 1-5 scale
- `distraction_level` (integer, required): 1-5 scale  
- `breaks_taken` (integer, required): Expected number of breaks

**Response**:
```json
{
  "predicted_hours": 2.45,
  "display_time": "2h 27m",
  "student_name": "Alex Johnson",
  "topic_name": "Linear Algebra Basics",
  "confidence": "High"
}
```

**Error Response**:
```json
{
  "error": "Student or topic not found"
}
```

**Status Codes**:
- `200`: Success
- `404`: Student or topic not found
- `500`: Internal server error

---

### Get Student Data
```
GET /api/student/<int:student_id>
```

**Description**: Retrieve student information in JSON format

**Parameters**:
- `student_id` (integer): Student ID

**Response**:
```json
{
  "id": 1,
  "name": "Alex Johnson",
  "email": "alex.johnson@university.edu",
  "age": 22,
  "education_level": "undergraduate",
  "learning_style": "visual",
  "avg_focus_duration": 75.5,
  "created_at": "2025-10-06T12:32:08"
}
```

---

### Get Topic Data
```
GET /api/topic/<int:topic_id>
```

**Description**: Retrieve topic information in JSON format

**Parameters**:
- `topic_id` (integer): Topic ID

**Response**:
```json
{
  "id": 1,
  "name": "Linear Algebra Basics",
  "subject": "Mathematics",
  "difficulty_level": 6,
  "estimated_base_time": 3.0,
  "prerequisites_count": 2,
  "content_type": "mixed"
}
```

---

### Model Management
```
GET /train-model
```

**Description**: Retrain the machine learning model with current data

**Response**: Redirects to home page with flash message about training status

**Note**: This is a web route that triggers model retraining. Training can take several minutes.

---

## Data Models

### Student Model
```json
{
  "id": integer,
  "name": string,
  "email": string,
  "age": integer,
  "education_level": "high_school" | "undergraduate" | "graduate",
  "learning_style": "visual" | "auditory" | "kinesthetic" | "reading",
  "avg_focus_duration": float,
  "created_at": datetime
}
```

### Topic Model
```json
{
  "id": integer,
  "name": string,
  "subject": string,
  "difficulty_level": integer (1-10),
  "estimated_base_time": float,
  "prerequisites_count": integer,
  "content_type": "reading" | "video" | "practice" | "mixed"
}
```

### Study Session Model
```json
{
  "id": integer,
  "student_id": integer,
  "topic_id": integer,
  "planned_duration": float,
  "actual_duration": float,
  "session_date": datetime,
  "time_of_day": string,
  "breaks_taken": integer,
  "distraction_level": integer (1-5),
  "energy_level": integer (1-5),
  "comprehension_score": float (0-100),
  "completion_percentage": float (0-100),
  "satisfaction_rating": integer (1-5),
  "study_method": string,
  "environment": string
}
```

---

## Usage Examples

### JavaScript (Fetch API)
```javascript
// Make a prediction
const predictionData = {
  student_id: 1,
  topic_id: 5,
  time_of_day: "morning",
  study_method: "reading",
  environment: "library",
  energy_level: 4,
  distraction_level: 2,
  breaks_taken: 1
};

fetch('/api/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(predictionData)
})
.then(response => response.json())
.then(data => {
  console.log('Predicted time:', data.display_time);
})
.catch(error => {
  console.error('Error:', error);
});
```

### Python (Requests)
```python
import requests

# Make a prediction
url = "http://localhost:5000/api/predict"
data = {
    "student_id": 1,
    "topic_id": 5,
    "time_of_day": "morning",
    "study_method": "reading",
    "environment": "library",
    "energy_level": 4,
    "distraction_level": 2,
    "breaks_taken": 1
}

response = requests.post(url, json=data)
result = response.json()
print(f"Predicted time: {result['display_time']}")
```

### cURL
```bash
# Make a prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": 1,
    "topic_id": 5,
    "time_of_day": "morning",
    "study_method": "reading",
    "environment": "library",
    "energy_level": 4,
    "distraction_level": 2,
    "breaks_taken": 1
  }'
```

---

## Error Handling

All API endpoints return appropriate HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid input)
- **404**: Not Found (student/topic doesn't exist)
- **500**: Internal Server Error

Error responses include a JSON object with an `error` field describing the issue:

```json
{
  "error": "Description of the error"
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing rate limiting to prevent abuse.

---

## CORS Support

CORS is not explicitly configured. If you need cross-origin requests, you may need to configure Flask-CORS.

---

## Versioning

This is version 1.0 of the API. Future versions will maintain backward compatibility where possible.