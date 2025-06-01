import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Define the model directory for SageMaker
MODEL_DIR = os.environ.get('SM_MODEL_DIR', '.')
DATA_DIR = os.environ.get('SM_CHANNEL_TRAINING', '.')

def model_fn(model_dir):
    """Load the model and scaler from the model directory"""
    model_path = os.path.join(model_dir, 'model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions using the loaded model"""
    model, scaler = model
    features = ['energy', 'tempo', 'danceability', 'loudness', 
               'liveness', 'valence', 'speechiness', 'acousticness']
    
    # Get song features
    song_features = np.array([input_data[feature] for feature in features]).reshape(1, -1)
    song_features_scaled = scaler.transform(song_features)
    
    # Calculate similarities
    similarities = cosine_similarity(song_features_scaled, model)
    
    # Get top 5 similar songs
    top_indices = np.argsort(similarities[0])[-5:][::-1]
    top_similarities = similarities[0][top_indices]
    
    return {
        'indices': top_indices.tolist(),
        'similarities': top_similarities.tolist()
    }

def output_fn(prediction, content_type):
    """Format the prediction output"""
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# For local testing
if __name__ == "__main__":
    from music_recommender import MusicRecommender
    
    # Initialize and save the model
    print("Initializing model...")
    recommender = MusicRecommender('spotifymusic.csv')
    
    # Save the model
    print("Saving model...")
    joblib.dump(recommender, os.path.join(MODEL_DIR, 'music_recommender.joblib'))
    
    # Test the endpoint
    print("Testing endpoint...")
    model = model_fn(MODEL_DIR)
    
    # Test song recommendations
    test_input = {'song_name': 'Die With A Smile'}
    parsed_input = input_fn(json.dumps(test_input), 'application/json')
    prediction = predict_fn(parsed_input, model)
    output = output_fn(prediction, 'application/json')
    print("Test output:", output) 