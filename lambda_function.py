import json
import os
import boto3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Initialize S3 client
s3 = boto3.client('s3')

def load_model_from_s3():
    """Load model files from S3"""
    bucket = os.environ['BUCKET_NAME']
    
    # Download model files
    s3.download_file(bucket, 'model/model.joblib', '/tmp/model.joblib')
    s3.download_file(bucket, 'model/scaler.joblib', '/tmp/scaler.joblib')
    s3.download_file(bucket, 'model/song_metadata.csv', '/tmp/song_metadata.csv')
    
    # Load the files
    model = joblib.load('/tmp/model.joblib')
    scaler = joblib.load('/tmp/scaler.joblib')
    metadata = pd.read_csv('/tmp/song_metadata.csv')
    
    return model, scaler, metadata

def lambda_handler(event, context):
    try:
        # Load model and data
        model, scaler, metadata = load_model_from_s3()
        
        # Get input features
        features = ['energy', 'tempo', 'danceability', 'loudness', 
                   'liveness', 'valence', 'speechiness', 'acousticness']
        
        # Get song features from input
        song_features = np.array([event[feature] for feature in features]).reshape(1, -1)
        song_features_scaled = scaler.transform(song_features)
        
        # Calculate similarities
        similarities = cosine_similarity(song_features_scaled, model)
        
        # Get top 5 similar songs
        top_indices = np.argsort(similarities[0])[-5:][::-1]
        top_similarities = similarities[0][top_indices]
        
        # Prepare response
        recommendations = []
        for idx, similarity in zip(top_indices, top_similarities):
            song = metadata.iloc[idx]
            recommendations.append({
                'track_name': song['track_name'],
                'artist': song['track_artist'],
                'album': song['track_album_name'],
                'popularity': int(song['track_popularity']),
                'genre': song['playlist_genre'],
                'similarity_score': float(similarity)
            })
        
        return {
            'statusCode': 200,
            'body': json.dumps(recommendations)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        } 