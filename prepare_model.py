import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def prepare_model():
    # Load the dataset
    data = pd.read_csv('spotifymusic.csv')
    
    # Define features
    features = ['energy', 'tempo', 'danceability', 'loudness', 
               'liveness', 'valence', 'speechiness', 'acousticness']
    
    # Scale the features
    scaler = StandardScaler()
    song_features = scaler.fit_transform(data[features])
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Save the model and scaler
    joblib.dump(song_features, 'model/model.joblib')
    joblib.dump(scaler, 'model/scaler.joblib')
    
    # Save song metadata
    metadata = data[['track_name', 'track_artist', 'track_album_name', 
                    'track_popularity', 'playlist_genre', 'duration_ms']]
    metadata.to_csv('model/song_metadata.csv', index=False)
    
    print("Model preparation complete. Files saved in 'model' directory.")

if __name__ == '__main__':
    prepare_model() 