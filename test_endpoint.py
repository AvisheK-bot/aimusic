import boto3
import json
import pandas as pd

def test_endpoint(endpoint_name):
    # Initialize SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime')
    
    # Load song metadata
    metadata = pd.read_csv('model/song_metadata.csv')
    
    # Test data - using features from the first song
    test_data = {
        'energy': 0.8,
        'tempo': 120,
        'danceability': 0.7,
        'loudness': -5.0,
        'liveness': 0.1,
        'valence': 0.6,
        'speechiness': 0.05,
        'acousticness': 0.2
    }
    
    # Make prediction request
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(test_data)
    )
    
    # Parse response
    result = json.loads(response['Body'].read().decode())
    
    # Get recommended songs
    recommended_songs = []
    for idx, similarity in zip(result['indices'], result['similarities']):
        song = metadata.iloc[idx]
        recommended_songs.append({
            'track_name': song['track_name'],
            'artist': song['track_artist'],
            'album': song['track_album_name'],
            'popularity': song['track_popularity'],
            'genre': song['playlist_genre'],
            'similarity_score': similarity
        })
    
    return recommended_songs

if __name__ == '__main__':
    endpoint_name = 'music-recommender-endpoint'  # Replace with your endpoint name
    recommendations = test_endpoint(endpoint_name)
    print("Recommended Songs:")
    for song in recommendations:
        print(f"\n{song['track_name']} by {song['artist']}")
        print(f"Album: {song['album']}")
        print(f"Genre: {song['genre']}")
        print(f"Popularity: {song['popularity']}")
        print(f"Similarity Score: {song['similarity_score']:.4f}") 