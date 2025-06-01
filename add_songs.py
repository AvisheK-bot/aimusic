import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

def setup_spotify():
    """Setup Spotify client"""
    load_dotenv()
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not (client_id and client_secret):
        raise ValueError("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env file")
    
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=auth_manager)

def get_audio_features(sp, track_id):
    """Get audio features for a track"""
    try:
        features = sp.audio_features([track_id])[0]
        return features
    except:
        return None

def search_and_add_songs(search_query, genre=None):
    """Search for songs and add them to the dataset"""
    try:
        # Setup Spotify client
        sp = setup_spotify()
        
        # Load existing dataset
        try:
            df = pd.read_csv('spotifymusic.csv')
        except:
            # Create new dataset if doesn't exist
            df = pd.DataFrame(columns=[
                'track_name', 'track_artist', 'track_album_name', 'track_popularity',
                'playlist_genre', 'duration_ms', 'energy', 'tempo', 'danceability',
                'loudness', 'liveness', 'valence', 'speechiness', 'acousticness'
            ])
        
        # Search for tracks
        results = sp.search(q=search_query, type='track', limit=10)
        
        new_songs = []
        for track in results['tracks']['items']:
            # Get audio features
            features = get_audio_features(sp, track['id'])
            if not features:
                continue
            
            # Create song entry
            song = {
                'track_name': track['name'],
                'track_artist': track['artists'][0]['name'],
                'track_album_name': track['album']['name'],
                'track_popularity': track['popularity'],
                'playlist_genre': genre if genre else 'Unknown',
                'duration_ms': track['duration_ms'],
                'energy': features['energy'],
                'tempo': features['tempo'],
                'danceability': features['danceability'],
                'loudness': features['loudness'],
                'liveness': features['liveness'],
                'valence': features['valence'],
                'speechiness': features['speechiness'],
                'acousticness': features['acousticness']
            }
            
            # Check if song already exists
            if not df[df['track_name'].str.lower() == track['name'].lower()].empty:
                print(f"Song '{track['name']}' already exists in dataset")
                continue
                
            new_songs.append(song)
            print(f"Added: {track['name']} by {track['artists'][0]['name']}")
        
        if new_songs:
            # Add new songs to dataset
            new_df = pd.DataFrame(new_songs)
            df = pd.concat([df, new_df], ignore_index=True)
            
            # Save updated dataset
            df.to_csv('spotifymusic.csv', index=False)
            print(f"\nAdded {len(new_songs)} new songs to the dataset")
        else:
            print("\nNo new songs were added")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    while True:
        print("\nAdd Songs to Dataset")
        print("-------------------")
        search = input("Enter song/artist to search (or 'quit' to exit): ")
        
        if search.lower() == 'quit':
            break
            
        genre = input("Enter genre (or press Enter to skip): ")
        if not genre:
            genre = None
            
        search_and_add_songs(search, genre) 