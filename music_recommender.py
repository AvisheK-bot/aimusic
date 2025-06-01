import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv

class MusicRecommender:
    def __init__(self, data_path):
        try:
            self.data = pd.read_csv(data_path)
            print(f"Loaded {len(self.data)} songs from dataset")
            
            # Ensure all required columns exist
            required_columns = ['energy', 'tempo', 'danceability', 'loudness', 
                              'liveness', 'valence', 'speechiness', 'acousticness',
                              'track_name', 'track_artist', 'track_album_name', 
                              'track_popularity', 'playlist_genre', 'duration_ms']
            
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Print first few song names for verification
            print("Sample of songs in dataset:")
            print(self.data['track_name'].head())
            
            self.features = ['energy', 'tempo', 'danceability', 'loudness', 
                           'liveness', 'valence', 'speechiness', 'acousticness']
            self.scaler = StandardScaler()
            self.song_features = None
            self._prepare_data()
            
            # Initialize Spotify client
            self.sp = None
            self._init_spotify()
        except Exception as e:
            raise Exception(f"Error initializing MusicRecommender: {str(e)}")
    
    def _init_spotify(self):
        """Initialize Spotify client with credentials"""
        try:
            load_dotenv()
            # Get credentials from environment variables
            client_id = os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
            redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8501')
            
            if client_id and client_secret:
                auth_manager = SpotifyOAuth(
                    client_id=client_id,
                    client_secret=client_secret,
                    redirect_uri=redirect_uri,
                    scope='user-library-read user-read-playback-state user-modify-playback-state'
                )
                self.sp = spotipy.Spotify(auth_manager=auth_manager)
        except Exception as e:
            print(f"Warning: Could not initialize Spotify client: {str(e)}")
            self.sp = None
    
    def get_song_info(self, track_name, artist_name):
        """Get complete Spotify information for a song"""
        if not self.sp:
            return None
            
        try:
            # Search for the track
            results = self.sp.search(q=f'track:{track_name} artist:{artist_name}', type='track', limit=1)
            
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                return {
                    'preview_url': track.get('preview_url'),
                    'spotify_url': track.get('external_urls', {}).get('spotify'),
                    'uri': track.get('uri'),
                    'album_art': track.get('album', {}).get('images', [{}])[0].get('url'),
                    'duration_ms': track.get('duration_ms'),
                    'popularity': track.get('popularity'),
                    'explicit': track.get('explicit')
                }
            return None
        except Exception as e:
            print(f"Error getting song info: {str(e)}")
            return None
    
    def play_song(self, track_uri):
        """Play a song on the user's active Spotify device"""
        if not self.sp:
            return False
            
        try:
            # Get user's active devices
            devices = self.sp.devices()
            if not devices['devices']:
                print("No active Spotify devices found")
                return False
            
            # Play the song on the first available device
            self.sp.start_playback(device_id=devices['devices'][0]['id'], uris=[track_uri])
            return True
        except Exception as e:
            print(f"Error playing song: {str(e)}")
            return False
    
    def _prepare_data(self):
        # Scale the features
        self.song_features = self.scaler.fit_transform(self.data[self.features])
    
    def get_mood_based_recommendations(self, mood, num_recommendations=5):
        """Get recommendations based on mood (happy, sad, energetic, calm)"""
        mood_filters = {
            'happy': {'valence': (0.7, 1.0), 'energy': (0.6, 1.0)},
            'sad': {'valence': (0.0, 0.3), 'energy': (0.0, 0.4)},
            'energetic': {'energy': (0.8, 1.0), 'tempo': (120, 200)},
            'calm': {'energy': (0.0, 0.3), 'acousticness': (0.7, 1.0)}
        }
        
        if mood.lower() not in mood_filters:
            return None
        
        filters = mood_filters[mood.lower()]
        filtered_data = self.data.copy()
        
        for feature, (min_val, max_val) in filters.items():
            filtered_data = filtered_data[
                (filtered_data[feature] >= min_val) & 
                (filtered_data[feature] <= max_val)
            ]
        
        if len(filtered_data) == 0:
            return None
        
        # Get random songs from filtered data
        random_indices = np.random.choice(len(filtered_data), 
                                        min(num_recommendations, len(filtered_data)), 
                                        replace=False)
        
        recommendations = []
        for idx in random_indices:
            song = filtered_data.iloc[idx]
            recommendations.append({
                'track_name': str(song['track_name']),
                'artist': str(song['track_artist']),
                'album': str(song['track_album_name']),
                'popularity': int(song['track_popularity']),
                'genre': str(song['playlist_genre']),
                'duration': int(song['duration_ms']) // 1000  # Convert to seconds
            })
        
        return recommendations

    def get_available_genres(self):
        """Get list of available genres in the dataset with counts"""
        genre_counts = self.data['playlist_genre'].value_counts()
        return {
            'genres': sorted(self.data['playlist_genre'].unique().tolist()),
            'counts': genre_counts.to_dict()
        }

    def get_genre_recommendations(self, genre, num_recommendations=5):
        """Get recommendations from a specific genre"""
        # Convert genre to lowercase for case-insensitive matching
        genre = genre.lower()
        
        # Find all matching genres (case-insensitive)
        matching_genres = [g for g in self.data['playlist_genre'].unique() 
                         if g.lower() == genre]
        
        if not matching_genres:
            # Try partial matching
            matching_genres = [g for g in self.data['playlist_genre'].unique() 
                             if genre in g.lower()]
        
        if not matching_genres:
            return None
        
        # Get all songs from matching genres
        genre_data = self.data[self.data['playlist_genre'].isin(matching_genres)]
        
        if len(genre_data) == 0:
            return None
        
        # Get random songs from genre
        random_indices = np.random.choice(len(genre_data), 
                                        min(num_recommendations, len(genre_data)), 
                                        replace=False)
        
        recommendations = []
        for idx in random_indices:
            song = genre_data.iloc[idx]
            recommendations.append({
                'track_name': str(song['track_name']),
                'artist': str(song['track_artist']),
                'album': str(song['track_album_name']),
                'popularity': int(song['track_popularity']),
                'genre': str(song['playlist_genre']),
                'duration': int(song['duration_ms']) // 1000  # Convert to seconds
            })
        
        return recommendations

    def get_random_songs(self, num_songs=5, min_popularity=0, genre=None):
        try:
            # Filter by popularity and genre if specified
            filtered_data = self.data.copy()
            if min_popularity > 0:
                filtered_data = filtered_data[filtered_data['track_popularity'] >= min_popularity]
            if genre:
                filtered_data = filtered_data[filtered_data['playlist_genre'].str.lower() == genre.lower()]
            
            # Ensure we don't try to get more songs than available
            num_songs = min(num_songs, len(filtered_data))
            
            # Get random songs from the filtered dataset
            random_indices = np.random.choice(len(filtered_data), num_songs, replace=False)
            random_songs = []
            
            for idx in random_indices:
                song = filtered_data.iloc[idx]
                random_songs.append({
                    'track_name': str(song['track_name']),
                    'artist': str(song['track_artist']),
                    'album': str(song['track_album_name']),
                    'popularity': int(song['track_popularity']),
                    'genre': str(song['playlist_genre']),
                    'duration': int(song['duration_ms']) // 1000  # Convert to seconds
                })
            
            return random_songs
        except Exception as e:
            print(f"Error in get_random_songs: {str(e)}")
            return [{
                'track_name': 'Error loading songs',
                'artist': 'System Error',
                'album': 'Please try again',
                'popularity': 0,
                'genre': 'Unknown',
                'duration': 0
            }]

    def get_recommendations(self, song_name, num_recommendations=5, min_popularity=0):
        try:
            # Convert song name to lowercase for case-insensitive search
            song_name = song_name.lower().strip()
            print(f"Searching for song: '{song_name}'")
            
            # First try exact match
            song_idx = self.data[self.data['track_name'].str.lower() == song_name].index
            print(f"Exact matches found: {len(song_idx)}")
            
            if len(song_idx) == 0:
                # Try partial match in song names
                song_idx = self.data[self.data['track_name'].str.lower().str.contains(song_name, na=False)].index
                print(f"Partial matches found: {len(song_idx)}")
                
                if len(song_idx) == 0:
                    # Try matching in artist names
                    song_idx = self.data[self.data['track_artist'].str.lower().str.contains(song_name, na=False)].index
                    print(f"Artist matches found: {len(song_idx)}")
                    
                    if len(song_idx) == 0:
                        # Try fuzzy matching
                        all_songs = self.data['track_name'].str.lower().tolist()
                        similar_songs = get_close_matches(song_name, all_songs, n=5, cutoff=0.6)
                        print(f"Similar songs found: {similar_songs}")
                        
                        if similar_songs:
                            # Try to find the most similar song
                            for similar_song in similar_songs:
                                song_idx = self.data[self.data['track_name'].str.lower() == similar_song].index
                                if len(song_idx) > 0:
                                    break
                        
                        if len(song_idx) == 0:
                            # If still no match, return suggestions
                            return {
                                'error': True,
                                'message': f"Song '{song_name}' not found. Did you mean:",
                                'suggestions': similar_songs[:3] if similar_songs else []
                            }
            
            song_idx = song_idx[0]
            found_song = self.data.iloc[song_idx]
            print(f"Found song: {found_song['track_name']} by {found_song['track_artist']}")
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity([self.song_features[song_idx]], self.song_features)[0]
            
            # Get top N recommendations (excluding the input song)
            top_indices = np.argsort(similarity_scores)[-num_recommendations-1:-1][::-1]
            
            recommendations = []
            for idx in top_indices:
                song = self.data.iloc[idx]
                try:
                    popularity = int(song['track_popularity'])
                except:
                    popularity = 0
                    
                if popularity >= min_popularity:
                    recommendations.append({
                        'track_name': str(song['track_name']),
                        'artist': str(song['track_artist']),
                        'album': str(song['track_album_name']),
                        'similarity_score': float(similarity_scores[idx]),
                        'popularity': popularity,
                        'genre': str(song['playlist_genre']),
                        'duration': int(song['duration_ms']) // 1000  # Convert to seconds
                    })
            
            print(f"Found {len(recommendations)} recommendations")
            
            # Get input song details
            input_song = self.data.iloc[song_idx]
            try:
                input_popularity = int(input_song['track_popularity'])
            except:
                input_popularity = 0
                
            return {
                'error': False,
                'recommendations': recommendations,
                'input_song': {
                    'track_name': str(input_song['track_name']),
                    'artist': str(input_song['track_artist']),
                    'album': str(input_song['track_album_name']),
                    'genre': str(input_song['playlist_genre']),
                    'popularity': input_popularity,
                    'duration': int(input_song['duration_ms']) // 1000
                }
            }
        except Exception as e:
            print(f"Error in get_recommendations: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'error': True,
                'message': f"Error getting recommendations: {str(e)}",
                'suggestions': []
            }

    def search_songs(self, query):
        """Search for songs that match the query in title, artist, or album"""
        try:
            query = str(query).lower()
            # Search in track names
            title_matches = self.data[self.data['track_name'].str.lower().str.contains(query)]
            # Search in artist names
            artist_matches = self.data[self.data['track_artist'].str.lower().str.contains(query)]
            # Search in album names
            album_matches = self.data[self.data['track_album_name'].str.lower().str.contains(query)]
            
            # Combine all matches
            all_matches = pd.concat([title_matches, artist_matches, album_matches]).drop_duplicates()
            
            if len(all_matches) > 0:
                return all_matches.head(10)  # Return top 10 matches
            return None
        except Exception as e:
            print(f"Error in search_songs: {str(e)}")
            return None

    def get_indian_music_recommendations(self, num_recommendations=5, min_popularity=0):
        """Get Indian music recommendations using combined approach"""
        try:
            # List of common Indian artist names and words
            indian_indicators = [
                'rahman', 'arijit', 'shreya', 'sonu', 'kumar', 'khan', 'kapoor',
                'bollywood', 'tollywood', 'kollywood', 'mollywood',
                'hindi', 'tamil', 'telugu', 'malayalam', 'bengali', 'punjabi',
                'kannada', 'marathi', 'gujarati', 'bhojpuri'
            ]
            
            # Create a mask for Indian songs
            indian_mask = (
                self.data['track_artist'].str.lower().str.contains('|'.join(indian_indicators)) |
                self.data['track_name'].str.lower().str.contains('|'.join(indian_indicators)) |
                self.data['track_album_name'].str.lower().str.contains('|'.join(indian_indicators)) |
                self.data['playlist_genre'].str.lower().str.contains('indian')
            )
            
            # Filter the data
            indian_songs = self.data[indian_mask]
            
            if len(indian_songs) == 0:
                return None
            
            # Apply popularity filter
            if min_popularity > 0:
                indian_songs = indian_songs[indian_songs['track_popularity'] >= min_popularity]
            
            # Get random songs
            random_indices = np.random.choice(len(indian_songs), 
                                            min(num_recommendations, len(indian_songs)), 
                                            replace=False)
            
            recommendations = []
            for idx in random_indices:
                song = indian_songs.iloc[idx]
                recommendations.append({
                    'track_name': str(song['track_name']),
                    'artist': str(song['track_artist']),
                    'album': str(song['track_album_name']),
                    'popularity': int(song['track_popularity']),
                    'genre': str(song['playlist_genre']),
                    'duration': int(song['duration_ms']) // 1000
                })
            
            return recommendations
        except Exception as e:
            print(f"Error in get_indian_music_recommendations: {str(e)}")
            return None 