import streamlit as st
from music_recommender import MusicRecommender
import os
from dotenv import load_dotenv
from streamlit_player import st_player
import pandas as pd

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Advanced Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# Initialize the recommender
@st.cache_resource
def load_recommender():
    return MusicRecommender('spotifymusic.csv')

recommender = load_recommender()

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .song-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .album-art {
        width: 200px;
        height: 200px;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .play-button {
        background-color: #1DB954;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        cursor: pointer;
        margin-top: 0.5rem;
    }
    .spotify-link {
        color: #1DB954;
        text-decoration: none;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("🎵 Advanced Music Recommender")
st.write("""
Discover new music based on your favorite songs, mood, or genre! This recommender system uses
audio features like energy, tempo, danceability, and more to find similar tracks.
""")

def display_song(song, show_play_button=True):
    """Display a song card with all information and controls"""
    with st.container():
        # Get Spotify information
        spotify_info = recommender.get_song_info(song['track_name'], song['artist'])
        
        # Create song card
        st.markdown(f"""
            <div class="song-card">
                <h3>{song['track_name']}</h3>
                <p>by {song['artist']}</p>
                <p>Album: {song['album']}</p>
                <p>Genre: {song['genre']}</p>
                <p>Duration: {song['duration']}s</p>
                <p>Popularity: {song['popularity']}/100</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display album art if available
        if spotify_info and spotify_info.get('album_art'):
            st.image(spotify_info['album_art'], width=200)
        
        # Display preview player if available
        if spotify_info and spotify_info.get('preview_url'):
            st_player(spotify_info['preview_url'])
        
        # Add play button if Spotify is connected
        if show_play_button and spotify_info and spotify_info.get('uri'):
            if st.button("▶️ Play on Spotify", key=f"play_{song['track_name']}"):
                if recommender.play_song(spotify_info['uri']):
                    st.success("Playing on your Spotify device!")
                else:
                    st.error("Could not play song. Make sure you have an active Spotify device.")
        
        # Add Spotify link if available
        if spotify_info and spotify_info.get('spotify_url'):
            st.markdown(f"""
                <a href="{spotify_info['spotify_url']}" target="_blank" class="spotify-link">
                    Open in Spotify
                </a>
            """, unsafe_allow_html=True)

# Sidebar for filters and random songs
with st.sidebar:
    st.header("🎛️ Filters")
    
    # Genre selection with statistics
    genre_info = recommender.get_available_genres()
    genres = genre_info['genres']
    genre_counts = genre_info['counts']
    
    # Create genre selection with counts
    genre_options = ["All"] + [f"{genre} ({count} songs)" for genre, count in genre_counts.items()]
    selected_genre = st.selectbox("Select Genre", genre_options)
    
    # Extract actual genre name from selection
    actual_genre = selected_genre.split(" ")[0] if selected_genre != "All" else None
    
    # Popularity filter
    min_popularity = st.slider("Minimum Popularity", 0, 100, 0)
    
    # Mood selection
    mood = st.selectbox("Select Mood", ["All", "Happy", "Sad", "Energetic", "Calm"])
    
    st.header("🎲 Random Songs")
    if st.button("Get Random Songs", key="random_button"):
        try:
            random_songs = recommender.get_random_songs(
                num_songs=5,
                min_popularity=min_popularity,
                genre=actual_genre
            )
            st.subheader("Random Song Suggestions")
            for song in random_songs:
                display_song(song)
        except Exception as e:
            st.error(f"Error loading random songs: {str(e)}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Find Similar Songs")
    
    # Show available songs
    with st.expander("📝 Click here to see available songs"):
        # Get all unique songs with artists
        available_songs = pd.DataFrame({
            'Song': recommender.data['track_name'],
            'Artist': recommender.data['track_artist'],
            'Genre': recommender.data['playlist_genre']
        }).sort_values('Song')
        
        # Display as a searchable table
        st.dataframe(
            available_songs,
            column_config={
                "Song": "Song Name",
                "Artist": "Artist",
                "Genre": "Genre"
            },
            hide_index=True,
        )
    
    # Song input with clear button
    song_name = st.text_input(
        "Enter a song name from the dataset:", 
        placeholder="e.g., Die With A Smile", 
        key="song_input",
        help="Type a song name from the dataset above"
    )
    
    if song_name:
        try:
            # Get recommendations
            result = recommender.get_recommendations(song_name, min_popularity=min_popularity)
            
            if result['error']:
                st.error(result['message'])
                st.write("Try one of these:")
                
                # Create columns for suggestions
                cols = st.columns(min(3, len(result['suggestions'])))
                for i, suggestion in enumerate(result['suggestions']):
                    with cols[i % len(cols)]:
                        if st.button(suggestion, key=f"suggestion_{suggestion}"):
                            st.session_state.song_name = suggestion
                            st.experimental_rerun()
            else:
                # Show the input song
                st.success(f"Found song: **{result['input_song']['track_name']}**")
                display_song(result['input_song'])
                
                st.write("---")
                
                # Show recommendations
                st.subheader("Similar Songs")
                for song in result['recommendations']:
                    display_song(song)
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")

with col2:
    st.header("🎯 Recommendations")
    
    # Indian Music Section
    st.subheader("🇮🇳 Indian Music")
    if st.button("Get Indian Music Recommendations", key="indian_button"):
        try:
            indian_songs = recommender.get_indian_music_recommendations(
                num_recommendations=5,
                min_popularity=min_popularity
            )
            if indian_songs:
                for song in indian_songs:
                    display_song(song)
            else:
                st.warning("No Indian songs found matching the criteria.")
        except Exception as e:
            st.error(f"Error getting Indian music recommendations: {str(e)}")
    
    # Mood-Based Recommendations
    st.subheader("😊 Mood-Based")
    if mood != "All":
        try:
            mood_songs = recommender.get_mood_based_recommendations(mood)
            if mood_songs:
                for song in mood_songs:
                    display_song(song)
            else:
                st.warning(f"No {mood} songs found matching the criteria.")
        except Exception as e:
            st.error(f"Error getting mood-based recommendations: {str(e)}")

# Footer
st.write("---")
st.write("Made with ❤️ using Spotify data") 