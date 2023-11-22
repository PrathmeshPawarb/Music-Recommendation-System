# Install necessary libraries
# pip install spotipy streamlit Pillow

import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Set up your Spotify API credentials
client_id = '2ac1f721087b49cd8458a8f33a50703a'
client_secret = '85c41bf2ac4e43efa3a384d6e9197abd'

# Authenticate with the Spotify API
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

def search_songs(query):
    results = sp.search(q=query, type='track', limit=20)
    return results['tracks']['items']

def recommend_songs(mood, tempo=None, atmosphere=None, religion=None):
    if mood == 'Yoga':
        query = f'yoga {tempo} music'
    elif mood == 'Meditation':
        query = f'meditation {atmosphere} music'
    elif mood == 'Devotional':
        query = f'devotional {religion} music'
    else:
        query = f'{mood} music'

    recommended_songs = search_songs(query)

    return recommended_songs

def display_recommendations(recommended_songs):
    st.subheader("Recommended Songs:")
    for i, song in enumerate(recommended_songs, start=1):
        # Display song information
        st.write(
            f"{i}. "
            f"<span style='color:red; font-weight:bold;'>{song['name']}</span> "
            f"by {', '.join([artist['name'] for artist in song['artists']])}",
            unsafe_allow_html=True
        )

        # Create a layout with two columns
        col1, col2 = st.columns(2)

        # Display album cover image with a smaller size in the first column
        image_url = song['album']['images'][0]['url']
        image = Image.open(BytesIO(requests.get(image_url).content))
        col1.image(image, caption=f"Album Cover: {song['name']}", width=100)

        # Display audio preview and add a Play button in the second column
        col2.audio(song['preview_url'], format='audio/mp3', start_time=0)


def main():
    st.title("Yoga, Meditation, and Devotional Songs Recommendation System")

    # Initialize variables
    tempo = atmosphere = religion = None

    # Get user input
    mood_options = ['Select Mood', 'Yoga', 'Meditation', 'Devotional']
    mood = st.selectbox("Select Mood:", mood_options)

    if mood != 'Select Mood':
        if mood == 'Yoga':
            tempo_options = ['Select Tempo', 'Slow', 'Moderate', 'Fast']
            tempo = st.selectbox("Select Tempo:", tempo_options)
        elif mood == 'Meditation':
            atmosphere_options = ['Select Atmosphere', 'Calm', 'Serene', 'Mystical']
            atmosphere = st.selectbox("Select Atmosphere:", atmosphere_options)
        elif mood == 'Devotional':
            religion_options = ['Select Religion', 'Hindu', 'Christian', 'Buddhist', 'Other']
            religion = st.selectbox("Select Religion:", religion_options)

        if st.button("Recommend"):
            # Get song recommendations based on user input
            recommended_songs = recommend_songs(mood, tempo, atmosphere, religion)

            # Display recommendations
            display_recommendations(recommended_songs)

if __name__ == "__main__":
    main()
