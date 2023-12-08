# Importing necessary libraries
import random
import pandas as pd
import pickle
from PIL import Image
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Content-based recommendation function
def get_content_based_recommendations(track_name, your_dataset, cosine_sim):
    if track_name not in your_dataset['Track Name'].values:
        print(f"Track '{track_name}' not found in the dataset.")
        return []

    idx = your_dataset[your_dataset['Track Name'] == track_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 recommendations

    # Extract only the indices for shuffling
    track_indices = [i[0] for i in sim_scores]

    # Shuffle the indices
    random.shuffle(track_indices)

    # Get the shuffled recommendations
    recommendations = your_dataset['Track Name'].iloc[track_indices]

    # Display accuracy and recommended genre
    accuracy = sim_scores[0][1]  # Cosine similarity of the top recommendation
    recommended_genre = your_dataset['Genre'].iloc[track_indices[0]]

    print(f"\033[91mAccuracy: {accuracy:.2f}\033[0m")  # Red color for accuracy
    print(f"\033[94mRecommended Genre: {recommended_genre}\033[0m")  # Blue color for recommended genre

    return recommendations

# Load the content-based recommendation model
with open('my_model.pkl', 'rb') as file:
    content_based_model = pickle.load(file)

# Load your dataset (replace 'your_dataset.csv' with your actual dataset file)
your_dataset = pd.read_csv('shuffled_data.csv')

# Calculate or load the cosine similarity matrix
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(your_dataset['Genre'])
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Set Streamlit app title and page configuration
st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styles for a more attractive appearance
st.markdown("""
    <style>
        /* Your custom styles here */
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.markdown('<div class="stTitle"><img src="https://cdn.iconscout.com/icon/free/png-512/free-spotify-11-432546.png?f=webp&w=64"> Spotify Genere Based Song Recommender</div>', unsafe_allow_html=True)

# User input for song selection using a dropdown in the sidebar
track_to_recommend = st.sidebar.selectbox("Select a song", your_dataset['Track Name'])

# Additional features
num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=3)
show_preview = st.sidebar.checkbox("Show Recommendations", value=True)
shuffle_recommendations = st.sidebar.checkbox("Shuffle Recommendations", value=False)
show_artist_info = st.sidebar.checkbox("Show Artist Information", value=True)
show_genre = st.sidebar.checkbox("Show Genre", value=True)

# Button to trigger recommendations in the sidebar
if st.sidebar.button("Get Recommendations"):
    # Get content-based recommendations
    content_based_recommendations = get_content_based_recommendations(track_to_recommend, your_dataset, cosine_sim)

    # Shuffle recommendations if the option is selected
    if shuffle_recommendations:
        content_based_recommendations = content_based_recommendations.sample(frac=1)

    # Display recommended songs
    st.subheader("Recommended Songs:")

    # Display song cover images and previews for the selected number of recommendations
    for recommended_track in content_based_recommendations.head(num_recommendations):
        track_info = your_dataset[your_dataset['Track Name'] == recommended_track].iloc[0]
        cover_image_url = track_info['Cover Image']
        preview_url = track_info['Audio Preview']

        # Get additional information like similarity score and genre
        idx = your_dataset[your_dataset['Track Name'] == recommended_track].index[0]
        genre = your_dataset['Genre'][idx]

        # Create a container for each recommendation
        container = st.container()

        # Display cover image or audio preview based on the user's choice
        if show_preview:
            col1, col2, col3, col4, col5 = container.columns([1, 2, 2, 2, 2])

            # Apply CSS styling to center text
            col1.markdown("<style> div.textInput {text-align: center;}</style>", unsafe_allow_html=True)
            col2.markdown("<style> div.textInput {text-align: center;}</style>", unsafe_allow_html=True)
            col3.markdown("<style> div.textInput {text-align: center;}</style>", unsafe_allow_html=True)
            col4.markdown("<style> div.textInput {text-align: center;}</style>", unsafe_allow_html=True)
            col5.markdown("<style> div.textInput {text-align: center;}</style>", unsafe_allow_html=True)

            cover_image = Image.open(requests.get(cover_image_url, stream=True).raw)
            col1.image(cover_image, caption="", width=90)  # Adjust the width as needed

            # Display song name, artist, and additional information in the second column
            col2.write(f"**Song:** {recommended_track}")

            # Display artist information if the option is selected
            if show_artist_info:
                col3.write(f"**Artist(s):** {track_info['Artist(s)']}")

            # Display the genre in the third column
            if show_genre:
                col4.write(f"**Genre:** {genre}")

            col5.audio(preview_url, format='audio/mp3', start_time=0)
