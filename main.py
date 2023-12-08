import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import requests
from recommendation_model import get_content_based_recommendations

# Load the content-based recommendation model
with open('my_model.pkl', 'rb') as file:
    content_based_model = pickle.load(file)

# Load your dataset (replace 'your_dataset.csv' with your actual dataset file)
your_dataset = pd.read_csv('shuffled_data.csv')

# Calculate or load the cosine similarity matrix
# Example using scikit-learn's CountVectorizer and cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
st.markdown(
    """
    <style>
        body {
            background-color: #1E1E1E;  /* Dark background color */
            color: #FFFFFF;  /* White text color */
            font-family: 'Helvetica Neue', sans-serif;  /* Use a clean sans-serif font */
        }
        .stApp {
            background-color: #1E1E1E;  /* Dark background color */
            color: #FFFFFF;  /* White text color */
        }
        .stSelectbox {
            background-color: #292929;  /* Darker selectbox background color */
            color: #FFFFFF;  /* White text color */
        }
        .stSlider, .stCheckbox, .stButton {
            color: #1DB954;  /* Spotify green color */
        }
        .stSlider .stSliderHandle, .stCheckbox input:checked + div::before, .stButton button {
            background-color: #1DB954;  /* Spotify green color */
        }
        .stSlider .stSliderProgressBar {
            background-color: #1DB954;  /* Spotify green color */
        }
        .stDataFrame .dataframe th, .stDataFrame .dataframe td {
            border: 1px solid #535353;  /* Border color */
        }
        .stDataFrame .dataframe th {
            background-color: #535353;  /* Header background color */
            color: #FFFFFF;  /* White text color */
        }
        .stDataFrame .dataframe tbody tr:hover {
            background-color: #434343;  /* Hover background color */
        }
        .stContainer {
            max-width: 1200px;  /* Set max width for the entire app */
        }

        .stSidebar {
            background-color: #121212;  /* Dark sidebar background color */
        }
        .stSidebar label, .stSidebar .stMarkdown, .stSidebar  {
            color: #FFFFFF;  /* White text color in the sidebar */
        }
        .stSidebar .stSelectbox {
            color: #FFFFFF;  /* White text color in the sidebar selectbox */
        }
        .stSidebar .stSlider, .stSidebar .stCheckbox {
            color: #1DB954;  /* Spotify green color in the sidebar */
        }
        .stSidebar .stSlider .stSliderHandle, .stSidebar .stCheckbox input:checked + div::before {
            background-color: #1DB954;  /* Spotify green color in the sidebar */
        }
        .stTitle {
            display: flex;
            align-items: center;
        }
        .stTitle img {
            width: 30px;  /* Adjust the width as needed */
            margin-right: 10px;
        }
        .stMarkdown {
            font-size: 40px;  /* Adjust the font size as needed */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app
#st.title("🎵 Spotify Genere Based Song Recommender")
st.markdown('<div class="stTitle"><img src="https://uxwing.com/wp-content/themes/uxwing/download/brands-and-social-media/spotify-icon.png"> Spotify Genere Based Song Recommender</div>', unsafe_allow_html=True)

# User input for song selection using a dropdown in the sidebar
track_to_recommend = st.sidebar.selectbox("Select a song", your_dataset['Track Name'])

# Additional features
num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=3)
show_preview = st.sidebar.checkbox("Show Recommendations", value=True)
shuffle_recommendations = st.sidebar.checkbox("Shuffle Recommendations", value=False)
show_artist_info = st.sidebar.checkbox("Show Artist Information", value=True)
show_genere = st.sidebar.checkbox("Show Genere", value=True)


# Button to trigger recommendations in the sidebar
if st.sidebar.button("Get Recommendations"):
    # Get content-based recommendations
    content_based_recommendations = content_based_model(track_to_recommend, your_dataset, cosine_sim)

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

            col1, col2, col3 ,col4,col5= container.columns([1, 2, 2,2,2])

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
            if show_genere:
                col4.write(f"**Genre:** {genre}")

            col5.audio(preview_url, format='audio/mp3', start_time=0)
