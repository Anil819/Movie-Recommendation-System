import streamlit as st
import pandas as pd
import pickle
import requests
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)
OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

HF_REPO_ID = "anilohar2325/movie-recommendation-model"


@st.cache_data(ttl=3600)
def fetch_poster(movie_title):
    url = "https://www.omdbapi.com/"

    params = {
        "t": movie_title,
        "apikey": OMDB_API_KEY,
        "type": "movie"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        data = response.json()

        if data.get("Response") == "True":
            poster = data.get("Poster")

            if poster and poster != "N/A":
                return poster

        return "https://placehold.co/300x450?text=Poster+Not+Available"

    except requests.RequestException:
        return "https://placehold.co/300x450?text=Poster+Not+Available"


@st.cache_resource
def load_data():

    with open("movies_dict.pkl", "rb") as file:
        movies_dict = pickle.load(file)

    movies = pd.DataFrame(movies_dict)

    similarity_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="similarity.pkl",
        repo_type="model"
    )

    with open(similarity_path, "rb") as file:
        similarity = pickle.load(file)

    return movies, similarity


movies, similarity = load_data()


def recommend(movie):

    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_movies_posters = []

    for i in movies_list:

        movie_title = movies.iloc[i[0]]["title"]

        recommended_movies.append(movie_title)

        poster = fetch_poster(movie_title)

        recommended_movies_posters.append(poster)

    return recommended_movies, recommended_movies_posters


st.title("🎬 Movie Recommender System")

st.write(
    "Discover movies based on your favourite movie."
)

selected_movie_name = st.selectbox(
    "Choose a movie",
    movies["title"].values
)

if st.button("Recommend Movies"):

    with st.spinner("Finding recommended movies..."):

        names, posters = recommend(
            selected_movie_name
        )

    cols = st.columns(5)

    for col, name, poster in zip(
        cols,
        names,
        posters
    ):

        with col:

            st.image(
                poster,
                width="stretch"
            )

            st.markdown(
                f"**{name}**"
            )