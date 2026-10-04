import os
import pickle
import pandas as pd
import requests
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
)

try:
    OMDB_API_KEY = st.secrets["OMDB_API_KEY"]
except Exception:
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "")

HF_REPO_ID = "anilohar2325/movie-recommendation-model"


@st.cache_data(ttl=3600)
def fetch_poster(movie_title):
    if not OMDB_API_KEY:
        return None

    try:
        response = requests.get(
            "https://www.omdbapi.com/",
            params={
                "apikey": OMDB_API_KEY,
                "t": str(movie_title).strip(),
            },
            timeout=15,
        )

        data = response.json()

        if data.get("Response") != "True":
            return None

        poster_url = data.get("Poster")

        if not poster_url or poster_url == "N/A":
            return None

        image_response = requests.get(
            poster_url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        if image_response.status_code == 200:
            return image_response.content

        return None

    except Exception as e:
        print(f"Poster error for {movie_title}: {e}")
        return None


@st.cache_resource
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(base_dir, "movies_dict.pkl")
    similarity_path = os.path.join(base_dir, "similarity.pkl")

    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"Missing file: {movies_path}")

    with open(movies_path, "rb") as file:
        movies_dict = pickle.load(file)

    movies = pd.DataFrame(movies_dict)

    if not os.path.exists(similarity_path):
        similarity_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="similarity.pkl",
            repo_type="model",
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
        key=lambda x: x[1],
    )[1:6]

    recommended_movies = []
    recommended_movies_posters = []

    for i in movies_list:
        movie_title = movies.iloc[i[0]]["title"]
        recommended_movies.append(movie_title)
        recommended_movies_posters.append(fetch_poster(movie_title))

    return recommended_movies, recommended_movies_posters


st.title("🎬 Movie Recommender System")
st.write("Discover movies based on your favourite movie.")

selected_movie_name = st.selectbox(
    "Choose a movie",
    movies["title"].values,
)

names = []
posters = []

if st.button("Recommend Movies"):
    with st.spinner("Finding recommended movies..."):
        names, posters = recommend(selected_movie_name)

    cols = st.columns(5)

    for col, name, poster in zip(cols, names, posters):
        with col:
            if poster:
                st.image(poster, width=180)
            else:
                st.write("Poster Not Available")
            st.markdown(f"**{name}**")