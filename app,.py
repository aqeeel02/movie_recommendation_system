import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords
nltk.download('stopwords')


@st.cache_data
def load_data():
    """Load movies and ratings dataset"""
    movies = pd.read_csv("C:\\Users\\aqeel\\Downloadst\\movies.csv")
    ratings = pd.read_csv("C:\\Users\\aqeel\\Downloadst\\ratings.csv")
    return movies, ratings


movies, ratings = load_data()


@st.cache_data
def compute_similarity_matrix():
    """Precompute the similarity matrix"""
    def tokenizer(text):
        return [PorterStemmer().stem(word).lower() for word in text.split('|') if word not in stopwords.words('english')]

    tfid = TfidfVectorizer(analyzer='word', tokenizer=tokenizer)
    tfidf_matrix = tfid.fit_transform(movies['genres'])
    cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cos_sim


cos_sim = compute_similarity_matrix()

# Mapping dictionaries
movie_map = pd.Series(movies.movieId.values, index=movies.title).to_dict()
reverse_movie_map = {v: k for k, v in movie_map.items()}
movieId_to_index_map = pd.Series(movies.index.values, index=movies.movieId).to_dict()
movieId_all_array = movies['movieId'].unique()


def get_movieId(movie_name):
    """Get movieId from movie title using fuzzy matching"""
    if movie_name in movie_map:
        return movie_map[movie_name]
    else:
        similar = [(title, movie_id, fuzz.ratio(title.lower(), movie_name.lower())) for title, movie_id in
                   movie_map.items()]
        similar = sorted(similar, key=lambda x: x[2], reverse=True)
        return similar[0][1] if similar else None


@st.cache_data
def train_model():
    """Train the SVD recommendation model"""
    reader = Reader(rating_scale=(ratings.rating.min(), ratings.rating.max()))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    model_svd = SVD()
    model_svd.fit(data.build_full_trainset())
    return model_svd


model_svd = train_model()


def make_recommendation_item_based(fav_movie, n_recommendations=10):
    """Recommend movies based on item similarity"""
    movieId = get_movieId(fav_movie)
    if movieId is None:
        return []

    index = movieId_to_index_map[movieId]
    cos_sim_scores = list(enumerate(cos_sim[index]))
    cos_sim_scores = sorted(cos_sim_scores, key=lambda x: x[1], reverse=True)

    topn_movies = [movies.iloc[i[0]].title for i in cos_sim_scores[1:n_recommendations + 1]]
    return topn_movies


def make_recommendation_user_based(fav_movie, n_recommendations=10):
    """Recommend movies based on user preferences"""
    movieId = get_movieId(fav_movie)
    if movieId is None:
        return []

    userId = ratings.userId.max() + 1
    predictions = [model_svd.predict(userId, m_id) for m_id in movieId_all_array if m_id != movieId]
    predictions.sort(key=lambda x: x.est, reverse=True)

    topn_movies = [reverse_movie_map[p.iid] for p in predictions[:n_recommendations]]
    return topn_movies


@st.cache_data
def get_popular_movies(n_recommendations=10):
    """Recommend top N popular movies"""
    userId = ratings.userId.max() + 1  # Simulate a new user

    predictions = [model_svd.predict(userId, m_id) for m_id in movieId_all_array]
    predictions.sort(key=lambda x: x.est, reverse=True)  # Sort by highest estimated rating

    topn_movies = [reverse_movie_map[p.iid] for p in predictions[:n_recommendations]]
    return topn_movies


# Streamlit UI
st.title("🎬 Movie Recommendation System")

option = st.radio(
    "Do you have a favourite movie?",
    ("Yes", "No"),
    index=None
)
st.write("You selected:", option)

if option == "Yes":
    fav_movie = st.text_input("Enter your favorite movie: (preferably with the year)")
    if fav_movie:
        st.subheader("Recommended Movies")
        st.write("Movies similar to your favourite movie:")
        st.write(make_recommendation_item_based(fav_movie))
        st.write("Users who liked your movie also liked:")
        st.write(make_recommendation_user_based(fav_movie))

elif option == "No":
    st.subheader("Popular Movies")
    st.write(get_popular_movies())

elif option == "None" or option is None:
    st.write("Please make a selection.")
