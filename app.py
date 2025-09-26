import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load DataSet
##Dataset_link(https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies['overview']=movies['overview'].fillna('')
    movies['genres'] = movies['genres'].fillna('')
    movies['keywords'] = movies['keywords'].fillna('')
    movies['content'] = movies['overview'] + " " + movies['genres'] + " " + movies ['keywords']
    return movies

movies = load_data()

# ------------------------------
# Compute Similarity
# ------------------------------
@st.cache_resource
def compute_similarity(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)
    indices =pd.Series(movies.index,index=movies['title']).drop_duplicates()
    return cosine_sim,indices

cosine_sim,indices = compute_similarity(movies)

# ------------------------------
# Recommendation Function
# ------------------------------

def recommend_movie(title,num_recommendations=5):
    if title not in indices:
        return pd.DataFrame(columns=['title','vote_average','vote_count'])
    
    idx =indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)

    sim_scores= sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]

    return movies[['title','vote_average','vote_count']].iloc[movie_indices]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on your favorite movie!")

# Movie dropdown
movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("Recommend"):
    st.subheader(f"Movies similar to {selected_movie}:")
    recommendations = recommend_movie(selected_movie, 5)
    if recommendations.empty:
        st.warning("Movie not found in dataset!")
    else:
        for idx, row in recommendations.iterrows():
            st.write(f"🎥 **{row['title']}**  | ⭐ {row['vote_average']} (Votes: {row['vote_count']})")