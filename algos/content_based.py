import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from load_data import movies_df, genre_columns


def build_similarity_matrix():
    genre_matrix = movies_df[genre_columns].values
    return cosine_similarity(genre_matrix)

similarity_matrix = build_similarity_matrix()

movie_index = {
    movie_id: idx
    for idx, movie_id in enumerate(movies_df["movie_id"])
}


def similar_movies(movie_id: int, top_n: int = 10):
    movie_idx = movie_index[movie_id]
    similarity_scores = similarity_matrix[movie_idx]
    similar_indices = np.argsort(similarity_scores)[::-1]
    similar_indices = similar_indices[similar_indices != movie_idx][:top_n]

    recommendations = movies_df.iloc[similar_indices][["movie_id", "title"]].copy()
    recommendations["similarity"] = similarity_scores[similar_indices]
    return recommendations.reset_index(drop=True)


if __name__ == "__main__":
    movie_id = 1
    movie_title = movies_df.loc[movies_df["movie_id"] == movie_id, "title"].iloc[0]
    print(f"\nMovies similar to '{movie_title}':\n")
    print(similar_movies(movie_id))