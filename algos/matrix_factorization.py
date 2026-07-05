import numpy as np
from load_data import movies_df


def train_matrix_factorization(ratings_df, k=20, epochs=20, alpha=0.01, seed=42):
    np.random.seed(seed)

    user_ids = ratings_df["user_id"].unique()
    movie_ids = ratings_df["movie_id"].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    n_users = len(user_ids)
    n_movies = len(movie_ids)

    U = np.random.normal(scale=0.1, size=(n_users, k))
    V = np.random.normal(scale=0.1, size=(n_movies, k))

    train_data = [
        (user_to_idx[row.user_id], movie_to_idx[row.movie_id], row.rating)
        for row in ratings_df.itertuples()
    ]

    for epoch in range(epochs):
        total_squared_error = 0
        for u_idx, m_idx, actual_rating in train_data:
            predicted = np.dot(U[u_idx], V[m_idx])
            error = actual_rating - predicted

            u_vector = U[u_idx].copy()
            U[u_idx] += alpha * error * V[m_idx]
            V[m_idx] += alpha * error * u_vector

            total_squared_error += error ** 2

        rmse = np.sqrt(total_squared_error / len(train_data))
        print(f"Epoch {epoch + 1}/{epochs} - RMSE: {rmse:.4f}")

    return U, V, user_to_idx, movie_to_idx


def recommend_matrix_factorization(user_id, ratings_df, U, V, user_to_idx, movie_to_idx, top_n=10):
    user_idx = user_to_idx[user_id]
    already_rated = set(ratings_df.loc[ratings_df['user_id'] == user_id, 'movie_id'])

    scores = {}
    for movie_id, movie_idx in movie_to_idx.items():
        if movie_id in already_rated:
            continue
        scores[movie_id] = np.dot(U[user_idx], V[movie_idx])

    top_predictions = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    recommendations = []
    for movie_id, score in top_predictions:
        title = movies_df.loc[movies_df["movie_id"] == movie_id, "title"].values[0]
        recommendations.append((movie_id, title, score))
    return recommendations


if __name__ == "__main__":
    from load_data import ratings
    U, V, user_to_idx, movie_to_idx = train_matrix_factorization(ratings)
    print()
    for movie_id, title, score in recommend_matrix_factorization(5, ratings, U, V, user_to_idx, movie_to_idx):
        print(f"  {title}: score {score:.2f}")