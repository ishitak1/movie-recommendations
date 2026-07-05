import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from load_data import movies_df


def build_user_item_matrix(ratings_df):
    return ratings_df.pivot(index='user_id', columns='movie_id', values='rating')


def user_similarity(user_a, user_b, matrix):
    ratings_a = matrix.loc[user_a]
    ratings_b = matrix.loc[user_b]

    common_movies = ratings_a.notna() & ratings_b.notna()
    ratings_a = ratings_a[common_movies]
    ratings_b = ratings_b[common_movies]

    if common_movies.sum() == 0:
        return 0

    similarity = cosine_similarity(
        ratings_a.values.reshape(1, -1),
        ratings_b.values.reshape(1, -1)
    )
    return similarity[0][0]


def recommend_for_user(user_id, matrix, top_k_neighbors=20, top_n=10, min_neighbors=3):
    target_ratings = matrix.loc[user_id]
    unrated_movies = target_ratings[target_ratings.isna()].index

    similarities = {}
    for other_user in matrix.index:
        if other_user == user_id:
            continue
        similarities[other_user] = user_similarity(user_id, other_user, matrix)

    top_neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k_neighbors]

    predictions = {}
    for movie in unrated_movies:
        numerator = 0
        denominator = 0
        voters = 0

        for neighbor_id, sim_score in top_neighbors:
            neighbor_rating = matrix.loc[neighbor_id, movie]
            if pd.isna(neighbor_rating):
                continue
            numerator += sim_score * neighbor_rating
            denominator += abs(sim_score)
            voters += 1

        if denominator > 0 and voters >= min_neighbors:
            predictions[movie] = numerator / denominator

    top_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]

    recommendations = []
    for movie_id, predicted_rating in top_predictions:
        title = movies_df.loc[movies_df["movie_id"] == movie_id, "title"].values[0]
        recommendations.append((movie_id, title, predicted_rating))
    return recommendations


def item_similarity(movie_a, movie_b, item_matrix):
    ratings_a = item_matrix.loc[movie_a]
    ratings_b = item_matrix.loc[movie_b]

    common_users = ratings_a.notna() & ratings_b.notna()
    ratings_a = ratings_a[common_users]
    ratings_b = ratings_b[common_users]

    if common_users.sum() == 0:
        return 0

    similarity = cosine_similarity(
        ratings_a.values.reshape(1, -1),
        ratings_b.values.reshape(1, -1)
    )
    return similarity[0][0]


def recommend_for_user_item_based(user_id, matrix, item_matrix, top_n=10, rating_threshold=4):
    target_ratings = matrix.loc[user_id]
    seed_movies = target_ratings[target_ratings >= rating_threshold].index
    already_rated = target_ratings[target_ratings.notna()].index

    scores = {}
    for seed in seed_movies:
        for candidate in matrix.columns:
            if candidate in already_rated:
                continue
            sim = item_similarity(seed, candidate, item_matrix)
            scores[candidate] = scores.get(candidate, 0) + sim

    top_predictions = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    recommendations = []
    for movie_id, score in top_predictions:
        title = movies_df.loc[movies_df["movie_id"] == movie_id, "title"].values[0]
        recommendations.append((movie_id, title, score))
    return recommendations


if __name__ == "__main__":
    from load_data import ratings
    matrix = build_user_item_matrix(ratings)
    item_matrix = matrix.T

    print("Similarity between user 1 and user 2:", user_similarity(1, 2, matrix))
    print()
    print("Recommendations for user 5:")
    for movie_id, title, pred in recommend_for_user(5, matrix):
        print(f"  {title}: predicted {pred:.2f}")
    print()
    print("Item-based recommendations for user 5:")
    for movie_id, title, score in recommend_for_user_item_based(5, matrix, item_matrix):
        print(f"  {title}: score {score:.2f}")