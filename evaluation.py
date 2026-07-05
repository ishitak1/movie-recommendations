import numpy as np
import pandas as pd
import torch
from load_data import ratings, movies_df
from algos.collaborative import build_user_item_matrix, user_similarity, item_similarity
from algos.popularity import popularity_recommender
from algos.matrix_factorization import train_matrix_factorization
from algos.neural_cf import train_neural_cf

np.random.seed(42)
shuffled = ratings.sample(frac=1, random_state=42).reset_index(drop=True)
split_point = int(len(shuffled) * 0.8)
train_ratings = shuffled.iloc[:split_point]
test_ratings = shuffled.iloc[split_point:]

print(f"Train size: {len(train_ratings)}")
print(f"Test size: {len(test_ratings)}")


def predict_popularity(movie_id, popularity_table):
    row = popularity_table.loc[popularity_table["movie_id"] == movie_id]
    if row.empty:
        return None
    return row["weighted_rating"].values[0]


def predict_matrix_factorization(user_id, movie_id, U, V, user_to_idx, movie_to_idx):
    if user_id not in user_to_idx or movie_id not in movie_to_idx:
        return None
    return np.dot(U[user_to_idx[user_id]], V[movie_to_idx[movie_id]])


def predict_user_cf(user_id, movie_id, matrix, top_k_neighbors=20, min_neighbors=3):
    similarities = {}
    for other_user in matrix.index:
        if other_user == user_id:
            continue
        similarities[other_user] = user_similarity(user_id, other_user, matrix)

    top_neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k_neighbors]

    numerator, denominator, voters = 0, 0, 0
    for neighbor_id, sim_score in top_neighbors:
        neighbor_rating = matrix.loc[neighbor_id, movie_id]
        if pd.isna(neighbor_rating):
            continue
        numerator += sim_score * neighbor_rating
        denominator += abs(sim_score)
        voters += 1

    if voters < min_neighbors or denominator == 0:
        return None
    return numerator / denominator


def predict_item_cf(user_id, movie_id, matrix, item_matrix, top_k_neighbors=20, min_neighbors=3):
    user_ratings = matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings.notna()].index

    similarities = {}
    for other_movie in rated_movies:
        if other_movie == movie_id:
            continue
        similarities[other_movie] = item_similarity(movie_id, other_movie, item_matrix)

    top_neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k_neighbors]

    numerator, denominator, voters = 0, 0, 0
    for neighbor_movie, sim_score in top_neighbors:
        rating = user_ratings[neighbor_movie]
        if pd.isna(rating):
            continue
        numerator += sim_score * rating
        denominator += abs(sim_score)
        voters += 1

    if voters < min_neighbors or denominator == 0:
        return None
    return numerator / denominator


def predict_neural_cf(user_id, movie_id, model, user_to_idx, movie_to_idx):
    if user_id not in user_to_idx or movie_id not in movie_to_idx:
        return None
    u_idx = torch.tensor([user_to_idx[user_id]], dtype=torch.long)
    m_idx = torch.tensor([movie_to_idx[movie_id]], dtype=torch.long)
    with torch.no_grad():
        prediction = model(u_idx, m_idx)
    return prediction.item()


print("Training popularity:")
pop_table = popularity_recommender(train_ratings, top_n=1682)

print("Building CF matrices:")
matrix = build_user_item_matrix(train_ratings)
item_matrix = matrix.T

print("Training matrix factorization:")
U, V, mf_user_idx, mf_movie_idx = train_matrix_factorization(train_ratings, epochs=20)

print("Training neural CF:")
nn_model, nn_user_idx, nn_movie_idx = train_neural_cf(train_ratings, epochs=50, lr=0.001)

test_sample = test_ratings.sample(n=1500, random_state=42)

results = {"popularity": [], "user_cf": [], "item_cf": [], "matrix_factorization": [], "neural_cf": []}

for row in test_sample.itertuples():
    u, m, actual = row.user_id, row.movie_id, row.rating

    p = predict_popularity(m, pop_table)
    if p is not None:
        results["popularity"].append((actual, p))

    if u in matrix.index and m in matrix.columns:
        p = predict_user_cf(u, m, matrix)
        if p is not None:
            results["user_cf"].append((actual, p))

        p = predict_item_cf(u, m, matrix, item_matrix)
        if p is not None:
            results["item_cf"].append((actual, p))

    p = predict_matrix_factorization(u, m, U, V, mf_user_idx, mf_movie_idx)
    if p is not None:
        results["matrix_factorization"].append((actual, p))

    p = predict_neural_cf(u, m, nn_model, nn_user_idx, nn_movie_idx)
    if p is not None:
        results["neural_cf"].append((actual, p))

print("\n=== RMSE Comparison ===")
for algo, pairs in results.items():
    if len(pairs) == 0:
        print(f"{algo}: no predictions made")
        continue
    actuals = np.array([a for a, p in pairs])
    preds = np.array([p for a, p in pairs])
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    print(f"{algo}: RMSE={rmse:.4f}  (on {len(pairs)} test predictions)")