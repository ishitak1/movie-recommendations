from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd

from load_data import ratings, movies_df, genre_columns
from algos.collaborative import build_user_item_matrix, item_similarity, user_similarity
from algos.content_based import similarity_matrix as genre_similarity_matrix, movie_index
from algos.popularity import popularity_recommender

app = FastAPI(title="Recommendation Lab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

matrix = build_user_item_matrix(ratings)
item_matrix = matrix.T


class RecommendRequest(BaseModel):
    liked_movie_ids: List[int]
    algorithm: str  # "popularity" | "content" | "item_cf" | "user_cf"
    top_n: int = 10


@app.get("/movies")
def get_movies():
    cols = ["movie_id", "title"] + genre_columns
    return movies_df[cols].to_dict(orient="records")


@app.get("/stats")
def get_stats():
    return {
        "n_users": int(ratings["user_id"].nunique()),
        "n_movies": int(ratings["movie_id"].nunique()),
        "n_ratings": int(len(ratings)),
        "sparsity": round(1 - len(ratings) / (ratings["user_id"].nunique() * ratings["movie_id"].nunique()), 4),
    }


def _title(movie_id: int) -> str:
    return movies_df.loc[movies_df["movie_id"] == movie_id, "title"].values[0]


@app.post("/recommend")
def recommend(req: RecommendRequest):
    liked = set(req.liked_movie_ids)
    algo = req.algorithm

    if not liked and algo != "popularity":
        return {"error": "Tick at least one movie you like first."}

    if algo == "popularity":
        table = popularity_recommender(ratings, top_n=req.top_n + len(liked) + 5)
        table = table[~table["movie_id"].isin(liked)].head(req.top_n)
        results = [
            {
                "movie_id": int(r.movie_id),
                "title": r.title,
                "score": round(float(r.weighted_rating), 2),
                "reason": f"Rated highly by {int(r.rating_count)} people overall",
            }
            for r in table.itertuples()
        ]

    elif algo == "content":
        scores = {}
        for mid in liked:
            if mid not in movie_index:
                continue
            idx = movie_index[mid]
            sims = genre_similarity_matrix[idx]
            for other_idx, s in enumerate(sims):
                other_id = int(movies_df.iloc[other_idx]["movie_id"])
                if other_id in liked:
                    continue
                scores[other_id] = scores.get(other_id, 0) + s
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: req.top_n]
        results = [
            {"movie_id": mid, "title": _title(mid), "score": round(s, 2), "reason": "Similar genres to your picks"}
            for mid, s in top
        ]

    elif algo == "item_cf":
        MIN_SHARED_RATERS = 8
        scores = {}
        for seed in liked:
            if seed not in item_matrix.index:
                continue
            seed_ratings = item_matrix.loc[seed]
            for candidate in matrix.columns:
                if candidate in liked:
                    continue
                candidate_ratings = item_matrix.loc[candidate]
                shared = (seed_ratings.notna() & candidate_ratings.notna()).sum()
                if shared < MIN_SHARED_RATERS:
                    continue
                sim = item_similarity(seed, candidate, item_matrix)
                if sim > 0:
                    scores[candidate] = scores.get(candidate, 0) + sim
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: req.top_n]
        results = [
            {"movie_id": mid, "title": _title(mid), "score": round(s, 2), "reason": "Loved by people with similar taste"}
            for mid, s in top
        ]

    elif algo == "user_cf":
        synthetic_id = -1
        synthetic_row = pd.Series(np.nan, index=matrix.columns)
        for mid in liked:
            if mid in synthetic_row.index:
                synthetic_row[mid] = 5.0

        temp_matrix = matrix.copy()
        temp_matrix.loc[synthetic_id] = synthetic_row

        MIN_SHARED_FOR_NEIGHBOR = min(2, len(liked))
        sims = {}
        for other_user in matrix.index:
            other_ratings = matrix.loc[other_user]
            shared = (synthetic_row.notna() & other_ratings.notna()).sum()
            if shared < MIN_SHARED_FOR_NEIGHBOR:
                continue
            sims[other_user] = user_similarity(synthetic_id, other_user, temp_matrix)

        neighbors = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:20]

        numerator, denominator, voters = {}, {}, {}
        for nb_id, sim in neighbors:
            if sim <= 0:
                continue
            for mid, r in matrix.loc[nb_id].dropna().items():
                if mid in liked:
                    continue
                numerator[mid] = numerator.get(mid, 0) + sim * r
                denominator[mid] = denominator.get(mid, 0) + sim
                voters[mid] = voters.get(mid, 0) + 1

        MIN_VOTERS = 3
        preds = {
            mid: numerator[mid] / denominator[mid]
            for mid in numerator
            if denominator[mid] > 0 and voters[mid] >= MIN_VOTERS
        }
        top = sorted(preds.items(), key=lambda x: x[1], reverse=True)[: req.top_n]
        results = [
            {"movie_id": mid, "title": _title(mid), "score": round(s, 2), "reason": "People with taste like yours loved this"}
            for mid, s in top
        ]

    else:
        return {"error": f"Unknown algorithm '{algo}'"}

    return {"algorithm": algo, "count": len(results), "recommendations": results}