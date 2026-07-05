import pandas as pd
from load_data import movies_df


def popularity_recommender(ratings_df, top_n: int = 10, min_votes: int = 50) -> pd.DataFrame:
    movie_stats = (
        ratings_df
        .groupby("movie_id")
        .agg(
            average_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    global_average = ratings_df["rating"].mean()

    movie_stats = movie_stats.assign(
        weighted_rating=lambda df: (
            (df["rating_count"] / (df["rating_count"] + min_votes)) * df["average_rating"]
            + (min_votes / (df["rating_count"] + min_votes)) * global_average
        )
    )

    recommendations = (
        movie_stats
        .merge(movies_df, on="movie_id")
        .sort_values("weighted_rating", ascending=False)
        .head(top_n)
        [["movie_id", "title", "average_rating", "rating_count", "weighted_rating"]]
    )

    return recommendations


if __name__ == "__main__":
    from load_data import ratings
    print(popularity_recommender(ratings))