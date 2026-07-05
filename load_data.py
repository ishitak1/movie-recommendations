import pandas as pd

ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
ratings = ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})

movies_raw = pd.read_csv('data/ml-latest-small/movies.csv')
movies_raw = movies_raw.rename(columns={'movieId': 'movie_id'})

all_genres = set()
for genre_string in movies_raw['genres']:
    if genre_string == '(no genres listed)':
        continue
    for g in genre_string.split('|'):
        all_genres.add(g)

genre_columns = sorted(g.replace('-', '_').replace("'", "") for g in all_genres)

movies_df = movies_raw[['movie_id', 'title']].copy()

for col in genre_columns:
    movies_df[col] = movies_raw['genres'].apply(
        lambda genres, c=col: 1 if any(
            g.replace('-', '_').replace("'", "") == c for g in genres.split('|')
        ) else 0
    )

n_users = ratings["user_id"].nunique()
n_movies = ratings["movie_id"].nunique()
n_ratings = len(ratings)

print(f"Users: {n_users}")
print(f"Movies: {n_movies}")
print(f"Ratings: {n_ratings}")
print(f"Rating range: {ratings['rating'].min()} to {ratings['rating'].max()}")

possible_ratings = n_users * n_movies
sparsity = 1 - (n_ratings / possible_ratings)

print(f"Possible (user, movie) pairs: {possible_ratings}")
print(f"Sparsity: {sparsity:.4%}")

print("\nMovie sample:")
print(movies_df.head())