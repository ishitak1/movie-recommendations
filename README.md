# Recommendation Lab

A movie recommendation engine built from scratch. Six algorithms, implemented without using any recommendation library, plus a FastAPI backend and a simple frontend to try them live.

## Algorithms

1. Popularity — Bayesian-weighted average rating
2. Content-based filtering — cosine similarity on genre vectors
3. User-based collaborative filtering
4. Item-based collaborative filtering
5. Matrix factorization — gradient descent implemented from scratch
6. Neural collaborative filtering — PyTorch

Algorithms 1–4 are available in the live demo. 5 and 6 require a fixed set of pre-trained users and are evaluated offline instead (`evaluation.py`).

## Setup

1. Clone the repo
2. Download the dataset from https://files.grouplens.org/datasets/movielens/ml-latest-small.zip and unzip it into:
   ```
   data/ml-latest-small/movies.csv
   data/ml-latest-small/ratings.csv
   ```
3. Install dependencies:
   ```
   pip3 install fastapi uvicorn pandas numpy scikit-learn torch
   ```
4. Start the backend:
   ```
   uvicorn main:app --reload --port 8000
   ```
5. Open `index.html` in a browser.

## Evaluation

```
python3 evaluation.py
```

Trains on 80% of ratings, tests on the remaining 20%, reports RMSE per algorithm.

## Structure

```
algos/            the 6 algorithms
load_data.py      loads and parses the dataset
main.py           FastAPI backend
index.html        frontend
evaluation.py     train/test split + RMSE comparison
```

## Dataset

MovieLens Latest-Small (100,836 ratings, 9,724 movies, 610 users), from GroupLens Research. Not included in this repo — see setup instructions above.
