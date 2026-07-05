import torch
import torch.nn as nn
from load_data import movies_df


class NeuralCF(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=20):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_idx, movie_idx):
        u = self.user_embedding(user_idx)
        m = self.movie_embedding(movie_idx)
        x = torch.cat([u, m], dim=1)
        out = self.fc_layers(x)
        return out.squeeze()


def train_neural_cf(ratings_df, embedding_dim=20, epochs=50, lr=0.001):
    user_ids = ratings_df["user_id"].unique()
    movie_ids = ratings_df["movie_id"].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    n_users = len(user_ids)
    n_movies = len(movie_ids)

    user_tensor = torch.tensor([user_to_idx[u] for u in ratings_df["user_id"]], dtype=torch.long)
    movie_tensor = torch.tensor([movie_to_idx[m] for m in ratings_df["movie_id"]], dtype=torch.long)
    rating_tensor = torch.tensor(ratings_df["rating"].values, dtype=torch.float32)

    model = NeuralCF(n_users, n_movies, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(user_tensor, movie_tensor)
        loss = loss_fn(predictions, rating_tensor)
        loss.backward()
        optimizer.step()
        rmse = torch.sqrt(loss).item()
        print(f"Epoch {epoch+1}/{epochs} - RMSE: {rmse:.4f}")

    return model, user_to_idx, movie_to_idx


if __name__ == "__main__":
    from load_data import ratings
    model, user_to_idx, movie_to_idx = train_neural_cf(ratings)