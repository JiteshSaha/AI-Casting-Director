
import torch
import torch.nn as nn
import torch.nn.functional as F

class CastRatingRegressor(nn.Module):
    def __init__(self, num_people, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_people, embedding_dim=embedding_dim)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output: single rating
        )

    def forward(self, x):
        """
        x: (batch_size, 5) - 4 actors + 1 director (all as IDs)
        """
        embeddings = self.embedding(x)  # (batch_size, 5, embedding_dim)
        pooled = embeddings.mean(dim=1)  # (batch_size, embedding_dim)
        output = self.fc_layers(pooled)  # (batch_size, 1)
        output = output.squeeze(1)  # (batch_size,)

        return torch.clamp(output, min=0.0, max=100.0)
