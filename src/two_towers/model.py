
import torch
from torch import nn

from .config import config

class Two_towers(nn.Module):
    def __init__(self, user_features_in:int, book_features_in:int):
        super().__init__()

        self.user_tower = nn.Sequential(
            nn.Linear(
                in_features=user_features_in, 
                out_features=config.TWO_TOWERS_PARAMS["usertower_inner_lenght"]
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=config.TWO_TOWERS_PARAMS["usertower_inner_lenght"], 
                out_features=config.TWO_TOWERS_PARAMS["usertower_embeddings_lenght"]
            ),
        )

        self.book_tower = nn.Sequential(
            nn.Linear(
                in_features=book_features_in, 
                out_features=config.TWO_TOWERS_PARAMS["booktower_inner_lenght"]
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=config.TWO_TOWERS_PARAMS["booktower_inner_lenght"], 
                out_features=config.TWO_TOWERS_PARAMS["booktower_embeddings_lenght"]
            )
        )


    def forward(self, user, book):

        user_emb = self.user_tower(user)
        user_emb = nn.functional.normalize(user_emb)

        book_emb = self.book_tower(book)
        book_emb = nn.functional.normalize(book_emb)

        return torch.sum(book_emb * user_emb, dim=1)




