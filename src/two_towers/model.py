
import torch
from torch import nn

import config


class Two_towers(nn.Module):
    def __init__(self, user_features_in:int, book_features_in:int):
        super().__init__()

        self.user_tower = nn.Sequential(
            nn.Linear(
                in_features=user_features_in, 
                out_features=config.TWO_TOWER_PARAMS.usertower_inner_lenght,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=config.TWO_TOWER_PARAMS.usertower_inner_lenght,
                out_features=config.TWO_TOWER_PARAMS.usertower_embedding_lenght,
            ),
        )

        self.book_tower = nn.Sequential(
            nn.Linear(
                in_features=book_features_in, 
                out_features=config.TWO_TOWER_PARAMS.booktower_inner_lenght
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=config.TWO_TOWER_PARAMS.booktower_inner_lenght, 
                out_features=config.TWO_TOWER_PARAMS.booktower_embeddings_lenght
            )
        )

        self.merge_net = nn.Sequential(
            nn.Linear(
                in_features=config.TWO_TOWER_PARAMS.usertower_embedding_lenght + config.TWO_TOWER_PARAMS.booktower_inner_lenght,
                out_features=config.TWO_TOWER_PARAMS.mergelayer_inner
                ),
            nn.ReLU(),
            nn.Linear(
                in_features=config.TWO_TOWER_PARAMS.mergelayer_inner,
                out_features=config.TWO_TOWER_PARAMS.mergelayer_out,
            )     
        )

        self.final_layer = nn.Sequential(
            nn.Linear(
                in_features=config.TWO_TOWER_PARAMS.mergelayer_out + 1, # dot product
                out_features=3
            ),
            nn.Softmax(dim=1)
        )


    def forward(self, user, book):

        user_emb = self.user_tower(user)
        user_emb = nn.functional.normalize(user_emb)

        book_emb = self.book_tower(book)
        book_emb = nn.functional.normalize(book_emb)

        dot_prod = torch.sum(book_emb * user_emb, dim=1)

        merge_res = self.merge_net(torch.cat(book_emb, user_emb, dim=1)) # NOT SURE IF IT SHOULD BE dim=1

        final_res = self.final_layer(torch.cat(merge_res, dot_prod, dim=1))

        return final_res




