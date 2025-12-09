import torch
from torch import nn
import torch.nn.functional as F

from . import TWO_TOWER_PARAMS

class TwoTowersModel(nn.Module):
    
    def __init__(self, user_input_dim, book_input_dim):
        super().__init__()
        
        # User Tower
        user_layers = []
        prev_dim = user_input_dim
        for hidden_dim in TWO_TOWER_PARAMS.user_tower_hidden:
            user_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(TWO_TOWER_PARAMS.dropout_rate)
            ])
            prev_dim = hidden_dim
        user_layers.append(nn.Linear(prev_dim, TWO_TOWER_PARAMS.user_embedding_dim))
        self.user_tower = nn.Sequential(*user_layers)
        
        # Book Tower
        book_layers = []
        prev_dim = book_input_dim
        for hidden_dim in TWO_TOWER_PARAMS.book_tower_hidden:
            book_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(TWO_TOWER_PARAMS.dropout_rate)
            ])
            prev_dim = hidden_dim
        book_layers.append(nn.Linear(prev_dim, TWO_TOWER_PARAMS.book_embedding_dim))
        self.book_tower = nn.Sequential(*book_layers)
        
        # Merge Network
        self.merge_net = nn.Sequential(
            nn.Linear(
                TWO_TOWER_PARAMS.user_embedding_dim + TWO_TOWER_PARAMS.book_embedding_dim,
                TWO_TOWER_PARAMS.merge_hidden
            ),
            nn.BatchNorm1d(TWO_TOWER_PARAMS.merge_hidden),
            nn.ReLU(),
            nn.Dropout(TWO_TOWER_PARAMS.dropout_rate),
            nn.Linear(TWO_TOWER_PARAMS.merge_hidden, TWO_TOWER_PARAMS.merge_output),
            nn.BatchNorm1d(TWO_TOWER_PARAMS.merge_output),
            nn.ReLU()
        )
        
        # Classification Head
        self.regressor = nn.Sequential(
            nn.Linear(TWO_TOWER_PARAMS.merge_output + 1, 64),
            nn.ReLU(),
            nn.Dropout(TWO_TOWER_PARAMS.dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  
        )
        
        # Инициализация весов
        self.sigmoid = nn.Sigmoid()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_features, book_features):
        # Проход через башни
        user_emb = self.user_tower(user_features)
        book_emb = self.book_tower(book_features)
        
        # Нормализация эмбеддингов
        user_emb = F.normalize(user_emb, p=2, dim=1)
        book_emb = F.normalize(book_emb, p=2, dim=1)
        
        # Скалярное произведение (сходство)
        dot_product = torch.sum(user_emb * book_emb, dim=1, keepdim=True)
        
        # Объединение эмбеддингов
        merged = torch.cat([user_emb, book_emb], dim=1)
        merged = self.merge_net(merged)
        
        # Регрессия
        rating = self.regressor(combined)
        rating = self.sigmoid(rating) * 2  
        
        return rating
    
    def predict_rating(self, user_features, book_features):
        """Предсказание рейтинга (0-2)"""
        with torch.no_grad():
            rating = self.forward(user_features, book_features)
            return rating
    
    def get_embeddings(self, user_features, book_features):
        """Получение эмбеддингов для анализа"""
        with torch.no_grad():
            user_emb = self.user_tower(user_features)
            book_emb = self.book_tower(book_features)
            return user_emb, book_emb


# Для обратной совместимости
Two_towers = TwoTowersModel