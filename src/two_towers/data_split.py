import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from .features import TwoTowersFeatureEngineer
from .config import Config, TWO_TOWER_PARAMS
from . import constants


class BookRankingDataset(Dataset):
    """Датасет для обучения Two-Towers модели"""
    
    def __init__(self, user_features: pd.DataFrame, book_features: pd.DataFrame, 
                 pairs: pd.DataFrame, text_features: pd.DataFrame = None):
        self.user_features = user_features
        self.book_features = book_features
        self.text_features = text_features
        self.pairs = pairs
        
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(user_features.index)}
        self.book_id_to_idx = {bid: idx for idx, bid in enumerate(book_features.index)}
        
        valid_pairs = []
        for _, row in pairs.iterrows():
            if (row['user_id'] in self.user_id_to_idx and 
                row['book_id'] in self.book_id_to_idx):
                valid_pairs.append(row)
        self.pairs = pd.DataFrame(valid_pairs)
        
        self.user_features_tensor = torch.FloatTensor(user_features.values)
        self.book_features_tensor = torch.FloatTensor(book_features.values)
        
        if text_features is not None:
            self.text_features_tensor = torch.FloatTensor(text_features.values)
            self.book_features_tensor = torch.cat(
                [self.book_features_tensor, self.text_features_tensor], dim=1
            )
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        
        user_idx = self.user_id_to_idx[pair['user_id']]
        book_idx = self.book_id_to_idx[pair['book_id']]
        
        user_feat = self.user_features_tensor[user_idx]
        book_feat = self.book_features_tensor[book_idx]
        label = torch.tensor(pair['label'], dtype=torch.long)
        
        return user_feat, book_feat, label


def create_training_pairs(train_df: pd.DataFrame, books_df: pd.DataFrame, 
                         negative_ratio: float = 2.0) -> pd.DataFrame:
    positive_pairs = train_df[['user_id', 'book_id', 'has_read']].copy()
    positive_pairs['label'] = positive_pairs['has_read'].apply(
        lambda x: constants.READ_CLASS if x == 1 else constants.PLANNED_CLASS
    )
    
    all_books = set(books_df['book_id'].unique())
    user_interactions = train_df.groupby('user_id')['book_id'].apply(set)
    
    negative_samples = []
    for user_id, interacted_books in user_interactions.items():
        non_interacted = list(all_books - interacted_books)
        
        n_positive = len(interacted_books)
        n_negative = min(int(n_positive * negative_ratio), len(non_interacted))
        
        if n_negative > 0:
            sampled_books = np.random.choice(non_interacted, n_negative, replace=False)
            for book_id in sampled_books:
                negative_samples.append({
                    'user_id': user_id,
                    'book_id': book_id,
                    'label': constants.COLD_CLASS
                })
    
    negative_pairs = pd.DataFrame(negative_samples)
    
    # Объединяем
    all_pairs = pd.concat([positive_pairs[['user_id', 'book_id', 'label']], 
                          negative_pairs], ignore_index=True)
    
    print(f"Created {len(positive_pairs)} positive/planned pairs")
    print(f"Created {len(negative_pairs)} negative pairs")
    print(f"Total pairs: {len(all_pairs)}")
    
    return all_pairs


def split_data_temporal(pairs_df: pd.DataFrame, train_df: pd.DataFrame, 
                       split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Performing temporal split...")
    
    user_timestamps = train_df.groupby('user_id')['timestamp'].max()
    split_time = user_timestamps.quantile(split_ratio)
    val_interactions = train_df[train_df['timestamp'] > split_time]
    val_pairs = set(zip(val_interactions['user_id'], val_interactions['book_id']))
    
    # Разделяем пары
    train_mask = pairs_df.apply(
        lambda row: (row['user_id'], row['book_id']) not in val_pairs, axis=1
    )
    
    train_pairs = pairs_df[train_mask]
    val_pairs = pairs_df[~train_mask]
    
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    
    return train_pairs, val_pairs


def prepare_datasets(config: Config, data: Dict[str, pd.DataFrame]) -> Tuple[Dataset, Dataset]:
    train_df = data['train']
    books_df = data['books']
    
    # Создаем фичи
    engineer = TwoTowersFeatureEngineer(config)
    engineer.create_all_features(data)
    
    all_pairs = create_training_pairs(train_df, books_df)
    
    # Разделяем на train/val
    train_pairs, val_pairs = split_data_temporal(all_pairs, train_df, 
                                                config.TEMPORAL_SPLIT_RATIO)
    
    train_dataset = BookRankingDataset(
        engineer.user_features,
        engineer.book_features,
        train_pairs,
        engineer.text_features
    )
    
    val_dataset = BookRankingDataset(
        engineer.user_features,
        engineer.book_features,
        val_pairs,
        engineer.text_features
    )
    
    TWO_TOWER_PARAMS.user_input_dim = train_dataset.user_features_tensor.shape[1]
    TWO_TOWER_PARAMS.book_input_dim = train_dataset.book_features_tensor.shape[1]
    
    print(f"User features dimension: {TWO_TOWER_PARAMS.user_input_dim}")
    print(f"Book features dimension: {TWO_TOWER_PARAMS.book_input_dim}")
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset: Dataset, val_dataset: Dataset, 
                      batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


def final() -> Tuple[DataLoader, DataLoader]:
    from .data_processing import load_and_merge_data
    
    print("=" * 60)
    print("PREPARING DATA FOR TWO-TOWERS MODEL")
    print("=" * 60)
    
    data = load_and_merge_data()
    train_dataset, val_dataset = prepare_datasets(Config, data)
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, TWO_TOWER_PARAMS.batch_size
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("=" * 60)
    
    return train_loader, val_loader