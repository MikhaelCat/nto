"""
Configuration file for the NTO ML competition baseline.
"""

from pathlib import Path
import torch

# --- DIRECTORIES ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

# --- PARAMETERS ---
RANDOM_STATE = 42

# --- TEMPORAL SPLIT CONFIG ---
TEMPORAL_SPLIT_RATIO = 0.8

# --- TRAINING CONFIG ---
EARLY_STOPPING_ROUNDS = 50
MODEL_FILENAME = "two_towers_model.pth"

# --- TWO-TOWERS MODEL CONFIG ---
class TWO_TOWER_PARAMS:
    # Размерности входов (будут вычислены автоматически)
    user_input_dim = None  # Будет вычислено
    book_input_dim = None  # Будет вычислено
    
    # Архитектура башен
    user_tower_hidden = [256, 128]
    book_tower_hidden = [256, 128]
    
    # Размерности эмбеддингов
    user_embedding_dim = 64
    book_embedding_dim = 64
    
    # Объединяющий слой
    merge_hidden = 128
    merge_output = 64
    
    # Финальный классификатор
    classifier_hidden = 64
    num_classes = 3  # 0: cold, 1: planned, 2: read
    
    # Обучение
    learning_rate = 0.001
    batch_size = 256
    num_epochs = 20
    dropout_rate = 0.3

# --- LIGHTGBM CONFIG (для сравнения) ---
LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
    "boosting_type": "gbdt",
    "max_bin": 255,
    "force_row_wise": True,
}

# --- FEATURE ENGINEERING CONFIG ---
TFIDF_MAX_FEATURES = 500
BERT_EMBEDDING_DIM = 768
USE_TEXT_FEATURES = True
CACHE_FEATURES = True

# --- PATHS FOR FEATURE ENGINEERING ---
paths = {
    'features': PROCESSED_DATA_DIR / 'features.pkl',
    'text_features': PROCESSED_DATA_DIR / 'text_features.pkl',
    'book_genres': RAW_DATA_DIR / 'book_genres.csv',
    'descriptions': RAW_DATA_DIR / 'book_descriptions.csv'
}

# --- FEATURE ENGINEERING SETTINGS ---
cache_features = True
cache_embeddings = True
use_gpu = torch and torch.cuda.is_available()


class Config:
    """Configuration class for easy access"""
    ROOT_DIR = ROOT_DIR
    DATA_DIR = DATA_DIR
    RAW_DATA_DIR = RAW_DATA_DIR
    INTERIM_DATA_DIR = INTERIM_DATA_DIR
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
    OUTPUT_DIR = OUTPUT_DIR
    MODEL_DIR = MODEL_DIR
    SUBMISSION_DIR = SUBMISSION_DIR
    RANDOM_STATE = RANDOM_STATE
    TEMPORAL_SPLIT_RATIO = TEMPORAL_SPLIT_RATIO
    EARLY_STOPPING_ROUNDS = EARLY_STOPPING_ROUNDS
    MODEL_FILENAME = MODEL_FILENAME
    TWO_TOWER_PARAMS = TWO_TOWER_PARAMS
    TFIDF_MAX_FEATURES = TFIDF_MAX_FEATURES
    BERT_EMBEDDING_DIM = BERT_EMBEDDING_DIM
    USE_TEXT_FEATURES = USE_TEXT_FEATURES
    CACHE_FEATURES = CACHE_FEATURES
    paths = paths
    cache_features = cache_features
    cache_embeddings = cache_embeddings
    use_gpu = use_gpu