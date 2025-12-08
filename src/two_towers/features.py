
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import joblib
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ML и NLP зависимости
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from transformers import AutoTokenizer, AutoModel
import torch

# Константы — ОБЯЗАТЕЛЬНО подключите их!
from src.baseline.constants import (
    USER_ID,
    BOOK_ID,
    HAS_READ,
    RATING,
    TIMESTAMP,
    AUTHOR_ID,
    DESCRIPTION,
    READ_CLASS,
    PLANNED_CLASS,
    COLD_CLASS
)

# Конфигурация
from src.baseline.config import Config
from .constants import (
    USER_ID, BOOK_ID, HAS_READ, RATING, TIMESTAMP,
    AUTHOR_ID, DESCRIPTION, READ_CLASS, PLANNED_CLASS, COLD_CLASS
)

class TwoTowersFeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        self.user_features = None
        self.book_features = None
        self.text_features = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.tfidf_vectorizer = None
    
    def load_precomputed_features(self) -> bool:
        features_path = self.config.paths['features']
        text_features_path = self.config.paths['text_features']
        
        if features_path.exists() and text_features_path.exists():
            print("Loading precomputed features...")
            features = joblib.load(features_path)
            self.user_features = features['user_features']
            self.book_features = features['book_features']
            
            if text_features_path.exists():
                self.text_features = pd.read_pickle(text_features_path)
                print("Loaded text features")
            
            print(f"Loaded user features for {len(self.user_features)} users")
            print(f"Loaded book features for {len(self.book_features)} books")
            return True
        return False
    
    def save_features(self) -> None:
        features_path = self.config.paths['features']
        text_features_path = self.config.paths['text_features']
        
        # Create output directories if they don't exist
        features_path.parent.mkdir(parents=True, exist_ok=True)
        text_features_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main features
        features = {
            'user_features': self.user_features,
            'book_features': self.book_features
        }
        joblib.dump(features, features_path)
        print(f"Saved features to {features_path}")
        
        # Save text features
        if self.text_features is not None:
            self.text_features.to_pickle(text_features_path)
            print(f"Saved text features to {text_features_path}")
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing for TF-IDF."""
        if not isinstance(text, str) or pd.isna(text) or text.strip() == '':
            return ''
        
        # Lowercase, remove special characters and extra spaces
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _get_bert_embeddings(self, descriptions: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate BERT embeddings for book descriptions."""
        if self.bert_tokenizer is None or self.bert_model is None:
            print("Loading BERT model...")
            model_name = "DeepPavlov/rubert-base-cased"
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_model.to(self.device)
            self.bert_model.eval()
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(descriptions), batch_size), desc="Generating BERT embeddings"):
            batch = descriptions[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.bert_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def create_user_features(self, train_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive user features for the user tower."""
        print("Creating user features...")

        user_features = users_df.set_index(USER_ID).copy()
        
        # Temporal features from interactions
        user_temporal = self._create_user_temporal_features(train_df)
        
        # Reading behavior features
        user_behavior = self._create_user_behavior_features(train_df)
        
        # Genre preference features
        user_genres = self._create_user_genre_preferences(train_df)
        
        # Merge all user features
        user_features = user_features.join(user_temporal, how='left')
        user_features = user_features.join(user_behavior, how='left')
        user_features = user_features.join(user_genres, how='left')
        
        # Fill missing values
        user_features.fillna(0, inplace=True)
        
        # Normalize numerical features
        numerical_cols = user_features.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols:
            if user_features[col].std() > 0:
                user_features[col] = (user_features[col] - user_features[col].mean()) / user_features[col].std()
        
        print(f"Created {len(user_features.columns)} features for {len(user_features)} users")
        return user_features
    
    def _create_user_temporal_features(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features for users."""
        # Ensure timestamp is datetime
        if TIMESTAMP in train_df.columns:
            train_df[TIMESTAMP] = pd.to_datetime(train_df[TIMESTAMP])
            
            # Last interaction timestamp
            last_interaction = train_df.groupby(USER_ID)[TIMESTAMP].max()
            
            # Days since last interaction
            current_time = train_df[TIMESTAMP].max()
            days_since_last = (current_time - last_interaction).dt.days
            
            # Activity frequency
            interaction_count = train_df.groupby(USER_ID).size()
            first_interaction = train_df.groupby(USER_ID)[TIMESTAMP].min()
            days_active = (last_interaction - first_interaction).dt.days
            
            # Handle division by zero
            days_active = days_active.replace(0, 1)
            activity_frequency = interaction_count / days_active
            
            # Time of day preferences
            train_df['hour'] = train_df[TIMESTAMP].dt.hour
            user_hour_stats = train_df.groupby(USER_ID)['hour'].agg(['mean', 'std']).fillna(0)
            
            # Weekend activity ratio
            train_df['is_weekend'] = train_df[TIMESTAMP].dt.dayofweek.isin([5, 6]).astype(int)
            weekend_ratio = train_df.groupby(USER_ID)['is_weekend'].mean()
            
            # Create features DataFrame
            temporal_features = pd.DataFrame({
                'days_since_last_interaction': days_since_last,
                'activity_frequency': activity_frequency,
                'avg_interaction_hour': user_hour_stats['mean'],
                'hour_std': user_hour_stats['std'],
                'weekend_activity_ratio': weekend_ratio
            })
            
            return temporal_features
        
        return pd.DataFrame(index=train_df[USER_ID].unique())
    
    def _create_user_behavior_features(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Create features describing user reading behavior."""
        # Reading ratio
        read_ratio = train_df.groupby(USER_ID)[HAS_READ].mean()
        
        # Total interactions
        interaction_count = train_df.groupby(USER_ID).size()
        
        # Average rating given
        avg_rating = train_df.groupby(USER_ID)[RATING].mean()
        
        # Reading speed (books read per day)
        if TIMESTAMP in train_df.columns:
            # Get timestamps for read books only
            read_books = train_df[train_df[HAS_READ] == 1].copy()
            if not read_books.empty:
                first_read = read_books.groupby(USER_ID)[TIMESTAMP].min()
                last_read = read_books.groupby(USER_ID)[TIMESTAMP].max()
                reading_span = (last_read - first_read).dt.days
                reading_span = reading_span.replace(0, 1)  # Avoid division by zero
                read_count = read_books.groupby(USER_ID).size()
                reading_speed = read_count / reading_span
            else:
                reading_speed = pd.Series(0, index=interaction_count.index)
        else:
            reading_speed = pd.Series(0, index=interaction_count.index)
        
        # Consistency of reading (ratio of read to planned)
        read_count = train_df.groupby(USER_ID)[HAS_READ].sum()
        planned_count = interaction_count - read_count
        read_to_planned_ratio = read_count / (planned_count + 1)  # +1 to avoid division by zero
        
        # Create features DataFrame
        behavior_features = pd.DataFrame({
            'read_ratio': read_ratio,
            'total_interactions': interaction_count,
            'avg_rating_given': avg_rating.fillna(0),
            'reading_speed': reading_speed.fillna(0),
            'read_to_planned_ratio': read_to_planned_ratio.fillna(0)
        })
        
        return behavior_features
    
    def _create_user_genre_preferences(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Create features describing user genre preferences."""
        if 'book_genres' not in self.config.paths:
            print("Warning: book_genres data not available for genre preference features")
            return pd.DataFrame(index=train_df[USER_ID].unique())
        
        try:
            # Load book-genre mapping
            book_genres = pd.read_csv(self.config.paths['book_genres'], sep=';')
            
            # Merge with train data to get user-genre interactions
            user_book_genres = train_df.merge(book_genres, on=BOOK_ID, how='left')
            
            # Count interactions per user-genre
            user_genre_counts = user_book_genres.groupby([USER_ID, 'genre_id']).size().reset_index(name='count')
            
            # Pivot to get genre counts per user
            genre_pivot = user_genre_counts.pivot(index=USER_ID, columns='genre_id', values='count').fillna(0)
            
            # Rename columns to be more descriptive
            genre_pivot.columns = [f'genre_{col}_count' for col in genre_pivot.columns]
            
            # Calculate genre preference strength (entropy)
            def calculate_entropy(counts):
                total = counts.sum()
                if total == 0:
                    return 0
                probs = counts / total
                return -np.sum(probs * np.log2(probs + 1e-10))
            
            genre_counts = user_genre_counts.groupby(USER_ID)['count'].apply(calculate_entropy).rename('genre_entropy')
            
            # Top genres (most interacted with)
            top_genres = user_genre_counts.loc[user_genre_counts.groupby(USER_ID)['count'].idxmax()]
            top_genres = top_genres.set_index(USER_ID)['genre_id'].rename('top_genre')
            
            # Merge all genre features
            genre_features = genre_pivot.join(genre_counts, how='left')
            genre_features = genre_features.join(top_genres, how='left')
            
            # Create genre diversity metric
            genre_features['genre_diversity'] = (genre_features > 0).sum(axis=1) / len(genre_pivot.columns)
            
            return genre_features.fillna(0)
            
        except Exception as e:
            print(f"Error creating genre features: {e}")
            return pd.DataFrame(index=train_df[USER_ID].unique())
    
    def create_book_features(self, train_df: pd.DataFrame, books_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive book features for the item tower."""
        print("Creating book features...")
        
        # Base book features from metadata
        book_features = books_df.set_index(BOOK_ID).copy()
        
        # Statistical features from interactions
        book_stats = self._create_book_statistical_features(train_df)
        
        # Author features
        author_stats = self._create_author_features(train_df, books_df)
        
        # Merge all book features
        book_features = book_features.join(book_stats, how='left')
        book_features = book_features.join(author_stats, how='left')
        
        # Derived features
        book_features['publication_decade'] = (book_features['publication_year'] // 10) * 10
        book_features['is_recent'] = (book_features['publication_year'] >= 2010).astype(int)
        
        # Handle missing values
        for col in book_features.select_dtypes(include=[np.number]).columns:
            if book_features[col].isna().any():
                book_features[col] = book_features[col].fillna(book_features[col].median())
        
        # Normalize numerical features
        numerical_cols = book_features.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols:
            if book_features[col].std() > 0:
                book_features[col] = (book_features[col] - book_features[col].mean()) / book_features[col].std()
        
        print(f"Created {len(book_features.columns)} features for {len(book_features)} books")
        return book_features
    
    def _create_book_statistical_features(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features for books based on interactions."""
        # Total interactions
        interaction_count = train_df.groupby(BOOK_ID).size()
        
        # Read ratio
        read_ratio = train_df.groupby(BOOK_ID)[HAS_READ].mean()
        
        # Average rating
        avg_rating = train_df.groupby(BOOK_ID)[RATING].mean()
        
        # Popularity metrics
        unique_users = train_df.groupby(BOOK_ID)[USER_ID].nunique()
        
        # Create features DataFrame
        book_stats = pd.DataFrame({
            'total_interactions': interaction_count,
            'read_ratio': read_ratio,
            'avg_rating_received': avg_rating.fillna(0),
            'unique_readers': unique_users
        })
        
        return book_stats.fillna(0)
    
    def _create_author_features(self, train_df: pd.DataFrame, books_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for books based on their authors' statistics."""
        # Merge train data with book-author mapping
        train_with_authors = train_df.merge(books_df[[BOOK_ID, AUTHOR_ID]], on=BOOK_ID, how='left')
        
        # Author statistics
        author_stats = train_with_authors.groupby(AUTHOR_ID).agg(
            author_total_books=('book_id', 'count'),
            author_avg_has_read=(HAS_READ, 'mean'),
            author_avg_rating=(RATING, 'mean'),
            author_unique_readers=(USER_ID, 'nunique')
        ).rename(columns={
            'author_avg_has_read': 'author_read_ratio',
            'author_avg_rating': 'author_avg_rating',
            'author_unique_readers': 'author_popularity'
        })
        
        # Merge author stats back to books
        book_author_stats = books_df[[BOOK_ID, AUTHOR_ID]].merge(
            author_stats, on=AUTHOR_ID, how='left'
        ).set_index(BOOK_ID)
        
        # Drop author_id column to avoid duplication
        if AUTHOR_ID in book_author_stats.columns:
            book_author_stats = book_author_stats.drop(columns=[AUTHOR_ID])
        
        return book_author_stats.fillna(0)
    
    def create_text_features(self, books_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
        """Create text-based features for books using TF-IDF and BERT."""
        print("Creating text features...")
        
        # Merge books with descriptions
        books_with_desc = books_df.merge(descriptions_df, on=BOOK_ID, how='left')
        books_with_desc[DESCRIPTION] = books_with_desc[DESCRIPTION].fillna('')
        
        # Preprocess text
        books_with_desc['processed_description'] = books_with_desc[DESCRIPTION].apply(self._preprocess_text)
        
        # TF-IDF features
        tfidf_features = self._create_tfidf_features(books_with_desc['processed_description'].tolist(), books_with_desc[BOOK_ID].tolist())
        
        # BERT features (computationally expensive, so we cache them)
        if self.config.paths['text_features'].exists() and self.config.cache_embeddings:
            print("Loading cached BERT embeddings...")
            bert_features = pd.read_pickle(self.config.paths['text_features'])
        else:
            print("Generating BERT embeddings (this may take a while)...")
            bert_features = self._create_bert_features(books_with_desc)
            if self.config.cache_embeddings:
                bert_features.to_pickle(self.config.paths['text_features'])
        
        # Combine text features
        text_features = tfidf_features.join(bert_features, how='inner')
        
        print(f"Created {text_features.shape[1]} text features for {text_features.shape[0]} books")
        return text_features
    
    def _create_tfidf_features(self, descriptions: List[str], book_ids: List[int]) -> pd.DataFrame:
        """Create TF-IDF features from book descriptions."""
        print("Creating TF-IDF features...")
        
        # Initialize or load TF-IDF vectorizer
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.9,
                stop_words=['и', 'в', 'на', 'с', 'по', 'для', 'не', 'что', 'это', 'как', 'то', 'а', 'о', 'у', 'к', 'я']
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(descriptions)
        
        # Reduce dimensionality with SVD
        n_components = min(100, tfidf_matrix.shape[1])
        svd = TruncatedSVD(n_components=n_components, random_state=self.config.random_seed)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)
        
        # Create DataFrame
        tfidf_features = pd.DataFrame(
            tfidf_reduced,
            columns=[f'tfidf_{i}' for i in range(n_components)],
            index=book_ids
        )
        
        return tfidf_features
    
    def _create_bert_features(self, books_with_desc: pd.DataFrame) -> pd.DataFrame:
        """Create BERT embeddings for book descriptions."""
        # Filter to only books with non-empty descriptions
        valid_books = books_with_desc[books_with_desc['processed_description'].str.len() > 10]
        
        if len(valid_books) == 0:
            print("Warning: No valid book descriptions found for BERT embeddings")
            return pd.DataFrame(index=books_with_desc[BOOK_ID])
        
        print(f"Generating BERT embeddings for {len(valid_books)} books with descriptions...")
        
        # Get embeddings
        embeddings = self._get_bert_embeddings(valid_books['processed_description'].tolist())
        
        # Create DataFrame for valid books
        bert_features = pd.DataFrame(
            embeddings,
            columns=[f'bert_{i}' for i in range(embeddings.shape[1])],
            index=valid_books[BOOK_ID]
        )
        
        # Add zero vectors for books without descriptions
        all_book_ids = books_with_desc[BOOK_ID].unique()
        missing_books = [bid for bid in all_book_ids if bid not in bert_features.index]
        
        if missing_books:
            zero_vectors = np.zeros((len(missing_books), embeddings.shape[1]))
            missing_df = pd.DataFrame(
                zero_vectors,
                columns=[f'bert_{i}' for i in range(embeddings.shape[1])],
                index=missing_books
            )
            bert_features = pd.concat([bert_features, missing_df])
        
        # Reindex to match original book order
        bert_features = bert_features.reindex(all_book_ids)
        
        return bert_features.fillna(0)
    
    def create_interaction_features(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for user-book interactions."""
        print("Creating interaction features...")
        
        # Create interaction matrix
        interaction_df = train_df[[USER_ID, BOOK_ID, HAS_READ, RATING]].copy()
        
        # Add interaction type label
        interaction_df['interaction_type'] = interaction_df[HAS_READ].apply(
            lambda x: READ_CLASS if x == 1 else PLANNED_CLASS
        )
        
        # Create user-book pairs
        user_book_pairs = interaction_df.groupby([USER_ID, BOOK_ID]).agg(
            has_read=(HAS_READ, 'max'),
            rating=(RATING, 'mean'),
            interaction_count=(HAS_READ, 'count')
        ).reset_index()
        
        # Add label for classification (read=2, planned=1)
        user_book_pairs['label'] = user_book_pairs['has_read'].apply(lambda x: READ_CLASS if x == 1 else PLANNED_CLASS)
        
        print(f"Created interaction features for {len(user_book_pairs)} user-book pairs")
        return user_book_pairs
    
    def create_two_towers_features(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for two towers architecture from candidate pools."""
        print("Creating two towers features for candidates...")
        
        all_features = []
        
        for _, row in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="Processing candidates"):
            user_id = row[USER_ID]
            candidate_books = [int(bid) for bid in str(row['book_id_list']).split(',') if bid.strip()]
            
            # Get user features
            user_feat = self.user_features.loc[user_id].to_dict() if user_id in self.user_features.index else {}
            
            # Create feature set for each candidate book
            for book_id in candidate_books:
                features = {
                    USER_ID: user_id,
                    BOOK_ID: book_id
                }
                
                # Add user features
                for col, value in user_feat.items():
                    features[f'user_{col}'] = value
                
                # Add book features
                if book_id in self.book_features.index:
                    book_feat = self.book_features.loc[book_id].to_dict()
                    for col, value in book_feat.items():
                        features[f'book_{col}'] = value
                else:
                    # Default values for unknown books
                    for col in self.book_features.columns:
                        features[f'book_{col}'] = 0
                
                # Add text features if available
                if self.text_features is not None and book_id in self.text_features.index:
                    text_feat = self.text_features.loc[book_id].to_dict()
                    for col, value in text_feat.items():
                        features[f'text_{col}'] = value
                
                # Add interaction flag (whether this user interacted with this book in training)
                features['has_interaction'] = 0  # Will be set later based on actual data
                
                all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        print(f"Created features for {len(features_df)} candidate pairs")
        return features_df
    
    def add_interaction_flags(self, features_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
        """Add flags indicating whether user interacted with book in training data and interaction type."""
        print("Adding interaction flags...")
        
        # Create a set of (user_id, book_id) pairs from training data
        train_interactions = set(zip(train_df[USER_ID], train_df[BOOK_ID]))
        
        # Create a mapping of (user_id, book_id) to has_read value
        interaction_map = {}
        for _, row in train_df.iterrows():
            interaction_map[(row[USER_ID], row[BOOK_ID])] = row[HAS_READ]
        
        # Add interaction flags
        features_df['has_interaction'] = features_df.apply(
            lambda row: 1 if (row[USER_ID], row[BOOK_ID]) in train_interactions else 0,
            axis=1
        )
        
        # Add has_read flag for interactions
        features_df['interaction_has_read'] = features_df.apply(
            lambda row: interaction_map.get((row[USER_ID], row[BOOK_ID]), -1),
            axis=1
        )
        
        # Determine if book is a "cold candidate"
        # Cold candidates are books that:
        # 1. User has never interacted with in training
        # 2. AND book has low popularity OR doesn't match user's genre preferences
        features_df['is_cold_candidate'] = features_df['has_interaction'].apply(lambda x: 1 if x == 0 else 0)
        
        return features_df
    
    def filter_cold_candidates(self, features_df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
        """
        Filter out cold candidates that are unlikely to be relevant.
        This is a placeholder method - in a real implementation, you'd use a trained model
        to predict the likelihood that a cold candidate is actually relevant.
        """
        print("Filtering cold candidates...")
        
        # Placeholder logic - in reality, this would use a model to score cold candidates
        if 'cold_candidate_score' not in features_df.columns:
            # Simple heuristic: keep all candidates that have interactions
            # For cold candidates with no interactions, keep only those with book popularity above threshold
            if 'book_total_interactions' in features_df.columns:
                features_df['cold_candidate_score'] = features_df.apply(
                    lambda row: 1.0 if row['has_interaction'] == 1 else 
                               (row['book_total_interactions'] / features_df['book_total_interactions'].max()),
                    axis=1
                )
            else:
                # Default to keeping all candidates if we don't have popularity data
                features_df['cold_candidate_score'] = 1.0
        
        # Create a flag for candidates to keep
        features_df['keep_candidate'] = features_df.apply(
            lambda row: True if row['has_interaction'] == 1 or row['cold_candidate_score'] >= threshold else False,
            axis=1
        )
        
        # Count how many candidates we're filtering out
        total_candidates = len(features_df)
        filtered_out = total_candidates - features_df['keep_candidate'].sum()
        print(f"Filtered out {filtered_out} cold candidates out of {total_candidates} total candidates")
        
        return features_df
    
    def create_all_features(self, data: Dict[str, pd.DataFrame]) -> None:
        """Create all features for the two towers architecture."""
        print("=" * 80)
        print("STARTING FEATURE ENGINEERING FOR TWO TOWERS ARCHITECTURE")
        print("=" * 80)
        
        train_df = data['train']
        books_df = data['books']
        users_df = data['users']
        book_descriptions_df = data.get('book_descriptions', pd.DataFrame())
        
        # Create user features
        self.user_features = self.create_user_features(train_df, users_df)
        
        # Create book features
        self.book_features = self.create_book_features(train_df, books_df)
        
        # Create text features if available
        if not book_descriptions_df.empty:
            self.text_features = self.create_text_features(books_df, book_descriptions_df)
        
        if self.config.cache_features:
            self.save_features()
        
        print("=" * 80)
        print("FEATURE ENGINEERING COMPLETE")
        print(f"User features shape: {self.user_features.shape}")
        print(f"Book features shape: {self.book_features.shape}")
        if self.text_features is not None:
            print(f"Text features shape: {self.text_features.shape}")
        print("=" * 80)
