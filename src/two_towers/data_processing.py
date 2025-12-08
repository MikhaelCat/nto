
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from . import constants
from .config import Config


def load_and_merge_data() -> Dict[str, pd.DataFrame]:
    print("=" * 60)
    print("LOADING AND MERGING DATA")
    print("=" * 60)
    
    data_dict = {}
    
    try:
        print("Loading train data...")
        train_path = Config.RAW_DATA_DIR / constants.TRAIN_FILENAME
        
        if not train_path.exists():
            raise FileNotFoundError(f"Train data not found at {train_path}")
        
        train_df = pd.read_csv(train_path)
        print(f"Train data shape: {train_df.shape}")
        
        if constants.COL_TIMESTAMP in train_df.columns:
            train_df[constants.COL_TIMESTAMP] = pd.to_datetime(train_df[constants.COL_TIMESTAMP])
        
        data_dict['train'] = train_df
        
        print("Loading books data...")
        books_path = Config.RAW_DATA_DIR / constants.BOOK_DATA_FILENAME
        
        if not books_path.exists():
            raise FileNotFoundError(f"Books data not found at {books_path}")
        
        books_df = pd.read_csv(books_path)
        print(f"Books data shape: {books_df.shape}")
        
        books_df = _clean_books_data(books_df)
        data_dict['books'] = books_df
        
        print("Loading users data...")
        users_path = Config.RAW_DATA_DIR / constants.USER_DATA_FILENAME
        
        if not users_path.exists():
            raise FileNotFoundError(f"Users data not found at {users_path}")
        
        users_df = pd.read_csv(users_path)
        print(f"Users data shape: {users_df.shape}")
        
        users_df = _clean_users_data(users_df)
        data_dict['users'] = users_df

        print("Loading genres data...")
        genres_path = Config.RAW_DATA_DIR / constants.GENRES_FILENAME
        
        if genres_path.exists():
            genres_df = pd.read_csv(genres_path)
            print(f"Genres data shape: {genres_df.shape}")
            data_dict['genres'] = genres_df
        else:
            print("Genres data not found, continuing without it...")
            data_dict['genres'] = pd.DataFrame()
        
        print("Loading book_genres data...")
        book_genres_path = Config.RAW_DATA_DIR / constants.BOOK_GENRES_FILENAME
        
        if book_genres_path.exists():
            book_genres_df = pd.read_csv(book_genres_path)
            print(f"Book genres data shape: {book_genres_df.shape}")
            data_dict['book_genres'] = book_genres_df
        else:
            print("Book genres data not found, continuing without it...")
            data_dict['book_genres'] = pd.DataFrame()
        
        print("Loading book descriptions...")
        descriptions_path = Config.RAW_DATA_DIR / constants.BOOK_DESCRIPTIONS_FILENAME
        
        if descriptions_path.exists():
            try:
                descriptions_df = pd.read_csv(descriptions_path, sep=';', on_bad_lines='skip')
                if len(descriptions_df.columns) == 1:
                    descriptions_df = pd.read_csv(descriptions_path, sep=',', on_bad_lines='skip')
                print(f"Descriptions data shape: {descriptions_df.shape}")
                data_dict['book_descriptions'] = descriptions_df
            except Exception as e:
                print(f"Error loading descriptions: {e}")
                data_dict['book_descriptions'] = pd.DataFrame()
        else:
            print("Book descriptions not found, continuing without them...")
            data_dict['book_descriptions'] = pd.DataFrame()
        
        print("Loading candidates data...")
        candidates_path = Config.RAW_DATA_DIR / constants.CANDIDATES_FILENAME
        
        if candidates_path.exists():
            candidates_df = pd.read_csv(candidates_path)
            print(f"Candidates data shape: {candidates_df.shape}")
            data_dict['candidates'] = candidates_df
        else:
            print("Candidates data not found (may be needed for inference)...")
            data_dict['candidates'] = pd.DataFrame()
        
        print("Loading targets data...")
        targets_path = Config.RAW_DATA_DIR / constants.TARGETS_FILENAME
        
        if targets_path.exists():
            targets_df = pd.read_csv(targets_path)
            print(f"Targets data shape: {targets_df.shape}")
            data_dict['targets'] = targets_df
        else:
            print("Targets data not found (may be needed for inference)...")
            data_dict['targets'] = pd.DataFrame()
        
        print("\nData loading complete!")
        print(f"Loaded {len(data_dict)} datasets")
        
        return data_dict
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def _clean_books_data(books_df: pd.DataFrame) -> pd.DataFrame:
    df = books_df.copy()
    
    if constants.COL_PUBLICATION_YEAR in df.columns:
        current_year = pd.Timestamp.now().year
        df[constants.COL_PUBLICATION_YEAR] = df[constants.COL_PUBLICATION_YEAR].clip(1800, current_year)
        
        median_year = df[constants.COL_PUBLICATION_YEAR].median()
        df[constants.COL_PUBLICATION_YEAR] = df[constants.COL_PUBLICATION_YEAR].fillna(median_year)

    if constants.COL_AVG_RATING in df.columns:
        df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(0)
        df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].clip(0, 10)
    
    if constants.COL_AUTHOR_ID in df.columns:
        df[constants.COL_AUTHOR_ID] = df[constants.COL_AUTHOR_ID].fillna(-1).astype(int)
    
    for col in [constants.COL_LANGUAGE, constants.COL_PUBLISHER]:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int)
    
    return df


def _clean_users_data(users_df: pd.DataFrame) -> pd.DataFrame:
    df = users_df.copy()
    if constants.COL_GENDER in df.columns:
        df[constants.COL_GENDER] = df[constants.COL_GENDER].fillna(0).astype(int)
    
    if constants.COL_AGE in df.columns:
        df[constants.COL_AGE] = df[constants.COL_AGE].clip(10, 100)
        
        
        median_age = df[constants.COL_AGE].median()
        df[constants.COL_AGE] = df[constants.COL_AGE].fillna(median_age)
    
    return df


def expand_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
    print("Expanding candidates...")
    
    all_pairs = []
    
    for _, row in candidates_df.iterrows():
        user_id = row[constants.COL_USER_ID]
        book_list_str = row[constants.COL_BOOK_ID_LIST]
        

        if isinstance(book_list_str, str):
            book_ids = [int(bid.strip()) for bid in book_list_str.split(',') if bid.strip().isdigit()]
        else:
            book_ids = []
        
        for book_id in book_ids:
            all_pairs.append({
                constants.COL_USER_ID: user_id,
                constants.COL_BOOK_ID: book_id
            })
    
    expanded_df = pd.DataFrame(all_pairs)
    print(f"Expanded {len(candidates_df)} rows to {len(expanded_df)} pairs")
    
    return expanded_df


def create_user_book_interaction_matrix(train_df: pd.DataFrame) -> pd.DataFrame:
    print("Creating user-book interaction matrix...")
    interaction_df = train_df.copy()
    interaction_df['interaction_type'] = interaction_df[constants.HAS_READ].map({
        1: 'read',
        0: 'planned'
    })
    
    pivot_df = interaction_df.pivot_table(
        index=constants.USER_ID,
        columns=constants.BOOK_ID,
        values='interaction_type',
        aggfunc='first'  
    ).fillna('none')
    
    print(f"Interaction matrix shape: {pivot_df.shape}")
    
    return pivot_df


def prepare_training_samples(train_df: pd.DataFrame, books_df: pd.DataFrame, 
                            negative_ratio: float = 2.0) -> pd.DataFrame:
    print("Preparing training samples...")
    
    positive_samples = train_df[[constants.USER_ID, constants.BOOK_ID, constants.HAS_READ]].copy()
    positive_samples['label'] = positive_samples[constants.HAS_READ].map({
        1: constants.READ_CLASS,    
        0: constants.PLANNED_CLASS  
    })

    all_users = train_df[constants.USER_ID].unique()
    all_books = books_df[constants.BOOK_ID].unique()
    
    user_book_pairs = set(zip(
        train_df[constants.USER_ID], 
        train_df[constants.BOOK_ID]
    ))
    

    negative_samples = []
    n_positive = len(positive_samples)
    n_negative_target = int(n_positive * negative_ratio)
    
    print(f"Generating {n_negative_target} negative samples...")
    
    for user_id in all_users:
        user_interacted_books = set(
            train_df[train_df[constants.USER_ID] == user_id][constants.BOOK_ID]
        )
        
        available_books = [b for b in all_books if b not in user_interacted_books]
        
        if available_books:
            n_samples = min(10, len(available_books))  
            sampled_books = np.random.choice(available_books, n_samples, replace=False)
            
            for book_id in sampled_books:
                negative_samples.append({
                    constants.USER_ID: user_id,
                    constants.BOOK_ID: book_id,
                    'label': constants.COLD_CLASS
                })
    
    negative_df = pd.DataFrame(negative_samples)
    

    if len(negative_df) < n_negative_target:
        additional_needed = n_negative_target - len(negative_df)
        print(f"Generating additional {additional_needed} negative samples...")
        
        additional_samples = []
        attempts = 0
        
        while len(additional_samples) < additional_needed and attempts < 1000:
            user_id = np.random.choice(all_users)
            book_id = np.random.choice(all_books)
            
            if (user_id, book_id) not in user_book_pairs:
                additional_samples.append({
                    constants.USER_ID: user_id,
                    constants.BOOK_ID: book_id,
                    'label': constants.COLD_CLASS
                })
            
            attempts += 1
        
        if additional_samples:
            additional_df = pd.DataFrame(additional_samples)
            negative_df = pd.concat([negative_df, additional_df], ignore_index=True)
    
    all_samples = pd.concat([
        positive_samples[[constants.USER_ID, constants.BOOK_ID, 'label']],
        negative_df
    ], ignore_index=True)
    
    print(f"Total samples: {len(all_samples)}")
    print(f"  - Positive (read/planned): {len(positive_samples)}")
    print(f"  - Negative (cold): {len(negative_df)}")
    
    return all_samples


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame, books_df: pd.DataFrame = None) -> pd.DataFrame:
    print("Adding aggregate features...")
    
    result_df = df.copy()
    
    print("  Computing user statistics...")
    user_stats = train_df.groupby(constants.USER_ID).agg({
        constants.HAS_READ: ['mean', 'sum', 'count'],
        constants.RATING: ['mean', 'std']
    }).reset_index()
    
    user_stats.columns = [
        constants.USER_ID,
        'user_read_ratio',
        'user_read_count',
        'user_interaction_count',
        'user_mean_rating',
        'user_rating_std'
    ]
    
    user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)

    print("  Computing book statistics...")
    book_stats = train_df.groupby(constants.BOOK_ID).agg({
        constants.HAS_READ: ['mean', 'sum', 'count'],
        constants.RATING: ['mean', 'std'],
        constants.USER_ID: 'nunique'
    }).reset_index()
    
    book_stats.columns = [
        constants.BOOK_ID,
        'book_read_ratio',
        'book_read_count',
        'book_interaction_count',
        'book_mean_rating',
        'book_rating_std',
        'book_unique_users'
    ]
    
    book_stats['book_rating_std'] = book_stats['book_rating_std'].fillna(0)
    
    if books_df is not None and constants.AUTHOR_ID in books_df.columns:
        print("  Computing author statistics...")
        
        train_with_author = train_df.merge(
            books_df[[constants.BOOK_ID, constants.AUTHOR_ID]], 
            on=constants.BOOK_ID, 
            how='left'
        )
        
        if not train_with_author.empty and constants.AUTHOR_ID in train_with_author.columns:
            author_stats = train_with_author.groupby(constants.AUTHOR_ID).agg({
                constants.HAS_READ: ['mean', 'sum', 'count'],
                constants.RATING: ['mean', 'std'],
                constants.BOOK_ID: 'nunique'
            }).reset_index()
            
            author_stats.columns = [
                constants.AUTHOR_ID,
                'author_read_ratio',
                'author_read_count',
                'author_interaction_count',
                'author_mean_rating',
                'author_rating_std',
                'author_unique_books'
            ]
            
            author_stats['author_rating_std'] = author_stats['author_rating_std'].fillna(0)
            
            book_author_stats = books_df[[constants.BOOK_ID, constants.AUTHOR_ID]].merge(
                author_stats, on=constants.AUTHOR_ID, how='left'
            )
            if constants.AUTHOR_ID in result_df.columns or constants.BOOK_ID in result_df.columns:
                if constants.BOOK_ID in result_df.columns and constants.BOOK_ID in book_author_stats.columns:
                    result_df = result_df.merge(
                        book_author_stats.drop(columns=[constants.AUTHOR_ID]), 
                        on=constants.BOOK_ID, 
                        how='left'
                    )
    
    result_df = result_df.merge(user_stats, on=constants.USER_ID, how='left')
    result_df = result_df.merge(book_stats, on=constants.BOOK_ID, how='left')
    
    for col in result_df.columns:
        if col.startswith(('user_', 'book_', 'author_')) and result_df[col].dtype in ['float64', 'int64']:
            if result_df[col].notna().any():
                median_val = result_df[col].median()
            else:
                median_val = 0
            result_df[col] = result_df[col].fillna(median_val)
    
    print(f"Added aggregate features. Total columns: {len(result_df.columns)}")
    
    return result_df


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame = None) -> pd.DataFrame:
    print("Handling missing values...")
    
    result_df = df.copy()
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in numeric_cols:
        if result_df[col].isnull().any():
            if result_df[col].notna().any():
                median_val = result_df[col].median()
            else:
                median_val = 0
            result_df[col] = result_df[col].fillna(median_val)
    
    for col in categorical_cols:
        if result_df[col].isnull().any():
            if not result_df[col].mode().empty:
                mode_val = result_df[col].mode()[0]
            else:
                mode_val = 'unknown'
            result_df[col] = result_df[col].fillna(mode_val)
    
    if train_df is not None:
        for col in numeric_cols:
            if col in train_df.columns:
                if train_df[col].notna().any():
                    train_median = train_df[col].median()
                else:
                    train_median = 0
                if col in result_df.columns:
                    if result_df[col].notna().any():
                        q1 = result_df[col].quantile(0.25)
                        q3 = result_df[col].quantile(0.75)
                        iqr = q3 - q1
                        if iqr > 0:  
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            
                            outliers = (result_df[col] < lower_bound) | (result_df[col] > upper_bound)
                            if outliers.any():
                                result_df.loc[outliers, col] = train_median
    
    remaining_nans = result_df.isna().sum().sum()
    if remaining_nans > 0:
        print(f"Warning: {remaining_nans} missing values remaining after cleanup")
    
    return result_df


def prepare_features_for_training(train_df: pd.DataFrame, books_df: pd.DataFrame, 
                                 users_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    print("=" * 60)
    print("PREPARING FEATURES FOR TRAINING")
    print("=" * 60)
    samples_df = prepare_training_samples(train_df, books_df)
    samples_df = samples_df.merge(users_df, on=constants.USER_ID, how='left')
    samples_df = samples_df.merge(books_df, on=constants.BOOK_ID, how='left')
    samples_df = add_aggregate_features(samples_df, train_df, books_df)
    samples_df = handle_missing_values(samples_df, train_df)
    
    print(f"\nFinal training dataset shape: {samples_df.shape}")
    print(f"Feature columns: {len(samples_df.columns)}")
    
    if 'label' in samples_df.columns:
        print(f"Target distribution:\n{samples_df['label'].value_counts().sort_index()}")
    
    return {
        'features': samples_df,
        'user_features': users_df.set_index(constants.USER_ID),
        'book_features': books_df.set_index(constants.BOOK_ID)
    }


def save_processed_data(data_dict: Dict[str, pd.DataFrame], output_dir: Path = None):
    if output_dir is None:
        output_dir = Config.PROCESSED_DATA_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving processed data to {output_dir}...")
    
    for name, df in data_dict.items():
        if name == 'features':
            output_path = output_dir / constants.PROCESSED_DATA_FILENAME
            df.to_parquet(output_path, engine='pyarrow')
            print(f"  Saved {name} to {output_path}")
        else:
            output_path = output_dir / f"{name}.parquet"
            df.to_parquet(output_path, engine='pyarrow')
            print(f"  Saved {name} to {output_path}")


def load_processed_data(input_dir: Path = None) -> Dict[str, pd.DataFrame]:
    if input_dir is None:
        input_dir = Config.PROCESSED_DATA_DIR
    
    data_dict = {}
    
    print(f"Loading processed data from {input_dir}...")
    
    features_path = input_dir / constants.PROCESSED_DATA_FILENAME
    if features_path.exists():
        data_dict['features'] = pd.read_parquet(features_path, engine='pyarrow')
        print(f"  Loaded features from {features_path}")
    
    for file_path in input_dir.glob("*.parquet"):
        if file_path.name != constants.PROCESSED_DATA_FILENAME:
            name = file_path.stem
            data_dict[name] = pd.read_parquet(file_path, engine='pyarrow')
            print(f"  Loaded {name} from {file_path}")
    
    return data_dict


if __name__ == "__main__":
    print("Testing data processing module...")
    data = load_and_merge_data()
    features_dict = prepare_features_for_training(
        data['train'],
        data['books'],
        data['users']
    )
    
    save_processed_data(features_dict)
    
    print("\nData processing completed successfully!")