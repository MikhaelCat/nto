import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .model import TwoTowersModel
from .features import TwoTowersFeatureEngineer
from .config import Config, TWO_TOWER_PARAMS
from . import constants
from .data_processing import load_and_merge_data, expand_candidates

class TwoTowersPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.engineer = None
        self.user_features = None
        self.book_features = None
        self.text_features = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        print(f"Loading model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        config = checkpoint['config']
        TWO_TOWER_PARAMS.user_input_dim = config['user_input_dim']
        TWO_TOWER_PARAMS.book_input_dim = config['book_input_dim']
        
        self.model = TwoTowersModel(
            TWO_TOWER_PARAMS.user_input_dim,
            TWO_TOWER_PARAMS.book_input_dim
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded. Best NDCG: {checkpoint['best_ndcg']:.4f}")
    
    def prepare_features(self, data_dict):
        if self.engineer is None:
            self.engineer = TwoTowersFeatureEngineer(Config)
            self.engineer.create_all_features(data_dict)
        
        # Преобразуем строковые столбцы в числовые
            self.user_features = self._convert_features_to_numeric(self.engineer.user_features)
            self.book_features = self._convert_features_to_numeric(self.engineer.book_features)
            self.text_features = self._convert_features_to_numeric(self.engineer.text_features) if self.engineer.text_features is not None else None
        
            print(f"User features shape: {self.user_features.shape}")
            print(f"Book features shape: {self.book_features.shape}")
            if self.text_features is not None:
                print(f"Text features shape: {self.text_features.shape}")
    
    def predict_batch(self, user_ids, book_ids):
        if self.model is None:
            raise ValueError("Model not loaded")
            #fix
        user_id_to_idx = {uid: idx for idx, uid in enumerate(self.user_features.index)}
        book_id_to_idx = {bid: idx for idx, bid in enumerate(self.book_features.index)}

        user_indices = [user_id_to_idx[uid] for uid in user_ids if uid in user_id_to_idx]
        book_indices = [book_id_to_idx[bid] for bid in book_ids if bid in book_id_to_idx]

        if len(user_indices) == 0 or len(book_indices) == 0:
            return np.array([])
    
        user_features_numeric = self.user_features.select_dtypes(include=[np.number])
        book_features_numeric = self.book_features.select_dtypes(include=[np.number])
    

        if user_features_numeric.shape[1] == 0 or book_features_numeric.shape[1] == 0:
            raise ValueError("No numeric features found!")
    
        print(f"Using {user_features_numeric.shape[1]} user features and {book_features_numeric.shape[1]} book features")
   

        user_features_tensor = torch.FloatTensor(user_features_numeric.values[user_indices].astype(np.float32))
        book_features_tensor = torch.FloatTensor(book_features_numeric.values[book_indices].astype(np.float32))

        if self.text_features is not None:
            text_features_tensor = torch.FloatTensor(self.text_features.values[book_indices].astype(np.float32))
            book_features_tensor = torch.cat([book_features_tensor, text_features_tensor], dim=1)

        with torch.no_grad():
            user_features_tensor = user_features_tensor.to(self.device)
            book_features_tensor = book_features_tensor.to(self.device)
    
            logits = self.model(user_features_tensor, book_features_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs
    def _convert_features_to_numeric(self, df):
        """Преобразует все столбцы DataFrame в числовые"""
        if df is None:
            return None
    
        df = df.copy()

        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                # Для строковых значений используем хэш
                df[col] = df[col].astype(str).apply(lambda x: hash(str(x)) % 10000)
            elif df[col].dtype.name == 'category':
                # Для категориальных значений используем коды
                 df[col] = df[col].cat.codes
        
            elif df[col].dtype == bool:
                # Булевы значения в 0/1
                df[col] = df[col].astype(int)
        
        return df

    def rank_candidates(self, user_id, candidate_books, k=20):
        if len(candidate_books) == 0:
            return []
        
        # Предсказание вероятностей
        user_ids = [user_id] * len(candidate_books)
        probs = self.predict_batch(user_ids, candidate_books)
        
        if len(probs) == 0:
            return candidate_books[:k]
        
        scores = probs[:, 1] * 1.0 + probs[:, 2] * 2.0
        
        # по убыванию скора
        sorted_indices = np.argsort(scores)[::-1]
        ranked_books = [candidate_books[i] for i in sorted_indices]
        
        return ranked_books[:k]
    
    def apply_hierarchical_constraint(self, ranked_books, probs):
        if len(ranked_books) == 0:
            return ranked_books
        
        # предсказанные классы
        pred_classes = np.argmax(probs, axis=1)
        
        # Группируем книги по классам
        books_by_class = {0: [], 1: [], 2: []}
        for book_id, cls in zip(ranked_books, pred_classes):
            if book_id in books_by_class:
                books_by_class[cls].append(book_id)
        
        reordered = []
        for cls in [2, 1, 0]:  
            reordered.extend(books_by_class[cls])
        
        return reordered


def predict():
    print("=" * 60)
    print("GENERATING PREDICTIONS WITH TWO-TOWERS MODEL")
    print("=" * 60)
    
    # Загрузка модели
    model_path = Config.MODEL_DIR / Config.MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please train the model first."
        )
    
    predictor = TwoTowersPredictor(model_path)
    
    # Загрузка данных
    print("\nLoading data...")
    data = load_and_merge_data()
    
    # Загрузка targets и candidates
    targets_df = pd.read_csv(
        Config.RAW_DATA_DIR / constants.TARGETS_FILENAME,
        dtype={constants.COL_USER_ID: "int32"}
    )
    
    candidates_df = pd.read_csv(
        Config.RAW_DATA_DIR / constants.CANDIDATES_FILENAME,
        dtype={constants.COL_USER_ID: "int32"}
    )
    
    print(f"Targets: {len(targets_df):,} users")
    print(f"Candidates: {len(candidates_df):,} users")
    
    # Подготовка признаков
    print("\nPreparing features...")
    predictor.prepare_features(data)
    
    # Предсказания для каждого пользователя
    print("\nGenerating predictions...")
    submission_rows = []
    
    for user_id in tqdm(targets_df[constants.COL_USER_ID], desc="Processing users"):
        user_candidates = candidates_df[
            candidates_df[constants.COL_USER_ID] == user_id
        ]
        
        if len(user_candidates) == 0:
            submission_rows.append({
                constants.COL_USER_ID: user_id,
                constants.COL_BOOK_ID_LIST: ""
            })
            continue
        
        # Разбираем список кандидатов
        candidate_list = user_candidates.iloc[0]['book_id_list']
        candidate_books = [int(bid) for bid in str(candidate_list).split(',') if bid.strip()]
        
        # Ранжируем кандидатов
        ranked_books = predictor.rank_candidates(
            user_id, candidate_books, constants.MAX_RANKING_LENGTH
        )
        
        if len(ranked_books) > 0:
            user_ids = [user_id] * len(ranked_books)
            probs = predictor.predict_batch(user_ids, ranked_books)
            if len(probs) > 0:
                ranked_books = predictor.apply_hierarchical_constraint(ranked_books, probs)
        
        book_id_list = ",".join(str(book_id) for book_id in ranked_books)
        submission_rows.append({
            constants.COL_USER_ID: user_id,
            constants.COL_BOOK_ID_LIST: book_id_list
        })
    
    # Создаем submission DataFrame
    submission_df = pd.DataFrame(submission_rows)
    
    # Сохраняем
    Config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = Config.SUBMISSION_DIR / "submission_two_towers.csv"
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n{'='*60}")
    print("PREDICTIONS COMPLETE")
    print(f"Submission saved to: {submission_path}")
    print(f"Submission shape: {submission_df.shape}")
    
    # Статистика
    non_empty = submission_df[submission_df[constants.COL_BOOK_ID_LIST] != ""].shape[0]
    avg_books = submission_df[constants.COL_BOOK_ID_LIST].apply(
        lambda x: len(x.split(',')) if x else 0
    ).mean()
    
    print(f"Users with recommendations: {non_empty}/{len(submission_df)}")
    print(f"Average books per user: {avg_books:.1f}")
    print(f"{'='*60}")
    
    return submission_df


if __name__ == "__main__":
    submission = predict()