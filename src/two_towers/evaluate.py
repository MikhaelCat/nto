import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .config import Config
from . import constants


def calculate_ndcg_at_k(submission_df: pd.DataFrame, solution_df: pd.DataFrame, k: int = 20) -> float:
    print("Calculating NDCG@20...")
    
    ndcg_scores = []
    
    for _, user_row in submission_df.iterrows():
        user_id = user_row[constants.COL_USER_ID]
        pred_books_str = user_row[constants.COL_BOOK_ID_LIST]
        user_solution = solution_df[solution_df[constants.COL_USER_ID] == user_id]
        
        if len(user_solution) == 0:
            continue
        
        read_books = set()
        planned_books = set()
        
        if not pd.isna(user_solution.iloc[0]['book_id_list_read']):
            read_books = set(int(b) for b in str(user_solution.iloc[0]['book_id_list_read']).split(',') if b.strip())
        
        if not pd.isna(user_solution.iloc[0]['book_id_list_planned']):
            planned_books = set(int(b) for b in str(user_solution.iloc[0]['book_id_list_planned']).split(',') if b.strip())
        
        if pd.isna(pred_books_str) or pred_books_str == '':
            pred_books = []
        else:
            pred_books = [int(b) for b in str(pred_books_str).split(',') if b.strip()]
        
        relevance_scores = []
        for book_id in pred_books[:k]:  
            if book_id in read_books:
                relevance_scores.append(2.0)  
            elif book_id in planned_books:
                relevance_scores.append(1.0)  
            else:
                relevance_scores.append(0.0)  
        
        dcg = 0.0
        for i, rel in enumerate(relevance_scores, 1):
            dcg += rel / np.log2(i + 1)
        
        ideal_relevance = []
        ideal_relevance.extend([2.0] * len(read_books))
        ideal_relevance.extend([1.0] * len(planned_books))
        ideal_relevance = ideal_relevance[:k]
        
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance, 1):
            idcg += rel / np.log2(i + 1)
        
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0.0
        
        ndcg_scores.append(ndcg)
    
    if ndcg_scores:
        avg_ndcg = np.mean(ndcg_scores)
        print(f"Average NDCG@{k}: {avg_ndcg:.6f}")
        print(f"Evaluated {len(ndcg_scores)} users")
        return avg_ndcg
    else:
        print("No users evaluated")
        return 0.0


def evaluate_submission(submission_path: Path = None, solution_path: Path = None):
    if submission_path is None:
        submission_path = Config.SUBMISSION_DIR / "submission_two_towers.csv"
    
    if solution_path is None:
        solution_path = Config.PROCESSED_DATA_DIR / "validation_solution.csv"
    
    print("=" * 60)
    print("EVALUATING SUBMISSION")
    print("=" * 60)
    
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")
    
    submission_df = pd.read_csv(submission_path)
    print(f"Submission shape: {submission_df.shape}")
    
    if not solution_path.exists():
        print(f"Warning: Solution file not found at {solution_path}")
        print("Creating dummy solution for testing...")
        
        solution_df = pd.DataFrame({
            constants.COL_USER_ID: submission_df[constants.COL_USER_ID].unique(),
            'book_id_list_read': '',
            'book_id_list_planned': '',
            'stage': 'public'
        })
    else:
        solution_df = pd.read_csv(solution_path)
        print(f"Solution shape: {solution_df.shape}")
    
    ndcg20 = calculate_ndcg_at_k(submission_df, solution_df, k=20)
    print("\nAdditional Metrics:")
    
    avg_length = submission_df[constants.COL_BOOK_ID_LIST].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) and x != '' else 0
    ).mean()
    print(f"Average recommendation length: {avg_length:.2f}")
    
    users_with_recs = submission_df[
        submission_df[constants.COL_BOOK_ID_LIST].notna() & 
        (submission_df[constants.COL_BOOK_ID_LIST] != '')
    ].shape[0]
    coverage = users_with_recs / len(submission_df) * 100
    print(f"User coverage: {coverage:.2f}% ({users_with_recs}/{len(submission_df)})")
    
    all_books = []
    for book_list in submission_df[constants.COL_BOOK_ID_LIST]:
        if pd.notna(book_list) and book_list != '':
            all_books.extend([int(b) for b in str(book_list).split(',')])
    
    unique_books = len(set(all_books))
    print(f"Unique books recommended: {unique_books}")
    
    print("\n" + "=" * 60)
    print(f"FINAL NDCG@20 SCORE: {ndcg20:.6f}")
    print("=" * 60)
    
    return ndcg20


if __name__ == "__main__":
    score = evaluate_submission()
    print(f"\nEvaluation complete. Score: {score:.6f}")