import pandas as pd
import numpy as np
from src.two_towers.predict import predict
from src.two_towers.config import Config

def test_predictions():
    print("Testing prediction pipeline...")
    
    # Запускаем предсказания
    submission_df = predict()
    
    # Проверяем результат
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    # Проверяем, что файл создан
    submission_path = Config.SUBMISSION_DIR / "submission_two_towers.csv"
    if submission_path.exists():
        print(f"✓ Submission file created: {submission_path}")
    else:
        print(f"✗ Submission file NOT created!")
        return
    
    # Читаем файл
    df = pd.read_csv(submission_path)
    print(f"✓ File shape: {df.shape}")
    
    # Проверяем колонки
    expected_columns = ['user_id', 'book_id_list']
    if all(col in df.columns for col in expected_columns):
        print(f"✓ All expected columns present")
    else:
        missing = [col for col in expected_columns if col not in df.columns]
        print(f"✗ Missing columns: {missing}")
    
    # Проверяем, что book_id_list не пустой
    empty_lists = df['book_id_list'].isna().sum() + (df['book_id_list'] == '').sum()
    if empty_lists == 0:
        print(f"✓ All users have recommendations")
    else:
        print(f"✗ {empty_lists} users have empty recommendations")
    
    # Проверяем длину рекомендаций
    df['rec_length'] = df['book_id_list'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) and x != '' else 0)
    avg_length = df['rec_length'].mean()
    
    if avg_length > 0:
        print(f"✓ Average recommendation length: {avg_length:.1f}")
        print(f"  Min: {df['rec_length'].min()}")
        print(f"  Max: {df['rec_length'].max()}")
    else:
        print(f"✗ All recommendations are empty!")
    
    # Выводим несколько примеров
    print("\nSample recommendations (first 5 users):")
    for i, row in df.head().iterrows():
        books = str(row['book_id_list']).split(',') if row['book_id_list'] else []
        print(f"User {row['user_id']}: {len(books)} books")
        if books and len(books) > 0:
            print(f"  First 3 books: {books[:3]}")

if __name__ == "__main__":
    test_predictions()