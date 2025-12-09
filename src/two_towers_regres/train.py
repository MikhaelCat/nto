import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.two_towers import constants, config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.two_towers.model import TwoTowersModel
from src.two_towers.data_split import final
from src.two_towers import TWO_TOWER_PARAMS, Config


def calculate_ndcg(predictions: torch.Tensor, labels: torch.Tensor, k: int = 20) -> float:
    weights = torch.tensor([0.0, 1.0, 2.0], device=predictions.device)
    pred_scores = predictions @ weights
    _, indices = torch.sort(pred_scores, descending=True)
    sorted_labels = labels[indices][:k]
    dcg = 0.0
    for i, rel in enumerate(sorted_labels, 1):
        gain = 2.0 ** float(rel) - 1.0
        dcg += gain / np.log2(i + 1)
    
    # Рассчитываем идеальный DCG (IDCG)
    sorted_true = torch.sort(labels, descending=True)[0][:k]
    idcg = 0.0
    for i, rel in enumerate(sorted_true, 1):
        gain = 2.0 ** float(rel) - 1.0
        idcg += gain / np.log2(i + 1)
    
    return dcg / idcg if idcg > 0 else 0.0


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Обучение на одной эпохе для регрессии"""
    model.train()
    total_loss = 0
    total_mae = 0
    total_samples = 0
    
    for batch_idx, (user_feat, book_feat, labels) in enumerate(pbar):
        user_feat = user_feat.to(device)
        book_feat = book_feat.to(device)
        labels = labels.to(device).float()  # Преобразуем во float для регрессии
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(user_feat, book_feat)
        
        # Регрессионная loss
        loss = criterion(predictions.squeeze(), labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Метрики
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.abs(predictions.squeeze() - labels).sum().item()
        total_samples += batch_size
    
    return total_loss / total_samples, total_mae / total_samples


def validate_epoch(model, val_loader, criterion, device):
    """Валидация на одной эпохе"""
    model.eval()
    total_loss = 0
    total_ndcg = 0
    total_correct = 0
    total_samples = 0
    
    # Матрица ошибок для 3 классов
    confusion_matrix = torch.zeros(3, 3, device=device)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for user_feat, book_feat, labels in pbar:
            user_feat = user_feat.to(device)
            book_feat = book_feat.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(user_feat, book_feat)
            loss = criterion(logits, labels)
            
            # Метрики
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Accuracy
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            
            # NDCG
            probs = torch.softmax(logits, dim=1)
            ndcg = calculate_ndcg(probs, labels, k=min(20, batch_size))
            total_ndcg += ndcg * batch_size
            
            # Confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    # Выводим матрицу ошибок
    print("\nConfusion Matrix:")
    print(confusion_matrix.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    avg_ndcg = total_ndcg / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, avg_ndcg, accuracy


def train():
    """Основная функция обучения"""
    print("=" * 60)
    print("TRAINING TWO-TOWERS MODEL")
    print("=" * 60)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Загружаем данные
    train_loader, val_loader = final()
    
    # Создаем модель
    model = TwoTowersModel(
        TWO_TOWER_PARAMS.user_input_dim,
        TWO_TOWER_PARAMS.book_input_dim
    ).to(device)
    
    print(f"Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=TWO_TOWER_PARAMS.learning_rate,
        weight_decay=1e-5
    )
    
    # Плато-шедулер
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3, 
       # verbose=True
    )
    
    # Ранняя остановка
    best_ndcg = 0
    patience_counter = 0
    patience = 7
    
    # История обучения
    history = {
        'train_loss': [], 'train_ndcg': [],
        'val_loss': [], 'val_ndcg': [], 'val_acc': []
    }
    
    # Цикл обучения
    for epoch in range(TWO_TOWER_PARAMS.num_epochs):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1}/{TWO_TOWER_PARAMS.num_epochs}")
        print(f"{'='*40}")
        
        # Обучение
        train_loss, train_ndcg = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Валидация
        val_loss, val_ndcg, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Обновляем историю
        history['train_loss'].append(train_loss)
        history['train_ndcg'].append(train_ndcg)
        history['val_loss'].append(val_loss)
        history['val_ndcg'].append(val_ndcg)
        history['val_acc'].append(val_acc)
        
        # Обновляем шедулер
        scheduler.step(val_ndcg)
        
        # Сохраняем лучшую модель
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            patience_counter = 0
            
            # Сохраняем   
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ndcg': best_ndcg,
                'history': history,
                'config': {
                    'user_input_dim': TWO_TOWER_PARAMS.user_input_dim,
                    'book_input_dim': TWO_TOWER_PARAMS.book_input_dim
                }
            }, Config.MODEL_DIR / Config.MODEL_FILENAME)
            
            print(f"\n✓ New best model saved with NDCG: {best_ndcg:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Выводим статистику эпохи
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train NDCG: {train_ndcg:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val NDCG: {val_ndcg:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Best NDCG: {best_ndcg:.4f} | Patience: {patience_counter}/{patience}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best Validation NDCG: {best_ndcg:.4f}")
    print("=" * 60)
    
    # Загружаем лучшую модель для возврата
    checkpoint = torch.load(Config.MODEL_DIR / Config.MODEL_FILENAME, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history

def calculate_ndcg_regression(predictions: torch.Tensor, labels: torch.Tensor, 
                            has_read: torch.Tensor, k: int = 20) -> float:
    weights = torch.where(has_read == 1, 2.0, 1.0)
    
    # Сортируем по предсказаниям
    _, indices = torch.sort(predictions, descending=True)
    sorted_weights = weights[indices][:k]
    sorted_labels = labels[indices][:k]
    
    # DCG
    dcg = 0.0
    for i, (w, l) in enumerate(zip(sorted_weights, sorted_labels), 1):
        gain = w if l == 1 else 0  # Только если книга была прочитана
        dcg += gain / np.log2(i + 1)
    
    # IDCG (идеальная сортировка по has_read)
    ideal_indices = torch.sort(has_read, descending=True)[1]
    ideal_weights = weights[ideal_indices][:k]
    ideal_labels = labels[ideal_indices][:k]
    
    idcg = 0.0
    for i, (w, l) in enumerate(zip(ideal_weights, ideal_labels), 1):
        gain = w if l == 1 else 0
        idcg += gain / np.log2(i + 1)
    
    return dcg / idcg if idcg > 0 else 0.0

if __name__ == "__main__":
    model, history = train()