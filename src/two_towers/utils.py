"""
Utility functions for the project.
"""

import numpy as np
import pandas as pd
import torch
import random
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(state: Dict[str, Any], filename: str):
    """Save model checkpoint."""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename: str) -> Dict[str, Any]:
    """Load model checkpoint."""
    if Path(filename).exists():
        checkpoint = torch.load(filename, map_location='cpu')
        print(f"Checkpoint loaded from {filename}")
        return checkpoint
    else:
        raise FileNotFoundError(f"Checkpoint not found: {filename}")


def plot_training_history(history: Dict[str, List[float]], save_path: Path = None):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axes[0, 0].plot(history.get('train_loss', []), label='Train Loss')
    axes[0, 0].plot(history.get('val_loss', []), label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot NDCG
    axes[0, 1].plot(history.get('train_ndcg', []), label='Train NDCG')
    axes[0, 1].plot(history.get('val_ndcg', []), label='Val NDCG')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('NDCG@20')
    axes[0, 1].set_title('Training and Validation NDCG')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'val_acc' in history:
        axes[1, 0].plot(history['val_acc'], label='Val Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], label='Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def analyze_feature_importance(model, feature_names: List[str], save_path: Path = None):
    """Analyze and plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        # LightGBM model
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        print("Model doesn't have feature importances attribute")
        return
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()
    
    # Save to CSV
    if save_path:
        csv_path = save_path.with_suffix('.csv')
        importance_df.to_csv(csv_path, index=False)
        print(f"Feature importance data saved to {csv_path}")
    
    return importance_df


def memory_usage_info():
    """Print memory usage information."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    print(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    print(f"Virtual memory: {mem_info.vms / 1024 / 1024:.2f} MB")
    
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")


def log_experiment(config: Dict[str, Any], metrics: Dict[str, float], 
                   model_path: Path, log_dir: Path = None):
    """Log experiment configuration and results."""
    if log_dir is None:
        log_dir = Path("experiments")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment ID
    import datetime
    exp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = log_dir / exp_id
    exp_dir.mkdir(exist_ok=True)
    
    # Save config
    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save metrics
    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Copy model if exists
    if model_path.exists():
        import shutil
        shutil.copy2(model_path, exp_dir / model_path.name)
    
    print(f"Experiment logged to {exp_dir}")
    
    # Create summary
    summary = {
        'experiment_id': exp_id,
        'timestamp': datetime.datetime.now().isoformat(),
        'config': config,
        'metrics': metrics
    }
    
    # Update main log
    log_file = log_dir / "experiment_log.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(summary) + '\n')
    
    return exp_id


class ProgressLogger:
    """Custom progress logger."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = None
        import time
        self.time = time
    
    def __enter__(self):
        self.start_time = self.time.time()
        print(f"{self.desc}...")
        return self
    
    def update(self, n: int = 1):
        self.current += n
        elapsed = self.time.time() - self.start_time
        progress = self.current / self.total
        if progress > 0:
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
            
            print(f"\r{self.desc}: {self.current}/{self.total} "
                  f"({progress:.1%}) | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"Remaining: {remaining:.1f}s", end="")
    
    def __exit__(self, *args):
        elapsed = self.time.time() - self.start_time
        print(f"\r{self.desc}: Complete! | "
              f"Total time: {elapsed:.1f}s")