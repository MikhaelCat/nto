# src/two_towers/prepare_data.py
"""
Script to prepare data for training.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.two_towers.data_processing import (
    load_and_merge_data,
    prepare_features_for_training,
    save_processed_data
)
from src.two_towers.config import Config
from src.two_towers.features import TwoTowersFeatureEngineer


def prepare_data():
    """Main function to prepare all data for training."""
    print("=" * 80)
    print("DATA PREPARATION PIPELINE")
    print("=" * 80)
    
    # Step 1: Load raw data
    print("\n1. Loading raw data...")
    raw_data = load_and_merge_data()
    
    # Step 2: Create features using TwoTowersFeatureEngineer
    print("\n2. Creating features with TwoTowersFeatureEngineer...")
    config = Config()
    engineer = TwoTowersFeatureEngineer(config)
    engineer.create_all_features(raw_data)
    
    # Step 3: Prepare training samples
    print("\n3. Preparing training samples...")
    features_dict = prepare_features_for_training(
        raw_data['train'],
        raw_data['books'],
        raw_data['users']
    )
    
    # Step 4: Save processed data
    print("\n4. Saving processed data...")
    save_processed_data(features_dict)
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    prepare_data()