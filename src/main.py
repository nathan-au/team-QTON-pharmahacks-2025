# src/main.py

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from feature_engineering import build_training_examples
from model import train, test
from puzzle import Puzzle
from simulation import simulate_alignment

def main():
    # Define the path to the training pickle file.
    train_file = os.path.join('../..', 'data', 'train_data.pickle')

    # Load the full dataset from the pickle file.
    with open(train_file, 'rb') as f:
        full_df = pd.read_pickle(f)

    print(f"Full dataset size: {len(full_df)}")

    # Sample 10% of the full dataset for training.
    train_df = full_df.sample(frac=0.05, random_state=42)
    remaining_df = full_df.drop(train_df.index)
    # Sample 10% of the remaining data for testing.
    test_df = remaining_df.sample(frac=0.01, random_state=42)

    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # Build training features from the sampled train data.
    print("Building training features...")
    train_features = build_training_examples(train_df, context_window=2)
    print(f"Training features shape: {train_features.shape}")

    # Build test features similarly.
    print("Building test features...")
    test_features = build_training_examples(test_df, context_window=2)
    print(f"Test features shape: {test_features.shape}")

    # Train the Random Forest model (with GridSearchCV hyperparameter tuning).
    print("Training Random Forest model...")
    rf_model = train(train_features)

    # Evaluate the model using classification accuracy.
    print("Evaluating model on classification accuracy...")
    test_acc = test(rf_model, test_features)
    print(f"Final Test Accuracy (classification): {test_acc:.4f}")

    # Simulate full alignment on one test puzzle.
    print("Simulating full alignment for one test puzzle...")
    sample_puzzle_data = test_df.iloc[0]
    puzzle_instance = Puzzle({
        'start': sample_puzzle_data['start'],
        'steps': sample_puzzle_data['steps'],
        'solution': sample_puzzle_data['solution'],
        'score': sample_puzzle_data['score'],
        'accepted_pair': sample_puzzle_data['accepted_pair']
    })

    final_state, final_score = simulate_alignment(rf_model, puzzle_instance)
    print("Final simulated alignment score:", final_score)
    print("Ground truth alignment score:", puzzle_instance.score)

if __name__ == '__main__':
    main()