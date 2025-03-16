# src/simulation.py

import pandas as pd
from feature_engineering import puzzle_state_to_features, build_puzzle_to_end

def simulate_alignment(model, puzzle_instance, max_steps=None):
    """
    Simulate the full alignment process using the trained model.
    model: Trained RandomForestClassifier.
    puzzle_instance: An instance of the Puzzle class.
    max_steps: Maximum number of moves to simulate (default: len(puzzle_instance.steps)).
    Returns the final puzzle state (list of sequences) and the final alignment score.
    """
    current_state = list(puzzle_instance.start)
    num_steps = max_steps if max_steps is not None else len(puzzle_instance.steps)

    for step in range(num_steps):
        # Build global features from the current state.
        features = puzzle_state_to_features(current_state, puzzle_instance.accepted_pair)
        # Add default local context features (to match training feature names)
        features['local_context_mean'] = 0.0
        features['local_context_sum'] = 0.0
        features['local_context_len'] = 0
        features['local_context_var'] = 0.0

        feature_df = pd.DataFrame([features])
        pred_code = model.predict(feature_df)[0]
        pred_label = model.label_categories_[pred_code]

        try:
            seq_idx, pos = map(int, pred_label.split(':'))
        except Exception as e:
            print("Error parsing predicted label:", pred_label, e)
            break

        current_state = puzzle_instance._apply_step_to_puzzle(current_state, (seq_idx, pos))

    final_score = puzzle_instance.gearbox_score(current_state)
    return current_state, final_score