# src/feature_engineering.py

import numpy as np
import pandas as pd
from puzzle import DICT_SYMBOLS

def get_local_context(sequence, position, window=2):
    """
    Extract a substring from 'sequence' around 'position' with length up to 2*window+1.
    """
    start_idx = max(position - window, 0)
    end_idx = min(position + window + 1, len(sequence))
    return sequence[start_idx:end_idx]

def get_local_context_stats(local_encoded):
    """
    Compute statistics for a list of numeric values representing a local context.
    Returns the mean and variance.
    """
    if not local_encoded:
        return 0.0, 0.0
    mean_val = float(np.mean(local_encoded))
    var_val = float(np.var(local_encoded))
    return mean_val, var_val

def build_puzzle_to_end(puzzle):
    """Pad all sequences in the puzzle so they have equal length."""
    if not puzzle:
        return []
    max_len = max(len(row) for row in puzzle)
    return [row.ljust(max_len, '-') for row in puzzle]

def count_mismatches_with_accepted(puzzle_list, accepted_pairs):
    """
    Calculate the total number of mismatches and the mismatch ratio.
    Each column is compared against the accepted pairs for that column.
    """
    puzzle_list = build_puzzle_to_end(puzzle_list)
    if not puzzle_list:
        return 0, 0.0
    total_mismatches = 0
    total_positions = 0
    max_len = len(puzzle_list[0])
    for col_ind in range(max_len):
        col_chars = [row[col_ind] for row in puzzle_list]
        valid_set = accepted_pairs[col_ind] if col_ind < len(accepted_pairs) else set()
        for ch in col_chars:
            total_positions += 1
            if ch == '-' or ch not in valid_set:
                total_mismatches += 1
    mismatch_ratio = total_mismatches / total_positions if total_positions else 0
    return total_mismatches, mismatch_ratio

def puzzle_state_to_features(puzzle_list, accepted_pairs):
    """
    Convert a puzzle state (list of sequences) into a feature vector.
    Global features include: number of sequences, mean and std of lengths and gap counts,
    total mismatches, mismatch ratio, and gap ratio.
    """
    puzzle_list = build_puzzle_to_end(puzzle_list)
    n_sequences = len(puzzle_list)
    lengths = [len(seq) for seq in puzzle_list]
    gaps_each = [seq.count('-') for seq in puzzle_list]
    total_chars = sum(lengths)
    total_gaps = sum(gaps_each)
    gap_ratio = total_gaps / total_chars if total_chars else 0.0
    total_mismatches, mismatch_ratio = count_mismatches_with_accepted(puzzle_list, accepted_pairs)

    features = {
        'n_sequences': n_sequences,
        'mean_length': float(np.mean(lengths)) if lengths else 0.0,
        'std_length': float(np.std(lengths)) if lengths else 0.0,
        'mean_gaps': float(np.mean(gaps_each)) if gaps_each else 0.0,
        'std_gaps': float(np.std(gaps_each)) if gaps_each else 0.0,
        'total_length': total_chars,
        'total_mismatches': total_mismatches,
        'mismatch_ratio': mismatch_ratio,
        'gap_ratio': gap_ratio
    }
    return pd.Series(features)

def build_training_examples(full_df, context_window=2):
    """
    Given the full DataFrame (each row is one puzzle), generate training examples.
    For each puzzle, iterate over each move, extract global features and local context features,
    and label the example with the next move in the format "seq_idx:pos".
    Returns a DataFrame with one row per training example.
    """
    rows = []
    for idx, row in full_df.iterrows():
        # Ensure 'steps' exists and is iterable
        steps = row.get('steps', None)
        if steps is None:
            continue  # Skip rows that do not have steps

        puzzle_data = {
            'start': row['start'],
            'steps': steps,
            'solution': row['solution'],
            'score': row['score'],
            'accepted_pair': row['accepted_pair']
        }
        current_puzzle = list(puzzle_data['start'])
        for step_i, next_step in enumerate(steps):
            seq_idx, pos = next_step
            # Global features from the current puzzle state
            feat_series = puzzle_state_to_features(current_puzzle, row['accepted_pair'])
            feat_dict = feat_series.to_dict()

            # Local context features for the specific sequence (if valid)
            if 0 <= seq_idx < len(current_puzzle):
                seq_str = current_puzzle[seq_idx]
                local_str = get_local_context(seq_str, pos, window=context_window)
                local_encoded = [DICT_SYMBOLS.get(ch, 0) for ch in local_str]
                local_mean, local_var = get_local_context_stats(local_encoded)
                feat_dict['local_context_mean'] = local_mean
                feat_dict['local_context_sum'] = float(np.sum(local_encoded)) if local_encoded else 0.0
                feat_dict['local_context_len'] = len(local_encoded)
                feat_dict['local_context_var'] = local_var
            else:
                feat_dict['local_context_mean'] = 0.0
                feat_dict['local_context_sum'] = 0.0
                feat_dict['local_context_len'] = 0
                feat_dict['local_context_var'] = 0.0

            # Label for the training example: "seq_idx:pos"
            feat_dict['action_label'] = f"{seq_idx}:{pos}"
            rows.append(feat_dict)

            # Update the current puzzle state by applying the move.
            if 0 <= seq_idx < len(current_puzzle):
                row_str = current_puzzle[seq_idx]
                if pos >= 0 and pos <= len(row_str):
                    new_row = row_str[:pos] + '-' + row_str[pos:]
                    current_puzzle[seq_idx] = new_row
    training_df = pd.DataFrame(rows)
    return training_df