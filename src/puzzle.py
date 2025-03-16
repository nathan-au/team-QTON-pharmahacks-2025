# src/puzzle.py

import copy
import numpy as np

# Mapping nucleotides/characters to integers (for feature encoding)
DICT_SYMBOLS = {
    'A': 1,
    'T': 2,
    'C': 3,
    'G': 4,
    '-': 0
}

class Puzzle:
    def __init__(self, data):
        """
        data: dict containing:
          - 'start': list of strings (initial unaligned sequences)
          - 'steps': list of (sequence_index, position) moves for gap insertions
          - 'solution': final aligned sequences
          - 'score': final alignment score (numerical)
          - 'accepted_pair': structure defining accepted nucleotide pairs per column (e.g., a list of sets)
        """
        self.start = list(data['start'])
        self.steps = copy.deepcopy(data.get('steps', []))
        self.solution = list(data.get('solution', []))
        self.score = data.get('score', 0)
        self.accepted_pair = data.get('accepted_pair', [])

    def build_puzzle_to_end(self, puzzle):
        """Pad each sequence in the puzzle with '-' so that all sequences have equal length."""
        if not puzzle:
            return []
        max_len = max(len(row) for row in puzzle)
        return [row.ljust(max_len, '-') for row in puzzle]

    def gearbox_score(self, puzzle, bonus=1.15):
        """
        Compute an alignment score for the given puzzle state.
        For each column, if all characters are in the accepted set for that column,
        apply a bonus; otherwise, simply sum the counts.
        """
        puzzle = self.build_puzzle_to_end(puzzle)
        if not puzzle:
            return 0.0
        score = 0.0
        max_len = len(puzzle[0])
        for col_ind in range(max_len):
            col_chars = [row[col_ind] for row in puzzle]
            valid_set = self.accepted_pair[col_ind] if col_ind < len(self.accepted_pair) else set()
            col_count = 0
            col_bonus = True
            for ch in col_chars:
                if ch == '-':
                    col_bonus = False
                elif ch in valid_set:
                    col_count += 1
                else:
                    col_bonus = False
            if col_bonus and col_count == len(col_chars):
                score += col_count * bonus
            else:
                score += col_count
        return score

    def _apply_step_to_puzzle(self, puzzle, step):
        """
        Apply a single move to the puzzle state.
        step: tuple (sequence_index, position) indicating where to insert a gap ('-').
        """
        new_puzzle = puzzle.copy()
        seq_idx, pos = step
        if seq_idx < 0 or seq_idx >= len(new_puzzle):
            return new_puzzle
        seq_str = new_puzzle[seq_idx]
        if pos < 0 or pos > len(seq_str):
            return new_puzzle
        new_seq = seq_str[:pos] + '-' + seq_str[pos:]
        new_puzzle[seq_idx] = new_seq
        return new_puzzle

    def apply_all_steps(self):
        """
        Apply all moves (steps) sequentially to the initial puzzle state.
        Returns the final puzzle state (list of sequences).
        """
        current_state = list(self.start)
        for step in self.steps:
            current_state = self._apply_step_to_puzzle(current_state, step)
        return current_state