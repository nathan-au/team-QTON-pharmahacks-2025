import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import copy
import gdown


# from google.colab import drive
#
# drive.mount('/content/drive')
#
# gdown.download('https://drive.google.com/uc?id=18HCzpRnUzMYwrWJ4k6qnkgqhlLY735NO', 'data_hackaton_train.data',
#                quiet=False)
#
# train_df = pd.read_pickle('/content/data_hackaton_train.data')
# train_df.shape


class Puzzle:
    def __init__(self, data):
        self.start = list(data["start"])
        self.accepted_pair = data["accepted_pair"]

        # self.moves = copy.deepcopy(data.get("moves"))
        # self.steps = copy.deepcopy(data.get("steps"))
        # self.solution = list(data["solution"])
        # self.score = data["score"]
        # self.padded_start = self.build_puzzle_to_end(self.start)
        # self.padded_solution = self.build_puzzle_to_end(self.solution)

    def build_puzzle_to_end(self, puzzle):
        """Pad each row with '-' to match the longest row for visualization."""
        max_len = max(len(row) for row in puzzle)
        return [row.ljust(max_len, '-') for row in puzzle]

    def plot_puzzle(self, puzzle, title):
        puzzle = self.build_puzzle_to_end(puzzle)
        num_puzzle = np.array([[DICT_SYMBOLS.get(char, 0) for char in row] for row in puzzle])
        rot_num_puzzle = np.rot90(num_puzzle, 1)
        puzzle_array = np.array([list(row) for row in puzzle])
        rot_labels = np.rot90(puzzle_array, 1)

        plt.figure(figsize=(8, 6))
        sns.heatmap(rot_num_puzzle, annot=rot_labels, fmt="", cmap="Pastel1_r", cbar=False)
        plt.title(title)
        plt.axis("off")
        plt.show()

    def gearbox_score(self, puzzle, bonus=1.15):
        consensus = self.accepted_pair
        score = 0
        for col_ind in range(len(puzzle[0])):
            col_bonus = True
            col_tot = 0
            column_chars = [row[col_ind] for row in puzzle]
            for char in column_chars:
                if char == "-":
                    col_bonus = False
                    continue
                if char in consensus[col_ind]:
                    col_tot += 1
                else:
                    col_bonus = False
            column_score = col_tot * bonus if col_bonus else col_tot
            score += column_score
        # print(f"Total Gearbox Score: {score}")
        return score

    def apply_step_to_puzzle(self, puzzle, step):
        """Apply a single step to the puzzle."""
        new_puzzle = puzzle.copy()
        row_index = step[0] - 1
        col_index = step[1]
        if row_index < 0 or row_index >= len(new_puzzle):
            return new_puzzle
        row_str = new_puzzle[row_index]
        if col_index < 0 or col_index > len(row_str):
            return new_puzzle
        new_row = row_str[:col_index] + '-' + row_str[col_index:]
        new_row = new_row[:len(row_str)]
        new_puzzle[row_index] = new_row
        return new_puzzle

    def apply_all_steps(self, steps):
        """Apply all steps on a copy of the puzzle and plot states."""
        current_puzzle = list(self.start)
        updated_puzzles = []
        scores = []

        for step in steps:
            current_puzzle = self.apply_step_to_puzzle(current_puzzle, step)
            padded_current = self.build_puzzle_to_end(current_puzzle)
            score = self.gearbox_score(padded_current)

            updated_puzzles.append(padded_current)
            scores.append(score)

        # plot, no need
        '''
        n_steps = len(updated_puzzles)
        fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 6))
        if n_steps == 1:
            axes = [axes]
        for idx, (puzzle_state, score) in enumerate(zip(updated_puzzles, scores)):
            num_puzzle = np.array([[DICT_SYMBOLS.get(char, 0) for char in row] for row in puzzle_state])
            rot_num_puzzle = np.rot90(num_puzzle, 1)
            puzzle_array = np.array([list(row) for row in puzzle_state])
            rot_labels = np.rot90(puzzle_array, 1)
            ax = axes[idx]
            sns.heatmap(rot_num_puzzle, annot=rot_labels, fmt="", cmap="Pastel1_r", cbar=False, ax=ax)
            ax.set_title(f"Step {idx + 1}\nScore: {score}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        '''

        # return current_puzzle
        return padded_current, score

    def get_valued_tot(self, puzzle):
        num_puzzle = np.array([[DICT_SYMBOLS.get(char, 0) for char in row] for row in puzzle])
        num_accepted_pair = np.array([[DICT_SYMBOLS.get(char, 0) for char in row] for row in self.accepted_pair]).T
        valued_tot = np.zeros(num_puzzle.shape)
        for row_i in range(num_puzzle.shape[0]):
            for col_i, val in enumerate(num_puzzle[row_i]):
                # '-' is valued
                valued_tot[row_i, col_i] = val in num_accepted_pair[:, col_i]
                # # '-' is not valued
                # valued_tot[row_i, col_i] = val in num_accepted_pair[:, col_i] and val != 0
                # # '-' is always valued
                # valued_tot[row_i, col_i] = val in num_accepted_pair[:, col_i] or val == 0
        return valued_tot

    def generate_steps(self, current_puzzle, improve_ratio):
        padded_current = puzzle.build_puzzle_to_end(current_puzzle)
        row_num = len(padded_current)
        col_num = len(padded_current[0])

        steps = []
        scores = []
        valued_tot = puzzle.get_valued_tot(padded_current)
        for rep_i in range(row_num):
            for col_i in range(col_num):
                max_score = 0
                max_row_i=np.nan
                for row_i in range(row_num):
                    if not valued_tot[row_i, col_i]:
                        optimal_puzzle = puzzle.apply_step_to_puzzle(current_puzzle, [row_i + 1, col_i])
                        optimal_puzzle = puzzle.build_puzzle_to_end(optimal_puzzle)
                        score = puzzle.gearbox_score(optimal_puzzle)
                        if score > max_score:
                            max_score = score
                            max_row_i = row_i
                if not np.isnan(max_row_i):
                    if not len(scores) or score >= scores[-1] * improve_ratio:  # at least improve or keep??
                        steps.append([max_row_i + 1, col_i])
                        current_puzzle, score = puzzle.apply_all_steps(steps)
                        valued_tot = puzzle.get_valued_tot(current_puzzle)
                        # print(steps)
                        scores.append(score)
        return steps, scores


DICT_SYMBOLS = {'A': 1,
                'T': 2,
                'C': 3,
                'G': 4}


# test
test_df = pd.read_pickle('test_data.pickle')
self_scores = []
self_steps = []
self_solution = []
for test_i in np.arange(6,test_df.shape[0]):
    puzzle_data = {'start': test_df.iloc[test_i]['start'],
                   'accepted_pair': test_df.iloc[test_i]['accepted_pair']}

    puzzle = Puzzle(puzzle_data)
    steps, scores = puzzle.generate_steps(puzzle_data['start'], 0.95)

    self_scores.append(max(scores))
    # plt.plot(scores);plt.grid()
    self_steps.append(steps[:np.argmax(np.array(scores)) + 1])

    print('Test.' + str(test_i + 1) + ' Score: ' + str(round(self_scores[-1], 3)))

    padded_current, score = puzzle.apply_all_steps(self_steps[-1])
    self_solution.append(padded_current)


# train
train_df = pd.read_pickle('data_hackaton_train.data')
# Create puzzle
self_scores = []
ref_scores = []
self_steps = []
ref_steps = []
for train_i in range(train_df.shape[0]):
    # for train_i in range(500):
    puzzle_data = {'start': train_df.iloc[train_i]['start'],
                   'accepted_pair': train_df.iloc[train_i]['accepted_pair']}
    # 'moves': train_df.iloc[-1].get('moves'),
    ref_steps.append(train_df.iloc[train_i].get('steps'))
    ref_solution = train_df.iloc[train_i]['solution']
    ref_scores.append(train_df.iloc[train_i]['score'])
    # puzzle_data['steps'] = steps

    puzzle = Puzzle(puzzle_data)
    steps, scores = puzzle.generate_steps(puzzle_data['start'], 0.95)

    self_scores.append(max(scores))
    # plt.plot(scores);plt.grid()
    self_steps.append(steps[:np.argmax(np.array(scores)) + 1])

    print('Train.' + str(train_i + 1) + ' Self-Reference Score: ' + str((self_scores[-1] - ref_scores[-1]).round(3)))
    print('Average Self-Reference Score: ' + str(np.average(np.array(self_scores) - np.array(ref_scores)).round(3)))

    padded_current, score = puzzle.apply_all_steps(self_steps[-1])

    # plot
    '''
    pairs = []
    for i, pair in enumerate(puzzle_data['accepted_pair']):
        pairs.append(pair[0] + ' ' + pair[1])

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    for idx, (puzzle_state, score) in enumerate(
            zip([puzzle_data['start'], ref_solution, padded_current],
                ['start', 'reference score: ' + str(round(ref_scores[-1], 3)),
                 'self score: ' + str(round(self_scores[-1], 3))])):
        num_puzzle = np.array([[DICT_SYMBOLS.get(char, 0) for char in row] for row in puzzle_state])
        rot_num_puzzle = np.rot90(num_puzzle, 1)
        puzzle_array = np.array([list(row) for row in puzzle_state])
        rot_labels = np.rot90(puzzle_array, 1)
        ax = axes[idx]
        # if idx==2:
        #     sns.heatmap(rot_num_puzzle, annot=rot_labels, fmt="", cmap="Pastel1_r", cbar=False, ax=ax, yticklabels=pairs)
        #     ax.set_yticks(pairs)
        # else:
        sns.heatmap(rot_num_puzzle, annot=rot_labels, fmt="", cmap="Pastel1_r", cbar=False, ax=ax)
        ax.set_title(score)
        ax.axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig('Train.' + str(train_i + 1) +'all_steps.png')
    plt.close()
    '''

# '-'value + improve = 2.28, more than reference score, good
# '-'value + 95%improve = 2.213, more than reference score, good
# '-'alwaysvalue + improve = -0.474
# '-'value + noimprove +3columnsteps = -0.15, close to reference score
# '-'value + noimprove +columnsteps = -6, close to reference score
# '-'notvalue + 95%improve = 2.175, more than reference score, good
# '-'notvalue + improve = 2.005, less higher than reference score.
# '-'notvalue + improve +columnsteps = -6.31, smaller than reference score, not good
# '-'notvalue + noimprove +columnsteps = -5.99, smaller than reference score, not good

# plot self and reference scores
'''
plt.plot(self_scores, 'b', label='self')
plt.plot(ref_scores, 'r', label='reference')
plt.xlabel('Train')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.title('Avg Self-Reference Score: ' + str(np.average(np.array(self_scores) - np.array(ref_scores)).round(3)))
plt.savefig('all_steps.png')
plt.close()
'''

