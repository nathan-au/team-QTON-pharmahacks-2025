#LOAD THE PICKLE FILE

import pickle
import pandas as pd

# Load the .pickle file
with open('train_data.pickle', 'rb') as file:
    data = pickle.load(file)

# Limit the dataset to the first 1000 rows
data = data.head(20000)

# Inspect the structure of the data
print(f"Number of entries: {len(data)}")
print(f"Columns in the DataFrame: {data.columns.tolist()}")

def apply_gaps(sequences, steps):

    if steps is None:
        return sequences
    
    # Convert sequences to a list of lists for mutability
    sequences = [list(seq) for seq in sequences]
    
    for step in steps:
        seq_idx, pos = step  # Unpack the step (sequence index, position)
        # Adjust sequence index to 0-based
        seq_idx_0based = seq_idx - 1
        
        # Check if the sequence index and position are valid
        if 0 <= seq_idx_0based < len(sequences) and 0 <= pos <= len(sequences[seq_idx_0based]):
            # Insert a gap ('-') at the specified position
            sequences[seq_idx_0based].insert(pos, '-')
    
    # Convert back to strings
    sequences = [''.join(seq) for seq in sequences]
    return sequences

print("Applying steps to start sequences.")

# Apply apply_gaps to all 'start' columns in the DataFrame and add the results to a new column called 'end'
data['end'] = data.apply(lambda row: apply_gaps(row['start'], row['steps']), axis=1)

#PREPROCESS THE DATA

import numpy as np

def one_hot_encode(sequence):
    # Define the mapping for nucleotides and gaps
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        '-': [0, 0, 0, 0]  # Gap is represented as all zeros
    }
    # Encode each character in the sequence
    encoded_sequence = [mapping[char] for char in sequence if char != '[' and char != ']']
    return np.array(encoded_sequence)

def encode_sequences(sequences):
    encoded_sequences = []
    for seq in sequences:
        sub_sequences = seq.split(',')
        encoded_sequences.extend([one_hot_encode(sub_seq) for sub_seq in sub_sequences])
    return encoded_sequences

# Apply one_hot_encode to every 'start' and 'solution' in the DataFrame
print("One-hot encoding start sequences.")
data['encoded_start'] = data['start'].apply(encode_sequences)

print("One-hot encoding solution sequences.")
data['encoded_solution'] = data['solution'].apply(encode_sequences)

print("One-hot encoding end sequences.")
data['encoded_end'] = data['end'].apply(encode_sequences)

def flatten_sequence(encoded_sequence):
    return encoded_sequence.flatten()

def flatten_puzzle(encoded_sequences, max_num_subseqs):
    # Pad or truncate each sub-sequence to the fixed length
    padded_sequences = [np.pad(seq, ((0, max(0, max_length - seq.shape[0])), (0, 0)), mode='constant')[:max_length] for seq in encoded_sequences]
    
    # Pad or truncate the list of sub-sequences to the fixed number of sub-sequences
    padded_sequences = np.pad(padded_sequences, ((0, max(0, max_num_subseqs - len(padded_sequences))), (0, 0), (0, 0)), mode='constant')[:max_num_subseqs]
    
    # Flatten each sub-sequence
    flattened_sequences = [flatten_sequence(seq) for seq in padded_sequences]
    
    return np.array(flattened_sequences)

# Determine the fixed length for padding/truncating
max_length = 19  # You can adjust this value as needed

# Determine the maximum number of sub-sequences across all sequences
max_num_subseqs = max(len(seqs) for seqs in data['encoded_start'])

# Apply flattening to the entire dataset with the fixed length and number of sub-sequences
print("Flattening start sequences.")
data['flattened_start'] = data['encoded_start'].apply(lambda seqs: flatten_puzzle(seqs, max_num_subseqs))
print("Flattening end sequences.")
data['flattened_end'] = data['encoded_end'].apply(lambda seqs: flatten_puzzle(seqs, max_num_subseqs))
print("Flattening solution sequences.")
data['flattened_solution'] = data['encoded_solution'].apply(lambda seqs: flatten_puzzle(seqs, max_num_subseqs))

# Extract features and labels
print("Extracting feature/label data sets.")

X = np.stack(data['flattened_start'].values)  # Features: Initial sequences
y = np.stack(data['flattened_solution'].values)  # Labels: Solution sequences

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split

print("Extracting training/testing data sets.")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Flatten the data from (batch_size, num_sequences, sequence_length * num_features) to (batch_size, num_sequences * sequence_length * num_features)
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

print(f"Flattened training data shape: {x_train_flat.shape}")
print(f"Flattened validation data shape: {x_test_flat.shape}")

# Flatten the target labels
y_train_flat = y_train.reshape(y_train.shape[0], -1)
y_test_flat = y_test.reshape(y_test.shape[0], -1)

print(f"Flattened y_train shape: {y_train_flat.shape}")
print(f"Flattened y_test shape: {y_test_flat.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU

model = Sequential([
    Dense(256, input_shape=(x_train_flat.shape[1],)),  # Explicit input shape
    BatchNormalization(),
    LeakyReLU(alpha=0.01),  
    # Dropout(0.3),  

    Dense(128),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    # Dropout(0.3),

    Dense(64),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    # Dropout(0.2),

    Dense(y_train_flat.shape[1], activation='softmax')  # Changed to softmax for multi-class
])

# Compilation
model.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Summary of the model
model.summary()

# # Define the model
# model = Sequential([
#     Dense(256, activation='relu'),
#     Dense(128, activation='relu'),
#     Dense(64, activation='relu'),
#     Dense(y_train_flat.shape[1], activation='sigmoid'),  # Output layer
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Print model summary
# model.summary()

# Train the model
history = model.fit(
    x_train_flat, y_train_flat,
    epochs = 40,
    batch_size = 512,
    validation_data = (x_test_flat, y_test_flat)
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(x_test_flat, y_test_flat)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

print("PROCESS COMPLETED SUCCESSFULLY.")

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()