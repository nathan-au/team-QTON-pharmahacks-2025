import pickle
import pandas as pd

# Replace 'your_pickle_file.pkl' with the actual path to your pickle file
with open('/Users/nathanau/Downloads/train_data.pickle', 'rb') as f:
    data = pickle.load(f)

# If 'data' is already a DataFrame, you can simply display it
if isinstance(data, pd.DataFrame):
    print(data)
else:
# Otherwise, try creating a DataFrame from 'data' (assuming it is compatible)
    df = pd.DataFrame(data)
    print(df)