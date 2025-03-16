# src/model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def train_with_grid_search(train_data):
    """
    Train a Random Forest model using GridSearchCV to tune hyperparameters.
    train_data: DataFrame with features and 'action_label'.
    Returns the best model found.
    """
    X = train_data.drop(columns=['action_label'])
    y_str = train_data['action_label']
    y_codes = y_str.astype('category').cat.codes

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=2, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y_codes)

    best_model = grid_search.best_estimator_
    best_model.label_categories_ = list(y_str.astype('category').cat.categories)
    print("Best parameters found:", grid_search.best_params_)
    return best_model

def train(train_data):
    """
    Train the model using grid search for hyperparameter tuning.
    """
    return train_with_grid_search(train_data)

def test(model, test_data):
    """
    Evaluate the trained model on test data using classification accuracy.
    test_data: DataFrame with features and 'action_label'.
    Returns accuracy.
    """
    X_test = test_data.drop(columns=['action_label'])
    y_true_str = test_data['action_label']
    label_map = {cat: i for i, cat in enumerate(model.label_categories_)}
    y_true_codes = y_true_str.map(lambda x: label_map.get(x, -1))
    y_pred_codes = model.predict(X_test)

    valid_mask = (y_true_codes != -1)
    if valid_mask.sum() == 0:
        print("No valid labels in test set!")
        return 0.0
    accuracy = accuracy_score(y_true_codes[valid_mask], y_pred_codes[valid_mask])
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy