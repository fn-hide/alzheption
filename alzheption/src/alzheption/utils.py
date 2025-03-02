import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score


def custom_cross_val_score(model, X, y, n_splits=10, n_repeats=5, scoring=accuracy_score):
    """
    Custom implementation of cross-validation.

    Parameters:
    -----------
    model : object
        The machine learning model with .fit() and .predict() methods.
    X : np.array or pd.DataFrame
        Feature matrix.
    y : np.array or pd.Series
        Target labels.
    n_splits : int, optional (default=10)
        Number of splits for KFold.
    n_repeats : int, optional (default=5)
        Number of times cross-validation is repeated.
    scoring : function, optional (default=accuracy_score)
        Scoring function to evaluate model performance.

    Returns:
    --------
    list
        List of scores from each fold.
    """
    scores = []
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    for train_idx, test_idx in rkf.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        model.fit(X_train, y_train)

        # Predict & convert to NumPy
        y_pred = model.predict(X_test)
        if not isinstance(y_pred, np.ndarray):  # Convert to NumPy if not already
            y_pred = y_pred.numpy()

        # Calculate score
        score = scoring(y_test, y_pred)
        scores.append(score)

    return scores
