import numpy as np
import tensorflow as tf
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score

from TfELM.Resources.kernel_distances import calculate_pairwise_distances_vector


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


def calc_kernel_batch(x, kernel, batch_size=100):
    """
        Calculate the kernel matrix in batches.

        Args:
        -----------
            x (tensor): Input data.
            batch_size (int): Number of samples per batch.

        Returns:
        -----------
            tensor: Kernel matrix values.
    """
    x = tf.cast(x, dtype=tf.float32)  # Pastikan format float32
    
    # Jika tidak ada data sebelumnya, buat kernel identity
    identity = tf.ones((tf.shape(x)[0], 1), dtype=tf.float32)  
    
    num_samples = tf.shape(x)[0]
    kernel_matrix = []
    
    for i in range(0, num_samples, batch_size):
        x_batch = x[i : i + batch_size]  # Ambil batch data
        k_batch = calculate_pairwise_distances_vector(x_batch, identity, kernel.ev)  
        kernel_matrix.append(k_batch)  # Simpan hasil batch
    
    return tf.concat(kernel_matrix, axis=0)  # Gabungkan semua batch
