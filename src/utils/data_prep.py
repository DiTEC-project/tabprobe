"""
Data preparation utilities.

This module contains functions for preparing categorical data for rule mining.
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def prepare_categorical_data(X):
    """
    One-hot encode categorical data and track feature information.

    Args:
        X: pandas DataFrame with categorical features

    Returns:
        encoded_data: numpy array of one-hot encoded data
        classes_per_feature: list of number of classes per feature
        feature_names: list of original feature names
        encoder: fitted OneHotEncoder instance
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(X)

    # Get number of classes per feature
    classes_per_feature = [len(cats) for cats in encoder.categories_]
    feature_names = X.columns.tolist()

    return encoded_data, classes_per_feature, feature_names, encoder


def add_gaussian_noise(data, noise_factor=0.5):
    """
    Add Gaussian noise to data, matching PyAerial's training noise.

    PyAerial adds noise during training to improve robustness:
        noisy_batch = (batch + torch.randn_like(batch) * noise_factor).clamp(0, 1)

    This function replicates that behavior for numpy arrays.

    Args:
        data: numpy array of shape (n_samples, n_features)
        noise_factor: standard deviation of Gaussian noise (default=0.5, matching PyAerial)

    Returns:
        noisy_data: numpy array with added noise, clamped to [0, 1]
    """
    noise = np.random.randn(*data.shape) * noise_factor
    noisy_data = np.clip(data + noise, 0, 1)
    return noisy_data