"""
Data preparation utilities for Aerial.

This module contains functions for preparing categorical data for use with Aerial models.
"""

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
