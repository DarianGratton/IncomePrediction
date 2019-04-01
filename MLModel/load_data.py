import pandas as pd
import numpy as np


def load_full_data():
    train_data = pd.read_csv("./trainingset.csv")
    test_data = pd.read_csv("./testingset.csv")

    return pd.concat([train_data, test_data], axis=0)


def load(file_path, one_hot_encode=False, max_unique=20):
    """
    Load a .csv file
    :param file_path: path to .csv
    :param one_hot_encode: (opt) if true, applies one-hot encoding on columns with less than 'max_unique' values (ignores if 2 (1 and 0))
    :param max_unique: (opt) upper bound of one-hot encoded unique columns
    :return: one-hot encoded data in format pd.data
    """
    data = pd.read_csv(file_path)

    data.loc[data['income'] == '>50K', 'income'] = 1
    data.loc[data['income'] == '<=50K', 'income'] = 0

    if one_hot_encode:
        hot_encoding_indices = []
        for col in data:
            if 1 < data[col].nunique() < max_unique and str(col) != 'income':
                hot_encoding_indices.append(col)
        data = pd.get_dummies(data, columns=hot_encoding_indices, prefix=hot_encoding_indices)

    return data


def get_missing_features(train_data, test_data):
    """
    Makes the test set features the same as the training set features
    :param train_data: the training set data
    :param test_data: the testing set data
    :return: The test data with the features from the training data (Not Sorted)
    """
    # Drop income from train_data if income exists
    if 'income' in train_data:
        train_data = train_data.drop(['income'], axis=1, inplace=False)

    # Get features in each list
    train_features = list(train_data)
    test_features = list(test_data)

    # Get features that are in one but not the other
    missing_features = np.setdiff1d(train_features, test_features)
    not_needed_features = np.setdiff1d(test_features, train_features)

    # Drop columns in the testset not in the training set
    new_data = test_data.drop(not_needed_features, axis=1, inplace=False)

    # Add columns into the testset that are in the training set but not the test set
    for feature in missing_features:
        new_data[feature] = 0

    return new_data
