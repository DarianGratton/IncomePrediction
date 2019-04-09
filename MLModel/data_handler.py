import pandas as pd
import numpy as np
import pickle
import glob
import json


def load_full_data():
    """
    Loads the full data set given (currently split in a training set and test set)
    :return: Dataframe containing all the samples from the training set and test set
    """
    train_data = pd.read_csv("./trainingset.csv")
    test_data = pd.read_csv("./testingset.csv")

    return pd.concat([train_data, test_data], axis=0)


def load(file_path, one_hot_encode=False, max_unique=20):
    """
    Load a .csv file
    :param file_path: path to .csv
    :param one_hot_encode: (opt) if true, applies one-hot encoding on columns with less than 'max_unique' values
    :param max_unique: (opt) upper bound of one-hot encoded unique columns
    :return: one-hot encoded data in format pd.data
    """
    # Read the file
    data = pd.read_csv(file_path)

    # Convert income to binary
    data.loc[data['income'] == '>50K', 'income'] = 1
    data.loc[data['income'] == '<=50K', 'income'] = 0

    # Hot encode the features that have < max_unique
    if one_hot_encode:
        hot_encoding_indices = []
        for col in data:
            if 1 < data[col].nunique() < max_unique and str(col) != 'income':
                hot_encoding_indices.append(col)
        data = pd.get_dummies(data, columns=hot_encoding_indices, prefix=hot_encoding_indices)

    return data


def one_hot_encode(data, max_unique=20):
    """

    :param data:
    :param max_unique:
    :return:
    """
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


def print_results_to_csv(actual, predicted):
    """
    Prints results to a .csv file
    :param actual: actual values of the label
    :param predicted: predicted values of the label
    :return:
    """
    # Check if predicted is a dataframe
    if not isinstance(predicted, pd.DataFrame):
        d = {'income_pred': predicted}
        predicted = pd.DataFrame(data=d)

    # Combine the two Dataframes
    actual['income_pred'] = predicted['income_pred'].values

    filename = 'results.csv'
    files_present = glob.glob(filename)

    # Check if the file already exists
    if not files_present:
        actual.to_csv(filename)
    else:
        print('\nFunction:', print_results_to_csv.__name__)
        print('WARNING: This file already exists')


def save_model(trained_model):
    """
    Save trained model to a file
    :param trained_model: The trained model
    """
    filename = 'finalized_model.sav'
    pickle.dump(trained_model, open(filename, 'wb'))


def predict_income():

    data = \
        '{ "age": 43, ' \
        '"workclass": "Never-worked",' \
        '"fnlwgt": 70800,' \
        '"education": "Bachelors",' \
        '"education-num": 13,' \
        '"marital-status": "Never-married",' \
        '"occupation": "?",' \
        '"relationship": "Unmarried",' \
        '"race": "Black",' \
        '"sex": "Male",' \
        '"capital-gain": 0,' \
        '"capital-loss": 0,' \
        '"hours-per-week": 40,' \
        '"native-country": "United-States",' \
        '"income": 0' \
        '}'

    # Parse the Json file
    donor = json.loads(data)
    dataframe = pd.DataFrame(data=donor, index=[0])

    # Ensure that the received data has the same features as the full data
    full_data = load_full_data()
    new_data = pd.concat([full_data, dataframe], axis=0)
    new_data = one_hot_encode(new_data, max_unique=50)

    # Prepare features
    features = new_data.drop('income', axis=1, inplace=False)
    # Get the sample to predict
    features = features.iloc[-1]

    # Load the model and predict
    filename = "finalized_model.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    pred = loaded_model.predict([features])

    if pred[0] == 0:
        return "Regular Donor"
    else:
        return "High Donor"

