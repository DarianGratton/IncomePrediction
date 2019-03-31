import pandas as pd

def load_full_data():
    train_data = pd.read_csv("./trainingset.csv")
    test_data = pd.read_csv("./testingset.csv")

    return pd.concat([train_data, test_data], axis=0)

def load(file_path, one_hot_encode = False, max_unique = 20):
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
            if 2 < data[col].nunique() < max_unique:
                hot_encoding_indices.append(col)
        data = pd.get_dummies(data, columns=hot_encoding_indices, prefix=hot_encoding_indices)

    return data


