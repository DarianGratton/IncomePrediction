import pandas as pd

def load_full_data():
    train_data = pd.read_csv("./trainingset.csv")
    test_data = pd.read_csv("./testingset.csv")

    return pd.concat([train_data, test_data], axis=0)

def load(datafile):
    return None
