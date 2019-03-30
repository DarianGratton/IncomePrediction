import pandas as pd

def load_full_data():
    train_data = pd.read_csv("./trainingset")
    test_data = pd.read_csv("./testingset")

    return train_data

def load(datafile):
    return None
