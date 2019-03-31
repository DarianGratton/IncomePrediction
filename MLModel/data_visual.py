import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import load_data as loader

# Load the data
data = loader.load_full_data()

max_unique = 20     # max number of unique values before it is considered continuous
for feature in data:

    # Separate the dataset into possible high donors and regular donors
    hd = data[data['income'] == '>50K']
    rd = data[data['income'] == '<=50K']

    if data[feature].nunique() <= max_unique or str(feature) == 'native-country':

        # Create the first bar plot for high donors
        feature_name, counts = np.unique(hd.loc[:, str(feature)], return_counts=True)
        plt.subplot(2, 1, 1)
        plt.barh(feature_name, counts)
        plt.title("Number of each type of person >50K")
        plt.xlabel('count')
        plt.ylabel(str(feature))
        for i, v in enumerate(counts):
            plt.text(v + 3, i, str(v), color='green')

        # Create the second bar plot for regular donors
        feature_name, counts = np.unique(rd.loc[:, str(feature)], return_counts=True)
        plt.subplot(2, 1, 2)
        plt.barh(feature_name, counts)
        plt.title("Number of each type of person <=50K")
        plt.xlabel('count')
        plt.ylabel(str(feature))
        for i, v in enumerate(counts):
            plt.text(v + 3, i, str(v), color='green')

    else:
        x = hd.loc[:, str(feature)]
        plt.subplot(2, 1, 1)
        plt.title(str(feature) + " Histogram >50K")
        plt.xlabel(str(feature))
        plt.ylabel('Frequency')
        plt.hist(x, bins=20)

        x = rd.loc[:, str(feature)]
        plt.subplot(2, 1, 2)
        plt.title(str(feature) + " Histogram <=50K")
        plt.xlabel(str(feature))
        plt.ylabel('Frequency')
        plt.hist(x, bins=20)

    plt.subplots_adjust(hspace=0.5)
    plt.rcParams["figure.figsize"] = (8, 8)

    plt.show()
