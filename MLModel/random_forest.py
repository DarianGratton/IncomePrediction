import pandas as pd
import numpy as np
import data_handler as handler
import random
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# Load the training and testing data
train_data = handler.load('./trainingset.csv', one_hot_encode=True, max_unique=50)
test_data = handler.load('./testingset.csv', one_hot_encode=True, max_unique=50)

# # Remove unwanted features
# train_data = train_data.drop(['native-country', 'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black',
#                               'race_Other', 'race_White', 'capital-gain', 'capital-loss', 'fnlwgt'], axis=1, inplace=False)
# test_data = test_data.drop(['native-country', 'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black',
#                               'race_Other', 'race_White', 'capital-gain', 'capital-loss', 'fnlwgt'], axis=1, inplace=False)
#
# print(test_data.info())
# print(train_data.info())

# Number of samples in the train_data
num_rows = train_data.shape[0]
# Shuffle the data
shuffled_indices = list(range(num_rows))
random.seed(42)
random.shuffle(shuffled_indices)

# Get indices of training data
train_indices = shuffled_indices[:len(train_data)]
# Create new train_data
train_data = train_data.iloc[train_indices, :]
print(len(train_data), "training")

# Prepare features and labels
train_features = train_data.drop('income', axis=1, inplace=False)
train_labels = train_data.loc[:, ['income']]
test_features = test_data.drop('income', axis=1, inplace=False)
test_labels = test_data.loc[:, ['income']]

test_features = handler.get_missing_features(train_features, test_features)

# Initialize variables for Cross-Validation
n_estimators = [10, 20, 50, 100, 200, 500, 1000]    # Different estimators
K = 5                                               # K-fold
validation_error_rates = []
best_error_rate = 1                                 # Initialized to worst possible error rate
best_estimator = n_estimators[0]

for estimators in n_estimators:

    # Calculate the size of each subset
    validation_ratio = 1 / K
    validation_set_size = int(len(train_data) * validation_ratio)

    # Initialize variable to keep track of the correct predictions
    validation_correct_pred = 0
    print("\nEstimators: ", estimators)

    for i in range(K):

        # Calculate and locate the indexes for current validation subset
        validation_indices = list(range(i * validation_set_size, (i + 1) * validation_set_size))
        validation_data = train_data.iloc[validation_indices, :]
        # Prepare validation features and training labels
        validation_features = validation_data.drop('income', axis=1, inplace=False)
        validation_labels = validation_data.loc[:, ['income']]

        # Drop the validation set from the training set
        training_data = train_data.drop(train_data.index[validation_indices])
        # Prepare training features and training labels
        train_features = training_data.drop('income', axis=1, inplace=False)
        train_labels = training_data.loc[:, ['income']]

        # Create the random forest model
        clf = RandomForestClassifier(n_estimators=estimators, max_depth=12)
        # Fit the model
        clf.fit(train_features, train_labels)
        # Predict on the validation set
        pred = clf.predict(validation_features)

        # Check how many are correctly predicted
        j = 0
        for predicted in pred:
            validation_labels_val = validation_labels.iloc[j]
            if predicted == validation_labels_val.income:
                validation_correct_pred += 1
            j += 1

    # Calculate the Error Rate
    validation_error_rate = (len(train_data) - validation_correct_pred) / len(train_data)
    print("Error Rate: ", validation_error_rate)

    validation_error_rates.append(validation_error_rate)

    # Check for best C
    if validation_error_rate < best_error_rate:
        best_estimator = estimators
        best_error_rate = validation_error_rate

print("\nBest Estimators: ", best_estimator)
print("Best Error Rate: ", best_error_rate)

# Fit the SVC model
clf = RandomForestClassifier(n_estimators=best_estimator, max_depth=13)
clf.fit(train_features, train_labels)
# Predict the output
pred = clf.predict(test_features)

# Check how many are correctly predicted
j = 0
correct_pred = 0
for predicted in pred:
    validation_labels_val = test_labels.iloc[j]
    if predicted == validation_labels_val.income:
        correct_pred += 1
    j += 1

print("F1 Score: ", f1_score(test_labels.values, pred))
error_rate = (len(test_data) - correct_pred) / len(test_data)
print("Test Set Error Rate: ", error_rate)

# Print results to file
handler.print_results_to_csv(test_labels, pred)
