import pandas as pd
import numpy as np
import load_data as loader
import random
from sklearn.svm import SVC

# Load the training and testing data
train_data = loader.load('./trainingset.csv', one_hot_encode=True, max_unique=50)
test_data = loader.load('./testingset.csv', one_hot_encode=True, max_unique=50)

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

test_features = loader.get_missing_features(train_features, test_features)

clf = SVC(gamma='auto')
clf.fit(train_features, train_labels)
pred = clf.predict(test_features)

# Check how many are correctly predicted
j = 0
correct_pred = 0
for predicted in pred:
    validation_labels_val = test_labels.iloc[j]
    if predicted == validation_labels_val.income:
        correct_pred += 1
    j += 1

error_rate = (len(test_data) - correct_pred) / len(test_data)
print("Test Set Error Rate: ", error_rate)
