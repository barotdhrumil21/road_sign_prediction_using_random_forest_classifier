# Random Forest Algorithm on Sonar dataframe
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import pandas as pd


# Load a CSV file
def load_csv(filename):
    dataframe = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataframe.append(row)
    return dataframe


# Convert string column to float
def str_column_to_float(dataframe, column):
    for row in dataframe:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataframe, column):
    class_values = [row[column] for row in dataframe]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataframe:
        row[column] = lookup[row[column]]
    return lookup


# for converting dataframe_copy into new datframe so we can remove rows that are included in folds
def remove_list_items(list_of_records, rr):
    result = []
    for r in list_of_records.T:
        if not r == rr:
            result.append(r)
    return result


# Split a dataframe into k folds
def cross_validation_split(dataframe, n_folds):
    dataframe_split = list()
    dataframe_copy = dataframe
    #print((dataframe_copy))

    fold_size = int(len(dataframe) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            indexs = randrange(len(dataframe_copy))
            fold.append(dataframe_copy.iloc[indexs])
            dataframe_copy=dataframe_copy.drop(dataframe_copy.index[indexs])
            print(len(dataframe_copy))
        dataframe_split.append(fold)
        print(dataframe_split)
    return dataframe_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataframe, algorithm, n_folds, *args):
    folds = cross_validation_split(dataframe, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        print(train_set)
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            # row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Split a dataframe based on an attribute and an attribute value
def test_split(index, value, dataframe):
    left, right = list(), list()
    for row in dataframe:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataframe
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


# Select the best split point for a dataframe
def get_split(dataframe, n_features):
    class_values = list(set(row[-1] for row in dataframe))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataframe[0]) - 1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataframe:
            groups = test_split(index, row[index], dataframe)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Create a random subsample from the dataframe with replacement
def subsample(dataframe, ratio):
    sample = list()
    n_sample = round(len(dataframe['Id']) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataframe.T) - 1)
        sample.append(dataframe[index])
    return sample


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)


# Test the random forest algorithm
seed(1)
# load and prepare data
filename = 'a.csv'
dataframe = pd.read_csv(filename)

#id=dataframe.loc[:,'Id']
#dataframe=dataframe.drop('Id',axis=1)
#indices=[int(i) for i in range(len(dataframe['Id']))]
#dataframe.loc[:,'indices']=indices
# convert string attributes to integers
# for i in range(0, len(dataframe[0]) - 1):
# str_column_to_float(dataframe, i)
# convert class column to integers
# str_column_to_int(dataframe, len(dataframe[0]) - 1)
# evaluate algorithm
target = dataframe.loc[:, str('SignFacing (Target)')]
dataframe = dataframe.drop(['SignWidth', 'SignHeight'], axis=1)


def convert_to_int():
    str = dataframe.loc[:, ['DetectedCamera', 'SignFacing (Target)']].as_matrix()
    for i in range(len(str)):
        if str[i, 0] == 'Front':
            str[i, 0] = 1
        elif str[i, 0] == 'Rear':
            str[i, 0] = 2
        elif str[i, 0] == 'Left':
            str[i, 0] = 3
        elif str[i, 0] == 'Right':
            str[i, 0] = 4
        if str[i, 1] == 'Front':
            str[i, 1] = 1
        elif str[i, 1] == 'Rear':
            str[i, 1] = 2
        elif str[i, 1] == 'Right':
            str[i, 1] = 4
        elif str[i, 1] == 'Left':
            str[i, 1] = 3
    dataframe.loc[:, 'DetectedCamera'] = str[:, 0]
    dataframe.loc[:, 'SignFacing (Target)'] = str[:, 1]
    # print(len(dataframe.T)-1z)


convert_to_int()
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataframe['Id']) - 1))
for n_trees in [1]:
    scores = evaluate_algorithm(dataframe, random_forest, n_folds, max_depth, min_size, sample_size, n_trees,
                                n_features)
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
