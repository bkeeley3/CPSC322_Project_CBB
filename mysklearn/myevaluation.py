from mysklearn import myutils

from random import random
import numpy as np 
import random
import copy
import math
from mysklearn import myutils
def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!
    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    random.seed(random_state)
    if isinstance(test_size, int):
        num_in_test_set = test_size
    else:
        num_in_test_set = math.ceil((test_size * len(X)))
    if shuffle:
        shuffled_indexes = [index for index in range(len(X))]
        random.shuffle(shuffled_indexes)
        X = [X[index] for index in shuffled_indexes]
        y = [y[index] for index in shuffled_indexes]  
    X_copy = copy.deepcopy(X)
    y_copy = copy.deepcopy(y)
    X_test = []
    y_test = []
    for i in range(num_in_test_set):
        X_test.append(X_copy[-1])
        y_test.append(y_copy[-1])
        X_copy.pop()
        y_copy.pop()
    X_test.reverse()
    y_test.reverse()
    X_train = X_copy
    y_train = y_copy
    return X_train, X_test, y_train, y_test
def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold
    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    random.seed(random_state) # occurs if a random state is not set
    X_indexes = [index for index in range(len(X))]
    if shuffle: # if shuffle is true
        random.shuffle(X_indexes)
        X = [X[index] for index in X_indexes]
    folds = []
    n_samples = len(X)
    first_folds = n_samples % n_splits # first_folds is the number of folds that have a higher split size because of uneven splits
    last_index = 0 # initialize at 0 so we can iterate through
    for i in range(n_splits):
        X_train_indexes = [] # initialize to blank for every iteration
        X_test_indexes = [] # ^^
        if i <= first_folds and first_folds != 0: # if the splits are uneven
            size = (n_samples // n_splits) + 1
        else: # if the splits are even
            size = (n_samples // n_splits)
        for j in range(size): # iterates over the size of the split
            try:
                X_test_indexes.append(X_indexes[last_index])
            except:
                pass
            last_index += 1 # keeps track of the last index used
        for index in X_indexes: # looks through all indexes in X_indexes
            if index not in X_test_indexes: # if it's not in the test
                X_train_indexes.append(index) # append it to the train
        fold = (X_train_indexes, X_test_indexes) # creates an individual fold
        folds.append(fold) # append the fold
    return folds

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold
    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    random.seed(random_state) # occurs if a random state is not set
    X_indexes = [index for index in range(len(X))]
    if shuffle: # if shuffle is true
        random.shuffle(X_indexes)
        X = [X[index] for index in X_indexes]
    folds_strat = []
    n_samples = len(X)
    pos_val = y[0] # automatically assigns the first value as positive even though it is ambiguous whether or not it is actually positive
    first_folds = n_samples % n_splits # first_folds is the number of folds that have a higher split size because of uneven splits
    next_val = 0 # ends up determining where we put the positive and negative values in the arrays
    pos_values = []
    neg_values = []
    pos_and_neg_values = [] # combined array of the positive and negative values
    for i in range(len(X)):
        if y[i] == pos_val: # if it equals the ambiguous positive val, add that to the "positive" array
            pos_values.append(X_indexes[i])
        else:
            neg_values.append(X_indexes[i])
    pos_last_index = 0 # keeps track of the last positive index
    neg_last_index = 0 # keeps track of the last negative index
    for j in range(len(y)): # we first must add positive and negative values to the combined array in alternating order so splits are evenly distributed
        if next_val % 2 == 0: # this uses an algorithm to alternate by determining if the "next_val" term is even or odd
            try:
                pos_and_neg_values.append(pos_values[pos_last_index])
            except:
                pass
            pos_last_index += 1
            next_val += 1
        else:
            try:
                pos_and_neg_values.append(neg_values[neg_last_index])
            except:
                pass
            neg_last_index += 1
            next_val += 1
    last_index = 0 # initialized to 0 every time
    for n in range(n_splits):
        X_train_indexes = [] # initialize to blank for every iteration
        X_test_indexes = [] # ^^
        if n <= first_folds and first_folds != 0: # if the splits are uneven
            size = (n_samples // n_splits) + 1
        else: # if the splits are even
            size = (n_samples // n_splits)
        for m in range(size): # iterates over the size of the split
            try:
                X_test_indexes.append(pos_and_neg_values[last_index])
            except:
                pass
            last_index += 1 # keeps track of the last index used
        for index in X_indexes: # looks through all indexes in X_indexes
            if index not in X_test_indexes: # if it's not in the test
                X_train_indexes.append(index) # append it to the train
        fold = (X_train_indexes, X_test_indexes) # creates an individual fold
        folds_strat.append(fold) # append the fold    
    return folds_strat

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.
    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    random.seed(random_state)
    if n_samples is None:
        sample_size = len(X)
    else:
        sample_size = n_samples
    X_indexes = [index for index in range(len(X))]
    X_sample_indexes = np.random.choice(X_indexes, size=sample_size, replace=True)
    X_sample = [X[index] for index in X_sample_indexes]
    y_sample = [y[index] for index in X_sample_indexes]
    X_out_of_bag_indexes = []
    for i, index in enumerate(X_indexes):
        actual_index = [index]
        if actual_index not in X_sample_indexes:
            X_out_of_bag_indexes.append(index)
    X_out_of_bag = [X[index] for index in X_out_of_bag_indexes]
    y_out_of_bag = [y[index] for index in X_out_of_bag_indexes]
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    labels = np.array(labels) # because we are going to use numpy functions, we have to change all of lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    matrix = np.zeros((len(labels), len(labels))) # creates a blank matrix with the shape of labels x labels
    for i in range(len(labels)):
        for j in range(len(labels)):
            matrix[i, j] = np.sum((y_true == labels[i]) & (y_pred == labels[j])) # sums each time y_true equals current label and y_pred equals current label 
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).
    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    num_corr_predictions = 0
    for i in range(len(y_pred)):
        if y_pred[i][0] == y_true[i]:
            num_corr_predictions += 1
            
    print("Number of Correct Predictions:", num_corr_predictions)
    print("Total Predictions:", len(y_pred))
    score = 0.0
    if normalize:
        score = num_corr_predictions / len(y_true)
    else:
        score = num_corr_predictions
    return score

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        precision(float): Precision of the positive class
    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    matrix = confusion_matrix(y_true, y_pred, labels)
    if pos_label == labels[0]:
        tp = matrix[0][0]
        fp = matrix[1][0]
    else:
        tp = matrix[1][1]
        fp = matrix[0][1]
    dividend = tp+fp
    if dividend > 0:
        precision = tp / (tp+fp)
    else:
        precision = 0
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        recall(float): Recall of the positive class
    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    matrix = confusion_matrix(y_true, y_pred, labels)
    if pos_label == labels[0]:
        tp = matrix[0][0]
        fn = matrix[0][1]
    else:
        tp = matrix[1][1]
        fn = matrix[1][0]
    dividend = tp+fn
    if dividend > 0:
        recall = tp / (tp+fn)
    else:
        recall = 0
    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        f1(float): F1 score of the positive class
    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    matrix = confusion_matrix(y_true, y_pred, labels)
    if pos_label == labels[0]:
        tp = matrix[0][0]
        fn = matrix[0][1]
        fp = matrix[1][0]
    else:
        tp = matrix[1][1]
        fn = matrix[1][0]
        fp = matrix[0][1]
    dividend_r = tp+fn
    if dividend_r > 0:
        recall = tp / (tp+fn)
    else:
        recall = 0
    dividend_p = tp+fp
    if dividend_p > 0:
        precision = tp / (tp+fp)
    else:
        precision = 0
    dividend_f1 = precision+recall
    if dividend_f1 > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1
