import numpy as np # use numpy's random number generation
from math import ceil
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
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    np.random.seed(random_state)
    if type(test_size) == int:
        num_test_instances = test_size
    elif type(test_size) == float:
        num_test_instances = ceil(len(X) * test_size)
    if shuffle == True:
        myutils.randomize_in_place(X, y)
    
    for i in range(0, len(X) - num_test_instances):
        X_train.append(X[i])
        y_train.append(y[i])
    for j in range(len(X) - num_test_instances, len(X)):
        X_test.append(X[j])
        y_test.append(y[j])
    
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
    folds = []
    
    np.random.seed(random_state)
    indexes = [i for i in range(0, len(X))]
    if shuffle == True:
        myutils.randomize_in_place(X, indexes)
    
    sets = myutils.card_dealing(indexes, n_splits)
    for set in sets:
        test = set
        train = []
        for s in sets:
            if s != set:
                for num in s:
                    train.append(num)
            else:
                continue
        folds.append(tuple((train, test)))

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
    folds = []

    np.random.seed(random_state)
    indexes = []
    values = []
    for value in y:
        if value not in values:
            values.append(value)
    
    for value in values:
        for i in range(0, len(y)):
            if y[i] == value:
                indexes.append(i)
    
    if shuffle == True:
        myutils.randomize_in_place(X, indexes)
    
    sets = myutils.card_dealing(indexes, n_splits)
    for set in sets:
        test = set
        train = []
        for s in sets:
            if s != set:
                for num in s:
                    train.append(num)
            else:
                continue
        folds.append(tuple((train, test)))

    return folds 

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

    sample_indexes = []
    indexes = [i for i in range(0, len(X))]
    np.random.seed(random_state)

    if n_samples == None:
        n_samples = len(X)
    
    for _ in range(0, n_samples):
        sample_index = np.random.randint(0, len(X))
        sample_indexes.append(sample_index)
        
    X_sample = [X[i] for i in sample_indexes]
    y_sample = [y[i] for i in sample_indexes]

    X_out_of_bag = [X[value] for value in indexes if value not in sample_indexes]
    y_out_of_bag = [y[value] for value in indexes if value not in sample_indexes]

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
    matrix = []

    for _ in range(0, len(labels)):
        matrix.append([0 for _ in range(0, len(labels))])
    
    for i in range(0, len(y_true)):
        true_index = labels.index(y_true[i])
        pred_index = labels.index(y_pred[i])
        matrix[true_index][pred_index] += 1
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
    score = 0
    for i in range(0, len(y_true)):
        if y_true[i] == y_pred[i]:
            score += 1
    if normalize == True:
        return score / len(y_true)
    
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
    precision = 0.0
    matrix = confusion_matrix(y_true, y_pred, labels)
    if pos_label is not None:
        true_index = labels.index(pos_label)
    else:
        true_index = 0
   
    try:
        precision = (matrix[true_index][true_index]) / (matrix[true_index][true_index] \
             + matrix[(true_index + 1) % 2][true_index])
    except ZeroDivisionError:
        precision = 0.0
    
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
    recall = 0.0
    matrix = confusion_matrix(y_true, y_pred, labels)

    if pos_label is not None:
        true_index = labels.index(pos_label)
    else:
        true_index = 0
    
    try:
        recall = (matrix[true_index][true_index]) / (matrix[true_index][true_index] \
             + matrix[true_index][(true_index + 1) % 2])
    except ZeroDivisionError:
        recall = 0.0

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
    f1 = 0.0

    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    try:
        f1 = (2 * precision * recall) / (precision + recall)    
    except ZeroDivisionError:
        f1 = 0.0

    return f1