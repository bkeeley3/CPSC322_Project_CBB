import operator
import statistics

import numpy as np

from mysklearn import myutils
from mysklearn.mypytable import MyPyTable


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        neighbor_index = 0
        for neighbor_index, train_instance in enumerate(self.X_train):
            distance = myutils.compute_euclidean_distance(train_instance, X_test[0])
            distances.append([distance])
            neighbor_indices.append([neighbor_index])
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, neighbor_indices = self.kneighbors(X_test)
        distance_and_indexes = {}
        for i in range(len(distances)):
            distance_and_indexes[neighbor_indices[i][0]] = distances[i][0]
        sorted_distances = sorted(distance_and_indexes.items(), key=operator.itemgetter(-1))
        k_nearest = sorted_distances[:self.n_neighbors]
        y_predicted_k = []
        for index in k_nearest:
            y_predicted_k.append(self.y_train[index[0]])
            
        most_common = statistics.mode(y_predicted_k)
        return most_common

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.
    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.
        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        self.most_common_label = statistics.mode(y_train)
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return self.most_common_label

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None
        self.key_to_attribute_names = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        nb_table = MyPyTable()
        nb_data = []
        nb_column_names = []
        unique_y_vals = np.unique(y_train)
        num_samples = len(y_train)
        header_amount = len(X_train[0]) + 1 # the plus 1 is added for the class attribute
        self.priors = np.zeros(len(unique_y_vals))
        for i in range(header_amount):
            header_name = "att{}".format(i + 1)
            nb_column_names.append(header_name)
        nb_table.column_names = nb_column_names
        for i in range(len(y_train)):
            temp_row = []
            for j in range(len(X_train[0])):
                temp_row.append(X_train[i][j])
            temp_row.append(y_train[i])
            nb_data.append(temp_row)
        nb_table.data = nb_data
        for i, search_val in enumerate(unique_y_vals):
            num_found = y_train.count(search_val)
            curr_prior = num_found / num_samples
            self.priors[i] = curr_prior
        self.posteriors, self.key_to_attribute_names= myutils.create_posterior((header_amount-1), unique_y_vals, y_train, X_train, nb_table)
        

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        
        y_predicted = []
        for test in X_test:
            all_probabilities = {}
            for prior_index, key in enumerate(self.posteriors):
                curr_dict = self.posteriors[key]
                probabilities = []
                for i, val in enumerate(test):
                    for att_value in curr_dict:
                        actual_att_value = self.key_to_attribute_names[att_value]
                        if val == actual_att_value:
                            probabilities.append(curr_dict[att_value])
                probability = probabilities[0]
                for prob in probabilities[1:]:
                    probability *= prob
                probability *= self.priors[prior_index]
                all_probabilities[key] = probability
            max_value = 0
            for key in all_probabilities:
                if max_value < all_probabilities[key]:
                    max_value = all_probabilities[key]
                    prediction = key
            y_predicted.append(prediction)
        
        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.header = None
        self.attribute_domain = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        curr_header_val = []
        num_attributes = len(X_train[0])
        for i in range(num_attributes):
            header_att_val = "att{}".format(i)
            curr_header_val.append(header_att_val)
        self.header = curr_header_val
        
        
        self.attribute_domain = {}
        for i, header_val in enumerate(self.header):
            curr_col = myutils.get_column(X_train, i)
            unique_col_vals = np.unique(curr_col)
            unique_col_vals = np.ndarray.tolist(unique_col_vals)
            self.attribute_domain[header_val] = unique_col_vals
            
        
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = self.header.copy() # recall
        # to be removing attributes from a list of available attributes
        # python is pass by object reference!!
        self.tree = myutils.tdidt(train, available_attributes, self.header, self.attribute_domain, None)
        

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for test in X_test:
            prediction = myutils.getTreePrediction(self.tree, test, self.header)
            y_predicted.append(prediction)
        
        
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        decision_rules = []
        header_to_att_name = {}
        if attribute_names:
            for i, att_name in enumerate(attribute_names):
                header_to_att_name[self.header[i]] = att_name
        
        first_att = self.tree[1]
        for top_branch in self.tree[2:]:
            top_string = "IF {} == ".format(first_att)
            for lower_branch in top_branch[2:]:
                final_string = top_string + "{}".format(lower_branch[1])
                if "Leaf" in lower_branch:
                    final_string += "THEN {} == {}".format(class_name, lower_branch[1])
                curr_branch_index = 2
                for lowest_branch in lower_branch:
                    while "Leaf" not in lowest_branch:
                        if "Attribute" in lowest_branch:
                            curr_att = lowest_branch[1]
                            final_string += " AND IF {} == ".format(curr_att)
                            lowest_branch = lowest_branch[curr_branch_index]
                        elif "Value" in lowest_branch:
                            final_string += "{}".format(lowest_branch[1])
                            
                print("lower_branch:", lower_branch)
    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this
