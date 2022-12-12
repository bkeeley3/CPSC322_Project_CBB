# TODO: your reusable general-purpose functions here
import numpy as np # use numpy's random number generation
import mysklearn.myevaluation as myevaluation
import mysklearn.myclassifiers as myclassifiers
import math
import matplotlib.pyplot as plt
def do_random_sub_sampling(X, y, k, test_size):
    # k = 10, test_size = 0.5
    y_true = []
    k_y_pred = []
    dummy_y_pred = []
    k_clf = myclassifiers.MyKNeighborsClassifier(k)
    dummy_clf = myclassifiers.MyDummyClassifier()
    X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y, test_size, random_state=0, shuffle=True)
    dummy_clf.fit(X_train, y_train)
    k_clf.fit(X_train, y_train)
    for i in range(len(X_test)):
        test_X = [X_test[i]]
        k_mpg_prediction = k_clf.predict(test_X)
        k_class_prediction = get_prediction(k_mpg_prediction)
        class_actual = get_prediction(y_test[i])
        k_y_pred.append(k_class_prediction)
        y_true.append(class_actual)
        dummy_mpg_prediction = dummy_clf.predict(test_X)
        dummy_class_prediction = get_prediction(dummy_mpg_prediction)
        dummy_y_pred.append(dummy_class_prediction)
    k_accuracy = myevaluation.accuracy_score(y_true, k_y_pred)
    k_error = 1.0 - k_accuracy
    dummy_accuracy = myevaluation.accuracy_score(y_true, dummy_y_pred)
    dummy_error = 1.0 - dummy_accuracy
    print("======================================")
    print("STEP 1: Predictive Accuracy")
    print("======================================")
    print("Random Subsample (k=10, 2:1 Train/Test)")
    print("k Nearest Neighbors Classifier: accuracy = {:.2f}, error rate = {:.2f}".format(k_accuracy, k_error))
    print("Dummy Classifier: accuracy = {:.2f}, error rate = {:.2f}".format(dummy_accuracy, dummy_error))
    

def do_cross_validation(X, y, k=10, random_state=None, shuffle=None, stratify=False):
    y_true = []
    k_y_pred = []
    dummy_y_pred = []
    k_clf = myclassifiers.MyKNeighborsClassifier(k)
    dummy_clf = myclassifiers.MyDummyClassifier()
    folds = []
    if stratify:
        folds = myevaluation.stratified_kfold_split(X, y, k, random_state, shuffle)
    else:
        folds = myevaluation.kfold_split(X, k, random_state, shuffle)
        
    for i in range(len(folds)):
        train_indexes = folds[i][0]
        test_indexes = folds[i][1]
        X_train = [X[index] for index in train_indexes]
        X_test = [X[index] for index in test_indexes]
        y_train = [y[index] for index in train_indexes]
        y_test = [y[index] for index in test_indexes]
        k_clf.fit(X_train, y_train)
        dummy_clf.fit(X_train, y_train)
        for j in range(len(X_test)):
            test_X = [X_test[j]]
            class_actual = get_prediction(y_test[j])
            k_mpg_prediction = k_clf.predict(test_X)
            k_class_prediction = get_prediction(k_mpg_prediction)
            dummy_mpg_prediction = dummy_clf.predict(test_X)
            dummy_class_prediction = get_prediction(dummy_mpg_prediction)
            dummy_y_pred.append(dummy_class_prediction)
            k_y_pred.append(k_class_prediction)
            y_true.append(class_actual)
    k_accuracy = myevaluation.accuracy_score(y_true, k_y_pred)
    k_error = 1.0 - k_accuracy
    dummy_accuracy = myevaluation.accuracy_score(y_true, dummy_y_pred)
    dummy_error = 1.0 - dummy_accuracy
    return folds, k_accuracy, k_error, dummy_accuracy, dummy_error, k_y_pred, y_true

def compute_euclidean_distance(v1, v2):
    
    isitString = False
    for i in range(len(v1)):
        if isinstance(v1[i], str) or isinstance(v2[i], str):
            isitString = True
    
    dist = 0
    if not isitString:
        dist = math.dist(v1, v2)
    else:
        if v1 == v2:
            dist = 0
        else:
            dist = 1
    return dist
def get_prediction(value):
    '''Assigns a fuel economy rating based on the mpg values
        Attributes:
            table (2d list) - 2d list of any type to represent the dataset we are using
            col_name(str) - string of the column name we want to look through        
        Returns:
            ratings (list of int) - list of all the mpg values changed into fuel econ rating
    '''
    if value >= 45:
        prediction = 10
    elif 37 <= value < 45:
        prediction = 9
    elif 31 <= value < 37:
        prediction = 8
    elif 27 <= value < 31:
        prediction = 7
    elif 24 <= value < 72:
        prediction = 6
    elif 20 <= value < 24:
        prediction = 5
    elif 17 <= value < 20:
        prediction = 4
    elif 15 <= value < 17:
        prediction = 3
    elif value == 14:
        prediction = 2
    else:
        prediction = 1
    return prediction

def print_cross_validation(k_accuracy, k_error, dummy_accuracy, dummy_error, \
                           strat_k_accuracy, strat_k_error, strat_dummy_accuracy, strat_dummy_error):
    print("======================================")
    print("STEP 2: Predictive Accuracy")
    print("======================================")
    print("10-Fold Cross Validation")
    print("k Nearest Neighbors Classifier: accuracy = {:.2f}, error rate = {:.2f}".format(k_accuracy, k_error))
    print("Dummy Classifier: accuracy = {:.2f}, error rate = {:.2f}\n".format(dummy_accuracy, dummy_error))
    print("Stratified 10-Fold Cross Validation")
    print("k Nearest Neighbors Classifier: accuracy = {:.2f}, error rate = {:.2f}".format(strat_k_accuracy, strat_k_error))
    print("Dummy Classifier: accuracy = {:.2f}, error rate = {:.2f}\n".format(strat_dummy_accuracy, strat_dummy_error))

def do_bootstrap(X, y=None, k=None, random_state=None):
    y_true = []
    k_y_pred = []
    dummy_y_pred = []
    k_clf = myclassifiers.MyKNeighborsClassifier(k)
    dummy_clf = myclassifiers.MyDummyClassifier()
    X_train, X_test, y_train, y_test = myevaluation.bootstrap_sample(X, y, k, random_state)
    dummy_clf.fit(X_train, y_train)
    k_clf.fit(X_train, y_train)
    for i in range(len(X_test)):
        test_X = [X_test[i]]
        k_mpg_prediction = k_clf.predict(test_X)
        k_class_prediction = get_prediction(k_mpg_prediction)
        class_actual = get_prediction(y_test[i])
        k_y_pred.append(k_class_prediction)
        y_true.append(class_actual)
        dummy_mpg_prediction = dummy_clf.predict(test_X)
        dummy_class_prediction = get_prediction(dummy_mpg_prediction)
        dummy_y_pred.append(dummy_class_prediction)
    k_accuracy = myevaluation.accuracy_score(y_true, k_y_pred)
    k_error = 1.0 - k_accuracy
    dummy_accuracy = myevaluation.accuracy_score(y_true, dummy_y_pred)
    dummy_error = 1.0 - dummy_accuracy
    print("======================================")
    print("STEP 3: Predictive Accuracy")
    print("======================================")
    print("k=10 Bootstrap Method")
    print("k Nearest Neighbors Classifier: accuracy = {:.2f}, error rate = {:.2f}".format(k_accuracy, k_error))
    print("Dummy Classifier: accuracy = {:.2f}, error rate = {:.2f}".format(dummy_accuracy, dummy_error))
    
def create_posterior(header_amount, unique_y_vals, y_train, X_train, table):
    posterior = {}
    
    all_y_rows = find_all_y_rows(unique_y_vals, y_train, X_train, table)
    all_dicts = []
    key_to_attribute_names = {}
    for j, key in enumerate(all_y_rows):
        num_iterations = 0
        curr_dict = {}
        y_curr_table = all_y_rows[key]
        for col_index in range(header_amount):
            y_curr_col = get_column(y_curr_table, col_index)
            actual_col = get_column(X_train, col_index)
            unique_actual_col_vals = np.unique(actual_col)
            for i, val in enumerate(unique_actual_col_vals):
                num_of_curr_val = y_curr_col.count(val)
                proportion = num_of_curr_val / len(y_curr_col)
                curr_attribute_name = "att{}_{}".format(col_index+1, i+1)
                if j == 0: # first loop through keys so we only save it once
                    key_to_attribute_names[curr_attribute_name] = val
                curr_dict[curr_attribute_name] = proportion
                num_iterations += 1
        all_dicts.append(curr_dict)
        num_iterations = 0
    for i, val in enumerate(unique_y_vals):
        posterior[val] = all_dicts[i]
    return posterior, key_to_attribute_names


def find_all_y_rows(unique_y_vals, y_train, X_train, table):
    all_y_rows = {}
    for row in table.data: # this for loop finds the index of the class column in the dataset
        for val in unique_y_vals:
            try:
                index_of_y = row.index(val)
                break
            except:
                pass
        break
    for i, search_val in enumerate(unique_y_vals):
        curr_y_row = []
        for row in table.data:
            if row[index_of_y] == search_val:
                temp_row = []
                for curr_index, val in enumerate(row):
                    if curr_index != index_of_y:
                        temp_row.append(val)
                    else:
                        pass
                curr_y_row.append(temp_row)
                temp_row = []
        all_y_rows[search_val] = curr_y_row
    return all_y_rows
    

def get_column(table, col_index):
    col = []
    for i, row in enumerate(table):
        try:
            col.append(row[col_index])
        except:
            print("error")
            pass
    return col
    
def cross_val_predict(X, y, k, stratified, clf):
    all_y_test = []
    all_y_predicted = []
    if stratified == False:
        folds = myevaluation.kfold_split(X, k)
    if stratified == True:
        folds = myevaluation.stratified_kfold_split(X, y, k)
    
    for fold in folds:
        X_train = [X[i] for i in fold[0]]
        y_train = [y[i] for i in fold[0]]
        X_test = [X[j] for j in fold[1]]
        y_test = [y[j] for j in fold[1]]
        for value in y_test:
            all_y_test.append(value)

        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_test)
        for value in y_predicted:
            all_y_predicted.append(value)

    return all_y_test, all_y_predicted

def card_dealing(values, places):
    distribution = []
    for i in range(0, len(values)):
        if i < places:
            distribution.append([values[i]])
        else:
            distribution[i%places].append(values[i])
    return distribution

def get_list_frequencies(list):
    values = [] # 75, 76, 77
    counts = [] # 2, 1, 1
    for value in list:
        if value not in values:
            # first time seeing this value
            values.append(value)
            counts.append(1)
        else:
            # seen this value before
            counts[values.index(value)] += 1 
    return values, counts

def measure_classifier_performance(y_true, y_pred, name_classifier, step, labels, pos_label):
    print("======================================")
    print("STEP {}: {}".format(step, name_classifier))
    print("k=10 Stratified K-Fold Cross Validation")
    print("======================================")
    
    
    accuracy = myevaluation.accuracy_score(y_true, y_pred)
    error = 1 - accuracy
    print("ACCURACY: {:.4f}".format(accuracy))
    print("ERROR RATE: {:.4f}\n".format(error))
    precision = myevaluation.binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = myevaluation.binary_recall_score(y_true, y_pred, labels, pos_label)
    f1 = myevaluation.binary_f1_score(y_true, y_pred, labels, pos_label)
    print("PRECISION SCORE: {:.4f}".format(precision))
    print("RECALL SCORE: {:.4f}".format(recall))
    print("F1 SCORE: {:.4f}\n".format(f1))
    
    matrix = myevaluation.confusion_matrix(y_true, y_pred, labels)
    print("MATRIX:")
    print(matrix)

def most_likely_label(partition):
    class_col_index = len(partition[0]) - 1
    class_col = get_column(partition, class_col_index)
    unique_class_vals = np.unique(class_col)
    class_label_1 = unique_class_vals[0]
    try:
        class_label_2 = unique_class_vals[1]
    except:
        return class_label_1
    num_label_1 = 0
    num_label_2 = 0
    for row in partition:
        if row[-1] == class_label_1:
            num_label_1 += 1
        else:
            num_label_2 += 1
    if num_label_1 > num_label_2:
        return class_label_1
    elif num_label_2 > num_label_1:
        return class_label_2
    elif num_label_1 == num_label_2:
        alpha_label_1 = class_label_1[0]
        alpha_label_2 = class_label_2[0]
        if alpha_label_1 < alpha_label_2:
            return class_label_1
        else:
            return class_label_2
    
    
def tdidt(current_instances, available_attributes, header, attribute_domains, last_partitions):
    # basic approach (uses recursion!!):
    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes, header, attribute_domains) # select_attribute should use entropy to give us an attribute to split on
    available_attributes.remove(split_attribute)
    # cannot split on this attribute again in this branch of tree
    tree = ["Attribute", split_attribute]
    
    # group data by attribute domains (creates pairwise disjoint partitions)
    before_last_partitions = last_partitions
    partitions = partition_instances(current_instances, split_attribute, header, attribute_domains)
    last_partitions = partitions
    total_partitions_length = 0
    for key in partitions:
        for row in partitions[key]:
            total_partitions_length += 1
    
    # for each partition, repeat unless one of the following occurs (base case)
    for att_value, att_partition in partitions.items():
        value_subtree = ["Value", att_value]
        if len(att_partition) > 0 and same_class_label(att_partition):
            partition_length = len(att_partition)
            current_instance_length = len(current_instances)
            leaf_node = ["Leaf", att_partition[0][-1], partition_length, current_instance_length]
            value_subtree.append(leaf_node)
            #leaf_node = ["LEAF", etc, etc, etc]
            # MAKE A LEAF NODE (refer to tree answer for help)
            #    CASE 1: all class labels of the partition are the same => make a leaf node
            
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            partition_length = len(att_partition)
            current_instance_length = len(current_instances)
            clash_label = most_likely_label(att_partition)
            leaf_node = ["Leaf", clash_label, partition_length, current_instance_length]
            value_subtree.append(leaf_node)
            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            # MAKE A LEAF NODE BY DOING MAJORITY VOTING
            
        elif len(att_partition) == 0:
            partition_length = len(att_partition)
            current_instance_length = len(current_instances)
            all_total_keys = []
            all_partitions = []
            for key, value in partitions.items():
                if key == att_value:
                    pass
                else:
                    for row in value:
                        all_partitions.append(row)
                    total_key = len(value)
                    all_total_keys.append(total_key)
            total_curr_partitions = 0
            for key in before_last_partitions:
                for row in before_last_partitions[key]:
                    total_curr_partitions += 1
            
            sum_total_keys = np.sum(all_total_keys)
            empty_label = most_likely_label(all_partitions)
            leaf_node = ["Leaf", empty_label, sum_total_keys, total_curr_partitions]
            tree = leaf_node
            break
            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            # if an empty partition, splitting on attribute was a bad idea so change tree with majority vote leaf node
            # basically tree = leaf node
            
        else: # none of the base cases were true... recurse
            subtree = tdidt(att_partition, available_attributes.copy(), header, attribute_domains, last_partitions)
            value_subtree.append(subtree)
        tree.append(value_subtree)
    
    return tree


def same_class_label(instances):
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False
    # get here, all the same
    return True

def calc_entropy(num1, num2):
    negnum1 = -abs(num1)
    negnum2 = -abs(num2)
    if num1 == 0 or num2 == 0:
        entropy = 0.0
    else:
        entropy = (negnum1 * math.log(num1, 2)) + (negnum2 * math.log(num2, 2))
    return entropy

def calc_enew(all_entropies, all_total_keys):
    all_attributes_len = np.sum(all_total_keys)
    enew_values = []
    for i, entropy in enumerate(all_entropies):
        proportion = all_total_keys[i] / all_attributes_len
        enew_val = proportion * entropy
        enew_values.append(enew_val)
    
    eNew = np.sum(enew_values)
    return eNew
    

def select_attribute(instances, attributes, header, attribute_domains):
    # TODO: implement the Enew algoritm to select
    # attribute with the lowest entropy
    # for now, let's use random selection
    # DO NOT DO THIS FOR THE ACTUAL ASSIGNMENT, USE ENTROPY
    class_col_index = len(instances[0]) - 1
    class_col = get_column(instances, class_col_index)
    unique_class_vals = np.unique(class_col)
    class_label_1 = unique_class_vals[0]
    
    
    all_eNew = []
    for attribute in attributes:
        partitions = partition_instances(instances, attribute, header, attribute_domains)
        all_entropies = []
        all_total_keys = []
        for key, value in partitions.items():
            total_key = len(value)
            all_total_keys.append(total_key)
            if total_key == 0:
                curr_entropy = 0.0
            else:
                current_partition = partitions[key]
                amt_label_1 = 0
                amt_label_2 = 0
                for row in current_partition:
                    if row[-1] == class_label_1:
                        amt_label_1 += 1
                    else:
                        amt_label_2 += 1
                num1 = amt_label_1 / total_key
                num2 = amt_label_2 / total_key
                curr_entropy = calc_entropy(num1, num2)
            all_entropies.append(curr_entropy)
            
        eNew = calc_enew(all_entropies, all_total_keys)
        all_eNew.append(eNew)
    
    min_index = 0
    min_val = all_eNew[0]
    for i, eNewVal in enumerate(all_eNew):
        if min_val > eNewVal:
            min_index = i
            min_val = eNewVal
    
    selected_attribute = attributes[min_index]
    
    return selected_attribute

def partition_instances(instances, attribute, header, attribute_domains):
    # this is a group by attribute domain
    att_index = header.index(attribute)
    att_domain = attribute_domains["att" + str(att_index)]
    # let's use dictionaries
    partitions = {}
    for att_value in att_domain:
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)
    
    return partitions


def ifvalueinrow(X_test, row):
    isinrow = False
    for val in X_test:
        if val in row:
            isinrow = True
    
    return isinrow

def getTreePrediction(curr_row, X_test, header, curr_att=None, curr_index=None):
    
    prediction = None
    
    
    if curr_row[0] == "Attribute":
        curr_att = curr_row[1]
        curr_index = header.index(curr_att)
    for row in curr_row:
        try: # "in" operator doesn't allow non-string vals on the LHS, so we will simply use try except block to get around this
            if X_test[curr_index] in row:
                if "Leaf" in row[-1]:
                    prediction = row[-1][1]
                    return prediction
                else:
                    prediction = getTreePrediction(row[-1], X_test, header, curr_att, curr_index)
        except:
            pass
            
        
        #prediction = getTreePrediction(curr_row[-1], X_test, header, curr_att, curr_index)

    return prediction

def getDecisionRules(curr_row, class_label, header, decision_rules, header_to_att_name=None):
    print(curr_row)
    all_decision_rules = []
    for outer_row in curr_row:
        print("outer row:", outer_row)
        inner_row = curr_row
        ruleString = ""
        while "Leaf" not in inner_row:
            if "Attribute" in inner_row:
                curr_att = inner_row[1]
                curr_index = header.index(curr_att)
                if header_to_att_name:
                    curr_att = header_to_att_name[curr_att]
                ruleString += "IF {} == ".format(curr_att)
                inner_row = inner_row[-1]
            if "Value" in inner_row:
                val_to_str = str(inner_row[1])
                ruleString += val_to_str
                if "Leaf" in inner_row[-1]:
                    end_string = " THEN {} == {}".format(class_label, inner_row[-1][1])
                    ruleString += end_string
                    break
                else:
                    ruleString += " AND "
                    inner_row = inner_row[-1]
        print("ruleString:", ruleString)
        all_decision_rules.append(ruleString)
            
    return all_decision_rules
    
def bar_chart(x, y, xlabel=None, ylabel=None, tick_labels=None, title=None):
    plt.figure()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x, tick_labels, rotation=75, ha="right")
    plt.show()
    
def find_col_average(table, col_name):
    col = table.get_column(col_name)
    total = sum(col)
    avg = total / len(col)
    return avg
    
    
def do_fold_predictions(X, y, folds, clf, clf_name=None):
    
    y_true = []
    y_pred = []
    for i in range(len(folds)):
        train_indexes = folds[i][0]
        test_indexes = folds[i][1]
        X_train = [X[index] for index in train_indexes]
        X_test = [X[index] for index in test_indexes]
        y_train = [y[index] for index in train_indexes]
        y_test = [y[index] for index in test_indexes]
        clf.fit(X_train, y_train)
        for j in range(len(X_test)):
            test_X = [X_test[j]]
            class_actual = y_test[j]
            prediction = clf.predict(test_X)
            y_true.append(class_actual)
            if clf_name == "Naive Bayes":
                y_pred.append(prediction[0])
            else:
                y_pred.append(prediction)
            
            
    return y_true, y_pred


def make_tree():
    pass

def compute_random_subset(values, num_values):
    # could use np.random.choice()
    # we will use np.random.shuffle() and slicing
    values_copy = values.copy()
    np.random.shuffle(values_copy) # inplace shuffle
    return values_copy[:num_values]

def get_majority_vote(predictions):
    instance_predictions, frequencies = get_list_frequencies(predictions)
    most_votes = instance_predictions[0]
    num_votes = frequencies[0]
    for i in range(len(instance_predictions)):
        if frequencies[i] > num_votes:
            num_votes = frequencies[i]
            most_votes = instance_predictions[i]
    return most_votes