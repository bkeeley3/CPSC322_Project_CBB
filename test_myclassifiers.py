import numpy as np
import pytest

from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.myclassifiers import MyNaiveBayesClassifier
import numpy as np
from mysklearn.myclassifiers import MyKNeighborsClassifier,\
    MyDummyClassifier
    
def high_low_discretizer(value):
    if value <= 100:
        return "low"
    return "high"
# note: order is actual/received student value, expected/solution
def test_kneighbors_classifier_kneighbors():
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    lin_kn1 = MyKNeighborsClassifier()
    lin_kn1.fit(X_train_class_example1, y_train_class_example1)
    X_test = [[0.33, 1]]
    pred_kdistances_1, k_indexes_1 = lin_kn1.kneighbors(X_test)
    k_indexes_1 = None
    actual_kdistances_1 = [[0.67], [1.203], [1.0], [1.053]]
    assert np.allclose(pred_kdistances_1, actual_kdistances_1, rtol=.01)
# from in-class #2 (8 instances)
# assume normalized
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    lin_kn2 = MyKNeighborsClassifier()
    lin_kn2.fit(X_train_class_example2, y_train_class_example2)
    X_test = [[4, 2]]
    pred_kdistances_2, k_indexes_2 = lin_kn2.kneighbors(X_test)
    k_indexes_2 = None
    actual_kdistances_2 = [[1.0], [4.47], [1.0], [2.0], [3.0], [2.83], [4.12], [5.0]]
    assert np.allclose(pred_kdistances_2, actual_kdistances_2, rtol=.01)
# from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]
    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]
    lin_kn3 = MyKNeighborsClassifier()
    lin_kn3.fit(X_train_bramer_example, y_train_bramer_example)
    X_test = [[16.0, 7.2]]
    pred_kdistances_bramer, k_indexes_bramer = lin_kn3.kneighbors(X_test)
    k_indexes_bramer = None
    actual_kdistances_bramer = [[15.22], [14.63], [13.90], [15.16], [10.67], [7.66], [8.10], [5.73], [4.99], [3.67], [6.89], [12.86]\
        , [11.44], [10.21], [5.00], [0.78], [3.04], [2.22], [4.84], [5.31]]
    assert np.allclose(pred_kdistances_bramer, actual_kdistances_bramer, rtol=.01)
def test_kneighbors_classifier_predict():
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    lin_kn1 = MyKNeighborsClassifier()
    lin_kn1.fit(X_train_class_example1, y_train_class_example1)
    X_test = [[0.33, 1]]
    prediction1 = lin_kn1.predict(X_test)
    assert prediction1 == "good" # desk calculation
# from in-class #2 (8 instances)
# assume normalized
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    lin_kn2 = MyKNeighborsClassifier()
    lin_kn2.fit(X_train_class_example2, y_train_class_example2)
    X_test = [[4, 2]]
    prediction2 = lin_kn2.predict(X_test) 
    assert prediction2 == 'no' # desk calcuation
# from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]
    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"] 
    lin_kn3 = MyKNeighborsClassifier()
    lin_kn3.fit(X_train_bramer_example, y_train_bramer_example)
    X_test = [[16.0, 7.2]]
    prediction_bramer = lin_kn3.predict(X_test)
    assert prediction_bramer == "+" # desk calcuation

def test_naive_bayes_classifier_fit():
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    keylist1 = np.unique(y_train_inclass_example)
    
    priors_solution_1 = [0.375, 0.625]
    
    no_vals_dict1 = {"att1_1": 2/3, "att1_2": 1/3, "att2_1": 2/3, "att2_2": 1/3}
    yes_vals_dict1 = {"att1_1": 4/5, "att1_2": 1/5, "att2_1": 2/5, "att2_2": 3/5}
    post_solution_1 = {"no": no_vals_dict1, "yes": yes_vals_dict1}
    
    nb_clf1 = MyNaiveBayesClassifier()
    nb_clf1.fit(X_train_inclass_example, y_train_inclass_example)
    assert np.allclose(nb_clf1.priors, priors_solution_1)
    
    test_breakdown_dict_1 = np.array([nb_clf1.posteriors[key] for key in keylist1])
    combined_sol_dict1 = []
    combined_sol_dict1.append(no_vals_dict1)
    combined_sol_dict1.append(yes_vals_dict1)
    for i in range(len(test_breakdown_dict_1)):
        curr_test_dict = test_breakdown_dict_1[i]
        curr_sol_dict = combined_sol_dict1[i]
        for key in curr_test_dict:
            test_value = curr_test_dict[key]
            sol_value = curr_sol_dict[key]
            assert np.isclose(test_value, sol_value)
    
    
    # RQ5 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    keylist2 = np.unique(y_train_iphone)
    priors_solution_2 = [0.333333, 0.666667]
    
    no_vals_dict2 = {"att1_1": 3/5, "att1_2": 2/5, "att2_1": 1/5, "att2_2": 2/5, "att2_3": 2/5, "att3_1": 3/5, "att3_2": 2/5}
    yes_vals_dict2 = {"att1_1": 2/10, "att1_2": 8/10, "att2_1": 3/10, "att2_2": 4/10, "att2_3": 3/10, "att3_1": 3/10, "att3_2": 7/10}
    post_solution_2 = {"no": no_vals_dict2, "yes": yes_vals_dict2}
    
    nb_clf2 = MyNaiveBayesClassifier()
    nb_clf2.fit(X_train_iphone, y_train_iphone)
    
    assert np.allclose(nb_clf2.priors, priors_solution_2)
    test_breakdown_dict_2 = np.array([nb_clf2.posteriors[key] for key in keylist2])
    combined_sol_dict2 = []
    combined_sol_dict2.append(no_vals_dict2)
    combined_sol_dict2.append(yes_vals_dict2)
    for i in range(len(test_breakdown_dict_2)):
        curr_test_dict = test_breakdown_dict_2[i]
        curr_sol_dict = combined_sol_dict2[i]
        for key in curr_test_dict:
            test_value = curr_test_dict[key]
            sol_value = curr_sol_dict[key]
            assert np.isclose(test_value, sol_value)
    
    # Bramer 3.2 train dataset
    header_breamer = ["day", "season", "wind", "rain", "class"]
    X_train_bramer = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_bramer = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    keylist3 = np.unique(y_train_bramer)
    priors_solution_3 = [0.05, 0.1, 0.7, 0.15]
    
    on_time_vals_dict = {"att1_1": 2/14, "att1_2": 2/14, "att1_3": 1/14, "att1_4": 9/14, \
                            "att2_1": 2/14, "att2_2": 4/14, "att2_3": 6/14, "att2_4": 2/14, \
                            "att3_1": 4/14, "att3_2": 5/14, "att3_3": 5/14, \
                            "att4_1": 1/14, "att4_2": 5/14, "att4_3": 8/14}
    
    late_vals_dict = {      "att1_1": 0/2, "att1_2": 1/2, "att1_3": 0/2, "att1_4": 1/2, \
                            "att2_1": 0/2, "att2_2": 0/2, "att2_3": 0/2, "att2_4": 2/2, \
                            "att3_1": 1/2, "att3_2": 0/2, "att3_3": 1/2, \
                            "att4_1": 1/2, "att4_2": 1/2, "att4_3": 0/2}
    
    very_late_vals_dict = { "att1_1": 0/3, "att1_2": 0/3, "att1_3": 0/3, "att1_4": 3/3, \
                            "att2_1": 1/3, "att2_2": 0/3, "att2_3": 0/3, "att2_4": 2/3, \
                            "att3_1": 1/3, "att3_2": 0/3, "att3_3": 2/3, \
                            "att4_1": 2/3, "att4_2": 1/3, "att4_3": 0/3}
    
    cancelled_vals_dict = {"att1_1": 0/1, "att1_2": 1/1, "att1_3": 0/1, "att1_4": 0/1, \
                            "att2_1": 0/1, "att2_2": 1/1, "att2_3": 0/1, "att2_4": 0/1, \
                            "att3_1": 1/1, "att3_2": 0/1, "att3_3": 0/1, \
                            "att4_1": 1/1, "att4_2": 0/1, "att4_3": 0/1}
    
    post_solution_3 = {"cancelled": cancelled_vals_dict, "very late": very_late_vals_dict, \
                        "late": late_vals_dict, "on time": on_time_vals_dict}
    
    nb_clf3 = MyNaiveBayesClassifier()
    nb_clf3.fit(X_train_bramer, y_train_bramer)
    assert np.allclose(nb_clf3.priors, priors_solution_3)
    test_breakdown_dict_3 = np.array([nb_clf3.posteriors[key] for key in keylist3])
    combined_sol_dict3 = []
    combined_sol_dict3.append(cancelled_vals_dict)
    combined_sol_dict3.append(late_vals_dict)
    combined_sol_dict3.append(on_time_vals_dict)
    combined_sol_dict3.append(very_late_vals_dict)
    for i in range(len(test_breakdown_dict_3)):
        curr_test_dict = test_breakdown_dict_3[i]
        curr_sol_dict = combined_sol_dict3[i]
        for key in curr_test_dict:
            test_value = curr_test_dict[key]
            sol_value = curr_sol_dict[key]
            assert np.isclose(test_value, sol_value)
def test_naive_bayes_classifier_predict():
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test_inclass_example = [[1, 5]]
    inclass_prediction_solution = ["yes"]


    nb_clf1 = MyNaiveBayesClassifier()
    nb_clf1.fit(X_train_inclass_example, y_train_inclass_example)
    inclass_prediction_test = nb_clf1.predict(X_test_inclass_example)
    assert inclass_prediction_test == inclass_prediction_solution
    
    
    # RQ5 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    X_test_iphone_1 = [[2, 2, "fair"]]
    X_test_iphone_2 = [[1, 1, "excellent"]]
    
    iphone_1_prediction_solution = ["yes"]
    iphone_2_prediction_solution = ["no"]
    
    nb_clf2 = MyNaiveBayesClassifier()
    nb_clf2.fit(X_train_iphone, y_train_iphone)
    iphone_1_prediction_test = nb_clf2.predict(X_test_iphone_1)
    iphone_2_prediction_test = nb_clf2.predict(X_test_iphone_2)
    
    assert iphone_1_prediction_test == iphone_1_prediction_solution
    assert iphone_2_prediction_test == iphone_2_prediction_solution
    
    
    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain", "class"]
    X_train_bramer = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_bramer = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    X_test_bramer_1 = [["weekday", "winter", "high", "heavy"]]
    X_test_bramer_2 = [["weekday", "summer", "high", "heavy"]]
    X_test_bramer_3 = [["sunday", "summer", "normal", "slight"]]
    
    bramer_1_prediction_solution = ["very late"]
    bramer_2_prediction_solution = ["on time"]
    bramer_3_prediction_solution = ["on time"]
    
    nb_clf3 = MyNaiveBayesClassifier()
    nb_clf3.fit(X_train_bramer, y_train_bramer)
    bramer_1_prediction_test = nb_clf3.predict(X_test_bramer_1)
    bramer_2_prediction_test = nb_clf3.predict(X_test_bramer_2)
    bramer_3_prediction_test = nb_clf3.predict(X_test_bramer_3)
    
    assert bramer_1_prediction_test == bramer_1_prediction_solution
    assert bramer_2_prediction_test == bramer_2_prediction_solution
    assert bramer_3_prediction_test == bramer_3_prediction_solution


# TODO: copy your test_myclassifiers.py solution from PA4-6 here

def test_decision_tree_classifier_fit():
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_interview = \
            ["Attribute", "att0",
                ["Value", "Junior", 
                    ["Attribute", "att3",
                        ["Value", "no", 
                            ["Leaf", "True", 3, 5]
                        ],
                        ["Value", "yes", 
                            ["Leaf", "False", 2, 5]
                        ]
                    ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Senior",
                    ["Attribute", "att2",
                        ["Value", "no",
                            ["Leaf", "False", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "True", 2, 5]
                        ]
                    ]
                ]
            ]
    tree_clf_interview = MyDecisionTreeClassifier()
    tree_clf_interview.fit(X_train_interview, y_train_interview)
    
    assert tree_clf_interview.tree == tree_interview
    
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    
    tree_clf_iphone = MyDecisionTreeClassifier()
    tree_clf_iphone.fit(X_train_iphone, y_train_iphone)
    
    tree_iphone =  ['Attribute', 'att0', 
                    ['Value', 1, 
                        ['Attribute', 'att1', 
                            ['Value', 1, 
                                ['Leaf', 'yes', 1, 5]
                            ], 
                            ['Value', 2, 
                                ['Attribute', 'att2', 
                                    ['Value', 'excellent', 
                                        ['Leaf', 'yes', 1, 2]
                                    ], 
                                    ['Value', 'fair', 
                                        ['Leaf', 'no', 1, 2]
                                    ]
                                ]
                            ], 
                            ['Value', 3, 
                                ['Leaf', 'no', 2, 5]
                            ]
                        ]
                    ], 
                    ['Value', 2, 
                        ['Attribute', 'att2', 
                            ['Value', 'excellent', 
                                ['Leaf', 'no', 4, 10]
                            ], 
                            ['Value', 'fair', 
                                ['Leaf', 'yes', 6, 10]
                            ]
                            ]
                        ]
                    ]
    tree_clf_iphone = MyDecisionTreeClassifier()
    tree_clf_iphone.fit(X_train_iphone, y_train_iphone)
    
    assert tree_clf_iphone.tree == tree_iphone
    

def test_decision_tree_classifier_predict():
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    X_test_interview_1 = [["Junior", "Java", "yes", "no"]]
    X_test_interview_2= [["Junior", "Java", "yes", "yes"]]
    corr_prediction_interview_1 = ["True"]
    corr_prediction_interview_2 = ["False"]
    
    tree_clf_interview = MyDecisionTreeClassifier()
    tree_clf_interview.fit(X_train_interview, y_train_interview)
    test_prediction_interview_1 = tree_clf_interview.predict(X_test_interview_1)
    test_prediction_interview_2 = tree_clf_interview.predict(X_test_interview_2)
    
    assert test_prediction_interview_1 == corr_prediction_interview_1
    assert test_prediction_interview_2 == corr_prediction_interview_2
    
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    
    tree_clf_iphone = MyDecisionTreeClassifier()
    tree_clf_iphone.fit(X_train_iphone, y_train_iphone)
    
    X_test_iphone_1 = [[1, 2, "excellent"]] # test on the one extra split on the left side of tree
    corr_prediction_iphone_1 = ["yes"]
    
    X_test_iphone_2 = [[2, 3, "excellent"]] # a test on the big mess on right side of tree
    corr_prediction_iphone_2 = ["no"]
    
    test_prediction_iphone_1 = tree_clf_iphone.predict(X_test_iphone_1)
    test_prediction_iphone_2 = tree_clf_iphone.predict(X_test_iphone_2)
    
    assert test_prediction_iphone_1 == corr_prediction_iphone_1
    assert test_prediction_iphone_2 == corr_prediction_iphone_2