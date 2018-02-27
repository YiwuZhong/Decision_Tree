# ######### Guideline #########
# 1. Candidate splits for nominal features should have one branch per value of the nominal feature.
#    The branches should be ordered according to the order of the feature values listed in the ARFF file.
#    (construct_dt)
# 2. Candidate splits for numeric features should use thresholds that are midpoints between values
#    in the given set of instances.The left branch of such a split should represent values that are less than or
#    equal to the threshold.
#    (construct_dt)(info_gain)
# 3. Splits should be chosen using information gain. If there is a tie between two features in their information gain,
#    you should break the tie in favor of the feature listed first in the header section of the ARFF file.
#    (info_gain)(find_best_split)
#    If there is a tie between two different thresholds for a numeric feature,
#    you should break the tie in favor of the smaller threshold.
#    (info_gain)
# 4. The stopping criteria (for making a node into a leaf) are that
#    (construct_dt)
#    a. all of the training instances reaching the node belong to the same class, or
#    b. there are fewer than m training instances reaching the node, where m is provided as input to the program, or
#    c. no feature has positive information gain, or
#    d. there are no more remaining candidate splits at the node.
# 5. If the classes of the training instances reaching a leaf are equally represented,
#    the leaf should predict the most common class of instances reaching the parent node.
#    (classify)
# 6. If the number of training instances that reach a leaf node is 0,
#    the leaf should predict the the most common class of instances reaching the parent node.
#    (classify)

import re as pattern
import sys
import math
# import random
# import numpy as np
# import matplotlib.pyplot as plt


# read in training instances
credit_train = open(sys.argv[1])

# read in training instances
credit_test = open(sys.argv[2])

# threshold m : the leaf contains less than m instances, stop split tree
m = int(sys.argv[3])

# name list of different features : ['A1', 'A2', 'A3', 'A4', 'A5', 'A8', 'A14', 'A15']
Feature_Name = []

# type list for different features : ['nominal', 'real', 'real', 'nominal', 'nominal', 'real', 'real', 'real']
Feature_Type = []

# value lists for different features : len(value)=len(type), each inner list includes all value for one feature
# [ [..], [..], [..], [..]', [..], [..], [..], [..] ]
Feature_Value = []

# value list for different labels : ['+', '-']
Class_Value = []

# list of Credit objects
Train_Set = []

# list of Credit objects in test set
Test_Set = []

# list of prediction ( y_hat )
Predict_Result = []

# count for correct prediction in test set
num_correct = 0

# ROC curve list : [ [positive_confidence, actual_class], [...] , ... , [...]  ]
ROC_List = []


# each instance(line) is a Credit object, consisting training instances and testing instances
class Credit:
    def __init__(self, feature_value, class_value):
        for i in range(0, len(feature_value)):
            self.attribute = feature_value
            self.Class = class_value


# DT tree's node
class TreeNode:
    def __init__(self, split_feature, threshold, Subtree, father_tree, current_train_set):
        self.split_feature = split_feature
        self.threshold = threshold
        self.Subtree = Subtree
        self.father_tree = father_tree
        self.current_train_set = current_train_set


# the decision tree
class DT:
    def __init__(self):
        self.root = None

    def make_new_tree(self, split_feature, threshold, subtree, father_tree, current_train_set):
        self.root = TreeNode(split_feature, threshold, subtree, father_tree, current_train_set)


# Constructing the DT tree:
# generate and spread the tree first (from top to bottom), and then initiate/assign all nodes (from bottom up to root)
# consideration order: whether can make a new subtree node; whether best_info_gain>0; numerical or nominal feature
def construct_dt(train_set, dt, father_tree):
    if create_new_node(train_set):
        [best_info_gain, split_feature, threshold] = find_best_split(train_set)
        if best_info_gain > 0:
            # numerical feature
            if threshold is not None:
                # for the numerical type, candidate splits to left and right using threshold.
                left = DT()
                right = DT()
                left.make_new_tree(None, None, [], None, None)   # add a branch which is empty
                right.make_new_tree(None, None, [], None, None)   # add a branch which is empty
                dt.root.Subtree.append(left)
                dt.root.Subtree.append(right)
                new_subtree = [p for p in train_set if p.attribute[split_feature] <= threshold]
                construct_dt(new_subtree, dt.root.Subtree[0], dt)
                new_subtree = [p for p in train_set if p.attribute[split_feature] > threshold]
                construct_dt(new_subtree, dt.root.Subtree[1], dt)
            # nominal feature
            else:
                # for the nominal type, candidate splits should have one branch per value of the nominal feature.
                for i in range(0, len(Feature_Value[split_feature])):
                    one_sub = DT()
                    one_sub.make_new_tree(None, None, [], None, None)  # add a branch which is empty
                    dt.root.Subtree.append(one_sub)
                    # to construct_dt recursively: select the instances which have the same value in current feature,
                    # and use them as train_set, current dt become 'father_tree'. current dt's subtree[i] become 'dt'.
                    construct_dt([p for p in train_set if p.attribute[split_feature] == Feature_Value[split_feature][i]],
                                 dt.root.Subtree[i], dt)
            # 'Subtree' list has been filled by the code "if-else".
            # 'split_feature' and 'threshold' have been calculated by 'find_best_split'
            # It's time for all nodes except leaf to be initiated
            dt.make_new_tree(split_feature, threshold, dt.root.Subtree, father_tree, train_set)
        else:
            # 4.c. no feature has positive information gain (once a feature is used, the info_gain of it must decrease)
            # also, it's base case (leaf node) to be initiated
            dt.make_new_tree(None, None, None, father_tree, train_set)
    else:
        # 4.a. all of the training instances reaching the node belong to the same class
        # 4.b. there are fewer than m training instances reaching the node, where m is provided as input to the program
        # 4.d. there are no more remaining candidate splits at the node
        # also, it's base case (leaf node) to be initiated
        dt.make_new_tree(None, None, None, father_tree, train_set)


# Test whether the node can be made. If not, stop split decision tree
def create_new_node(train_set):
    global m
    # pos_list = [ex for ex in train_set if ex.Class != train_set[0].Class]
    # neg_list = [ex for ex in train_set if ex.Class != train_set[1].Class]
    # 4.b. there are fewer than m training instances reaching the node, where m is provided as input to the program
    # 4.d. there are no more remaining candidate splits at the node
    if len(train_set) < m:
        return False
    # train_set[0].Class = +
    # train_set[1].Class = -
    # elif len(pos_list) == 0 or len(neg_list) == 0:
        # return False
    # 4.a. all of the training instances reaching the node belong to the same class
    elif [ex for ex in train_set if ex.Class != train_set[0].Class] == []:
        return False
    elif [ex for ex in train_set if ex.Class != train_set[1].Class] == []:
        return False
    else:
        return True


# traversal all features to find the best split feature which has max information gain,
def find_best_split(train_set):
    max_gain = 0
    split_threshold = None
    split_feature = None
    # Splits should be chosen using information gain. If there is a tie between two features in their information gain,
    # you should break the tie in favor of the feature listed first in the header section of the ARFF file.
    # ( this requirement is automatically met, because 'i' is traversed from the first one value to last one. )
    for i in range(0, len(Feature_Name)):
        [temp_gain, temp_threshold] = info_gain(i, train_set)
        if temp_gain > max_gain:
            max_gain = temp_gain
            split_feature = i
            split_threshold = temp_threshold
    return [max_gain, split_feature, split_threshold]


# Return max_gain and threshold. Calculate the information gain including numerical and nominal features.
def info_gain(index, train_set):
    max_gain = 0
    threshold = None

    # split_feature is numerical type
    if Feature_Type[index] == "real":
        value_list = [ex.attribute[index] for ex in train_set]  # don't use set to save the unique element
        number_list = value_list
        number_list.sort()  # sort all values of the Feature_Name(index) within current node
        # for current feature, traversal all possible threshold, and find best one
        # If there is a tie between two different thresholds for a numeric feature,
        # you should break the tie in favor of the smaller threshold
        # ( this requirement is automatically met, because 'i' is traversed from the first one value to last one. )
        for i in range(0, len(number_list) - 1):
            # Candidate splits for numeric features should use thresholds that are midpoints between values
            # in the given set of instances.
            # The left branch of such a split should represent values that are less than or equal to the threshold
            temp_threshold = (float(number_list[i]) + float(number_list[i + 1])) / 2
            left_branch = [p for p in train_set if p.attribute[index] <= temp_threshold]
            right_branch = [p for p in train_set if p.attribute[index] > temp_threshold]
            left_ratio = float(len(left_branch)) / (len(left_branch) + len(right_branch))
            right_ratio = float(len(right_branch)) / (len(left_branch) + len(right_branch))
            temp_gain = calculate_entropy(train_set) - left_ratio * calculate_entropy(left_branch) - right_ratio * calculate_entropy(right_branch)
            if temp_gain > max_gain:
                max_gain = temp_gain
                threshold = temp_threshold

    # split_feature is nominal type
    else:
        value_list = list(set([ex.attribute[index] for ex in train_set]))  # use set to save the unique element
        samples = []
        entropy = 0
        total = 0
        ratio = []
        # for the nominal type, candidate splits should have one branch per value of the nominal feature.
        # The branches should be ordered according to the order of the feature values listed in the ARFF file
        # sort and save the samples which have same value in inner list, and all value is outer list
        for i in range(0, len(value_list)):
            temp = [p for p in train_set if p.attribute[index] == value_list[i]]
            samples.append(temp)  # gather the samples which has same feature value
            total = total + len(temp)
        # calculate the entropy after splitting ( per value corresponds one branch )
        for i in range(0, len(value_list)):
            ratio.append(float(len(samples[i])) / total)
            entropy = entropy + ratio[i] * calculate_entropy(samples[i])
        # get information gain in current splitting using current feature
        temp_gain = calculate_entropy(train_set) - entropy
        # because the default value for max_gain is None, if the temp_gain is negative or 0, then return None.
        # which means stopping splitting if use current nominal feature
        if temp_gain > 0:
            max_gain = temp_gain
    return [max_gain, threshold]


# calculate and return the entropy of given data
def calculate_entropy(data):
    entropy = 0
    total = float(len(data))
    pos = len([p for p in data if p.Class == Class_Value[0]])
    neg = len([p for p in data if p.Class == Class_Value[1]])
    if pos == 0 or neg == 0:
        return 0.0
    else:
        entropy = entropy - (pos / total) * math.log(pos / total, 2)
        entropy = entropy - (neg / total) * math.log(neg / total, 2)
        return entropy


# Predict on Test_Set
def test(test_set):
    global num_correct
    global my_DT
    # predict on each instance
    for i in range(0, len(test_set)):
        dt = my_DT
        # depth-first-search down to the leaf
        while (dt.root.Subtree is not None):
            # split on numerical feature
            if Feature_Type[dt.root.split_feature] == "real":
                if test_set[i].attribute[dt.root.split_feature] <= dt.root.threshold:
                    dt = dt.root.Subtree[0]
                else:
                    dt = dt.root.Subtree[1]
            # split on nominal feature
            else:
                index = Feature_Value[dt.root.split_feature].index(test_set[i].attribute[dt.root.split_feature])
                dt = dt.root.Subtree[index]
        # once reach leaf, classify according to the instances number of training data within current leaf
        Predict_Result.append(classify(dt))
        if Predict_Result[i] == test_set[i].Class:
            num_correct = num_correct + 1
        # calculate the positive confidence according to the instances number of training data within current leaf
        pos_cfd = calculate_positive_confidence(dt)
        one_instance = [pos_cfd, test_set[i].Class]
        ROC_List.append(one_instance)


def calculate_positive_confidence(dt):
    [pos, neg] = count_pos_neg(dt.root.current_train_set)
    #  Laplace estimates
    positive_confidence = (pos+1) / (pos+neg+1+1)
    return positive_confidence


# the test instance reaching a leaf is classified by the training data within this leaf
def classify(dt):
    current_train_set = dt.root.current_train_set
    [pos, neg] = count_pos_neg(current_train_set)
    if neg < pos:
        return Class_Value[0]
    elif neg > pos:
        return Class_Value[1]
    else:
        # 1. If the classes of the training instances reaching a leaf are equally represented,
        #    the leaf should predict the most common class of instances reaching the parent node.
        # 2. If the number of training instances that reach a leaf node is 0,
        #    the leaf should predict the the most common class of instances reaching the parent node
        return classify(dt.root.father_tree)


# count the number of positive and negative class within current node
def count_pos_neg(current_train_set):
        pos = 0
        neg = 0
        for i in range(0, len(current_train_set)):
            if current_train_set[i].Class == Class_Value[1]:
                neg = neg + 1
            else:
                pos = pos + 1
        return [pos, neg]


#  Display part
def display_dtree(dt, level):
    tab = []
    nu = ''
    for i in range(0, level):
        tab.append('|\t')
    if Feature_Type[dt.root.split_feature] == 'real':
        if dt.root.Subtree[0].root.Subtree is None:
            [pos, neg] = count_pos_neg(dt.root.Subtree[0].root.current_train_set)
            print("%s%s <= %f [%d %d]: %s" % (nu.join(tab), Feature_Name[dt.root.split_feature], dt.root.threshold,
                                              pos, neg, classify(dt.root.Subtree[0])))
        else:
            [pos, neg] = count_pos_neg(dt.root.Subtree[0].root.current_train_set)
            print("%s%s <= %f [%d %d]" % (nu.join(tab), Feature_Name[dt.root.split_feature], dt.root.threshold,
                                          pos, neg))
            display_dtree(dt.root.Subtree[0], level + 1)
        if dt.root.Subtree[1].root.split_feature is None:
            [pos, neg] = count_pos_neg(dt.root.Subtree[1].root.current_train_set)
            print("%s%s > %f [%d %d]: %s" % (nu.join(tab), Feature_Name[dt.root.split_feature], dt.root.threshold,
                                             pos, neg, classify(dt.root.Subtree[1])))
        else:
            [pos, neg] = count_pos_neg(dt.root.Subtree[1].root.current_train_set)
            print("%s%s > %f [%d %d]" % (nu.join(tab), Feature_Name[dt.root.split_feature], dt.root.threshold,
                                         pos, neg))
            display_dtree(dt.root.Subtree[1], level + 1)
    else:
        for i in range(0, len(dt.root.Subtree)):
            if dt.root.Subtree[i].root.split_feature is None:
                [pos, neg] = count_pos_neg(dt.root.Subtree[i].root.current_train_set)
                print("%s%s = %s [%d %d]: %s" % (nu.join(tab), Feature_Name[dt.root.split_feature],
                                                 Feature_Value[dt.root.split_feature][i],
                                                 pos, neg, classify(dt.root.Subtree[i])))
            else:
                [pos, neg] = count_pos_neg(dt.root.Subtree[i].root.current_train_set)
                print("%s%s = %s [%d %d]" % (nu.join(tab), Feature_Name[dt.root.split_feature],
                                             Feature_Value[dt.root.split_feature][i],
                                             pos, neg))
                display_dtree(dt.root.Subtree[i], level + 1)


# Parse the data from training data and test data ('arff' file)
# variable 'List' represents the actual content of each line
for line in credit_train:
    # the row of 'relation' and 'data'
    if (pattern.findall('@relation', line) != []) | (pattern.findall('@data', line) != []):
        pass
    # the rows of 'attribute'
    elif pattern.findall('@attribute', line) != []:
        List = line.split(None, 2)   # split twice on one or more whitespace
        List[1] = List[1].strip('\'')  # get feature name
        List[2] = List[2].replace('{', '')
        List[2] = List[2].replace('}', '')
        List[2] = List[2].replace(' ', '')
        List[2] = List[2].replace('\n', '')  # for safety no matter of Windows or Linux
        List[2] = List[2].replace('\r', '')  # for safety no matter of Windows or Linux
        List[2] = List[2].split(',')  # get items name of certain feature
        if List[1] != 'class':
            Feature_Name.append(List[1])
            if len(List[2]) == 1:  # only one value means it is numerical feature
                Feature_Type.append('real')  # numerical feature
                Feature_Value.append([])
            else:
                Feature_Type.append('nominal')  # nominal feature
                Feature_Value.append(List[2])
        else:
            # po = List[2][0]
            # List[2][0] = List[2][1]
            # List[2][1] = po
            Class_Value.extend(List[2])  # the last feature, namely [0]+  [1]-
    # the rows of training data
    else:
        line = line.strip('\n')
        line = line.strip('\r')
        line = line.replace(' ', '')
        List = line.split(',')
        for i in range(0, len(List) - 1):
            if Feature_Type[i] == 'real':
                List[i] = float(List[i])  # convert string into float
        # for i in range(0, len(List) - 1):
            # elif List[i] not in Feature_Value[i]:
                # Feature_Value[i].append(List[i])
        # create a Credit object which includes features and class,and add it into Train_Set list, for training
        Train_Set.append(Credit(List[0:len(Feature_Name)], List[len(Feature_Name)]))

for line in credit_test:
    if pattern.findall('@', line) != []:
        pass
    else:
        line = line.strip('\n')
        line = line.strip('\r')
        line = line.replace(' ', '')
        List = line.split(",")
        for i in range(0, len(List) - 1):
            if Feature_Type[i] == 'real':
                List[i] = float(List[i])    # convert string into float
        # create a Credit object which includes features and class,and add it into Test_Set list, for testing
        Test_Set.append(Credit(List[0:len(Feature_Name)], List[len(Feature_Name)]))


# Part 1 : construct decision tree, learning with all Train_Set
my_DT = DT()
my_DT.make_new_tree(None, None, [], None, None)  # initiate subtree using a empty list
construct_dt(Train_Set, my_DT, None)
test(Test_Set)
display_dtree(my_DT, 0)
print("<Predictions for the Test Set Instances>")
for i in range(0, len(Predict_Result)):
    print("%d: Actual: %s Predicted: %s" % (i + 1, Test_Set[i].Class, Predict_Result[i]))
print("Number of correctly classified: %d Total number of test instances: %d" % (num_correct, len(Test_Set)))



''' 
# Part 2.1 : plot learning curve, learning with part of Train_Set
count = [15, 25, 50, 125, len(Train_Set)]
min_acc = []
max_acc = []
mean_acc = []
for c in count:
    # index = np.random.randint(c, size=(c, 1))
    list_of_accuracy = []
    acc = []
    for i in range(1, 11):
        num_correct = 0
        Predict_Result = []  # attention: this is a global variable, which affects num_correct in 'test' function
        little_train = random.sample(Train_Set, c)
        my_DT = DT()
        my_DT.make_new_tree(None, None, [], None, None)  # initiate subtree using a empty list
        construct_dt(little_train, my_DT, None)
        test(Test_Set)
        accuracy = num_correct / len(Test_Set)
        list_of_accuracy.append(accuracy)
    max_acc.extend([max(list_of_accuracy)])
    min_acc.extend([min(list_of_accuracy)])
    mean_acc.extend([sum(list_of_accuracy) / len(list_of_accuracy)])

# scatter point, plot line, label, title, xlabel, ylabel
percentage = [5, 10, 20, 50, 100]
plt.scatter(percentage, min_acc, 8, 'r')  # '2' control the size of point
plt.scatter(percentage, mean_acc, 8, 'b')
plt.scatter(percentage, max_acc, 8, 'g')
plt.plot(percentage, min_acc, 'r', label='Min')
plt.plot(percentage, mean_acc, 'b', label='Mean')
plt.plot(percentage, max_acc, 'g', label='Max')
plt.ylim(min(min_acc) * 0.95, max(max_acc) * 1.05)  # control the range of axis
plt.xlim(0, 105)
plt.title('Learning Curve')
plt.xlabel('percentage of all training instances')
plt.ylabel('test accuracy')
plt.legend()  # show the 'label' of line
plt.show()
'''

'''
# Part 2.2 : plot ROC curve, learning with all Train_Set
my_DT = DT()
my_DT.make_new_tree(None, None, [], None, None)  # initiate subtree using a empty list
construct_dt(Train_Set, my_DT, None)
test(Test_Set)
display_dtree(my_DT, 0)
# Important : sorted( iterable object, key=lambda:..)
sorted_ROC_List = sorted(ROC_List, key=lambda instance: instance[0], reverse=True)
pos_label = 0
neg_label = 0
total_pos = 0
total_neg = 0
True_Positive = [0]    # create the ROC point at origin
False_Positive = [0]   # create the ROC point at origin
dummy = True  # track the status before this cut-off
for i in range(len(sorted_ROC_List)-1):
    if sorted_ROC_List[i][1] == '+':
        total_pos = total_pos+1
    else:
        total_neg = total_neg+1
# Traverse the sorted ROC list and find cut-off points from high confidence down to low confidence
for i in range(len(sorted_ROC_List)-1):
    # once reach a '+'or '-' after consecutive group of '-' or '+',
    # stop and calculate True Positive and False Positive, and then continue
    if sorted_ROC_List[i][1] == '-':
        if dummy is True:
            neg_label = neg_label + 1
            True_Positive.append(pos_label / total_pos)  # attention: use actual label instead predicted label
            False_Positive.append(neg_label / total_neg)  # attention: use actual label instead predicted label
            dummy = False
        else:
            neg_label = neg_label + 1
    else:
        if dummy is not True:
            pos_label = pos_label + 1
            True_Positive.append(pos_label / total_pos)  # attention: use actual label instead predicted label
            False_Positive.append(neg_label / total_neg)  # attention: use actual label instead predicted label
            dummy = True
        else:
            pos_label = pos_label + 1
True_Positive.append(1)     # create the ROC point at (1,1)
False_Positive.append(1)     # create the ROC point at (1,1)
for i in range(len(sorted_ROC_List)-1):
    print(sorted_ROC_List[i][1])
plt.scatter(False_Positive, True_Positive, 10, 'g')
plt.plot(False_Positive, True_Positive, 'g')
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
plt.title('ROC Curve')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.show()
'''


