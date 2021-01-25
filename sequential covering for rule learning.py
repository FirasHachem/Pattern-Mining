"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from gspan_mining import gSpan
from gspan_mining import GraphDatabase


class PatternGraphs:
    """
    This template class is used to define a task for the gSpan implementation.
    You should not modify this class but extend it to define new tasks
    """

    def __init__(self, database):
        # A list of subsets of graph identifiers.
        # Is used to specify different groups of graphs (classes and training/test sets).
        # The gid-subsets parameter in the pruning and store function will contain for each subset, all the occurrences
        # in which the examined pattern is present.
        self.gid_subsets = []

        self.database = database  # A graphdatabase instance: contains the data for the problem.

    def store(self, dfs_code, gid_subsets):
        """
        Code to be executed to store the pattern, if desired.
        The function will only be called for patterns that have not been pruned.
        In correlated pattern mining, we may prune based on confidence, but then check further conditions before storing.
        :param dfs_code: the dfs code of the pattern (as a string).
        :param gid_subsets: the cover (set of graph ids in which the pattern is present) for each subset in self.gid_subsets
        """
        print("Please implement the store function in a subclass for a specific mining task!")

    def prune(self, gid_subsets):
        """
        prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
        should be pruned.
        :param gid_subsets: A list of the cover of the pattern for each subset.
        :return: true if the pattern should be pruned, false otherwise.
        """
        print("Please implement the prune function in a subclass for a specific mining task!")

class task_three(PatternGraphs):
    """
    Finds the frequent (support >= minsup) subgraphs among the positive graphs.
    This class provides a method to build a feature matrix for each subset.
    """

    def __init__(self, k, minsup, database, subsets):
        """
        Initialize the task.
        :param minsup: the minimum positive support
        :param database: the graph database
        :param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
        """
        super().__init__(database)
        self.patterns = ("zz", [])  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
        self.k = k
        self.minsup = minsup
        self.gid_subsets = subsets
        self.thresh_conf = 0
        self.thresh_freq = 0

    # Stores any pattern found that has not been pruned
    def store(self, dfs_code, gid_subsets):

        #compute total support and the confidence
        freq = len(gid_subsets[0]) + len(gid_subsets[2])
        conf_pos = len(gid_subsets[0]) / freq
        conf_neg = len(gid_subsets[2]) / freq

        conf = max(conf_neg, conf_pos)

        #replace the best patterns by the current pattern if it is better
        if freq >= self.minsup:
            if conf > self.thresh_conf:
                self.patterns = (dfs_code, gid_subsets)
                self.thresh_conf = conf
                self.thresh_freq = freq

            elif conf == self.thresh_conf:
                if freq > self.thresh_freq:
                    self.patterns = (dfs_code, gid_subsets)
                    self.thresh_conf = conf
                    self.thresh_freq = freq
                elif freq == self.thresh_freq:
                    if dfs_code < self.patterns[0]:
                        self.patterns = (dfs_code, gid_subsets)
                        self.thresh_conf = conf
                        self.thresh_freq = freq

    # Prunes any pattern that is not frequent in the positive class
    def prune(self, gid_subsets):

        freq = len(gid_subsets[0]) + len(gid_subsets[2])

        if freq < self.minsup:
            return True


def example3():
    """
    Runs gSpan with the specified positive and negative graphs; finds all frequent subgraphs in the training subset of
    the positive class with a minimum support of minsup.
    Uses the patterns found to train a naive bayesian classifier using Scikit-learn and evaluates its performances on
    the test set.
    Performs a k-fold cross-validation.
    """

    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    k = int(args[3])  # Third parameter: k
    minsup = int(args[4])  # Fourth parameter: minimum support
    nfolds = int(args[5])  # Fifth parameter: number of folds to use in the k-fold cross-validation.

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(
        database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(
        database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

    # If less than two folds: using the same set as training and test set (note this is not an accurate way to evaluate the performances!)
    if nfolds < 2:
        subsets = [
            pos_ids,  # Positive training set
            pos_ids,  # Positive test set
            neg_ids,  # Negative training set
            neg_ids  # Negative test set
        ]
        # Printing fold number:
        print('fold {}'.format(1))
        train_and_evaluate2(minsup, graph_database, subsets)

    # Otherwise: performs k-fold cross-validation:
    else:
        pos_fold_size = len(pos_ids) // nfolds
        neg_fold_size = len(neg_ids) // nfolds
        for i in range(nfolds):
            # Use fold as test set, the others as training set for each class;
            # identify all the subsets to be maintained by the graph mining algorithm.
            subsets = [
                numpy.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),  # Positive training set
                pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
                numpy.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),  # Negative training set
                neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
            ]
            # Printing fold number:
            print('fold {}'.format(i+1))
            # print(subsets)
            train_and_evaluate2(minsup, graph_database, subsets, k)


def train_and_evaluate2(minsup, database, subsets, k):

    sub = subsets.copy()

    test_labels = numpy.concatenate((numpy.full(len(subsets[1]), 1, dtype=int), numpy.full(len(subsets[3]), -1, dtype=int)))  # Testing labels
    test_idx = subsets[1] + subsets[3]

    predictions = numpy.zeros(len(test_labels))

    #run the gSpan algorhtm k times to find the 'k' best patterns and build 'k' rules using them
    for i in range(k):

        task = task_three(1, minsup, database, sub)
        gSpan(task).run()  # Running gSpan

        best_pattern = task.patterns

        #classify the test patterns if the rule can be applied to them
        if len(best_pattern[1]):
            freq = len(best_pattern[1][0]) + len(best_pattern[1][2])
            conf = max(len(best_pattern[1][0]) / freq, len(best_pattern[1][2]) / freq)
            print('{} {} {}'.format(best_pattern[0], conf, freq))

            # DETERMINE THE CLASS TO USE
            if len(best_pattern[1][0]) > len(best_pattern[1][2]):
                predi = 1
            else:
                predi = -1
            # CLASSIFY POSITIVE
            for p in best_pattern[1][1]:
                for idx_p, pp in enumerate(test_idx):
                    if p == pp:
                        predictions[idx_p] = predi
            # CLASSIFY NEGATIVE
            for n in best_pattern[1][3]:
                for idx_n, nn in enumerate(test_idx):
                    if n == nn:
                        predictions[idx_n] = predi
            # REMOVE THE CLASSIFIED GRAPHS
            if len(best_pattern[1]):
                for j in range(0, 4):
                    if len(sub[j]) and len(best_pattern[1][j]):
                        sub[j] = list(set(sub[j]) - set(best_pattern[1][j]))

    #find the default class
    if len(sub[2]) > len(sub[0]):
        default_class = -1
    else:
        default_class = 1

    #assign the default class to the non-classified graphs
    predictions[predictions == 0] = default_class
    #compute accuracy
    accuracy = metrics.accuracy_score(test_labels, predictions)  # Computing accuracy:

    # printing classification results:
    predictions = predictions.astype(int)
    print(predictions.tolist())
    print('accuracy: {}'.format(accuracy))
    print()  # Blank line to indicate end of fold.


if __name__ == '__main__':
    # example1("C:/Users/richard/AppData/Local/Programs/Python/Python36/Scripts/mining_pattern/project_3/data/molecules-small.pos",
    # "C:/Users/richard/AppData/Local/Programs/Python/Python36/Scripts/mining_pattern/project_3/data/molecules-small.neg", 5, 5)
    # example2("C:/Users/richard/AppData/Local/Programs/Python/Python36/Scripts/mining_pattern/project_3/data/molecules-small.pos",
    # "C:/Users/richard/AppData/Local/Programs/Python/Python36/Scripts/mining_pattern/project_3/data/molecules-small.neg", 5, 5, 4)
    # 	# example1_test()
    # 	example2_test()
    example3()