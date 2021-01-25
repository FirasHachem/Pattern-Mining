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


class task_one(PatternGraphs):
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
        self.patterns = {}  # contains the patterns in the top k using the confidence as a key
        self.dico_thresh = {} #same dictionary as 'patterns' axcept that it stores frequencies of the patterns and not the patterns itself
        self.k = k #k value
        self.minsup = minsup #minimum total support (+ and -)
        self.gid_subsets = subsets
        self.thresh = 0 #lower bound of the confidence of the current top k sets
        self.curr_score = [] # a list containing only the current confidence score of the patterns

    # Stores any pattern found that has not been pruned
    def store(self, dfs_code, gid_subsets):

        freq = len(gid_subsets[0]) + len(gid_subsets[1])
        conf = len(gid_subsets[0]) / freq

        # store the patterns if it is a possible candidate
        if conf >= self.thresh and freq >= self.minsup:
            if not (conf in self.dico_thresh):
                self.patterns[conf] = [(dfs_code, gid_subsets)]
                self.dico_thresh[conf] = [freq]
                self.curr_score.append(conf)
            else:
                if not(freq in self.dico_thresh[conf]):
                    self.curr_score.append(conf)
                self.patterns[conf].append((dfs_code, gid_subsets))
                self.dico_thresh[conf].append(freq)

            #remove the patterns with the lowest score
            if len(self.curr_score) > self.k:
                # determine the confidence and freq (total support) of the worst patterns
                mini = min(self.dico_thresh[min(self.dico_thresh)])
                idx_minis = []
                for idx, zz in enumerate(self.dico_thresh[min(self.dico_thresh)]):
                    if zz == mini:
                        idx_minis.append(idx)
                idx_minis = sorted(idx_minis, reverse=True)

                #remove the worst pattern(s)
                for yy in idx_minis:
                    self.dico_thresh[min(self.dico_thresh)].pop(yy)
                    self.patterns[min(self.patterns)].pop(yy)

                if not self.patterns[min(self.patterns)]:
                    del self.patterns[min(self.patterns)]

                if not self.dico_thresh[min(self.dico_thresh)]:
                    del self.dico_thresh[min(self.dico_thresh)]

                self.curr_score.remove(min(self.curr_score))

                self.thresh = min(self.dico_thresh)

    # Prunes any pattern that has a total support smaller than the minsup
    def prune(self, gid_subsets):

        freq = len(gid_subsets[0]) + len(gid_subsets[1])

        if freq < self.minsup:
            return True



    # creates a column for a feature matrix
    def create_fm_col(self, all_gids, subset_gids):
        subset_gids = set(subset_gids)
        bools = []
        for i, val in enumerate(all_gids):
            if val in subset_gids:
                bools.append(1)
            else:
                bools.append(0)
        return bools

    # return a feature matrix for each subset of examples, in which the columns correspond to patterns
    # and the rows to examples in the subset.
    def get_feature_matrices(self):
        matrices = [[] for _ in self.gid_subsets]
        for pattern, gid_subsets in self.patterns:
            for i, gid_subset in enumerate(gid_subsets):
                matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
        return [numpy.array(matrix).transpose() for matrix in matrices]


# def example1(database_file_name_pos, database_file_name_neg, k, minsup):
def example1():

    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    k = int(args[3])  # Third parameter: k
    minsup = int(args[4])  # Fourth parameter: minimum support
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

    subsets = [pos_ids, neg_ids]  # The ids for the positive and negative labelled graphs in the database
    # print("subsets : ", subsets)
    # print("graph database :", graph_database)
    task = task_one(k, minsup, graph_database, subsets)  # Creating task


    gSpan(task).run()  # Running gSpan

    #print the top k patterns
    keyz = task.patterns.keys()
    keyz = sorted(keyz, reverse=True)

    for kk in keyz:
        for kk2 in task.patterns[kk]:
            freq = len(kk2[1][0]) + len(kk2[1][1])
            print('{} {} {}'.format(kk2[0], kk, freq))


if __name__ == '__main__':
    # example1(
    #     "C:/Users/richard/AppData/Local/Programs/Python/Python36/Scripts/0_PROJECT/mining_pattern/project_3/data/molecules-medium.pos",
    #     "C:/Users/richard/AppData/Local/Programs/Python/Python36/Scripts/0_PROJECT/mining_pattern/project_3/data/molecules-medium.neg",
    #     5, 20)

    example1()