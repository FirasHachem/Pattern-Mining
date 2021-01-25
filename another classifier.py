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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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
        """prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
		 should be pruned.
	    :param gid_subsets: A list of the cover of the pattern for each subset.
		:return: true if the pattern should be pruned, false otherwise.
		"""
        print("Please implement the prune function in a subclass for a specific mining task!")


class task_classification(PatternGraphs):
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
        freq = len(gid_subsets[0]) + len(gid_subsets[2])
        conf = max((len(gid_subsets[0]) / freq), (len(gid_subsets[2]) / freq))

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
            # remove the patterns with the lowest score
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

        freq = len(gid_subsets[0]) + len(gid_subsets[2])

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

        keyz = self.patterns.keys()
        for kk in keyz:
            for kk2 in self.patterns[kk]:
                for i, gid_subset in enumerate(kk2[1]):
                    matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
        return [numpy.array(matrix).transpose() for matrix in matrices]

class task_rule(PatternGraphs):
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

# def example4(database_file_name_pos, database_file_name_neg, nfolds):
def example4():
    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    nfolds = int(args[3])  # Fourth parameter: number of folds to use in the k-fold cross-validation.

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

    # print(pos_ids)
    # print(neg_ids)

    minsup = (len(pos_ids) + len(neg_ids)) / 10
    k = 10

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
        train_and_evaluate_rule(minsup, graph_database, subsets)

    # Otherwise: performs k-fold cross-validation:
    else:
        pos_fold_size = len(pos_ids) // nfolds
        neg_fold_size = len(neg_ids) // nfolds
        for i in range(nfolds):
            # Use fold as test set, the others as training set for each class;
            # identify all the subsets to be maintained by the graph mining algorithm.
            subsets = [
                numpy.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),
                # Positive training set
                pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
                numpy.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),
                # Negative training set
                neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
            ]
            # Printing fold number:
            print('fold {}'.format(i + 1))
            train_and_evaluate_rule(minsup, graph_database, subsets, k)


def train_and_evaluate_classification(minsup, database, subsets, k):
    # task = FrequentPositiveGraphs(minsup, database, subsets)  # Creating task
    task = task_classification(k, minsup, database, subsets)

    gSpan(task).run()  # Running gSpan

    # Creating feature matrices for training and testing:
    features = task.get_feature_matrices()
    train_fm = numpy.concatenate((features[0], features[2]))  # Training feature matrix
    train_labels = numpy.concatenate(
        (numpy.full(len(features[0]), 1, dtype=int), numpy.full(len(features[2]), -1, dtype=int)))  # Training labels
    test_fm = numpy.concatenate((features[1], features[3]))  # Testing feature matrix
    test_labels = numpy.concatenate(
        (numpy.full(len(features[1]), 1, dtype=int), numpy.full(len(features[3]), -1, dtype=int)))  # Testing labels

    # classifier = naive_bayes.GaussianNB()  # Creating model object
    # classifier = DecisionTreeClassifier(random_state=1)
    classifier = SVC()
    # classifier = RandomForestClassifier()
    # classifier = KNeighborsClassifier()
    classifier.fit(train_fm, train_labels)
    # Training model

    predicted = classifier.predict(test_fm)  # Using model to predict labels of testing data

    accuracy = metrics.accuracy_score(test_labels, predicted)  # Computing accuracy:

    # for pattern, gid_subsets in task.patterns:
    #     freq = len(gid_subsets[0]) + len(gid_subsets[2])
    #     conf = max(len(gid_subsets[0]) / freq, len(gid_subsets[2]) / freq)
    #     print('{} {} {}'.format(pattern, conf, freq))

    # printing classification results:
    # print(predicted.tolist())
    print(k, 'accuracy: {}'.format(accuracy))
    # print()  # Blank line to indicate end of fold.

def train_and_evaluate_rule(minsup, database, subsets, k):

    sub = subsets.copy()

    test_labels = numpy.concatenate((numpy.full(len(subsets[1]), 1, dtype=int), numpy.full(len(subsets[3]), -1, dtype=int)))  # Testing labels
    test_idx = subsets[1] + subsets[3]

    predictions = numpy.zeros(len(test_labels))

    #run the gSpan algorhtm k times to find the 'k' best patterns and build 'k' rules using them
    for i in range(k):

        task = task_rule(1, minsup, database, sub)
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
    # example4("C:/Users/richard/AppData/Local/Programs/Python/Python36/Scripts/0_PROJECT/mining_pattern/project_3/data/molecules-small.pos",
    #     "C:/Users/richard/AppData/Local/Programs/Python/Python36/Scripts/0_PROJECT/mining_pattern/project_3/data/molecules-small.neg",
    #     4)

    example4()