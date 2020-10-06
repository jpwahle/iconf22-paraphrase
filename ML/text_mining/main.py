import time
from datetime import timedelta

import os
import sys

# python module absolute path

pydir_name = os.path.dirname(os.path.abspath(__file__))
ppydir_name = os.path.dirname(pydir_name)
# python path definition
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from text_mining.CommandLine import CLClassification
from text_mining.Evaluation import Classification
from text_mining.FileUtil import FileUtil
from text_mining.GridSearch import GridSearch
from text_mining.BagOfWords import BagOfWords


# TODO: improve command line(fixed)

class Main:
    def main(self):
        start_time = time.monotonic()

        command_line = CLClassification()

        # create bag of words from raw text files
        if command_line.isBow:
            x_test, x_train, y_test, y_train = self.build_bag_of_words(command_line.train, command_line.test)

        # otherwise read dataset from a comma separated file
        else:
            x_test, x_train, y_test, y_train = self.read_comma_separated_file(command_line.train, command_line.test)

        classifier_selected = command_line.classifier

        # check if it was selected grid search
        if command_line.isGridSearch:
            searcher = GridSearch(classifier_selected, command_line.cpu)
            searcher.execute(x_train, y_train)
            estimator = searcher.get_estimator()
            classification = Classification(command_line.cpu, estimator=estimator)

        # otherwise just apply classification
        else:
            classification = Classification(command_line.cpu, classifier_selected)

        if command_line.test is not None:
            classification.evaluate(x_train, y_train, x_test, y_test)
        else:
            classification.cross_validate(x_train, y_train)

        print((timedelta(seconds=time.monotonic() - start_time)))

    def read_comma_separated_file(self, train, test):
        file_util = FileUtil()
        # read the whole train and separate features from labels
        train_folder_name = os.path.join(ppydir_name, train)
        train = file_util.read_table(train_folder_name)
        x_train = file_util.get_features(train)
        y_train = file_util.get_labels(train)
        x_test = None
        y_test = None
        # read the whole test and separate features from labels
        if test is not None:
            test_folder_name = os.path.join(ppydir_name, test)
            test = file_util.read_table(test_folder_name)
            x_test = file_util.get_features(test)
            y_test = file_util.get_labels(test)
        return x_test, x_train, y_test, y_train

    def build_bag_of_words(self, train, test):
        train_folder_name = os.path.join(ppydir_name, train)
        bow = BagOfWords()
        x_test = None
        y_test = None
        if test is not None:
            test_folder_name = os.path.join(ppydir_name, test)
            bow.build(train_folder_name, test_folder_name)
            x_train = bow.get_train_features()
            y_train = bow.get_train_labels()
            x_test = bow.get_test_features()
            y_test = bow.get_test_labels()
        else:
            bow.build(train_folder_name)
            x_train = bow.get_train_features()
            y_train = bow.get_train_labels()
        return x_test, x_train, y_test, y_train


if __name__ == "__main__":
    Main().main()
