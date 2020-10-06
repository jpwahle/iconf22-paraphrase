import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from text_mining.CommandLine import CLClassification

class GridSearch:
    selection = None
    classifier = None
    parameters = []
    cpu = 1
    estimator = None

    def __init__(self, classifier, cpu):
        self.selection = classifier
        self.cpu = cpu
        if self.selection == 'knn':
            self.classifier = KNeighborsClassifier()
            self.parameters = self.get_knn_parameters()

        elif self.selection == 'forest':
            self.classifier = RandomForestClassifier()
            self.parameters = self.get_forest_parameters()

        elif self.selection == 'svm':
            self.classifier = SVC()
            self.parameters = self.get_svm_parameters()

        elif self.selection == 'logistic':
            self.classifier = LogisticRegression()
            self.parameters = self.get_logistic_parameters()

        elif self.selection == 'naive':
            self.classifier = GaussianNB()
            self.parameters = self.get_naive_bayes_parameters()
        else:
            raise Exception("\n\nUnknown classifier to do grid search: " + self.selection + "\n\n")

    def get_naive_bayes_parameters(self):
        tuned_parameters = {}
        return tuned_parameters

    def get_knn_parameters(self):
        tuned_parameters = {'n_neighbors': [5, 15, 25, 35, 55]}
        return tuned_parameters

    def get_forest_parameters(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=325, stop=1000, num=4)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 100, num=5)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1]

        # Create the random grid
        tuned_parameters = {'n_estimators': n_estimators,
                            'max_features': max_features,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf}
        return tuned_parameters

    def get_svm_parameters(self):
        polynomial_kernel = {'kernel': ['poly'], 'C': [1, 10, 100], 'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9]}

        linear_kernel = {'kernel': ['linear'], 'C': [1, 10, 100]}

        rbf_kernel = {'kernel': ['rbf'], 'gamma': [1e-2], 'C': [1, 10, 100]}

        tuned_parameters = [rbf_kernel, linear_kernel]

        return tuned_parameters

    def get_logistic_parameters(self):
        solvers_ = ['newton-cg', 'sag', 'saga']
        max_iter = [500, 1000, 1500]
        tol_ = [1e-2, 1e-3]
        multi_class = ['ovr', 'multinomial']

        tuned_parameters = [{'solver': solvers_,
                             'max_iter': max_iter,
                             'tol': tol_,
                             'multi_class': multi_class}]
        return tuned_parameters

    def execute(self, x_train, y_train):
        self.grid_search(self.classifier, x_train, y_train, self.parameters)

    def grid_search(self, classifier, x_train, y_train, tuned_parameters):
        print("# Tuning hyper-parameters for accuracy")
        command_line = CLClassification()
        classifier_name = command_line.classifier
        clf = GridSearchCV(classifier, tuned_parameters, cv=3, scoring="accuracy", n_jobs=self.cpu, verbose=1)
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:\n")
        print(clf.best_params_)
        print("")
        self.estimator = clf.best_estimator_
        # pickle.dump(self.estimator, open(classifier_name + '-model', 'wb'))  #saving classifier model
        # print("\nGrid scores on development set:\n")
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    def get_estimator(self):
        return self.estimator
