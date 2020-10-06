from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class Classification:
    classifier = None
    cpu = 1

    def __init__(self, cpu, selection="knn", estimator=None):
        if estimator is not None:
            self.classifier = estimator
        else:
            self.classifier = self.get_classifier(selection)
        self.cpu = cpu

    def get_classifier(self, selection):
        if selection == "knn":
            return self.get_knn()
        if selection == "svm":
            return self.get_svm()
        if selection == 'logistic':
            return self.get_logistic_regression()
        if selection == 'lda':
            return self.get_lda()
        if selection == 'naive':
            return self.get_naive_bayes()
        if selection == 'forest':
            return self.get_random_forest()

        raise Exception("Unknown classifier: " + selection)

    def get_naive_bayes(self):
        return GaussianNB()

    def get_lda(self):
        return LinearDiscriminantAnalysis()

    def get_knn(self):
        return KNeighborsClassifier(
            n_neighbors=15,
            n_jobs=self.cpu)

    def get_svm(self):
        return SVC(
            kernel='poly',
            degree=1,
            gamma="auto")

    def get_logistic_regression(self):
        return LogisticRegression(
            solver='lbfgs',
            multi_class='ovr',
            max_iter=500,
            n_jobs=self.cpu)

    def get_random_forest(self):
        return RandomForestClassifier(
            n_estimators=100,
            n_jobs=self.cpu)

    def evaluate(self, x_train, y_train, x_test, y_test):
        print('evaluating train and test...')
        self.classifier.fit(x_train, y_train)
        y_pred = self.classifier.predict(x_test)

        print("Accuracy: %0.4f" % (metrics.accuracy_score(y_test, y_pred) * 100))
        print("F1-Micro: %0.4f" % (metrics.f1_score(y_test, y_pred, average="micro") * 100))
        print("F1-Macro: %0.4f" % (metrics.f1_score(y_test, y_pred, average="macro") * 100))

    def cross_validate(self, x_train, y_train):
        print('applying cross-validation...')
        scoring = ['accuracy', 'precision_micro', 'precision_macro']
        scores = cross_validate(self.classifier, x_train, y_train, scoring=scoring, cv=10, return_train_score=False)

        print("Accuracy: %0.4f (+/- %0.2f)" % (scores['test_accuracy'].mean() * 100, scores['test_accuracy'].std() * 2))
        print("F1-Micro: %0.4f (+/- %0.2f)" % (
            scores['test_precision_micro'].mean() * 100, scores['test_precision_micro'].std() * 2))
        print("F1-Macro: %0.4f (+/- %0.2f)" % (
            scores['test_precision_macro'].mean() * 100, scores['test_precision_macro'].std() * 2))
