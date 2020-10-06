from sklearn.feature_extraction.text import TfidfVectorizer

from text_mining.FileUtil import FileUtil


class BagOfWords:
    x_train = None
    y_train = []
    x_test = None
    y_test = []

    def build(self, train_folder, test_folder=None):
        train_data = self.read_text_files(train_folder, self.y_train)
        # create tfidf object
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=300)
        self.x_train = vectorizer.fit_transform(train_data)  # apply tfidf on train data
        self.x_train = self.x_train.toarray()

        if test_folder is not None:
            test_data = self.read_text_files(test_folder, self.y_test)
            self.x_test = vectorizer.transform(test_data)  # apply tfidf on test data
            self.x_test = self.x_test.toarray()

    def read_text_files(self, data_folder, labels):
        data = []
        file_util = FileUtil()
        folders = file_util.get_folders(data_folder)
        for folder in folders:  # reading class folders
            label = file_util.get_label(folder)
            files = file_util.get_files_path(folder)
            for file in files:
                data.append(file_util.read(file))
                labels.append(label)  # update label of the document
        return data

    def get_train_features(self):
        return self.x_train

    def get_train_labels(self):
        return self.y_train

    def get_test_features(self):
        return self.x_test

    def get_test_labels(self):
        return self.y_test
