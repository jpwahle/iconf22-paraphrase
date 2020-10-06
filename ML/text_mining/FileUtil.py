import errno
import glob
import os

import pandas as pd


class FileUtil:
    def get_labels(self, dataset):
        number_features = self.get_number_features(dataset)
        return dataset[number_features]

    def get_features(self, dataset):
        number_features = self.get_number_features(dataset)
        return dataset.iloc[:, 0:number_features]

    def get_number_features(self, dataset):
        return dataset.shape[1] - 1

    def read_table(self, dataset):
        return pd.read_csv(dataset, header=None, delimiter=',')

    def get_label(self, folder_path):
        data = folder_path.split(os.sep)
        return data[len(data) - 1]

    def read(self, file_name):
        text = ""
        try:
            f = open(file_name, errors="ignore")
            text = f.read()
            f.close()
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise ("problem reading file: " + file_name)
        return text

    def get_files_path(self, folder_name):
        current = folder_name + os.sep + "*.txt"
        files = glob.glob(current)
        return files

    def get_files_path_recursively(self, folder_name):
        current = folder_name + os.sep + '**' + os.sep + '*.txt'
        files = glob.glob(current)
        return files

    def get_folders(self, folder_root):
        folders = []
        for dirname, dirnames, filenames in os.walk(folder_root):
            for subdirname in dirnames:
                folders.append(os.path.join(dirname, subdirname))

        return folders
