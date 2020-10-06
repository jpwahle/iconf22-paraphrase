import os

from text_mining.FileUtil import FileUtil


class Result:
    technique = ""
    dataset = ""
    classifier = ""
    accuracy = ""
    f1_micro = ""
    f1_macro = ""

    def __init__(self, result_file):
        file_util = FileUtil()
        self.dataset = self.get_dataset(result_file)
        self.technique = self.get_technique(result_file)
        self.classifier = self.get_classifier(result_file)
        content = file_util.read(result_file)
        self.accuracy = self.get_accuracy(content)
        self.f1_micro = self.get_f1_micro(content)
        self.f1_macro = self.get_f1_macro(content)

    def get_file_name(self, file):
        data = file.split(os.sep)
        file_name = data[-1]
        return file_name

    def get_classifier(self, file):
        file_name = self.get_file_name(file)
        content = file_name.split("-")
        classifier = content[-1]
        classifier = classifier.replace(".txt", "")
        return classifier

    def get_dataset(self, file):
        file_name = self.get_file_name(file)
        content = file_name.split("-")
        return content[1]

    def get_technique(self, file):
        file_name = self.get_file_name(file)
        content = file_name.split("-")
        return content[0]

    def get_accuracy(self, content):
        lines = content.split("\n")
        for line in lines:
            if "Accuracy" not in line:
                continue
            words = line.split(" ")
            return words[1]

    def get_f1_micro(self, content):
        lines = content.split("\n")
        for line in lines:
            if "F1-Micro" not in line:
                continue
            words = line.split(" ")
            return words[1]

    def get_f1_macro(self, content):
        lines = content.split("\n")
        for line in lines:
            if "F1-Macro" not in line:
                continue
            words = line.split(" ")
            return words[1]

    def __str__(self):
        data = ""
        data += "dataset: " + self.dataset + "\n"
        data += "technique: " + self.technique + "\n"
        data += "classifier: " + self.classifier + "\n"
        data += "accuracy: " + self.accuracy + "\n"
        data += "f1-micro: " + self.f1_micro + "\n"
        data += "f1-macro: " + self.f1_macro + "\n"
        return data
