# python module absolute path
import os
import sys

pydir_name = os.path.dirname(os.path.abspath(__file__))
ppydir_name = os.path.dirname(pydir_name)
# python path definition
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

# TODO: report generate tex output
# TODO: ignore files that do not have all information (try catch)
# TODO: report generate comma separated output (done)
# TODO: input parameter to select: accuracy, fi-micro or f1-macro (done)
# TODO: input parameter to the folder with the content (done)
# TODO: input parameter to the dataset to get information (done)
# TODO: save results with the name of the dataset (done)

from text_mining.FileUtil import FileUtil
from text_mining.Result import Result
from text_mining.CommandLineReport import CLReport


class Report:
    dataset = ""
    folder = ""
    metric = ""

    def create_csv(self):
        commandLine = CLReport()

        # get command line information
        self.dataset = commandLine.dataset
        self.folder = commandLine.folder
        self.metric = commandLine.metric

        # read full path of all .txt files
        file_util = FileUtil()
        folder_path = os.path.join(ppydir_name, self.folder)
        files_path = file_util.get_files_path_recursively(folder_path)

        # create a list of Result objects
        results = self.get_results(files_path)

        # sort results by technique name
        results.sort(key=lambda x: x.technique)

        # build csv file
        csv = self.build_csv(results)

        self.save_csv(csv)

    def build_csv(self, results):
        csv = ""
        for classifier in ("knn", "logistic", "svm", "forest", "naive"):
            csv += classifier + "\n"
            technique = ""
            metric = ""

            for r in results:
                if r.classifier != classifier:
                    continue
                technique += r.technique + ","

                if self.metric == "Accuracy":
                    metric += r.accuracy + ","
                if self.metric == "F1-Micro":
                    metric += r.f1_micro + ","
                if self.metric == "F1-Macro":
                    metric += r.f1_macro + ","

            csv += technique + "\n" + metric + "\n\n\n"
        return csv

    def get_results(self, files_path):
        results = []
        for file in files_path:
            result = Result(file)
            if self.dataset != result.dataset:
                continue
            results.append(result)
        return results

    def save_csv(self, csv):
        print(csv)
        file = open(self.dataset + '.csv', 'w')
        file.write(csv)
        file.close()


if __name__ == "__main__":
    Report().create_csv()
