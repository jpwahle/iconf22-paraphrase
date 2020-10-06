import argparse


class CLReport:
    folder = ""
    dataset = ""
    metric = ""

    def __init__(self):
        parser = self.define_parser_parameters()
        args = parser.parse_args()

        self.folder = args.folder
        self.dataset = args.dataset
        self.metric = args.metric

    def str2bool(self, str_value):
        return str_value == "True"

    def define_parser_parameters(self):
        parser = argparse.ArgumentParser(
            description="##################   Generate a report file with all the results  ##################")

        parser.add_argument('--metric', '-m', type=str, action='store', dest='metric',
                            choices=['Accuracy', 'F1-Micro', 'F1-Macro'],
                            required=False, help='set which metric to get the results (default: %(default)s)',
                            default='Accuracy')

        required_args = parser.add_argument_group('required arguments')

        required_args.add_argument('--folder', "-f", type=str, action='store', dest='folder', metavar='<path>',
                                   required=True, help='relative path from your current folder to the results folder')

        required_args.add_argument('--dataset', '-d', type=str, action='store', dest='dataset', metavar='<value>',
                                   required=True, help='name of the dataset to generate the results')

        return parser
