import argparse


class CLClassification:
    train = None
    test = None
    classifier = None
    isGridSearch = False
    isBow = False
    cpu = 1

    def __init__(self):
        parser = self.define_parser_parameters()
        args = parser.parse_args()

        self.train = args.tr
        self.test = args.te
        self.classifier = args.cla
        self.isGridSearch = self.str2bool(args.grid)
        self.isBow = self.str2bool(args.bow)
        self.cpu = args.cpu

    def define_parser_parameters(self):
        parser = argparse.ArgumentParser(
            description="##################   Apply classification or grid search  ##################")

        parser.add_argument('--test', "-te", type=str, action='store', dest='te', metavar='<path>', required=False,
                            help='relative path from your current folder to the test file')

        parser.add_argument('--cpu', "-cp", type=int, action='store', dest='cpu', metavar='<value>', default=1,
                            required=False, help='number of cores to use (default: %(default)s)')

        parser.add_argument('--grid', '-g', type=str, action='store', dest='grid', choices=['True', 'False'],
                            required=False, help='set True to apply a grid search  (default: %(default)s)',
                            default=False)

        parser.add_argument('--bow', '-b', type=str, action='store', dest='bow', choices=['True', 'False'],
                            required=False, help='set True to create a bag of words (default: %(default)s)',
                            default=False)

        required_args = parser.add_argument_group('required arguments')

        required_args.add_argument('--train', "-tr", type=str, action='store', dest='tr', metavar='<path>',
                                   required=True, help='relative path from your current folder to the train file')

        required_args.add_argument('--classifier', '-cl', type=str, action='store', dest='cla', metavar='<value>',
                                   required=True, help='type of aggregator to use',
                                   choices=["knn", "svm", "logistic", 'forest', 'naive'])

        return parser

    def str2bool(self, str_value):
        return str_value == "True"
