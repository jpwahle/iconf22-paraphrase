# Classification, Grid Search and Report

## This tool provides classification, grid search and a report file to generate a csv with the results 

### To do grid search and/or classificatoin type:

python3 text_mining/main.py [options] 

  Options:

        --train,-tr: 
          Name of the train file. No header file with comma-separated-values. 
          The last column must be the label of the class

        --test,-te: [Optional] 
          Name of the test file. No header file with comma-separated-values. 
          The last column must be the label of the class.
          If no test file is provided then a cross-validation will set.

        --classifier,-cl: 
          Name of the classifier. Choices are: knn, logistic, svm, forest or naive.
          if no grid search was done the classification with be executed with the
          parameters hard coded.

        --grid,-g: [Optional] 
          Set True to do grid search before doing classification. 
          After applying grid search, the best parameters
          will be fed into a classifier to do classification. Default is False

        --cpu,-cp: [Optional]
          Set the number of cpus to work in parallel. Default is 1 
        
        --bow,-b: [Optional]
          Set True to build a bag of words from text files. Default is False
          The dataset must follow this tree structure:

          dataset_folder:
              |
              |---> folder_class1
              |      |---> file1.txt
              |      |---> file2.txt
              |      |---> filen.txt
              |
              |---> folder_class2
                     |---> file1.txt
                     |---> file2.txt
                     |---> filen.txt

### To generate a report file type:

python3 text_mining/Report.py [options]

  Options:
        
        --folder,-f:
          Name of the folder with the results

        --dataset,-d:
          Name of the dataset to filter the results

        --metric,-m: [Optional]
          choose which metric to retreive the results. 
          Supported metrics are: Accuracy, F1-Micro, F1-Macro
          Default is Accuracy
