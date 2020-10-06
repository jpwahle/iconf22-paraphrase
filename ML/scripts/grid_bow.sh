#!/bin/bash

# This script will call a python code to apply grid search and
# classification with the best set of parameters found
# it is mandatory to name the input dataset with  
# this specific convention: 
#
# if you only have a training corpus
# <technique>-<corpus>-<anythingYouWant>.<anyExtension>
# examples: elmo-scygenes-mean.arff
#           elmo-bbc.arff
#           elmo-scycluster-sum.txt
#
# if you have train and test set
# <technique>-<corpus>-<anythingYouWant>-<set>.<anyExtension>
# examples: fasttext-reuters-mean-train.arff
#           fasttext-reuters-mean-test.arff

#-----------------------------------------------------------------------------
# Train/Test: False
#-----------------------------------------------------------------------------
#BOW
#for classifier in naive knn logistic forest svm;do
#  for technique in bow;do
#    for dataset in machines;do
#      #save path into variable
#      path=grid_machines/$classifier/$technique-$dataset-mean-$classifier.txt
#      #create folder structure
#      mkdir -p $(dirname $path) && touch $path
#     #call python code
#      python3 text_mining/main.py \
#        --train $dataset \
#        --classifier $classifier \
#        --bow True \
#        --grid True \
#        --cpu 6 \
#        > $path
#    done
#  done
#done

#-----------------------------------------------------------------------------
# Train/Test: True
#-----------------------------------------------------------------------------
#BOW
for classifier in naive knn logistic forest svm;do
  for technique in bow;do
    for dataset in machined;do
      #save path into variable
      path=grid_test/$classifier/$technique-$dataset-mean-$classifier.txt
      #create folder structure
      mkdir -p $(dirname $path) && touch $path
      #call python code
      python3 text_mining/main.py \
        --train tmp/train/$dataset \
        --test  tmp/test/$dataset \
        --classifier $classifier \
		    --bow True \
        --grid True \
        --cpu 28 \
        > $path
    done
  done
done


