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
#for classifier in naive knn logistic forest svm;do
#  for technique in used elmod;do
#    for dataset in scycluster scygenes bbc ohsumed;do
	  #save path into variable
#      path=grid_d2v/$classifier/$technique-$dataset-mean-$classifier.txt
      #create folder structure
#      mkdir -p $(dirname $path) && touch $path
      #call python code
#      python3 text_mining/main.py \
#       --train datasets_vector/$technique-${dataset}-mean.arff \
#       --classifier $classifier \
#       --grid True \
#       --cpu 28 \
#       > $path
#    done
#  done
#done

#-----------------------------------------------------------------------------
# Train/Test: True
#-----------------------------------------------------------------------------
for classifier in svm;do
  for technique in glove;do
    for dataset in thesisp;do 
      #save path into variable
      path=grid_wiki_thesis/$classifier/$technique-$dataset-mean-$classifier.txt
      #create folder structure
      mkdir -p $(dirname $path) && touch $path
      #call python code
      python3 text_mining/main.py \
        --train datasets_vector_train_p/$technique-machinep-mean.arff \
        --test thesispd_vector_test/$technique-${dataset}-mean.arff \
        --classifier $classifier \
        --grid True \
        --cpu 12 \
        > $path
    done
  done
done



