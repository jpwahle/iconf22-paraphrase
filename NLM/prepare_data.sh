#!/bin/bash

# sh prepare_data.sh /path/to/data/dir

# This script will extract and move the paraphrase detection dataset
# to the folders used by the experiments of neural language models


export spinbot_dir=$1/automated_evaluation_up/spinbot/corpus/paragraphs
export spinnerchief_dir=$1/automated_evaluation_up/spinnerchief/corpus/paragraphs

wget -nc https://zenodo.org/record/3608000/files/automated_evaluation.zip?download=1 -O $1/automated_evaluation_up.zip
unzip $1/automated_evaluation_up.zip -d $1

for zip_archive in arxiv_paragraph thesis_paragraph wikipedia_paragraph; do
    export old_folder_name=$(unzip -qql $spinbot_dir/$zip_archive.zip | head -n1 | tr -s ' ' | cut -d' ' -f5-)
    unzip $spinbot_dir/$zip_archive.zip -d $spinbot_dir
    mv $spinbot_dir/$old_folder_name $spinbot_dir/$zip_archive
done

for targz in wikipedia_paragraphs_train; do
    export old_folder_name=$(tar -tzf $spinbot_dir/$targz.tar.gz | head -1 | cut -f1 -d"/")
    tar xvzf $spinbot_dir/$targz.tar.gz -C $spinbot_dir
    mv $spinbot_dir/$old_folder_name $spinbot_dir/$targz
done

for targz in arxiv_paragraphs_2w arxiv_paragraphs_4w thesis_paragraphs_2w thesis_paragraphs_4w wikipedia_paragraphs_2w wikipedia_paragraphs_4w; do
    export old_folder_name=$(tar -tzf $spinnerchief_dir/$targz.tar.gz | head -1 | cut -f1 -d"/")
    tar xvzf $spinnerchief_dir/$targz.tar.gz -C $spinnerchief_dir
    mv $spinnerchief_dir/$old_folder_name $spinnerchief_dir/$targz
done
