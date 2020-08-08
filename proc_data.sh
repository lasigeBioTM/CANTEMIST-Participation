#!/bin/bash
#Executes the code to process the data given by CANTEMIST organizers and stores them in the format required for X-Trasnformer
#
#Example run: ./proc_data.sh
#

#Processes the text data given
python proc_data_cantemist.py -i data/ -o proc_data_output/

#Creates the directories for the two datasets for CANTEMIST (CANTEMIST and CANTEMIST_2)
#CANTEMIST_2 has a larger train set with 750 documents, while CANTEMIST has only 501.
mkdir X-Transformer/datasets/CANTEMIST
mkdir X-Transformer/datasets/CANTEMIST/mapping

mkdir X-Transformer/datasets/CANTEMIST_2
mkdir X-Transformer/datasets/CANTEMIST_2/mapping

#Then, copies the files for those directories
cp proc_data_output/label_map.txt X-Transformer/datasets/CANTEMIST/mapping/label_map.txt
cp proc_data_output/label_map.txt X-Transformer/datasets/CANTEMIST_2/mapping/label_map.txt

cp proc_data_output/label_vocab.txt X-Transformer/datasets/CANTEMIST/label_vocab.txt
cp proc_data_output/label_vocab.txt X-Transformer/datasets/CANTEMIST_2/label_vocab.txt

cp proc_data_output/train.txt X-Transformer/datasets/CANTEMIST/train.txt
cp proc_data_output/train_raw_labels.txt X-Transformer/datasets/CANTEMIST/train_raw_labels.txt
cp proc_data_output/train_raw_texts.txt X-Transformer/datasets/CANTEMIST/train_raw_texts.txt
cp proc_data_output/train_750.txt X-Transformer/datasets/CANTEMIST_2/train.txt
cp proc_data_output/train_750_raw_labels.txt X-Transformer/datasets/CANTEMIST_2/train_raw_labels.txt
cp proc_data_output/train_750_raw_texts.txt X-Transformer/datasets/CANTEMIST_2/train_raw_texts.txt

cp proc_data_output/test.txt X-Transformer/datasets/CANTEMIST/test.txt
cp proc_data_output/test_raw_labels.txt X-Transformer/datasets/CANTEMIST/test_raw_labels.txt
cp proc_data_output/test_raw_texts.txt X-Transformer/datasets/CANTEMIST/test_raw_texts.txt
cp proc_data_output/test.txt X-Transformer/datasets/CANTEMIST_2/test.txt
cp proc_data_output/test_raw_labels.txt X-Transformer/datasets/CANTEMIST_2/test_raw_labels.txt
cp proc_data_output/test_raw_texts.txt X-Transformer/datasets/CANTEMIST_2/test_raw_texts.txt