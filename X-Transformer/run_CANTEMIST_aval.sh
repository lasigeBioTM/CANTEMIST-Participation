#!/bin/bash
#Specify the dataset and then the model before executing.
#Example run: ./run_CANTEMIST_aval.sh CANTEMIST DeCS_Titles_MER
#
#Currently allowed datasets: CANTEMIST, CANTEMIST_2 
#Currently allowed models: DeCS_Titles_MER, DeCS_Titles_MER_L, beto, bert-base-multilingual-cased
#

DATASET=$1
MODEL=$2

mkdir  -p datasets/CANTEMIST_Aval/
mkdir  -p datasets/CANTEMIST_Aval/mapping/

cd ..

if [ $DATASET == 'CANTEMIST' ]; then
	TRAIN_FILE=proc_data_output/train.txt
	echo $TRAIN_FILE
	
	cp proc_data_output/train_raw_labels.txt X-Transformer/datasets/CANTEMIST_Aval/train_raw_labels.txt
	cp proc_data_output/train_raw_texts.txt X-Transformer/datasets/CANTEMIST_Aval/train_raw_texts.txt
else
	TRAIN_FILE=proc_data_output/train_750.txt
	echo $TRAIN_FILE
	
	cp proc_data_output/train_750_raw_labels.txt X-Transformer/datasets/CANTEMIST_Aval/train_raw_labels.txt
	cp proc_data_output/train_750_raw_texts.txt X-Transformer/datasets/CANTEMIST_Aval/train_raw_texts.txt
fi

cp proc_data_output/label_vocab.txt X-Transformer/datasets/CANTEMIST_Aval/label_vocab.txt
cp proc_data_output/label_map.txt X-Transformer/datasets/CANTEMIST_Aval/mapping/label_map.txt
cd X-Transformer

for (( num = 1; num < 49 ; ++num))
do
	echo $num
	
	cd ..
	cp $TRAIN_FILE X-Transformer/datasets/CANTEMIST_Aval/train.txt
	mv test_aval/test_${num}.txt X-Transformer/datasets/CANTEMIST_Aval/test.txt
	mv test_aval/test_${num}_raw_labels.txt X-Transformer/datasets/CANTEMIST_Aval/test_raw_labels.txt
	mv test_aval/test_${num}_raw_texts.txt X-Transformer/datasets/CANTEMIST_Aval/test_raw_texts.txt
	cd X-Transformer
	
	taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19  python -m datasets.new_preprocess_tfidf -i datasets/CANTEMIST_Aval
	cd datasets
	python proc_data.py --dataset CANTEMIST_Aval

	python label_embedding.py -d CANTEMIST_Aval -e pifa
	cd ..

	mkdir -p save_models/CANTEMIST_Aval/pifa-a5-s0/indexer

	python -m xbert.indexer \
		-i datasets/CANTEMIST_Aval/L.pifa.npz \
		-o save_models/CANTEMIST_Aval/pifa-a5-s0/indexer \
		-d 6 --algo 5 --seed 0 --max-iter 20

	OUTPUT_DIR=save_models/CANTEMIST_Aval/pifa-a5-s0

	mkdir -p ${OUTPUT_DIR}/data-bin-cased

	python -m xbert.preprocess \
		-m bert \
		-n bert-base-multilingual-cased \
		-i datasets/CANTEMIST_Aval \
		-c ${OUTPUT_DIR}/indexer/code.npz \
		-o ${OUTPUT_DIR}/data-bin-cased


	mkdir -p ${OUTPUT_DIR}/ranker_${DATASET}_${MODEL}

	taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 python -m xbert.ranker train \
	  -x1 datasets/CANTEMIST_Aval/X.trn.npz \
	  -x2 save_models/${DATASET}/pifa-a5-s0/matcher/${MODEL}/trn_embeddings.npy \
	  -y datasets/CANTEMIST_Aval/Y.trn.npz \
	  -z save_models/${DATASET}/pifa-a5-s0/matcher/${MODEL}/C_trn_pred.npz \
	  -c save_models/${DATASET}/pifa-a5-s0/indexer/code.npz \
	  -o ${OUTPUT_DIR}/ranker_${DATASET}_${MODEL} -t 0.01 \
	  -f 0 -ns 0 --mode ranker

	taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 python -m xbert.ranker predict \
	  -m ${OUTPUT_DIR}/ranker_${DATASET}_${MODEL} \
	  -o ${OUTPUT_DIR}/ranker_${DATASET}_${MODEL}/tst_${num}.pred.npz \
	  -x1 datasets/CANTEMIST_Aval/X.tst.npz \
	  -x2 save_models/${DATASET}/pifa-a5-s0/matcher/${MODEL}/tst_embeddings.npy \
	  -y datasets/CANTEMIST_Aval/Y.tst.npz \
	  -z save_models/${DATASET}/pifa-a5-s0/matcher/${MODEL}/C_tst_pred.npz -t noop \
	  -f 0 -k 20
done