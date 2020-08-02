# CANTEMIST-Participation
## 1.Cantemist-NER
### 1.Train Spanish Biomedical Flair Embeddings
1.Pre process the MESINESP dataset for Flair Embeddings
```
pre_process_NER.py
```
Returns the output files in data/datasets/mesinesp/ directory 

2.Train the Flair Embeddings
```
train_flair_embeddings.py <mesinesp_subset> <direction> 
# mesinesp_subset is mesinesp_1, mesinesp_2, mesinesp_3 or mesinesp_4; direction is 'fwd' if training LM foward and 'bwd' if training backward LM
```
Returns Flair Embeddings in :
- /trained_embeddings/mesinesp_subset/fwd/ directory if chosen direction is foward
- /trained_embeddings/mesinesp_subset/bwd/ directory if chosen direction is backward
### 2.Train Spanish Biomedical NER Model
1.Convert the CANTEMIST dataset to IOB2
```
pre_process_NER.py
```
Returns a text file with the text of all documents in the subset tokenized and tagged according with the IOB2 schema in /data/datasets/iob2/ directory
2.Train the NER model
```
train_ner_model.py <model> # model is 'base', 'medium', 'large', 'pubmed'
```
Returns the trained model in /resources/taggers/' + ner_model directory 
### 3.Apply Spanish Biomedical NER Model to Cantemist Test Dataset
1.Apply NER model
```
predict_ner.py
```
Returns the submission files in the ./evaluation/NER/test/

## 2.Cantemist-NORM
1.Create candidates for each entity in ./tmp/norm_candidates_dir;
```
pre_process_norm.py <model> <ont_number> #  model is 'string_matching', 'ppr', 'ppr_ic', ont_number is number of used ontologies
```

2.Apply PPR algorithm over the candidate files
```
java ppr_for_ned_all <model> #  model is 'string_matching', 'ppr', 'ppr_ic'
```
3.Create the submission files in the ./evaluation/NORM/test/ dir
```
post_process_norm.py <model> <ont_number> #  ont_number is number of used ontologies 
```
## 3.Cantemist-Coding
1.Pre process the Cantemist dataset
```
python proc_data_cantemist.py -i cantemist_data/ -o proc_data_output/
````
2.Create a folder inside the X-Transformer/datasets/ and copy the proc_data_cantemist.py over there.
```
cd X-Transformer/datasets/
mkdir CANTEMIST
cd CANTEMIST
mkdir mapping
```
Returns tst_CANTEMIST.pred.npz in 'X-Transformer/save_models/CANTEMIST/pifa-a5-s0/ranker_bert-base-multilingual-cased/
3.Execute run_X-Transformer_CANTEMIST.sh 
```
./run_ X-Transformer_CANTEMIST.sh CANTEMIST bert-base-multilingual-cased
```
4.Rename of the output of run_X-Transformer_CANTEMIST.sh to tst.pred.npz and copy to X-Transformer_results

5.Run  proc_npz_cantemist.py
```
python proc_npz_cantemist.py -npz results_X-Transformer/tst.pred.npz -i1 cantemist_data/dev-set2-to-publish/cantemist-coding/txt/ -i2 proc_data_output/ -o predictions_final/
```


