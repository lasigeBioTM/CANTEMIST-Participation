# CANTEMIST-Participation
Code relative to the participation of the lasigeBioTM team in the [CANTEMIST-CANcer TExt Mining Shared Task – tumor named entity recognition](https://temu.bsc.es/cantemist/).

## Preparation
Get the necessary data to reproduce the experiments from:

```
./get_data.sh
```

To install all the necessary dependencies to run the code:

```
pip3 install -r requirements.txt
```

## 1.CANTEMIST-NER

- Train Spanish Biomedical Flair Embeddings: pre-processes Spanish PubMed abstracts and then train the FLAIR embeddings on the raw text.

```
python3 src/NER/train_flair_embeddings.py <mesinesp_subset> <direction> 
```

Args:
  - <mesinesp_subset> : mesinesp_1, mesinesp_2, mesinesp_3 or mesinesp_4; each subset contains the raw text of ~32500 PubMed abstracts 
  - <direction> : is 'fwd' if training foward language model and 'bwd' if training backward language model

The output will be in '/trained\_embeddings/<mesinesp_subset>/<direction>/' directory.


- Train Spanish Biomedical NER tagger: converts the CANTEMIST dataset to IOB2 schema and then train a NER tagger with the Spanish Biomedical Flair embeddings.

```
python3 src/NER/train_ner_model.py medium 
```

The output will be in '/resources/taggers/medium' directory 

- Apply the NER tagger to the test set of CANTEMIST corpus: predicts the entities present in the text, as well their location and build the annotation files.

```
python3 src/NER/predict_ner.py
```

The annotation files will be in './evaluation/NER/test/' directory


## 2. CANTEMIST-NORM
To find the CIE-O-3 code for each entity outputed by the NER tagger:

```
./norm.sh <ont_number> 
```

Arg:
  - <ont_number> : 'single_ont', to retrieve only CIE-O-3 candidates, 'multi-ont', to additionally retrieve candidates from the Spanish ICD10-CM and DeCS terminologies, which will improve the disambiguation graph and the application of the PPR algorithm..

The annotation files will be in './evaluation/NORM/<ont_number>' directory


## 3. CANTEMIST-CODING
1. Pre-process the Cantemist dataset.

```
python proc_data_cantemist.py -i cantemist_data/ -o proc_data_output/
````

2. Create a folder inside the X-Transformer/datasets/ and copy the proc_data_cantemist.py over there.

```
cd X-Transformer/datasets/
mkdir CANTEMIST
cd CANTEMIST
mkdir mapping
```
Returns tst_CANTEMIST.pred.npz in 'X-Transformer/save_models/CANTEMIST/pifa-a5-s0/ranker_bert-base-multilingual-cased/

3. Execute run_X-Transformer_CANTEMIST.sh 

```
./run_ X-Transformer_CANTEMIST.sh CANTEMIST bert-base-multilingual-cased
```

4. Rename of the output of run_X-Transformer_CANTEMIST.sh to tst.pred.npz and copy to X-Transformer_results.

5. Run  proc_npz_cantemist.py.

```
python proc_npz_cantemist.py -npz results_X-Transformer/tst.pred.npz -i1 cantemist_data/dev-set2-to-publish/cantemist-coding/txt/ -i2 proc_data_output/ -o predictions_final/
```

## Evaluation
To use the official scripts refer to [CANTEMIST EVALUATION LIBRARY](https://github.com/TeMU-BSC/cantemist-evaluation-library).




