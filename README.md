# CANTEMIST-Participation
Code relative to the participation of the lasigeBioTM team in the [CANTEMIST-CANcer TExt Mining Shared Task – tumor named entity recognition](https://temu.bsc.es/cantemist/).

## Preparation
Get the necessary data to reproduce the experiments:

```
./get_data.sh
```

To install all the necessary dependencies to run the code:

```
./install_requirements.sh
```

## 1. CANTEMIST-NER

#### Train Spanish Biomedical Flair Embeddings: 
pre-processes Spanish PubMed abstracts and then trains the FLAIR embeddings on the raw text.

```
python3 src/NER/train_flair_embeddings.py <mesinesp_subset> <direction>
```

Args:
  * <mesinesp_subset> : mesinesp_1, mesinesp_2, mesinesp_3 or mesinesp_4; each subset contains the raw text of ~32500 PubMed abstracts.
  - \<direction> : is 'fwd' if training foward language model and 'bwd' if training backward language model.

The output will be in '/trained\_embeddings/<mesinesp_subset>/\<direction>/' directory.


#### Train Spanish Biomedical NER tagger: 
converts the CANTEMIST dataset to IOB2 schema and then trains a NER tagger with the Spanish Biomedical Flair embeddings.

```
python3 src/NER/train_ner_model.py medium 
```

The output will be in '/resources/taggers/medium' directory.

#### Apply the NER tagger to the test set of CANTEMIST corpus: 
predicts the entities present in the text, as well their location and builds the annotation files.

```
python3 src/NER/predict_ner.py
```

The annotation files will be in './evaluation/NER/' directory.


## 2. CANTEMIST-NORM

#### To find the CIE-O-3 code for each entity outputed by the NER tagger:

```
./norm.sh <ont_number> 
```

Arg:
  - <ont_number> : 'single_ont', to retrieve only CIE-O-3 candidates, 'multi-ont', to additionally retrieve candidates from the Spanish ICD10-CM and DeCS terminologies, which will improve the disambiguation graph and the application of the PPR algorithm..

The annotation files will be in './evaluation/NORM/<ont_number>' directory


#### Alternative: 
uses [merpy](https://pypi.org/project/merpy/) to annotate the documents.

```
python3 src/NER/mer_annotate.py <task> <subset>
```

Args:
  - \<task> : 'ner' or 'norm'
  - \<subset> : 'train', 'dev1', 'dev2' or 'test'

The annotation files will be in './mer_annotations/<task>/<subset>' directory.


## 3. CANTEMIST-CODING
### Train the model
convert the CANTEMIST dataset to a suitable format to X-Transformer(https://github.com/OctoberChang/X-Transformer)
```
./proc_data.sh
```
returns 2 datasets: CANTEMIST and CANTEMIST_2. The last will have 249 more documents on the train set that are coming from the dev-set1.
Move to X-Transformer directory 
```
cd X-Transformer
```
Get the custom models
```
./get_custom_models
```
For this task the models available are:
- BERT Base Multilingual Cased
- BETO:Spanish BERT
- DeCS_Titles_MER
- DeCS_Titles_MER_L

Unlike NER and NORM in the next step it is necessary that the version of transformers is 2.2.2
```
pip install transformers==2.2.2
```
execute run_X-Transformer_CANTEMIST.sh specifying the model to use
```
./run_X-Transformer_CANTEMIST.sh CANTEMIST bert-base-multilingual-cased
```
convert the predictions of X-Transformer to .tsv format
```
python proc_npz_cantemist.py \
-npz X-Transformer/save_models/CANTEMIST/pifa-a5-s0/ranker_bert-base-multilingual-cased/tst.pred.npz \
-i1 cantemist_data/dev-set2-to-publish/cantemist-coding/txt/ \
-i2 proc_data_output/ \
-o predictions_CANTEMIST_bert-base-multilingual-cased/
```
Returns 4 .tsv files, each one with a different value of cofidence for the predictions. 3 files will have predictions based on the best values achieved in the precision. recall and f-score measures.
The 4th file ("baseline") uses a value of cofidence equal to 0.

### Predictions for test background set
 convert the test and background sets to a suitable format to X-Transformer
```
python proc_test_set_cantemist.py \
-i1 cantemist_data/ \
-i2 proc_data_output/ \
-o test_aval/
```
Returns 48 test files with the respective label and text files. Each test file is composed by 109 lines from the test-background set documents and the remaning 141 are from dev-set1 that will be used to chose the value of confidence to the predictions.
 
execute run_CANTEMIST_aval.sh specifying the model to use
```
./run_CANTEMIST_aval.sh CANTEMIST bert-base-multilingual-cased
```
Returns the prediction for each one of the 48 files

execute the proc_npz_test_set_cantemist.py:
```
python proc_npz_test_set_cantemist.py \
-npz X-Transformer/save_models/CANTEMIST_Aval/pifa-a5-s0/ranker_CANTEMIST_bert-base-multilingual-cased/ \
-i1 cantemist_data/ \
-i2 proc_data_output/ \
-o predictions_test_aval/
```
Returns the  .tsv files with the predictions for test background sets documents. 









## Evaluation
To use the official scripts refer to [CANTEMIST EVALUATION LIBRARY](https://github.com/TeMU-BSC/cantemist-evaluation-library).



