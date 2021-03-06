mkdir data data/datasets data/vocabularies

#Download CANTEMIST corpus:
cd data/datasets
wget https://zenodo.org/record/3952175/files/cantemist.zip

#Download CIEO3 valid codes
cd ../vocabularies
wget https://temu.bsc.es/cantemist/wp-content/uploads/2020/07/valid-codes.txt

#Download Spanish ICD10-CM
wget https://www.mscbs.gob.es/estadEstudios/estadisticas/normalizacion/CIE10/CIE10_2020_DIAGNOST_REFERENCIA_2019_10_04_1.xlsx

#Download Spanish DeCS
#wget

#Get embeddings trained on Spanish PubMed abstracts and decompress trained NER tagger
cd ../../
wget https://zenodo.org/record/3982487/files/resources.zip
unzip resources.zip
wget https://zenodo.org/record/3982487/files/trained_embeddings.zip
unzip trained_embeddings.zip

#Get the custom models for the Coding subtask
cd X-Transformer/
wget https://zenodo.org/record/3982487/files/custom_models.zip
unzip custom_models.zip




