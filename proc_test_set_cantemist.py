# -*- coding: utf-8 -*-
"""
Processes the test and background sets files given by the CANTEMIST organizers 
for task CODING and prepares them to be used by X-Transformer model to make predictions
from previously trained models.

@author: André
"""

import os
import copy
import argparse
import proc_utils as pu
import stem_utils as su


def main(args):
    data_path = args.data_path #'C:/Users/André/Documents/FCUL/2º Ano/CANTEMIST/organized version/cantemist_data/'
    vocab_path = args.vocab_path #'C:/Users/André/Documents/FCUL/2º Ano/CANTEMIST/organized version/output/'
    out_path = args.output_path #'C:/Users/André/Documents/FCUL/2º Ano/CANTEMIST/organized version/test_aval/'
    
    if not os.path.exists(out_path): 
        print('Creating path %s' % out_path)
        os.mkdir(out_path)
        
    #Generates the dictionary of labels from the label correspondence file
    flabels = open(vocab_path + 'label_correspondence.txt', encoding='utf-8')
    labels = flabels.readlines()
    flabels.close()
        
    #Dict with ECIE-O codes as keys
    dict_labels = {}
    for i in range(len(labels)):
        dict_labels[labels[i].split('=')[1]] = (labels[i].split('=')[0], 
                   labels[i].split('=')[2].replace('\n',''))
    
    #Reads dev data to fill part of the test files
    l_dev_txt, l_dev_labels = pu.read_files(data_path, 'dev1')
    l_dev_labels_ori = copy.deepcopy(l_dev_labels)
    l_dev_labels = pu.convert_labels(l_dev_labels, dict_labels)
        
    #Reads tst set data
    l_tst_aval_txt, l_tst_aval_labels = pu.read_test_set_files(data_path)
    l_tst_aval_labels_ori = copy.deepcopy(l_tst_aval_labels)     
    l_tst_aval_labels = pu.convert_labels(l_tst_aval_labels, dict_labels)
        
    #Stemms the data
    su.check_nltk_punkt()
    stemmer = su.set_stemmer('spanish')
    print('Stemming dev1 text...')
    l_stem_text_dev = su.list_stemming(l_dev_txt, stemmer)
    print('Stemming test aval text...')
    l_stem_text_tst_aval = su.list_stemming(l_tst_aval_txt, stemmer)
    
    #Creates the Test aval files
    #It is necessary to split the articles and respective labels in 48 sets, each with 250 articles, 
    #which is equal to the number of articles present in the test set of the trained X-Transformer models
    #The first 109 lines of each file correspond to text from the test&background set to classify.
    #The other 141 lines correspond to text from dev set1 that will be used to find best confidence threshold
    #for the predictions
    cnt = 1
    ini = 0
    fin = 109
    
    while cnt <= 48:
        l_chunk_txt = l_stem_text_tst_aval[ini:fin] + l_stem_text_dev[0:141]
        l_chunk_labels = l_tst_aval_labels[ini:fin] + l_dev_labels[0:141]
        l_chunk_labels_ori = l_tst_aval_labels_ori[ini:fin] + l_dev_labels_ori[0:141]
        
        pu.write_files(l_chunk_txt, l_chunk_labels, l_chunk_labels_ori, out_path, 'test_' + str(cnt))
    
        ini = fin
        fin = fin + 109
        cnt += 1



#Begin
parser = argparse.ArgumentParser()

#Required parameters
parser.add_argument('-i1', '--data-path', type=str, required=True,
                    help='Path to the folder that contains the data given by the CANTEMIST Competition.')
parser.add_argument('-i2', '--vocab-path', type=str, required=True,
                    help='Path to the folder that contains the label_correspondence file')
parser.add_argument('-o', '--output-path', type=str, required=True,
                    help='Path to the directory that will contain the output test files')


args = parser.parse_args()
print(args)
main(args)
print('Finished processing')