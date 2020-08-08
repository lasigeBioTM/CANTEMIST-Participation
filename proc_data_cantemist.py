# -*- coding: utf-8 -*-
"""
Processes the files given by the CANTEMIST organizers for task CODING and prepares them
to be used as input to train a X-Transformer model.

@author: André Neves
"""

import os
import copy
import argparse
import pandas as pd
import proc_utils as pu
import stem_utils as su

def main(args):
    data_path = args.input_path  #'C:/Users/André/Documents/FCUL/2º Ano/CANTEMIST/cantemist_new/'
    out_path = args.output_path  #'output/'
    
    if not os.path.exists(out_path): 
        print('Creating path %s' % out_path)
        os.mkdir(out_path)
        
    #Generates .tsv file with all unique labels present in the txt files
    pu.gen_vocab_tsv(data_path, out_path)
        
    #Reads the generated .tsv file
    ecie_data = pd.read_csv(out_path + 'cantemist_terms.tsv', sep='\t')
    
    #Stores the terms in spanish and the respective ECIE-O Code in lists        
    l_codes = ecie_data['Code'].astype(str).values.tolist()
    l_terms = ecie_data['Terms'].astype(str).values.tolist()
    
    #Generates vocab and label_correspondence files and returns dict with label correspondence
    dict_labels = pu.gen_vocab(l_terms, l_codes, out_path)
        
    #Reads training data
    l_trn_txt, l_trn_labels = pu.read_files(data_path, 'trn')
    #creates a copy of the original labels for it is needed for X-Transformer
    l_trn_labels_ori = copy.deepcopy(l_trn_labels)
    #converts the labels to their corresponding numeric identifier
    l_trn_labels = pu.convert_labels(l_trn_labels, dict_labels)
    
    #Reads dev data
    #l_dev_txt, l_dev_labels = pu.read_files(data_path, 'dev1')
    #it is using dev2 since dev1 has one unlabelled file, and that causes X-Transformer to fail. 
    #if the unlabelled file is removed, then the tst_aval processing would have to be changed, since the 
    #X-Transformer model would not have a test set with 250 documents.
    l_dev_txt, l_dev_labels = pu.read_files(data_path, 'dev2')
    l_dev_labels_ori = copy.deepcopy(l_dev_labels)
    l_dev_labels = pu.convert_labels(l_dev_labels, dict_labels)
    
    #Reads extra dev data
    #it uses the dev1 files to crete a larger train set
    #The file that has no assigned labels is removed.
    l_extra_txt, l_extra_labels = pu.read_files(data_path, 'dev1')
    l_extra_txt.pop(212) #text file with no assigned labels
    l_extra_labels.pop(212) #text file with no assigned labels
    l_extra_labels_ori = copy.deepcopy(l_extra_labels)     
    l_extra_labels = pu.convert_labels(l_extra_labels, dict_labels)
    
    #Stemms the data
    su.check_nltk_punkt()
    stemmer = su.set_stemmer('spanish')
    print('Stemming trn text...')
    l_stem_text_trn = su.list_stemming(l_trn_txt, stemmer)
    print('Stemming dev text...')
    l_stem_text_dev = su.list_stemming(l_dev_txt, stemmer)
    print('Stemming extra text...')
    l_stem_text_extra = su.list_stemming(l_extra_txt, stemmer)
    
    #Writes files
    pu.write_files(l_stem_text_trn, l_trn_labels, l_trn_labels_ori, out_path, 'train')
    pu.write_files(l_stem_text_dev, l_dev_labels, l_dev_labels_ori, out_path, 'test')
    
    #Joins the extra data to the train data
    for i,j,z in zip(l_stem_text_extra, l_extra_labels, l_extra_labels_ori):
        l_stem_text_trn.append(i)
        l_trn_labels.append(j)
        l_trn_labels_ori.append(z)
    #Writes larger train set    
    pu.write_files(l_stem_text_trn, l_trn_labels, l_trn_labels_ori, out_path, 'train_750')


#Begin
parser = argparse.ArgumentParser()

#Required parameters
parser.add_argument('-i', '--input-path', type=str, required=True,
                    help='Path to the folder that contains the data given by the competition organizers')
parser.add_argument('-o', '--output-path', type=str, required=True,
                    help='Path to the directory that will contain the output files')

args = parser.parse_args()
print(args)
main(args)
print('Finished processing')