# -*- coding: utf-8 -*-
"""
Functions and Methods used to proccess data for the CANTEMIST-CODING competition.

@author: André Neves
"""

import csv
import pandas as pd
from os import path, listdir

def gen_vocab(terms, codes, path):
    """Generates the label_vocab, label_correspondence and label_map files along with
    a dictionary containing the correspondence between the labels and numeric identifiers
    that will be used by X-Transformer.
    @Requires:
        terms = list containing the label terms
        codes = list containing the label codes
        path = path to store the output files
    @Ensures:
        -> The creation of the following files: 'label_vocab.txt', 'label_correspondence.txt'
        and 'label_map.txt'.
        -> Returns a dictionary with the correspondence between the labels and the 
        corresponding numeric identifiers used by X-Transformer.
        dictionary structure: {ECIE-O Code: (numeric identifier, ECIE-O Term)}
    """
    l_vocab, l_corr = [], []
    for i in range(len(terms)):
        l_vocab.append(str(i) + '\t' + codes[i])
        l_corr.append(str(i) + '=' + codes[i] + '=' + terms[i])
          
    #Label_vocab:
    #Label_number  ECIE-O-Term    
    with open(path+'label_vocab.txt', 'w', encoding='utf-8') as f:
        for i in l_vocab:
            f.write('%s\n' % i)

    #Label_correspondence:
    #Label_number=ECIE-O-Code=ECIE-O-Term         
    with open(path+'label_correspondence.txt', 'w', encoding='utf-8') as f:
        for i in l_corr:
            f.write('%s\n' % i)
            
    #Creates a dict to be returned with the label correspondence
    #Dict Format = {ECIE-O Code: (label number, ECIE-O Term)}
    dict_corr = {}
    for i in range(len(l_corr)):
        dict_corr[l_corr[i].split('=')[1]] = (l_corr[i].split('=')[0], 
                   l_corr[i].split('=')[2])

    #Label_map:
    #ECIE-O-Term -> sorted alphabetical
    with open(path+'label_map.txt', 'w', encoding='utf-8') as f:
        codes.sort()
        for i in range(len(codes)):
            f.write('%s\n' % codes[i])
            
    return dict_corr
    
def convert_labels(l_labels, dict_labels):
    """Converts the original labels to the corresponding numeric identifiers
    @Requires:
        l_labels = list of lists containing the labels that are assigned to each article
        dict_labels = dictionary with the correspondence between the labels and their 
        corresponding numeric identifier -> Generated through the gen_vocab function
    @Ensures:
        The list with the labels converted to their corresponding numeric identifier
    """
    for i in range(len(l_labels)):
        l_aux = []
        for l in range(len(l_labels[i])):
            l_aux.append(dict_labels.get(l_labels[i][l])[1].replace(' ', '_'))
            l_labels[i][l] = dict_labels.get(l_labels[i][l])[0]
    return l_labels
    

def get_codes(l_text):
    """Fetches the ECIE-O codes and terms that are present in the strings retrieved 
    from the .ann files given by the CANTEMIST competition organizers.
    @Requires:
        l_text = list of lists containing the lines of text from the .ann file 
        that contain the ECIE-O information (terms or codes). There are several contents
        in these lines of text separated by '\t' chars. The ECIE-O term or Code is the
        last content of each line, thus being present after the last '\t' character.
    @Ensures:
        Returns the list of lists with the found entities (ECIE-O terms or ECIE-O codes) 
        present in the texts from the .ann files 
    """
    l_entities = []        
    for i in l_text:
        str_aux=i.replace('\n','')
        
        for j in range(-1, -len(i), -1):
            if str_aux[j] == '\t':
                l_entities.append(str_aux[j+1:])
                break
            
    return l_entities


def gen_vocab_tsv(data_path, out_path):
    """Generates a vocabulary file containing the ECIE-O codes and ECIE-O terms in 
    the .tsv format. 
    @Requires: 
        data_path = path to the files given by the competition organizers
        out_path = path to store the resulting file
    @Ensures:
        The creation of a .tsv file containing the ECIE-O codes and corresponding ECIE-O terms
        that were found among the files given by the competition organizers.
    """
    l_codes, l_terms = [], []
    with open(data_path + 'valid-codes.tsv', 'r', encoding='utf-8') as in_file:
        df_codes = pd.read_csv(in_file, sep='\t', header=None)
        df_codes = df_codes.dropna()

    l_codes = df_codes[0].astype(str).values.tolist()
    l_terms = df_codes[1].astype(str).values.tolist()

    dict_final = {}
    for i, j in zip(l_codes, l_terms):
        dict_final[i] = [j]
    
    print('Generating .tsv vocab file ...')
    for x in range(1,1001):
        try:
            with open(data_path + 'train-set-to-publish/cantemist-norm/cc_onco'+str(x)+'.ann', 'r', encoding='utf-8') as finput:
                data=finput.readlines()
                
            l_codes_raw, l_names_raw = [], []
            
            for i in range(len(data)):
                if data[i][0] == '#': 
                    l_codes_raw.append(data[i])
                else:
                    l_names_raw.append(data[i])
            
            l_codes = get_codes(l_codes_raw)
            l_names = get_codes(l_names_raw)
                    
            for i, j in zip(l_codes, l_names):
                if i not in dict_final.keys():
                    dict_final[i] = []
                
                if j not in dict_final[i]:
                    dict_final[i].append(j)
        except FileNotFoundError:
            continue

    for x in range(1,1001):
        try:
            with open(data_path + 'dev-set1-to-publish/cantemist-norm/cc_onco'+str(x)+'.ann', 'r', encoding='utf-8') as finput:
                data=finput.readlines()
                
            l_codes_raw, l_names_raw = [], []
            
            for i in range(len(data)):
                if data[i][0] == '#': 
                    l_codes_raw.append(data[i])
                else:
                    l_names_raw.append(data[i])
            
            l_codes = get_codes(l_codes_raw)
            l_names = get_codes(l_names_raw)
                    
            for i, j in zip(l_codes, l_names):
                if i not in dict_final.keys():
                    dict_final[i] = []
                
                if j not in dict_final[i]:
                    dict_final[i].append(j)
        except FileNotFoundError:
            continue            

    for x in range(1,1501):
        try:
            with open(data_path + 'dev-set2-to-publish/cantemist-norm/cc_onco'+str(x)+'.ann', 'r', encoding='utf-8') as finput:
                data=finput.readlines()
                
            l_codes_raw, l_names_raw = [], []
            
            for i in range(len(data)):
                if data[i][0] == '#': 
                    l_codes_raw.append(data[i])
                else:
                    l_names_raw.append(data[i])
            
            l_codes = get_codes(l_codes_raw)
            l_names = get_codes(l_names_raw)
                    
            for i, j in zip(l_codes, l_names):
                if i not in dict_final.keys():
                    dict_final[i] = []
                
                if j not in dict_final[i]:
                    dict_final[i].append(j)
        except FileNotFoundError:
            continue           
        
    with open(out_path + 'cantemist_terms.tsv', 'wt', encoding='utf-8') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(['Code', 'Terms'])
        for i, j in dict_final.items():        
            tsv_writer.writerow([i, str(j).replace('[','').replace(']','').replace('\'','')])
        

def read_files(data_path, data_type):
    """Reads the .txt files present in the chosen set given by the competition organizers.
    @Requires:
        data_path = path to the files given by the competition organizers
        data_type = type of data set (train, dev-set1 or dev-set2).
                    Possible values: 'trn', 'dev1' or 'dev2'.
    @Ensures:
        l_strs = list of lists containing the text present in the .txt files
        l_codes_raw = lists of lists containing the ECIE-O codes found in the .ann files
    """
    if data_type == 'trn': 
        type_path = 'train-set-to-publish/cantemist-norm/cc_onco'
    elif data_type == 'dev1':
        type_path = 'dev-set1-to-publish/cantemist-norm/cc_onco'
    else:
        type_path = 'dev-set2-to-publish/cantemist-norm/cc_onco'

    l_strs, l_codes_raw = [], []
    
    print('Reading ' + type_path + ' data in ' + data_path + type_path + ' ...')
    for x in range(1,1501):
        try:
            #Fetches text
            with open(data_path + type_path + str(x) + '.txt', 'r', encoding='utf-8') as finput_txt:
                data_txt=finput_txt.read().replace('\n',' ')
                
            #Fetches ECIO codes/Labels
            with open(data_path + type_path + str(x) + '.ann', 'r', encoding='utf-8') as finput:
                data=finput.readlines()
                
            l_codes_aux = []
            for i in range(len(data)):
                if data[i][0] == '#': 
                    l_codes_aux.append(data[i])
            
            l_codes = get_codes(l_codes_aux)
            l_codes = list(dict.fromkeys(l_codes)) #Remove Duplicates
            
            #l_codes_raw.append(str(l_codes).replace('[','').replace(']','').replace('\'','').replace(', ',','))
            l_codes_raw.append(l_codes)
            l_strs.append(data_txt)
        except FileNotFoundError:
            continue
        
    return l_strs, l_codes_raw
    
    
def write_files(l_stem_text, l_labels, l_raw_labels, out_path, name):
    """Writes the .txt files required for X-Transformer.
    @Requires:
        l_stem_text = list of lists containing the stemmed texts
        l_labels = list of lists containing the labels assigned to each text, 
        converted to their corresponding numeric identifier.
        l_raw_labels = list of lists containing the labels assigned to each text in their
        original format.
        out_path = path to store the resulting files 
        name = dataset name. Possible values: 'train', 'test', 'test_aval_pt1' and 'test_aval_pt2'
    @Ensures:
        The creation of the following files for each dataset (train or test):
            dataset.txt = contains the texts and the assigned labels converted to their
            corresponding numeric identifier. Each line has the following structure:
                label1,label2,...,labelN '\t' stemmed_text '\n'
            dataset_raw_labels.txt = contains only the labels assigned to each text in their
            original format, i.e., contains the ECIE-O terms assigned to each text.
            dataset_raw_texts.txt = contains only the stemmed texts.
    """
    print('Writing ' + name + ' files ...')
    with open(out_path + name + '.txt', 'w', encoding='utf-8') as foutput:
        for i, j in zip(l_labels, l_stem_text):
            foutput.write(str(i).replace('[','').replace(']','').replace('\'','').replace(', ',',') + '\t' + j + '\n')

    with open(out_path + name + '_raw_labels.txt', 'w', encoding='utf-8') as foutput_labs:
        for i in l_raw_labels:
            foutput_labs.write(str(i).replace('[','').replace(']','').replace('\'','').replace(', ',' ') + '\n')
            
    with open(out_path + name + '_raw_texts.txt', 'w', encoding='utf-8') as foutput_txts:
        for i in l_stem_text:
            foutput_txts.write(i + '\n')
            

def write_results(data_path, out_path, l_pred_labs, dict_labels, eval_type):
    """Writes the algorithm prediction in the format required by the CANTEMIST competition.
    @Requires:
        data_path = path to the files of the test set to classify
        out_path = path to store the resulting file
        l_pred_labs = list of lists containing the predicted labels for each file
        dict_labels = dictionary containing the correspondence between the numeric identifiers
        and the ECIE-O codes.
        eval_type = type of evaluation focus used to make the predictions.
                    possible values: 'baseline', 'prec', 'rec' and 'f1'.
    @Ensures:
        The creation of a .tsv file containing the label predictions for each document
    """
    l_files = []
    for x in range(1,1501):
        if path.exists(data_path + 'cc_onco' + str(x) + '.txt') == True:
            l_files.append('cc_onco' + str(x))
    
    with open(out_path + 'output_' + eval_type + '.tsv', 'w', newline='') as foutput:
        tsv_writer = csv.writer(foutput, delimiter='\t')
        tsv_writer.writerow(['file', 'code'])
        for i, j in zip(l_files, l_pred_labs):
            for z in j:
                l_code = dict_labels[str(z)][0] #Converts from numeric identifier to eCIE-O code  
                tsv_writer.writerow([i, l_code])


def read_test_set_files(data_path):
    """Reads the .txt files present in the chosen set given by the competition organizers.
    @Requires:
        data_path = path to the files given by the competition organizers
        data_type = type of data set (train, dev-set1 or dev-set2).
                    Possible values: 'trn', 'dev1' or 'dev2'.
    @Ensures:
        l_strs = list of lists containing the text present in the .txt files
        l_codes_raw = lists of lists containing the ECIE-O codes found in the .ann files
    """
    
    data_path = data_path + 'test-background-set-to-publish/'
    
    #Fetches all files from the test set
    l_test_files = sorted(listdir(data_path))
    
    l_strs, l_codes_raw = [], []
    
    print('Number of files in ' + data_path + ' : ' + str(len(l_test_files)) + ' files.')
    print('Reading test set files stored in ' + data_path + ' ...')
    
    #At the same time it reads the text, it writes the file name in a file that 
    #will later be used when making the final prediction file
    with open(data_path + 'list_files.txt', 'w', encoding='utf-8') as foutput:
        for f in l_test_files:
            #Fetches text
            with open(data_path + f, 'r', encoding='utf-8') as finput_txt:
                data_txt=finput_txt.read().replace('\n',' ')
    
            l_codes_raw.append(['8000/0']) #placeholder label
            l_strs.append(data_txt)

            foutput.write(f + '\n')
            
    return l_strs, l_codes_raw

                
def write_test_set_results(file_list_path, out_path, l_pred_labs, dict_labels, eval_type):
    """Writes the algorithm prediction for the test set in the format required by the CANTEMIST competition.
    @Requires:
        file_list_path = path to the file that contains the list of files from the test and
                        background set in the order used to create the test_aval files.
                        This is required otherwise the predictions would not match the correct files.
        out_path = path to store the resulting file
        l_pred_labs = list of lists containing the predicted labels for each file
        dict_labels = dictionary containing the correspondence between the numeric identifiers
        and the ECIE-O codes.
        eval_type = type of evaluation focus used to make the predictions.
                    possible values: 'baseline', 'prec', 'rec' and 'f1'.
    @Ensures:
        The creation of a .tsv file containing the label predictions for each document
    """
    with open(file_list_path + 'list_files.txt', 'r', encoding='utf-8') as finput:
        l_files = finput.readlines()
    
    with open(out_path + 'test_set_preds_' + eval_type + '.tsv', 'w', newline='') as foutput:
        tsv_writer = csv.writer(foutput, delimiter='\t')
        tsv_writer.writerow(['file', 'code'])
        for i, j in zip(l_files, l_pred_labs):
            for z in j:
                l_code = dict_labels[str(z)][0] #Converts from numeric identifier to eCIE-O code  
                tsv_writer.writerow([i.replace('\n',''), l_code])