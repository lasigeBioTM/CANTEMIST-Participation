# -*- coding: utf-8 -*-
"""
Processes the predictions made by the X-Transformer model using the test and background sets
files and stores them in a file with the required format by the CANTEMIST CODING competition. 

@author: André
"""

import os
import argparse
import numpy as np
from os import listdir
import aval_utils as au
import proc_utils as pu

def main(args):
    files_path = args.data_path #'cantemist_data/'
    test_set_path = files_path + 'test-background-set-to-publish/'
    vocab_path =  args.vocab_path #'output/'
    npz_files = args.npz_path #'ranker_CANTEMIST_2_DeCS_Titles_MER/'
    out_path = args.output_path#'predictions_final/test_set/'
    
    if not os.path.exists(out_path): 
        print('Creating path %s' % out_path)
        os.mkdir(out_path)
        
    #Generates the dictionary of labels from the label correspondence file
    flabels = open(vocab_path + 'label_correspondence.txt', encoding='utf-8')
    labels = flabels.readlines()
    flabels.close()
        
    #Dict with the ECIE-O codes as keys and a reverse version with the 
    #numeric idenitifier as keys
    dict_labels, dict_labels_rev = {}, {}
    for i in range(len(labels)):
        dict_labels[labels[i].split('=')[1]] = (labels[i].split('=')[0], 
                   labels[i].split('=')[2].replace('\n',''))
        dict_labels_rev[labels[i].split('=')[0]] = (labels[i].split('=')[1], 
               labels[i].split('=')[2].replace('\n',''))
    
    #Reads dev data that filled part of the test files
    l_dev_txt, l_dev_labels = pu.read_files(files_path, 'dev1')
    l_dev_labels = pu.convert_labels(l_dev_labels, dict_labels)
    l_dev_labels = [list(map(int,i)) for i in l_dev_labels]
    
    #Creates list of all the .npz files in the given path
    #the list is being made manually since through sorted(listdir(npz_files)) the order 
    #was never the right one...
    l_npz_files = []
    for i in range(1,len(listdir(npz_files))-1):
        l_npz_files.append('tst_'+str(i)+'.pred.npz')
    
    #will contain the predictions for all files from the test and background sets using the 
    #threshold set for the different metrics: b = baseline, f = f1, p = precision, r = recall
    l_preds_test_b, l_preds_test_f, l_preds_test_p, l_preds_test_r = [], [], [], [] 
    
    
    #Reads the first .npz file in order to find the best threshold value for each measure
    #using the predictions made for the texts of the dev1 set that was used to fill the files.
    data = np.load(npz_files + l_npz_files[0])
    
    count = 0
    l_probs, l_aux = [], []
    for i, j in zip(data['indices'], data['data']):
        if count == 19: #number of predictions made by X-Transformer for each article
            l_probs.append(l_aux)
            l_aux = []
            count = 0
        else:
            l_aux.append((i,j))
            count += 1
    
    l_probs_dev1 = l_probs[109:]
    
    #Finds the best confidence threshold value that achieves the best score in the measures
    prob_baseline = 0
    prob_best_prec = au.check_prob_min(l_probs_dev1, l_dev_labels[0:141], 'prec')
    prob_best_rec = au.check_prob_min(l_probs_dev1, l_dev_labels[0:141], 'rec')
    prob_best_f1 = au.check_prob_min(l_probs_dev1, l_dev_labels[0:141], 'f1')
    
    #Using the previously calculated threshold scores, it iterates each .npz file 
    #and stores the predictions for the test-background set using those thresholds.
    print('Processing .npz files...')
    for f in l_npz_files:
        print(f) #DEBUG
        #loads npz prediction file and stores the predictions in list. 
        data = np.load(npz_files + f)
        
        count = 0
        l_probs, l_aux = [], []
        for i, j in zip(data['indices'], data['data']):
            if count == 19: #number of predictions made by X-Transformer for each article
                l_probs.append(l_aux)
                l_aux = []
                count = 0
            else:
                l_aux.append((i,j))
                count += 1
        
        l_probs_test = l_probs[0:109]
        
        #Stores the predictions for each test set file using the previously set threshold scores
        l_pred_labs = au.make_pred_list(l_probs_test, prob_baseline)
        for l in l_pred_labs:
            l_preds_test_b.append(l)
        
        l_pred_labs = au.make_pred_list(l_probs_test, prob_best_prec)
        for l in l_pred_labs:
            l_preds_test_p.append(l)
        
        l_pred_labs = au.make_pred_list(l_probs_test, prob_best_rec)    
        for l in l_pred_labs:
            l_preds_test_r.append(l)
            
        l_pred_labs = au.make_pred_list(l_probs_test, prob_best_f1)
        for l in l_pred_labs:
            l_preds_test_f.append(l)
    
    print('Writing files...')
    pu.write_test_set_results(test_set_path, out_path, l_preds_test_b, dict_labels_rev, 'baseline')
    pu.write_test_set_results(test_set_path, out_path, l_preds_test_p, dict_labels_rev, 'prec')
    pu.write_test_set_results(test_set_path, out_path, l_preds_test_r, dict_labels_rev, 'rec')
    pu.write_test_set_results(test_set_path, out_path, l_preds_test_f, dict_labels_rev, 'f1')



#Begin
parser = argparse.ArgumentParser()

#Required parameters
parser.add_argument('-npz', '--npz-path', type=str, required=True,
                    help='Path to the folder containing the .npz files with the X-Transformer predictions.')
parser.add_argument('-i1', '--data-path', type=str, required=True,
                    help='Path to the folder that contains the data given by the CANTEMIST Competition.')
parser.add_argument('-i2', '--vocab-path', type=str, required=True,
                    help='Path to the folder that contains the label_correspondence file')
parser.add_argument('-o', '--output-path', type=str, required=True,
                    help='Path to the directory that will contain the output prediction files')


args = parser.parse_args()
print(args)
main(args)
print('Finished processing')