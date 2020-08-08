# -*- coding: utf-8 -*-
"""
Processes the predictions made by the X-Transformer model and stores them in 
a file with the required format by the CANTEMIST CODING competition. 

@author: André Neves
"""

import os
import argparse
import numpy as np
import aval_utils as au
import proc_utils as pu

def main(args):
    files_path = args.files_path #'cantemist_new/dev-set2-to-publish/cantemist-coding/txt/'    
    vocab_path = args.vocab_path #'output/'
    npz_file = args.npz_file #'X-Transformer_results/tst_CANTEMIST.pred.npz'
    out_path = args.output_path #'predictions_final/'
    
    if not os.path.exists(out_path): 
        print('Creating path %s' % out_path)
        os.mkdir(out_path)
        
    flabels = open(vocab_path + 'label_correspondence.txt', encoding='utf-8')
    labels = flabels.readlines()
    flabels.close()
    
    ftest = open(vocab_path + 'test.txt', encoding='utf-8')
    test_lines = ftest.readlines()
    ftest.close()
    
    #Stores test file labels
    l_test_labs = []
    for i in range(len(test_lines)):
        l_aux = test_lines[i].split('\t')
        l_test_labs.append(l_aux[0].split(','))
        l_test_labs = [list(map(int,i)) for i in l_test_labs]
        test_lines[i] = l_aux[1] #Only the txts stay on the list
    
    #Removes possible duplicates
    for i in range(len(l_test_labs)):
        l_test_labs[i] = list(dict.fromkeys(l_test_labs[i]))
    
    #Generates dict with numeric identifiers as key
    dict_labels = {}
    for i in range(len(labels)):
        dict_labels[labels[i].split('=')[0]] = (labels[i].split('=')[1], 
                   labels[i].split('=')[2].replace('\n',''))
    
    #loads npz prediction file and stores the predictions in list. 
    data = np.load(npz_file)
    
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
    
    #Finds the best confidence threshold value to achieve the best score in the measures
    prob_baseline = 0
    prob_best_prec = au.check_prob_min(l_probs, l_test_labs, 'prec')
    prob_best_rec = au.check_prob_min(l_probs, l_test_labs, 'rec')
    prob_best_f1 = au.check_prob_min(l_probs, l_test_labs, 'f1')
    
    #Writes the predictions with a confidence value >= to the confidence threshold
    l_pred_labs = au.make_pred_list(l_probs, prob_baseline)
    pu.write_results(files_path, out_path, l_pred_labs, dict_labels, 'baseline')
        
    l_pred_labs = au.make_pred_list(l_probs, prob_best_prec)
    pu.write_results(files_path, out_path, l_pred_labs, dict_labels, 'prec')
    
    l_pred_labs = au.make_pred_list(l_probs, prob_best_rec)
    pu.write_results(files_path, out_path, l_pred_labs, dict_labels, 'rec')
    
    l_pred_labs = au.make_pred_list(l_probs, prob_best_f1)
    pu.write_results(files_path, out_path, l_pred_labs, dict_labels, 'f1')


#Begin
parser = argparse.ArgumentParser()

#Required parameters
parser.add_argument('-npz', '--npz-file', type=str, required=True,
                    help='Path to the .npz file containing the X-Transformer predictions.')
parser.add_argument('-i1', '--files-path', type=str, required=True,
                    help='Path to the folder that contains the test set text files given by the CANTEMIST Competition.')
parser.add_argument('-i2', '--vocab-path', type=str, required=True,
                    help='Path to the folder that contains the label_correspondence file and the test.txt file used by X-Transformer')
parser.add_argument('-o', '--output-path', type=str, required=True,
                    help='Path to the directory that will contain the output prediction files')

args = parser.parse_args()
print(args)
main(args)
print('Finished processing')