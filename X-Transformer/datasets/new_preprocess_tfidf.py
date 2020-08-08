# -*- coding: utf-8 -*-
#!/usr/bin/env python3 -u

import argparse
import os, sys
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
import pickle
import scipy.sparse as sp
from xbert.rf_util import smat_util

# mlc2seq format (each line):
# l_1,..,l_k \t w_1 w_2 ... w_t
def parse_mlc2seq_format(data_path):
    print(data_path)
    assert(os.path.isfile(data_path))
    #>>> ANEVES
    #with open(data_path) as fin:
    with open(data_path, 'r', encoding='utf-8') as fin:
    #<<< ANEVES
        lines = fin.readlines()
        labels, corpus = [], []
        #for line in fin:
        i=0
        for line in lines:
            #line = line.encode('utf-8')
            #print(line)
            tmp = line.strip().split('\t', 1)
            #print(tmp[0])
            #print(tmp[1])
            print(i)
            i+=1
            labels.append(tmp[0])
            corpus.append(tmp[1])            
    return labels, corpus


def convert_feat(X, feat_type=None):
    if feat_type == 0:
        X_ret = X.copy()
        X_ret[X_ret>0] = 1
    elif feat_type == 1:
        X_ret = X.copy()
    else:
        raise NotImplementedError('unknown feature_type!')
    #>>> ANEVES
    return X_ret
    #<<< ANEVES
            
# convert list of label strings into list of list of label
# label index start from 0
#>>> ANEVES
#def convert_label_to_Y(labels, K_in=None):
def convert_label_to_Y(labels, num_labels, K_in=None):
#<<< ANEVES
    rows, cols ,vals = [], [], []
    for i, label in enumerate(labels):
        label_list = list(map(int, label.split(',')))
        rows += [i] * len(label_list)
        cols += label_list
        vals += [1.0] * len(label_list)

    #>>> ANEVES
    print("DEBUG")
    print("rows:", len(rows))
    print("cols:", len(cols))
    print("max cols:", max(cols))
    print("num_labels:", num_labels)
    
    if K_in is None:
        if num_labels >= max(cols):
            #K_out = num_labels - 1 #to remove the empty line in the end
            K_out = num_labels
        else:
            K_out = max(cols) + 1
    else:
        K_out = K_in
        
    #K_out = max(cols) + 1 if K_in is None else K_in
    print("K_out:", K_out)        
    #<<< ANEVES
    
    Y = sp.csr_matrix( (vals, (rows,cols)), shape=(len(labels),K_out) )
    return Y
        
# save X and Y
def save_npz(X, Y, set_flag='trn'):
    X.sort_indices()
    sp.save_npz('{}/X.{}.npz'.format(args.input_data_dir, set_flag), X)
    sp.save_npz('{}/Y.{}.npz'.format(args.input_data_dir, set_flag), Y)

### MAIN ###
def main(args):
    # load data
    print('loading corpus and labels...')
    #trn_labels, trn_corpus = parse_mlc2seq_format('{}/mlc2seq/train.txt'.format(args.input_data_dir))
    #tst_labels, tst_corpus = parse_mlc2seq_format('{}/mlc2seq/test.txt'.format(args.input_data_dir))
    trn_labels, trn_corpus = parse_mlc2seq_format('{}/train.txt'.format(args.input_data_dir))
    tst_labels, tst_corpus = parse_mlc2seq_format('{}/test.txt'.format(args.input_data_dir))
    
    #>>> ANEVES
    #with open('{}/mlc2seq/label_vocab.txt'.format(args.input_data_dir), 'r', encoding='utf-8') as f:
    #    n_labels = len(f.readlines())
    with open('{}/label_vocab.txt'.format(args.input_data_dir), 'r', encoding='utf-8') as f:
        n_labels = len(f.readlines())
    #<<< ANEVES
    
    # create features
    print('creating features...')
    if args.feat_type == 0 or args.feat_type == 1:
        vectorizer = CountVectorizer(
                ngram_range=(1, args.ngram),
                min_df=args.min_df,)

        X_wordcount_trn = vectorizer.fit_transform(trn_corpus)
        X_wordcount_tst = vectorizer.transform(tst_corpus)
            
        X_ret_trn = convert_feat(X_wordcount_trn, feat_type=args.feat_type)
        X_ret_tst = convert_feat(X_wordcount_tst, feat_type=args.feat_type)

    elif args.feat_type == 2:
        vectorizer = TfidfVectorizer(
                ngram_range=(1, args.ngram),
                min_df=args.min_df,)
        X_ret_trn = vectorizer.fit_transform(trn_corpus)
        X_ret_tst = vectorizer.transform(tst_corpus)

    else:
        raise NotImplementedError('unknown feature_type!')

    # save vectorizer as pickle
    print('saving vectorized features into libsvm format and dictionary into pickle...')
    out_file_name = '{}/vectorizer.pkl'.format(args.input_data_dir)
    with open(out_file_name, 'wb') as fout:
        pickle.dump(vectorizer, fout, protocol=pickle.HIGHEST_PROTOCOL)

	# write file back into libsvm format
    #>>> ANEVES
    #Y_ret_trn = convert_label_to_Y(trn_labels, K_in=None)
    Y_ret_trn = convert_label_to_Y(trn_labels, n_labels, K_in=None)
    #<<< ANEVES
    #save_npz(X_ret_trn, Y_ret_trn, set_flag='trn')

    #>>> ANEVES
    #Y_ret_tst = convert_label_to_Y(tst_labels, K_in=Y_ret_trn.shape[1])
    Y_ret_tst = convert_label_to_Y(tst_labels, n_labels, K_in=Y_ret_trn.shape[1])
    #<<< ANEVES
    #save_npz(X_ret_tst, Y_ret_tst, set_flag='tst')
    
    #>>> ANEVES
    print('saving txt files in libsvm format...')
    X_ret_trn.sort_indices()
    X_ret_tst.sort_indices()
    dump_svmlight_file(X_ret_trn, Y_ret_trn, '{}/train.txt'.format(args.input_data_dir), multilabel=True)
    dump_svmlight_file(X_ret_tst, Y_ret_tst, '{}/test.txt'.format(args.input_data_dir), multilabel=True)
    #<<< ANEVES

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='read [train|test].txt and produce n-gram tfidf features')
    parser.add_argument('-i', '--input-data-dir', type=str, required=True, help='dataset directory: data_dir/*.txt')
    parser.add_argument('--feat-type', type=int, default=2, help='0: binary feature, 1: word count, 2: TF-IDF')
    parser.add_argument('--ngram', type=int, default=1, help='ngram features')
    parser.add_argument('--min-df', type=int, default=1, help='keep words whose df >= min_df (absolute count)')
    args = parser.parse_args()
    print(args)
    main(args)
