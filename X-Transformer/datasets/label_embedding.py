#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import numpy as np
from sklearn.datasets import load_svmlight_file
import scipy.sparse as sp
import pickle
from sklearn.preprocessing import normalize
from tqdm import tqdm

import torch
from transformers import *


def main(args):
    if args.embed_type == "text-emb":
        label_text_list = [line.strip() for line in open("./{}/mapping/label_map.txt".format(args.dataset), "r")]
        n_label = len(label_text_list)

        # xlnet-large-cased tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        # model = XLNetModel.from_pretrained('xlnet-base-cased')
        # tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
        # model = BertModel.from_pretrained('bert-large-cased-whole-word-masking')
        #>>> ANEVES
        #tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        #model = RobertaModel.from_pretrained("roberta-large")
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = BertModel.from_pretrained("bert-base-multilingual-cased")
        #<<< ANEVES
        model = model.to(device)

        # get label embedding
        label_embedding = []
        for idx in tqdm(range(n_label)):
            inputs = torch.tensor([tokenizer.encode(label_text_list[idx])])
            inputs = inputs.to(device)
            with torch.no_grad():
                last_hidden_states = model(inputs)[0]  # [1, seq_len, hidden_dim]
                seq_embedding = last_hidden_states.mean(dim=1)
            label_embedding.append(seq_embedding)
        label_embedding = torch.cat(label_embedding, dim=0)
        label_embedding = label_embedding.cpu().numpy()
        label_embedding = sp.csr_matrix(label_embedding)
        label_embedding = normalize(label_embedding, axis=1, norm="l2")

    elif args.embed_type == "pifa":
        # load TF-IDF and label matrix
        X = sp.load_npz("./{}/X.trn.npz".format(args.dataset))
        Y = sp.load_npz("./{}/Y.trn.npz".format(args.dataset))
        assert Y.getformat() == "csr"
        print("X", type(X), X.shape)
        print("Y", type(Y), Y.shape)
        # create label embedding
        Y_avg = normalize(Y, axis=1, norm="l2")
        label_embedding = sp.csr_matrix(Y_avg.T.dot(X))
        label_embedding = normalize(label_embedding, axis=1, norm="l2")

    elif args.embed_type == "pifa-neural":
        # load neural embedding from matcher
        X_trn = np.load(args.trn_embedding_npy)
        Y_trn = sp.load_npz("./{}/Y.trn.npz".format(args.dataset))
        print("X_trn", type(X_trn), X_trn.shape)
        print("Y_trn", type(Y_trn), Y_trn.shape)
        Y_avg = normalize(Y_trn, axis=1, norm="l2")
        label_embedding = sp.csr_matrix(Y_avg.T.dot(X_trn))
        label_embedding = normalize(label_embedding, axis=1, norm="l2")

    else:
        raise NotImplementedError("unknown embed_type {}".format(args.embed_type))

    # save label embedding
    print("label_embedding", type(label_embedding), label_embedding.shape)
    label_embedding_path = "{}/L.{}.npz".format(args.dataset, args.embed_type)
    sp.save_npz(label_embedding_path, label_embedding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="dataset name: [ Eurlex-4K | Wiki10-31K | AmazonCat-13K | Wiki-500K ]"
    )
    parser.add_argument(
        "-e", "--embed-type", type=str, required=True, help="label embedding type: [ pifa | pifa-tst | pifa-neural | text-emb"
    )
    parser.add_argument("-x1", "--trn_embedding_npy", type=str, default=None, help="train embedding extracted from neural matcher")
    parser.add_argument("-x2", "--tst_embedding_npy", type=str, default=None, help="test embedding extracted from neural matcher")
    args = parser.parse_args()
    print(args)
    main(args)
