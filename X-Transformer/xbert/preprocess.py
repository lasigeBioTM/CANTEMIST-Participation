#!/usr/bin/env python
# encoding: utf-8

import argparse
from collections import Counter
import itertools
import json
import os
from os import path
import logging
import numpy as np
import pickle
import time
from tqdm import tqdm
import scipy as sp
import scipy.sparse as smat

from xbert.data_utils import InputExample, InputFeatures

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig, DistilBertConfig,)),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer,),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
}


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Dataset is a set of tuple:
# D = {(x, c, y)_i}, i=1,...,N
# xseq is a list of string
# cseq is a list of list (cluster_ids)
# yseq is a list of list (label_ids)
def load_text_data(text_path, Y, csr_codes):
    xseq_list, cseq_list, yseq_list = [], [], []
    with open(text_path, "r") as fin:
        for idx, line in enumerate(tqdm(fin)):
            xseq = line.strip()
            if len(xseq) == 0:
                logger.info("WARNING: line {} has empty text".format(idx))
                xseq = ""
            # xseq
            xseq_list.append(xseq)
            # yseq
            yseq = Y.indices[Y.indptr[idx] : Y.indptr[idx + 1]]
            yseq_list.append(list(yseq))
            # cseq
            cseq = [csr_codes[y] for y in yseq]
            cseq_list.append(cseq)

    return xseq_list, cseq_list, yseq_list


# self-defined helper functions for preprocessing
# matcher=xbert
def create_examples(xseq_list, cseq_list, set_type):
    """Creates examples for the training and dev sets."""

    examples = []
    for i, (xseq, cseq) in enumerate(zip(xseq_list, cseq_list)):
        guid = "%s-%s" % (set_type, i)
        examples.append(InputExample(guid=guid, text=xseq, label=cseq))
    return examples


def convert_examples_to_features(
    examples,
    tokenizer,
    max_xseq_len=512,
    max_cseq_len=128,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):

    """Loads a data file into a list of `InputBatch`s."""
    features = []
    xseq_lens, cseq_lens = [], []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d" % (ex_index))

        # truncate long text by 8192 chars as they will exceed max_seq_len anyway
        inputs = tokenizer.encode_plus(text=example.text[:8192], text_pair=None, add_special_tokens=True, max_length=max_xseq_len,)

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        xseq_lens.append(len(input_ids))

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_xseq_len - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_xseq_len, "Error with input length {} vs {}".format(len(input_ids), max_xseq_len)
        assert len(attention_mask) == max_xseq_len, "Error with input length {} vs {}".format(len(attention_mask), max_xseq_len)
        assert len(token_type_ids) == max_xseq_len, "Error with input length {} vs {}".format(len(token_type_ids), max_xseq_len)

        # labels
        labels = example.label
        cseq_lens.append(len(labels))
        if len(labels) > max_cseq_len:
            labels = labels[:max_cseq_len]
        output_ids = labels
        output_mask = [1] * len(output_ids)
        padding = [0] * (max_cseq_len - len(output_ids))
        output_ids += padding
        output_mask += padding

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("output_ids: %s" % " ".join([str(x) for x in output_ids]))
            logger.info("output_mask: %s" % " ".join([str(x) for x in output_mask]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_ids=output_ids,
                output_mask=output_mask,
            )
        )
    # end for loop
    return features, xseq_lens, cseq_lens


def main(args):
    # set hyper-parameters
    input_data_dir = args.input_data_dir
    input_code_path = args.input_code_path
    output_data_dir = args.output_data_dir

    # load existing code
    C = smat.load_npz(input_code_path)
    csr_codes = C.nonzero()[1]

    # load label matrix
    trn_label_path = os.path.join(args.input_data_dir, "Y.trn.npz")
    tst_label_path = os.path.join(args.input_data_dir, "Y.tst.npz")
    Y_trn = smat.load_npz(trn_label_path)
    Y_tst = smat.load_npz(tst_label_path)
    assert Y_trn.shape[1] == C.shape[0]

    # a short glimpse at clustering result for fast debugging
    logger.info("NUM_LABELS: {}".format(C.shape[0]))
    logger.info("NUM_CLUSTERS: {}".format(C.shape[1]))

    # load data (text and label_ids)
    logger.info("loading data into quadruple set")
    trn_text_path = os.path.join(args.input_data_dir, "train_raw_texts.txt")
    tst_text_path = os.path.join(args.input_data_dir, "test_raw_texts.txt")
    trn_xseq_list, trn_cseq_list, trn_yseq_list = load_text_data(trn_text_path, Y_trn, csr_codes)
    tst_xseq_list, tst_cseq_list, tst_yseq_list = load_text_data(tst_text_path, Y_tst, csr_codes)

    # load pretrained model tokenizers
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    trn_examples = create_examples(trn_xseq_list, trn_cseq_list, "trn")
    tst_examples = create_examples(tst_xseq_list, tst_cseq_list, "tst")

    # create train set features
    trn_features, xseq_lens, cseq_lens = convert_examples_to_features(
        trn_examples,
        tokenizer,
        args.max_xseq_len,
        args.max_cseq_len,
        pad_on_left=bool(args.model_type in ["xlnet"]),
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )
    logger.info(
        "trn_xseq: min={} max={} mean={} median={}".format(np.min(xseq_lens), np.max(xseq_lens), np.mean(xseq_lens), np.median(xseq_lens),)
    )
    logger.info(
        "trn_cseq: min={} max={} mean={} median={}".format(np.min(cseq_lens), np.max(cseq_lens), np.mean(cseq_lens), np.median(cseq_lens),)
    )

    # create ttest set features
    tst_features, xseq_lens, cseq_lens = convert_examples_to_features(
        tst_examples,
        tokenizer,
        args.max_xseq_len,
        args.max_cseq_len,
        pad_on_left=bool(args.model_type in ["xlnet"]),
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )
    logger.info(
        "tst_xseq: min={} max={} mean={} median={}".format(np.min(xseq_lens), np.max(xseq_lens), np.mean(xseq_lens), np.median(xseq_lens),)
    )
    logger.info(
        "tst_cseq: min={} max={} mean={} median={}".format(np.min(cseq_lens), np.max(cseq_lens), np.mean(cseq_lens), np.median(cseq_lens),)
    )

    # save data dict
    data = {
        "args": args,
        "C": C,
        "trn": {"cseq": trn_cseq_list, "yseq": trn_yseq_list},
        "tst": {"cseq": tst_cseq_list, "yseq": tst_yseq_list},
        "tokenizer": tokenizer,
        "trn_features": trn_features,
        "tst_features": tst_features,
    }

    logger.info("Dumping the processed data to pickle file")
    output_data_path = path.join(output_data_dir, "data_dict.pt")
    with open(output_data_path, "wb") as fout:
        pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)
    output_config_path = path.join(output_data_dir, "config.json")
    with open(output_config_path, "w") as fout:
        json.dump(vars(args), fout)
    logger.info("Finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument(
        "-m", "--model-type", type=str, required=True, default="bert", help="preprocess for model-type [bert | xlnet | xlm | roberta]",
    )
    parser.add_argument(
        "-n",
        "--model_name_or_path",
        type=str,
        required=True,
        default="bert-base-uncased",
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "-i",
        "--input-data-dir",
        type=str,
        required=True,
        metavar="DIR",
        default="./datasets/Eurlex-4K",
        help="path to the dataset directory containing mls2seq/",
    )
    parser.add_argument(
        "-c",
        "--input-code-path",
        type=str,
        required=True,
        metavar="PATH",
        default="./save_models/Eurlex-4K/indexer/code.npz",
        help="path to the npz file of the indexing codes (CSR, nr_labels * nr_codes)",
    )
    parser.add_argument(
        "-o",
        "--output-data-dir",
        type=str,
        required=True,
        metavar="DIR",
        default="./save_models/Eurlex-4K/elmo-a0-s0/data-data-xbert",
        help="directory for storing data_dict.pkl",
    )
    ## Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--max_xseq_len",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--max_cseq_len",
        default=32,
        type=int,
        help="The maximum total output sequence length. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    args = parser.parse_args()
    print(args)
    main(args)
