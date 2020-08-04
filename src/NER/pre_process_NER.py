import json
import os
import re
import sys
import time
from flair.data import Sentence
from segtok.segmenter import split_single, split_multi

sys.path.append('./')


def cantemist_to_IOB2(subset):
    """Convert the CANTEMIST dataset into IOB2 FORMAT. Example of sentence with an annotation:
        
            Original text: "Carcinoma microcítico de pulmón."
            
            Converted text:"Carcinoma   B-MORF_NEO
                            microcítico I-MORF_NEO
                            de  O
                            pulmón  O
                            .   O"
        Requires:
            subset - str indicating the subset of the CANTEMIST dataset to convert
        Ensures: 
            A .txt file with the text of all documents in the subset tokenized and tagged according with the IOB2 schema
    """

    if subset == "train":
        cantemist_data_dir = './data/datasets/train-set-to-publish/cantemist-ner/'
    
    elif subset == "dev":
        cantemist_data_dir = './data/datasets/dev-set1-to-publish/cantemist-ner/'

    elif subset == "test":
        cantemist_data_dir = './data/datasets/dev-set2-to-publish/cantemist-ner/'

    total_doc_count = int(len(os.listdir(cantemist_data_dir))/2)
    doc_count = int()
    output = str()

    for doc in os.listdir(cantemist_data_dir):
        doc_count += 1

        if doc[-3:] == "txt":
            
            doc_annotations = dict()
            text = str()   
            
            with open(cantemist_data_dir+doc, 'r') as corpus_file:
                text = corpus_file.read()
                corpus_file.close()      
            
            # Import the annotations from annotations file into dict if train or dev sets
            annotations_filename = doc[:-3] + "ann"
            doc_annotations_begin, doc_annotations_end = list(), list()

            with open(cantemist_data_dir+annotations_filename, 'r') as annotations_file:
                annotations = annotations_file.readlines()
                annotations_file.close()

                for annotation in annotations:
                    begin_position = int(annotation.split("\t")[1].split(" ")[1])
                    end_position = int(annotation.split("\t")[1].split(" ")[2])
                    annotation_str = annotation.split("\t")[2]#.split(" ")[0]
                    doc_annotations_begin.append(begin_position)
                    doc_annotations_end.append(end_position)
                
            # Tag all annotations in text in the respective correct positions
            for begin in doc_annotations_begin:
                text = text[:begin-1] + "@" + text[begin:]
            
            for end in doc_annotations_end:
                text = text[:end] + "$" + text[end+1:]
        
            # Sentence segmentation followed by tokenization of each sentence
            sentences = [Sentence(sent, use_tokenizer=True) for sent in split_single(text)]
            next_token_is_annotation = False
            
            for sentence in sentences:
                
                for token in sentence: 
                    token_text = str(token).split(" ")[2]
                    
                    if len(token_text) > 0:
                        
                        if  "@" in token_text: # Next token belongs to an annotation
                            
                            next_token_is_annotation = True
                            previous_token_is_begin = True

                            if token_text != "@": # For example "»@"
                                temp = str(token_text).strip("@")

                                if temp != "$": # Sometimes a token can be '$@'
                                    output += temp + "\tO" + "\n"

                        elif "$" in token_text: # End of the annotation
                            temp = str(token_text).strip("$")

                            if temp != "@":
                                next_token_is_annotation = False
                            
                                if token_text != "$": 
                                    output +=  temp + "\tO" + "\n"
                        
                        else:
                            if next_token_is_annotation:
                                
                                if previous_token_is_begin: # The first token of the annotation
                                    output += str(token_text) + "\tB-MOR_NEO" + "\n"
                                    previous_token_is_begin = False
                                
                                else: # The intermediate tokens of the annotation
                                    output += str(token_text) + "\tI-MOR_NEO" + "\n"

                            else: # Tokens outside any annotation
                                output += str(token_text) + "\tO" + "\n"
                                next_token_is_annotation = False
                    
                output += "\n" # To add a separator between each sentence
    
    output = output[:-2] # To delete empty lines in the final of the doc

    #Create a file with the text of all documents tokenized and tagged
    if not os.path.exists("./data/datasets/iob2/"):
        os.makedirs("./data/datasets/iob2/")

    with open("./data/datasets/iob2/"+ subset + ".txt", 'w') as output_file:
        output_file.write(output)
        output_file.close()

cantemist_to_IOB2('test')


def pre_process_mesinesp_subset(file_dir):
    """Prepare MESINESP dataset for training the FLAIR embeddings."""

    with open(file_dir, 'r') as json_file:
        data = json.load(json_file)
        json_file.close()
    
    output = str()
    output_list = list()
    article_count = int()
    temp_article_count = int()
    excluded_articles_count = int()
    
    for article in data:
        
        try:
            article_count += 1
            temp_article_count += 1
            es_title = article['title_es']
            es_title = es_title.strip(" [")
            es_title = es_title.strip(".]")
            es_abstract = article['abstractText']['ab_es']
            output += es_title + "\n"
            output += es_abstract + "\n\n"

            if temp_article_count == 50: # Each split will only contain 50 article max
                output_list.append(output)
                temp_article_count = 0 
                output = "" # Reset the output string
        
        except:
            excluded_articles_count += 1
            continue
    
    print(str(excluded_articles_count), "articles excluded!")
    
    return output_list, article_count


def prepare_mesinesp_for_flair_embeds_training():
    """Prepare all necessary files to train FLAIR embeddings over the MESINESP dataset.
       Refer to https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md#preparing-a-text-corpus
       and https://github.com/flairNLP/flair/issues/80
    """
    start_time = time.time()

    #dataset_dir = "../../../mnt/data/pubmed_spanish/jsons/" #Change if merged_1.json is in different dir
    dataset_dir = "./data/datasets/"
    mesinesp_splits = ["merged_1.json"]#, "merged_0.json", "merged_2.json", "merged_3.json", "merged_4.json", "merged_5.json", "merged_6.json", "merged_7.json", "merged_9.json"]
    
    output_dir = "./data/datasets/mesinesp/"
    
    if not os.path.exists("./data/datasets/mesinesp/mesinesp_1"):
        os.makedirs("./data/datasets/mesinesp/mesinesp_1")
    
    if not os.path.exists("./data/datasets/mesinesp/mesinesp_2"):
        os.makedirs("./data/datasets/mesinesp/mesinesp_2")

    if not os.path.exists("./data/datasets/mesinesp/mesinesp_3"):
        os.makedirs("./data/datasets/mesinesp/mesinesp_3")
    
    if not os.path.exists("./data/datasets/mesinesp/mesinesp_4"):
        os.makedirs("./data/datasets/mesinesp/mesinesp_4")

    total_article_count, train_split_count, dev_split_count, test_split_count, token_count, split_count = int(), int(), int(), int(), int(), int()
   
    for part in mesinesp_splits:
        
        print("Pre-processing ", part, "...")
        file_dir = dataset_dir + part
        output_list, article_count = pre_process_mesinesp_subset(file_dir)        
        total_article_count += article_count

        mesinesp_1_token_count, mesinesp_1_train_token_count, mes_1_train_count = int(), int(), int()
        mesinesp_2_token_count, mesinesp_2_train_token_count, mes_2_train_count = int(), int(), int()
        mesinesp_3_token_count, mesinesp_3_train_token_count, mes_3_train_count = int(), int(), int()
        mesinesp_4_token_count, mesinesp_4_train_token_count, mes_4_train_count = int(), int(), int()

        test_output_1, valid_output_1 = str(), str()
        test_output_2, valid_output_2 = str(), str()
        test_output_3, valid_output_3 = str(), str()
        test_output_4, valid_output_4 = str(), str()
        
        for split in output_list:
            split_count += 1
    
            if split_count <= 650: # mesinesp_1 subcorpus: each split contains 50 articles, so mesinesp_1 contains 50*650
                
                mesinesp_1_token_count += len(split)

                if split_count <= 25:
                    test_output_1 += split

                elif 25 < split_count < 50: 
                    valid_output_1 += split

                elif 50 <split_count:
                    mes_1_train_count += 1
                    mesinesp_1_train_token_count += len(split)

                    train1_output_file = "train1_output_" + str(mes_1_train_count)

                    with open(output_dir + "mesinesp_1/train/train_split_" + str(mes_1_train_count) + ".txt", 'w') as train_output_file:
                        train_output_file.write(split)
                        train_output_file.close

            elif 650 < split_count <= 1300: # "mesinesp_2" subcorpus
                mesinesp_2_token_count += len(split)
                
                if split_count <= 675:
                    test_output_2 += split

                elif 675 < split_count < 700: 
                    valid_output_2 += split
                
                elif 700 <split_count:
                    mes_2_train_count += 1
                    mesinesp_2_train_token_count += len(split)
                    train2_output_file = "train2_output_" + str(mes_2_train_count)

                    with open(output_dir + "mesinesp_2/train/train_split_" + str(mes_2_train_count) + ".txt", 'w') as train2_output_file:
                        train2_output_file.write(split)
                        train2_output_file.close

            elif 1300 < split_count <= 1950: # mesinesp_3 sub_corpus 
                mesinesp_3_token_count += len(split)
                
                if split_count <= 1325:
                    test_output_3 += split

                elif 1325 < split_count < 1350:
                    valid_output_3 += split

                elif 1350 <split_count: 
                    mes_3_train_count += 1
                    mesinesp_3_train_token_count += len(split)
                    train3_output_file = "train3_output_" + str(mes_3_train_count)

                    with open(output_dir + "mesinesp_3/train/train_split_" + str(mes_3_train_count) + ".txt", 'w') as train3_output_file:
                        train3_output_file.write(split)
                        train3_output_file.close
                
            elif 1950 < split_count <= 2600: # mesinesp_4 subcorpus
                mesinesp_4_token_count += len(split)

                if split_count <= 1975:
                    test_output_4 += split
                    
                elif 1975 < split_count < 2000: 
                    valid_output_4 += split

                elif 2000 <split_count: 
                    mes_4_train_count += 1
                    mesinesp_4_train_token_count += len(split)
                    train4_output_file = "train4_output_" + str(mes_4_train_count)

                    with open(output_dir + "mesinesp_4/train/train_split_" + str(mes_4_train_count) + ".txt", 'w') as train4_output_file:
                        train4_output_file.write(split)
                        train4_output_file.close

    #Create test and valid files: 10% and 10% of the articles
    with open(output_dir + "mesinesp_1/test.txt", 'w') as test_file_1:
        test_file_1.write(test_output_1)
        test_file_1.close
    
    with open(output_dir + "mesinesp_1/valid.txt", 'w') as valid_file_1:
        valid_file_1.write(valid_output_1)
        valid_file_1.close

    with open(output_dir + "mesinesp_2/test.txt", 'w') as test_file_2:
        test_file_2.write(test_output_2)
        test_file_2.close

    with open(output_dir + "mesinesp_2/valid.txt", 'w') as valid_file_2:
        valid_file_2.write(valid_output_2)
        valid_file_2.close

    with open(output_dir + "mesinesp_3/test.txt", 'w') as test_file_3:
        test_file_3.write(test_output_3)
        test_file_3.close
    
    with open(output_dir + "mesinesp_3/valid.txt", 'w') as valid_file_3:
        valid_file_3.write(valid_output_3)
        valid_file_3.close

    with open(output_dir + "mesinesp_4/test.txt", 'w') as test_file_4:
        test_file_4.write(test_output_4)
        test_file_4.close

    with open(output_dir + "mesinesp_4/valid.txt", 'w') as valid_file_4:
        valid_file_4.write(valid_output_4)
        valid_file_4.close

    total_token_count = mesinesp_1_token_count + mesinesp_2_token_count + mesinesp_3_token_count + mesinesp_4_token_count
    total_train_token_count = mesinesp_1_train_token_count + mesinesp_2_train_token_count + mesinesp_3_train_token_count + mesinesp_4_train_token_count
            
    print("Total articles pre-processed: ", str(total_article_count))
    print("mesinesp_1_token_count: ", str(mesinesp_1_token_count)) 
    print("mesinesp_1_train_token_count: ", str(mesinesp_1_train_token_count)) 
    print("mesinesp_2_token_count: ", str(mesinesp_2_token_count))
    print("mesinesp_2_train_token_count: ", str(mesinesp_2_train_token_count))
    print("mesinesp_3_token_count: ", str(mesinesp_3_token_count)) 
    print("mesinesp_3_train_token_count: ", str(mesinesp_3_train_token_count))
    print("mesinesp_4_token_count: ", str(mesinesp_4_token_count))
    print("mesinesp_4_train_token_count: ", str(mesinesp_4_train_token_count)) 
    print("Total token count: ", str(total_token_count))
    print("Total training token count: ", str(total_train_token_count))
    print("Total time (aprox.): ", int((time.time() - start_time)/60.0), "minutes\n----------------------------------")




