import json
import merpy
import sys
import os
from cieo3 import load_cieo3
from es_decs import load_es_decs
from flair.data import Sentence
from icd10cm import load_spanish_icd10cm
from segtok.segmenter import split_single, split_multi

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append("./")


def build_relations_dict():
    """Iterates over all sentences in train, dev sets recognizes CIE-O-3, ICD10-CM and DeCS entities and establish a relation
        between two entities in a given sentece

    Ensures:
        dict stored in file './tmp/relations_cieo3_icd10cm.json' ES ICD10-CM <-> CIEO3 relations and in file 
        './tmp/relations_cieo3_esdecs.json' with ES DeCS <-> CIEO3 relations
    """
    
    #Create CIE-O-3 lexicon 
    lexicon_name = "cieo3"
    ontology_graph, name_to_id, synonym_to_id = load_cieo3()
    merpy.create_lexicon(name_to_id.keys(), lexicon_name)
    merpy.create_mappings(name_to_id, lexicon_name)
    merpy.process_lexicon(lexicon_name)

    #Create ICD10-CM lexicon
    lexicon_name = "icd10cmes"
    ontology_graph, name_to_id = load_spanish_icd10cm()
    merpy.create_lexicon(name_to_id.keys(), lexicon_name)
    merpy.create_mappings(name_to_id, lexicon_name)
    merpy.process_lexicon(lexicon_name)

    #Create DECS lexicon
    lexicon_name = "es_decs"
    ontology_graph, name_to_id, synonym_to_id = load_es_decs()
    merpy.create_lexicon(name_to_id.keys(), lexicon_name)
    merpy.create_mappings(name_to_id, lexicon_name)
    merpy.process_lexicon(lexicon_name)
    
    filenames_1 = ["./data/datasets/train-set-to-publish/cantemist-norm/"+input_file for input_file in os.listdir("./data/datasets/train-set-to-publish/cantemist-norm/")]
    filenames_2 = ["./data/datasets/dev-set1-to-publish/cantemist-norm/"+input_file for input_file in os.listdir("./data/datasets/dev-set1-to-publish/cantemist-norm/")]
    filenames_3 = ["./data/datasets/dev-set2-to-publish/cantemist-norm/"+input_file for input_file in os.listdir("./data/datasets/dev-set2-to-publish/cantemist-norm/")]
    filenames_4 = ["./data/datasets/test-background-set-to-publish/"+input_file for input_file in os.listdir("./data/datasets/test-background-set-to-publish/")]

    filenames = filenames_1 + filenames_2 + filenames_3# + filenames_4
    
    relations_1, relations_2 = dict(), dict()
    doc_count = int()
    
    for doc in filenames:
        
        if doc[-3:] == "txt":
        #if doc == "cc_onco1016.txt":
            doc_count += 1
            print("DOC_COUNT:", doc_count)
            with open(doc, 'r') as doc_file:
                text = doc_file.read()
                doc_file.close() 

            sentences = [Sentence(sent) for sent in split_single(text)]
            
            for sentence in sentences:
                sent_text = sentence.to_original_text()
                cieo3_entities = merpy.get_entities(sent_text, "cieo3")
                icd10cm_entities = merpy.get_entities(sent_text, "icd10cmes")
                es_decs_entities = merpy.get_entities(sent_text, "es_decs")
                
                if icd10cm_entities != [['']] and cieo3_entities != [['']]:
                    icd10cm_codes = [entity[3] for entity in icd10cm_entities]
                    cieo3_codes = [entity[3] for entity in cieo3_entities]
                    
                    for code in cieo3_codes:
                        
                        if code in relations_1:
                            current_values = relations_1[code]
                            current_values.extend(icd10cm_codes)
                            relations_1[code] = current_values
                        else:
                            relations_1[code] = icd10cm_codes
                                   
                if es_decs_entities != [['']] and cieo3_entities != [['']]:    
                    es_decs_codes = [entity[3] for entity in es_decs_entities]
                    cieo3_codes = [entity[3] for entity in cieo3_entities]
                  
                    for code in cieo3_codes:
                        
                        if code in relations_2:
                            current_values = relations_2[code]
                            current_values.extend(es_decs_codes)
                            relations_2[code] = es_decs_codes
                        else:
                            relations_2[code] = es_decs_codes
    
    #Output the relations into json files              
    d = json.dumps(relations_1)
    d_file = open("./tmp/relations_cieo3_icd10cm.json", 'w')
    d_file.write(d)
    d_file.close()

    b = json.dumps(relations_2)
    b_file = open("./tmp/relations_cieo3_esdecs.json", 'w')
    b_file.write(b)
    b_file.close()


if __name__== "__main__":
    build_relations_dict()