import atexit
import logging
import networkx as nx
import pickle
import sys
from fuzzywuzzy import fuzz, process
import os
import pandas as pd
import xml.etree.ElementTree as ET

sys.path.append("./")

# Import ICD10-CM cache storing the candidates list for each entity mention or create it if it does not exist
icd10cm_cache_file = "./tmp/icd10cm_cache.pickle"

if os.path.isfile(icd10cm_cache_file):
    logging.info("loading ICD10-CM dictionary...")
    icd10cm_cache = pickle.load(open(icd10cm_cache_file, "rb"))
    loadedeicd10cm = True
    logging.info("loaded ICD10-CM dictionary with %s entries", str(len(icd10cm_cache)))

else:
    icd10cm_cache = {}
    loadedeicd10cm = False
    logging.info("new ICD10-CM dictionary")


def exit_handler():
    print('Saving ICD10-CM dictionary...!')
    pickle.dump(icd10cm_cache, open(icd10cm_cache_file, "wb"))

atexit.register(exit_handler)


def load_spanish_icd10cm():
    """Load the 2nd chapter of ICD10-CM (Neoplasias) from local files into dict and builds the ontology graph with relations between codes
    
    Ensures: 
        name_to_id: is dict with mappings between each concept name and the respective ICD code;
        ontology_graph: is a MultiDiGraph object from Networkx representing the ICD10-CM vocabulary
    """
    
    print("Loading Spanish ICD10-CM...")

    name_to_id = dict()
            
    es_icd = pd.read_excel("./data/vocabularies/CIE10_2020_DIAGNOST_REFERENCIA_2019_10_04_1.xlsx", sheet_name = ["CIE10ES 2020 COMPLETA MARCADORE"], header=1)
    chapter_dict = {'Cap.02': 'C00-D49'} # The e-CIE-O-3 only have mappings to the Chapter 2 'Neoplasms' of the ICD10-CM

    #chapter_dict = {'Cap.01': 'A00-B99', 'Cap.02': 'C00-D49', 'Cap.03': 'D50-D89', 'Cap.04': 'E00-E89', 'Cap.05': 'F01-F99', 'Cap.06': 'G00-G99', 
    #            'Cap.07': 'H00-H59', 'Cap.08': 'H60-H95', 'Cap.09': 'I00-I99', 'Cap.10': 'J00-J99', 'Cap.11': 'K00-K95', 'Cap.12': 'L00-L99', 
    #            'Cap.13': 'M00-M99', 'Cap.14': 'N00-N99', 'Cap.15': 'O00-O9A', 'Cap.16': 'P00-P96' , 'Cap.17': 'Q00-Q99', 'Cap.18': 'R00-R99', 
    #            'Cap.19': 'S00-T88', 'Cap.20': 'V00-Y99', 'Cap.21': 'Z00-Z99'}

    edge_list = list()

    for row in es_icd["CIE10ES 2020 COMPLETA MARCADORE"].iterrows():
        code, description = row[1]['Tab.D'], row[1]['CIE-10-ES Diagnósticos 2020']
        
        if (code[0] == "C" or (code[0] == "D" and int(code[1]) <= 4)) and code[:3] != "Cap":

            if code in chapter_dict.keys():
                code = chapter_dict[code]

            string_to_remove = '(' + code + ')'
            description = description.replace(string_to_remove, '')
            name_to_id[description] = code
            parent_code = str()
            
            if "." not in code: # We need to find the parent code of the current code to build the ontology graph
                
                if "-" not in code:
                    if code[0] == "C":
                        
                        if code == "C7A":
                            parent_code = "C7A-C7A"
                        
                        elif code == "C7B":
                            parent_code = "C7B-C7B"
                        
                        elif code == "C50":
                            parent_code = "C50-C50"

                        elif code == "C4A":
                            parent_code = "C43-C44"

                        else:
                            sub_sub_sections = ["C00-C14", "C15-C26", "C30-C39", "C40-C41", "C43-C44", "C45-C49", "C51-C58", "C60-C63", "C64-C68", "C69-C72", "C73-C75", "C76-C80", "C81-C96"] 

                            for sub_code in sub_sub_sections:
                                tmp1 = sub_code.split("-")
                                
                                if int(code.strip("C")) >= int(tmp1[0].strip("C")) and int(code.strip("C")) <= int(tmp1[1].strip("C")):
                                    parent_code = sub_code

                    elif code[0] == "D":
                        parent_code = "D00-D09"
                        
                if "-" in code and code[0] == "C" and code != "C00-C96":
                    parent_code = "C00-C96"

                if "-" in code and code[0] == "D":
                    parent_code = "C00-D49"
                      
                if code == "C00-C96":
                    parent_code = "C00-D49" # The root node: neoplasias
            
            else:
                tmp2 = code.split(".")

                if len(tmp2[1]) == 1: # Ex: D47 is the parent code of D47.3
                    parent_code = tmp2[0]
                
                elif len(tmp2[1]) > 1: # Ex: D41.22 is the child code of D41.2
                    parent_code = tmp2[0] + "." + tmp2[1][:-1]

            edge_list.append((code, parent_code))      

    ontology_graph = nx.MultiDiGraph([edge for edge in edge_list])
    print("Is ontology_graph acyclic:", nx.is_directed_acyclic_graph(ontology_graph))
    print("Spanish ICD10-CM loading complete")
    
    return ontology_graph, name_to_id 


def map_to_spanish_icd10cm(entity_text, name_to_id):
    """Get best matches for entity text according to lexical similarity (edit distance).

    Ensures: 
        matches: is list; each match is dict with the respective properties
    """
    
    global icd10cm_cache
    codes = list()

    if entity_text in name_to_id: # There is an exact match for this entity
        codes = [entity_text]
    
    if entity_text.endswith("s") and entity_text[:-1] in icd10cm_cache: # Removal of suffix -s 
        codes = icd10cm_cache[entity_text[:-1]]
    
    elif entity_text in icd10cm_cache: # There is already a candidate list stored in cache file
        codes = icd10cm_cache[entity_text]

    else:
        # Get first 5 candidates according to lexical similarity with entity_text
        codes = process.extract(entity_text, name_to_id.keys(), scorer=fuzz.token_sort_ratio, limit=5)
        
        if codes[0][1] == 100: # There is an exact match for this entity
            codes = [codes[0]]
        
        icd10cm_cache[entity_text] = codes
    
    # Build the candidates list with each match id, name and matching score with entity_text
    matches = []
   
    for d in codes:
        term_name = d[0]
        
        if term_name in name_to_id.keys():
            term_id = name_to_id[term_name]
            #print(term_name, term_id)        
        else:
            term_id = "NIL"

        match = {"ontology_id": term_id,
                 "name": term_name,
                 "match_score": d[1]/100}
    
        matches.append(match)
  
    return matches
    



