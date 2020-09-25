import os
import networkx as nx
import sys
import xml.etree.ElementTree as ET
from annotations import parse_ner_output
from math import log

sys.path.append("./")


def build_extrinsic_information_content_dict(annotations):
    """Dict with extrinsic information content (Resnik's) for each term in a corpus.""" 

    # Get the term frequency in the corpus
    term_counts, extrinsic_ic = {}, {}
    
    for document in annotations: 
        
        for annotation in annotations[document]: 
            term_id = annotation
            
            if term_id not in term_counts.keys():
                term_counts[term_id] = 1
            else:
                term_counts[term_id] += 1
            
    max_freq = max(term_counts.values()) # Frequency of the most frequent term in dataset
    
    for term_id in term_counts:
        term_frequency = term_counts[term_id] 
        term_probability = (term_frequency + 1)/(max_freq + 1)
        information_content = -log(term_probability) + 1
        extrinsic_ic[term_id] = information_content + 1
    
    return extrinsic_ic


def generate_ic_file(annotations, ontology_graph, ont_number):
    """Generate file with information content of all CIEO3 entities appearing in candidates files."""

    ontology_pop_string = str()
    candidates_dir = "./tmp/norm_candidates/{}/".format(ont_number)
    ic_dict = build_extrinsic_information_content_dict(annotations) 
    url_temp = list()

    for doc in os.listdir(candidates_dir): 
        data = ''
        path = candidates_dir + doc
        candidate_file = open(path, 'r', errors="ignore")
        data = candidate_file.read()
        candidate_file.close()
        candidate_count = int()

        for line in data.split('\n'):
            url, surface_form = str(), str()
            
            if line[0:6] == "ENTITY":
                surface_form = line.split('\t')[1].split('text:')[1]
                url = line.split('\t')[8].split('url:')[1]
                #predicted_type = line.split('\t')[3][14:]
            
            elif line[0:9] == "CANDIDATE" and candidate_count <= 10:
                surface_form = line.split('\t')[6].split('name:')[1]
                url = line.split('\t')[5].split('url:')[1]
                #predicted_type = line.split('\t')[9][14:]     
            
            if url in url_temp:
                continue
                
            else:
                if url != "" and "/" in url: #Only consider CIEO3 codes   
                    url_temp.append(url)

                    if url in ic_dict.keys():
                        ic = ic_dict[url]                                 
                    else:
                        ic = 1.0       
                 
                    ontology_pop_string += url +'\t' + str(ic) + '\n'
                        
    # Create file ontology_pop with information content for all entities in candidates file
    with open("./tmp/ecieo3_ic", 'w') as ontology_pop:
        ontology_pop.write(ontology_pop_string)
        ontology_pop.close()


