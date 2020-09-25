import json
import os
import sys
import time
import xml.etree.ElementTree as ET

from annotations import parse_ner_output
from candidates import write_candidates, generate_candidates_for_entity
from cieo3 import load_cieo3
from es_decs import load_es_decs
from icd10cm import load_spanish_icd10cm
from information_content import generate_ic_file
#from strings import entity_string

sys.path.append("./")

entity_string = "ENTITY\ttext:{0}\tnormalName:{1}\tpredictedType:{2}\tq:true"
entity_string += "\tqid:Q{3}\tdocId:{4}\torigText:{0}\turl:{5}\n"


def build_entity_candidate_dict(annotations, model, min_match_score, ontology_graph, ontology_graph_2, ontology_graph_3, name_to_id, name_to_id_2, name_to_id_3, synonym_to_id, ont_number):
    """Builds a dict with candidates for all entity mentions in each document."""

    doc_count, nil_count, total_entities, total_unique_entities, no_solution, solution_is_first_count = int(), int(),int(), int(), int(), int()
    documents_entity_list = dict() 
    doc_total = len(annotations.keys())

    for document in annotations.keys(): 
        doc_count += 1 
        percent = round((doc_count/doc_total*100), 2)
        print("Parsing document", document, "(", doc_count, "/", doc_total, " - ", str(percent), "%)...")
        entity_dict = dict() 
        document_entities = list()

        for annotation in annotations[document]:
            normalized_text = annotation[0].lower().replace(" ", "_")
            total_entities += 1
            
            if normalized_text in document_entities: # Repeated instances of the same entity count as one instance
                continue
                
            else:    
                document_entities.append(normalized_text)
                total_unique_entities += 1
                     
                # Get candidates for current entity
                entity_dict[normalized_text]= generate_candidates_for_entity(normalized_text, name_to_id, name_to_id_2, name_to_id_3, \
                                                                            synonym_to_id, min_match_score, ontology_graph, \
                                                                            ontology_graph_2, ontology_graph_3, ont_number)

            if len(entity_dict[normalized_text]) == 0: # Do not consider this entity if no candidate was found
                del entity_dict[normalized_text]
                no_solution += 1
                                     
            else: # The entity has candidates, so it is added to entity_dict  
                entity_str = entity_string.format(annotation[0], normalized_text, "MOR_NEO", doc_count, document.strip(".ann"), "NaN")
                current_values = list()
                    
                if model != "string_matching":
                    current_values = entity_dict[normalized_text]
                
                else:
                    current_values = [entity_dict[normalized_text][0]] # Only the first solution will be considered (the most similar)
                    
                current_values.insert(0, entity_str)
                entity_dict[normalized_text] = current_values
                
        documents_entity_list[document] = entity_dict
    
    # Output statistics
    statistics = "\nNumber of documents: " + str(len(documents_entity_list.keys())) 
    statistics += "\nTotal entities: " + str(total_entities) + "\nNILs: " + str(nil_count)
    statistics += "\nValid entities: " + str(total_entities-nil_count)
    valid_entities_perc = ((total_entities-nil_count)/total_entities)*100
    statistics += "\n % of valid entities: " + str(valid_entities_perc)
    statistics += "\n\nTotal unique entities: " + str(total_unique_entities)
    print(statistics)

    return documents_entity_list
    

def pre_process():

    start_time = time.time()
    min_match_score = 0.2 # min lexical similarity between entity text and candidate text
    ont_number = str(sys.argv[1]) # ontologies to consider: single_ont (cieo3) or multi_ont (CIEO3, ICD10CM and DeCS)
    model = "ppr_ic" 
    
    if not os.path.exists(".tmp/norm_results/" + ont_number):
        os.makedirs(".tmp/norm_results/" + ont_number)

    if ont_number == "multi_ont": 
        ontology_graph_2, name_to_id_2 = load_spanish_icd10cm()
        ontology_graph_3, name_to_id_3, synonym_to_id_3 = load_es_decs()
    
    elif ont_number == "single_ont":
        ontology_graph_2, name_to_id_2,  ontology_graph_3, name_to_id_3 = list(), list(), list(), list()

    ontology_graph, name_to_id, synonym_to_id  = load_cieo3() 
    annotations = parse_ner_output(False) 
    documents_entity_list = build_entity_candidate_dict(annotations, model, min_match_score, ontology_graph, ontology_graph_2,\
                                                                ontology_graph_3, name_to_id, name_to_id_2, name_to_id_3, \
                                                                synonym_to_id, ont_number)
    
    print("Parsing time (aprox.):", int((time.time() - start_time)/60.0), "minutes\n----------------------------------")
    
    # Get relations CIEO3<->ICD10CM and CIEO3<->DeCS from json files
    extracted_relations_1, extracted_relations_2 = dict(), dict()

    if model != "string_matching" and ont_number == "multi-ont":
        
        with open('./tmp/relations_cieo3_icd10cm.json', 'r') as json_file1:
            extracted_relations_1 = json.load(json_file1)
        json_file1.close()

        with open('./tmp/relations_cieo3_esdecs.json', 'r') as json_file2:
            extracted_relations_2 = json.load(json_file2)
        json_file2.close()
    
    # Create a candidates file for each initial document
    document_count, entities_writen = int(), int()
        
    for document in documents_entity_list:
        document_count += 1
        candidates_filename = "./tmp/norm_candidates/{}/{}".format(ont_number, document.strip(".ann"))
        print("Writing candidates:\t", document_count, "/", len(documents_entity_list.keys()))
        entities_writen += write_candidates(documents_entity_list[document], candidates_filename, ontology_graph, ont_number, extracted_relations_1, extracted_relations_2)
        
    print("Entities writen in the candidates files:", entities_writen)
        
    # Create file with the information content of each CIEO3 candidate appearing in candidates files (to use in ppr_ic model) 
    if model != "string_matching":
        generate_ic_file(parse_ner_output(True), ontology_graph, ont_number)

    print("Total time (aprox.):", int((time.time() - start_time)/60.0), "minutes\n----------------------------------")   
    

if __name__ == "__main__":
    pre_process()
         
       
