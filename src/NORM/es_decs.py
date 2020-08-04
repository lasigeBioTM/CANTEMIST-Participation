import atexit
import logging
import networkx as nx
import obonet
import os
import pickle
import sys
from fuzzywuzzy import fuzz, process

sys.path.append("./")

# Import ES DeCS cache storing the candidates list for each entity mention or create it if it does not exist
es_decs_cache_file = "./tmp/es_decs_cache.pickle"

if os.path.isfile(es_decs_cache_file):
    logging.info("loading ES DeCS dictionary...")
    es_decs_cache = pickle.load(open(es_decs_cache_file, "rb"))
    loadedes_decs = True
    logging.info("loaded ES DeCS dictionary with %s entries", str(len(es_decs_cache)))

else:
    es_decs_cache = {}
    loadedees_decs = False
    logging.info("new ES DeCS dictionary")


def exit_handler():
    print('Saving ES DeCS dictionary...!')
    pickle.dump(es_decs_cache, open(es_decs_cache_file, "wb"))

atexit.register(exit_handler)


def load_es_decs():
    """Load ChEBI ontology from local file 'chebi.obo' or from online source.
    
    Ensures: 
        ontology_graph: is a MultiDiGraph object from Networkx representing ChEBI ontology;
        name_to_id: is dict with mappings between each ontology concept name and the respective ChEBI id;
        synonym_to_id: is dict with mappings between each ontology concept name and the respective ChEBI id;
    """

    print("Loading ES DeCS...")
    
    graph = obonet.read_obo("./data/vocabularies/DeCS_2019.obo") # Load the ontology from local file
    graph = graph.to_directed()
    name_to_id, synonym_to_id, edges = dict(), dict(), list()

    for node in  graph.nodes(data=True):
        node_id, node_name = node[0], node[1]["name"]
        name_to_id[node_name] = node_id
        
        if 'is_a' in node[1].keys(): # The root node of the ontology does not have is_a relationships
                
            for related_node in node[1]['is_a']: # Build the edge_list with only "is-a" relationships
                edges.append((node[0], related_node)) 
            
        if "synonym" in node[1].keys(): # Check for synonyms for node (if they exist)
                
            for synonym in node[1]["synonym"]:
                synonym_name = synonym.split("\"")[1]
                synonym_to_id[synonym_name] = node_id

    ontology_graph = nx.MultiDiGraph([edge for edge in edges])
    print("Is ontology_graph acyclic:", nx.is_directed_acyclic_graph(ontology_graph))
    print("ES DeCS loading complete")
    
    return ontology_graph, name_to_id, synonym_to_id


def map_to_es_decs(entity_text, name_to_id):
    """Get best spanish DECS matches for entity text according to lexical similarity (edit distance).
    
    Ensures: 
        matches: is list; each match is dict with the respective properties
    """
    
    global es_decs_cache
    
    if entity_text in name_to_id:# or entity_text in synonym_to_id: # There is an exact match for this entity
        codes = [entity_text]
    
    if entity_text.endswith("s") and entity_text[:-1] in es_decs_cache: # Removal of suffix -s 
        codes = es_decs_cache[entity_text[:-1]]
    
    elif entity_text in es_decs_cache: # There is already a candidate list stored in cache file
        codes = es_decs_cache[entity_text]

    else:
        # Get first ten candidates according to lexical similarity with entity_text
        codes = process.extract(entity_text, name_to_id.keys(), scorer=fuzz.token_sort_ratio, limit=5)
        
        if codes[0][1] == 100: # There is an exact match for this entity
            codes = [codes[0]]
    
        #elif codes[0][1] < 100: # Check for synonyms of this entity
        #    drug_syns = process.extract(entity_text, synonym_to_id.keys(), limit=10, scorer=fuzz.token_sort_ratio)

        #    for synonym in drug_syns:

        #        if synonym[1] == 100:
        #            codes = [synonym]
                
        #        else:
        #            if synonym[1] > codes[0][1]:
        #                codes.append(synonym)
        
        es_decs_cache[entity_text] = codes
    
    # Build the candidates list with each match id, name and matching score with entity_text
    matches = []

    for d in codes:
        term_name = d[0]
        
        if term_name in name_to_id.keys():
            term_id = name_to_id[term_name]
        
        #elif term_name in synonym_to_id.keys():
        #    term_id = synonym_to_id[term_name]
        
        else:
            term_id = "NIL"

        match = {"ontology_id": term_id,
                 "name": term_name,
                 "match_score": d[1]/100}
    
        matches.append(match)
  
    return matches