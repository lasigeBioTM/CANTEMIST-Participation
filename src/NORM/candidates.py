import networkx as nx

from cieo3 import map_to_cieo3
from es_decs import map_to_es_decs
from icd10cm import map_to_spanish_icd10cm
from fuzzywuzzy import fuzz, process
#from strings import candidate_string

candidate_string = "CANDIDATE\tid:{0}\tinCount:{1}\toutCount:{2}\tlinks:{3}\t"
candidate_string += "url:{4}\tname:{5}\tnormalName:{6}\tnormalWikiTitle:{7}\tpredictedType:{8}\n"


def write_candidates(entity_list, candidates_filename, ontology_graph, ont_number, extracted_relations_1, extracted_relations_2):
    """Write the entities and respective candidate properties of a given document to a candidate file."""
    
    entities_used = int() # entitites with at least one candidate
    candidates_links = dict() # (url: links)
    candidates_file = open(candidates_filename, 'w')
    
    for e in entity_list:
    
        if len(entity_list[e]) > 0:
            entity_str = entity_list[e][0]
            candidates_file.write(entity_str)
            entities_used += 1
        
        for ic, c in enumerate(entity_list[e][1:]): # iterate over the candidates for current entity
            
            if c["url"] in candidates_links:
                c["links"] = candidates_links[c["url"]]
    
            else: 
                links, other_candidates = [], []
                    
                for e2 in entity_list: # Iterate over candidates for every other entity except current one
                    
                    if e2 != e:
                        
                        for ic2, c2 in enumerate(entity_list[e2][1:]):
                            other_candidates.append((c2["url"], c2["id"]))
    
                for c2 in other_candidates: # 
                    c1 = c["url"]
                    relation_tuple_1 = (str(c1), str(c2[0]))
                    relation_tuple_2 = (str(c2[0]), str(c1))

                    # Check for Is-a relations between CIEO3 codes
                    if c1 == c2[0] or relation_tuple_1 in ontology_graph.edges() or relation_tuple_2 in ontology_graph.edges(): 
                        candidates_linked = True
                        links.append(str(c2[1]))
                    
                    if ont_number == "multi_ont": 
                        
                        if c1 in extracted_relations_1.keys(): # Check for CIEO3-ICD10CM relations
                            relations_with_c1 = extracted_relations_1[c1]
                            
                            if c2[0] in relations_with_c1: 
                                links.append(str(c2[1]))
                        
                        if c1 in extracted_relations_2.keys(): # Check for CIEO3-DeCS relations
                            relations_with_c1 = extracted_relations_2[c1]
                            
                            if c2[0] in relations_with_c1: 
                                links.append(str(c2[1]))

                c["links"] = ";".join(set(links))
                candidates_links[c["url"]] = c["links"][:]
            
            entity_modified_str = c["name"].lower().replace(" ", "_")
            candidates_file.write(candidate_string.format(c["id"], c["incount"],
                                                          c["outcount"], c["links"],
                                                          c["url"], c["name"], entity_modified_str, entity_modified_str, "MOR_NEO"))
    
    candidates_file.close()
    
    return entities_used


def generate_candidates_for_entity(entity_text, name_to_id, name_to_id_2, named_to_id_3, synonym_to_id, min_match_score, ontology_graph, ontology_graph_2, ontology_graph_3, ont_number):
    """Get the structured candidates list for given entity."""

    structured_candidates = list()
    less_than_min_score = int()

    candidate_names = map_to_cieo3(entity_text, name_to_id, synonym_to_id)

    if ont_number == "multi_ont": # Add the candidates from ICD10-CM
        icd10_cm_candidates = map_to_spanish_icd10cm(entity_text, name_to_id_2)
        decs_candidates = map_to_es_decs(entity_text, named_to_id_3)
        candidate_names = candidate_names + icd10_cm_candidates + decs_candidates
        
    # Get properties for each retrieved candidate 
    for i, candidate_match in enumerate(candidate_names): 
    
        if candidate_match["ontology_id"] != "NIL":

            if candidate_match["match_score"] > min_match_score:
                incount, outcount, candidate_id = int(), int(), str()

                if candidate_match["ontology_id"][0] == "D" or candidate_match["ontology_id"][0] == "C": #ICD10-CM candidate
                    outcount = ontology_graph_2.out_degree(candidate_match["ontology_id"])
                    incount = ontology_graph_2.in_degree(candidate_match["ontology_id"])
                    
                    numbs = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                    tmp = str()
                    
                    for char in candidate_match["ontology_id"]:
                        if char in numbs:
                            tmp += char
                        else:
                            tmp += "0"
                    
                    candidate_id = int(tmp)
       
                else: # CIEO3 or DeCS candidate

                    if "/" in candidate_match["ontology_id"]: #CIEO3 candidate
                        outcount = ontology_graph.out_degree(candidate_match["ontology_id"])
                        incount = ontology_graph.in_degree(candidate_match["ontology_id"])
                        candidate_id = int(candidate_match["ontology_id"].replace("/", ""))
                    
                    else: #DeCS candidate
                        outcount = ontology_graph_3.out_degree(candidate_match["ontology_id"])
                        incount = ontology_graph_3.in_degree(candidate_match["ontology_id"])
                        
                        try:
                            candidate_id = int(candidate_match["ontology_id"])
                        except:
                            candidate_id = "0000001"
                
                if incount == [] or outcount ==[] or candidate_match["name"] == "ENFERMEDADES" or candidate_match["name"] == "DeCS":
                    incount = 0
                    outcount = 0
                
                # The first candidate in candidates list should be the correct solution
                structured_candidates.append({"url": candidate_match["ontology_id"], "name": candidate_match["name"],
                                                "outcount": outcount, "incount": incount,
                                                "id": candidate_id, "links": [],
                                                "score": candidate_match["match_score"]})
            
            else: 
                less_than_min_score += 1
    
    return structured_candidates
