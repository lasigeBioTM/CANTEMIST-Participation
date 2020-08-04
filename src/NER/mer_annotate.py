
import merpy
import os
import sys
import time
from cieo3 import load_cieo3

sys.path.append("./")


def annotate_documents(task, subset, name_to_id, output_dir):
    """Recognise entities (NER) and link them to the respective CIE-O-3 code (Normalisation), if available, using MERPY"""
    
    lexicon_name = "cieo3"
    merpy.create_lexicon(name_to_id.keys(), lexicon_name)
    merpy.create_mappings(name_to_id, lexicon_name)
    merpy.process_lexicon(lexicon_name)
    
    dataset_dir = str()
    
    if subset == "train":
        dataset_dir = "./data/datasets/train-set-to-publish/cantemist-norm/" 
    
    elif subset == "dev1":
        dataset_dir = "./data/datasets/dev-set1-to-publish/cantemist-norm/" 
    
    elif subset == "dev2":
        dataset_dir = "./data/datasets/dev-set2-to-publish/cantemist-norm/" 
    
    elif subset == "test":
        dataset_dir = "./data/datasets/test-background-set-to-publish"
    
    doc_count = int()
    doc_annot_ratio = int()
    doc_w_annotations_count = int()
    total_entity_count = int()
    linked_mentions = int()
    total_doc_count = int(len(os.listdir(dataset_dir))/2)

    for doc in os.listdir(dataset_dir):

        if doc[-3:] == "txt":

            doc_count += 1
            output_string = str()
            print("Annotating " + str(doc_count) + " of " + str(total_doc_count) + " documents")
            
            with open(dataset_dir + doc, 'r') as input_file:
                text = input_file.read()
                input_file.close()
                doc_entity_count = int()

                entities = merpy.get_entities(text, lexicon_name)
                    
                for entity in entities:

                    if entity != ['']:
                        total_entity_count += 1
                        doc_entity_count += 1
                           
                        if len(entity) == 4: # linked mentions with CIE-O-3 code
                            linked_mentions += 1
                            output_string += "T" + str(doc_entity_count) + "\tMORFOLOGIA_NEOPLASIA " + entity[0] + " " + entity[1] + "\t" + entity[2] + "\n"
                            
                            if task == "norm":
                                output_string += "#" + str(doc_entity_count) + "\tAnnotatorNotes\tT" + str(doc_entity_count) + "\t" + entity[3] + "\n"
                            
                        elif len(entity) == 3: # mentions without CIE-O-3 code
                            output_string += "T" + str(doc_entity_count) + "\tMORFOLOGIA_NEOPLASIA " + entity[0] + " " + entity[1] + "\t" + entity[2] + "\n"
                            
                            if task == "norm":
                                output_string += "#" + str(doc_entity_count) + "\tAnnotatorNotes T" + str(doc_entity_count) + "\tNA\n"

                if doc_entity_count > 0:
                    doc_w_annotations_count += 1
                
                elif doc_entity_count == 0:
                    output_string = "NA\tNA NA NA\tNA\n"
                         
                    if task == "norm":
                        output_string += "#" + str(doc_entity_count) + "\tAnnotatorNotes T" + str(doc_entity_count) + "\tNA\n"
                        
            
            # output annotations file
            output_filename = output_dir + doc[:-4] + ".ann"

            with open(output_filename, 'w') as output_file:
                output_file.write(output_string)
                output_file.close()

    try:
        doc_annot_ratio = float(doc_w_annotations_count/total_doc_count)
        mentions_ratio = float(total_entity_count/doc_w_annotations_count)
        doc_linked_ratio = float(linked_mentions/doc_w_annotations_count)
        linked_ratio = float(linked_mentions/total_entity_count)
    
    except:
        mentions_ratio = 0.0
        doc_linked_ratio = 0.0
        linked_ratio = 0.0
    
    output_str = "TOTAL DOCUMENTS: " + str(total_doc_count) + "\n"
    output_str += "DOCS WITH ANNOTATIONS: " + str(doc_w_annotations_count) + "\n"
    output_str += "RATIO OF DOCS WITH ANNOTATIONS: " + str(doc_annot_ratio) + "\n"
    output_str += "TOTAL ENTITY MENTIONS: " + str(total_entity_count) + "\n"
    output_str += "ENTITY MENTIONS PER DOCUMENT: " + str(mentions_ratio) + "\n"
    output_str += "LINKED ENTITY MENTIONS: " + str(linked_mentions) + "\n"
    output_str += "LINKED ENTITY MENTIONS PER DOCUMENT: " + str(doc_linked_ratio) + "\n"
    output_str += "RATIO OF LINKED ENTITY MENTIONS: " + str(linked_ratio)

    file_name = "./mer_annotations/" + task + "/" + task + "_" + subset + "_stats"    
    
    with open(file_name, "w") as output:
        output.write(output_str)
        output.close()
    

if __name__ == "__main__":
    start_time = time.time()
    ontology_graph, name_to_id, synonym_to_id = load_cieo3()
    task = str(sys.argv[1])
    subset = str(sys.argv[2])

    print("Starting annotation with MER...")
    
    output_dir = "./mer_annotations/" + task + "/" + subset + "/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    annotate_documents(task, subset, name_to_id, output_dir)

    print("...Done!")
    print("Total time (aprox.):", int((time.time() - start_time)/60.0), "minutes\n----------------------------------")
