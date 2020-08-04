import os
import sys
from annotations import parse_ner_output

sys.path.append("./")


def get_ppr_output(ont_number, model):
    """ Get the results outputted by the PPR algorithm into dict."""

    results_dict, results_file = dict(), str()
    
    with open("./tmp/norm_results/" + ont_number + "/ppr_ic_all_candidates") as results:
        data = results.read()
        results.close()
    
    doc_id = str()
    temp_dict = dict()
    line_count = int()
    
    split_results = data.split("======= ")

    def get_score(candidate):
        return candidate[1]

    for element in split_results:

        if element != "" and element != "\n":
                
            if element[0] != "\n":
                doc_name = element.split(" =")[0]
                
            else:
                doc_data = element.split("\n")
                temp_dict = dict()

                if len(doc_data) and doc_data[0] == "" and doc_data[1] == "" and doc_data[2] == "":
                    continue

                else:
                    for element_2 in doc_data:

                        if element_2 != "" and element_2 != "==============================================":

                            if element_2[0] == "=":
                                entity_text = element_2.split("= ")[1].split("\t")[0]
                                candidate_count = 0
                                candidates = list()
                                
                            else:
                                if element_2[:5] == "\tbest": #The selected candidate from CIEO3 or ICD10-CM or DeCS
                                    best_candidate_url= element_2.split("(")[1].split(")")[0]
                                        
                                    if best_candidate_url[0] == "C" or best_candidate_url[0] == "D" or ("/" not in best_candidate_url[0]): 
                                        #We need to find the best scoring candidate from the CIEO3 vocabulary instead of ICD10-CM or DeCS
                                        candidates.sort(key=get_score, reverse=True)
                                        found_new_best_candidate = False
                                               
                                        for candidate in candidates:
                                                
                                            if candidate[0][0] != "C" and candidate[0][0] != "D" and found_new_best_candidate == False:
                                                new_best_candidate = candidate[0]
                                                    
                                                found_new_best_candidate = True
                                                temp_dict[entity_text] = new_best_candidate
                                        
                                    else:  # The best candidate is from CIEO3 vocabulary
                                        temp_dict[entity_text] = best_candidate_url    
                                        
                                else:
                                    candidate_url= element_2.split("(")[1].split(")")[0] 
                                    score = float(element_2.split("(")[1].split(")")[1].split("=>")[1].strip(" "))
                                    candidates.append((candidate_url, score))
                        
                results_dict[doc_name] = temp_dict           
    
    return results_dict            


def build_annotation_files(results_dict, ont_number):
    """Create the annotation files containing the NER annotations + CIE-O-3 codes (submission to the CANTEMIST-NORM subtask)"""
    
    output_dir = "./evaluation/NORM/" + ont_number + "/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ner_annotations = parse_ner_output()
    
    for doc in ner_annotations.keys():
        output = str()

        if doc.strip(".ann") in results_dict.keys():
            doc_results = results_dict[doc.strip(".ann")]
            doc_annotations = ner_annotations[doc]

            for annotation in doc_annotations:
                tmp_annotation_text = annotation[0].lower().replace(" ", "_")
               
                output += annotation[1] + "\t" + annotation[2] + " " + annotation[3] + " " + annotation[4] + "\t" + annotation[0] + "\n"

                if tmp_annotation_text in doc_results.keys():
                    ecieo3_code = doc_results[tmp_annotation_text]
                    annotation_number = annotation[1].strip("T")
                    output += "#" + annotation_number + "\tAnnotatorNotes" + " " + annotation[1] + "\t" + ecieo3_code + "\n"
        
        with open(output_dir + doc, 'w') as output_file:
            output_file.write(output)
            output_file.close()
               
                
if __name__ == "__main__":
    ont_number = str(sys.argv[1])
    model = "ppr_ic"
    
    results_dict = get_ppr_output(ont_number, model)
    build_annotation_files(results_dict, ont_number)
    
    output_filenames = os.listdir("./evaluation/NORM/" + ont_number)
    initial_filenames = os.listdir("./data/datasets/test-background-set-to-publish")
    
    for filename in initial_filenames: #Create empty files for submission
        ann_filename = filename[:-3]+"ann"
        
        if  ann_filename not in output_filenames:
            with open("./evaluation/NORM/" + ont_number + "/" + ann_filename, "w") as new_file:
                new_file.close()
    
    print("Post-processing complete. Submission files in '.evaluation/NORM/'" + ont_number + " dir")