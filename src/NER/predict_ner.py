import os 
import sys
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import FlairEmbeddings
from flair.data import Sentence
import spacy
from spacy.lang.es import Spanish
from spacy.pipeline import SentenceSegmenter

sys.path.append("./")


def build_annotation_file(doc, doc_filepath, model):

    entity_type = "MORFOLOGIA_NEOPLASIA"
    output = str()

    with open(doc_filepath, 'r', encoding='utf-8') as doc_file:
        text = doc_file.read()
        doc_file.close()
    
    # Sentence segmentation followed by tokenization of each sentence
    doc_spacy = nlp(text)
    sent_begin_position = int(0)
    annot_number = int()

    for spacy_sent in doc_spacy.sents:
        
        if len(spacy_sent.text) >= 2 and spacy_sent.text[0] == "\n":
            sent_begin_position -= 1
        
        sentence = Sentence(spacy_sent.text, use_tokenizer=True)
        model.predict(sentence)
            
        for entity in sentence.get_spans('ner'):
            
            if str(entity.tag).strip("\n") == "MOR_NEO":
                annot_number += 1
                begin_entity = str(sent_begin_position+entity.start_pos)
                end_entity = str(sent_begin_position+entity.end_pos)
                entity_text = "" 
                output += "T" + str(annot_number) + "\t" + entity_type + " " + begin_entity + " " + end_entity + "\t" + str(entity.text) + "\n"
                        
        sent_begin_position += len(spacy_sent.text) + 1
 
    # Output the annotation file
    annotation_filepath = './evaluation/NER/' + doc[:-3] + 'ann'

    with open(annotation_filepath, 'w', encoding='utf-8') as annotation_file:
        annotation_file.write(output)
        annotation_file.close()


if __name__ == "__main__":
    # load the model you trained
    model = SequenceTagger.load('resources/taggers/medium_updated/final-model.pt')

    ## Create spanish sentence segmenter with Spacy
    # the sentence segmentation by spacy is non-destructive, i.e., the empty lines are considered when getting a span of a given word/entity
    nlp = Spanish()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    
    test_dir = "./data/datasets/test-background-set-to-publish/"

    if not os.path.exists("./evaluation/NER/"):
        os.makedirs("./evaluation/NER/")

    for doc in os.listdir(test_dir): #For each document in test_dir build the respective annotation file with predicted entities
        doc_filepath = test_dir + doc
        build_annotation_file(doc, doc_filepath, model)

