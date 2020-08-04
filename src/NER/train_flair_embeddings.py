import os
import sys
import time
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from pre_process_NER import prepare_mesinesp_for_flair_embeds_training

sys.path.append("./")

## Set the gpus that will be used
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7,8"

start_time = time.time()

## see tutorial: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md

## are you training a forward or backward LM?
direction = sys.argv[2]

is_forward_lm = bool()

if direction == "fwd":
    is_forward_lm = True 

elif direction == "bwd":
    is_forward_lm = False

## load the default character dictionary
dictionary: Dictionary = Dictionary.load('chars')

## get your corpus, process forward and at the character level
prepare_mesinesp_for_flair_embeds_training() # prepare raw text from Spanish PubMed Abstracts for training
mesinesp_subset = sys.argv[1]
corpus_path = "./data/datasets/mesinesp/" + str(mesinesp_subset) + "/"

corpus = TextCorpus(corpus_path,
                    dictionary,
                    is_forward_lm,
                    character_level=True)

## instantiate your language model, set hidden size and number of layers (hidden_size=1024-small model, (hidden_size=2048-large model)
language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=1024, 
                               nlayers=1,
                               dropout=0.1)

## train your language model
trainer = LanguageModelTrainer(language_model, corpus)

#trainer.num_workers = 4 #Flair auto-detects whether you have a GPU available. If there is a GPU, it will automatically run training there.
output_dir = str()

if is_forward_lm:
    
    if not os.path.exists('./trained_embeddings/' + str(mesinesp_subset) + '/fwd/'):
        os.makedirs('./trained_embeddings/' + str(mesinesp_subset) + '/fwd/')
    
    output_dir = './trained_embeddings/' + str(mesinesp_subset) + '/fwd/'

else:
    if not os.path.exists('./trained_embeddings/' + str(mesinesp_subset) + '/bwd/'):
        os.makedirs('./trained_embeddings/' + str(mesinesp_subset) + '/bwd/')

    output_dir = './trained_embeddings/' + str(mesinesp_subset) + '/bwd/'

trainer.train(output_dir,
              sequence_length=250,
              mini_batch_size=100,
              max_epochs=2000,
              patience=25,
              checkpoint=True)

## To see the training process:
#from flair.visual.training_curves import Plotter
#plotter = Plotter()

#f is_forward_lm:
#    plotter.plot_training_curves('resources/taggers/fwd_embeds/loss.tsv')
#    plotter.plot_weights('resources/taggers/fwd_embeds/weights.txt')
#else:
#    plotter.plot_training_curves('resources/taggers/bwd_embeds/loss.tsv')
#    plotter.plot_weights('resources/taggers/bwd_embeds/weights.txt')

print("Total time (aprox.):", int((time.time() - start_time)), "hours\n----------------------------------")

