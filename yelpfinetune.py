import numpy as np
import argparse
import pandas as pd

import torch
import torch.nn as nn
import random
from torch.optim import lr_scheduler, AdamW

from dataset import create_dataset_object, load_agnews_dataset, load_imdb_dataset, load_topic_dataset, load_yelp_dataset
from dataloader import get_dataloaders

from prompt import PROMPTEmbedding
from model import APT
from utils import get_accuracy, count_parameters, freeze_params

from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from train import test_model, train_model

import os

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

RANDOM_SEED = 42
set_all_seeds(RANDOM_SEED)


dataset = 'yelp'   #imdb

model_type = 'roberta-base'   #roberta

number_of_tokens = 20

mode = 'finetune'


batch_size = 16

learning_rate = 1e-5

epochs = 30

tokenizer = RobertaTokenizer.from_pretrained(model_type)

train_text, train_labels, test_text, test_labels, valid_text, valid_labels = load_yelp_dataset(dataset)

train_data_object = create_dataset_object(train_text, train_labels, number_of_tokens, tokenizer, dataset, mode)

test_data_object  = create_dataset_object(test_text, test_labels, number_of_tokens, tokenizer, dataset, mode)

val_data_object = create_dataset_object(valid_text, valid_labels, number_of_tokens, tokenizer, dataset, mode)

dataloaders = get_dataloaders(train_data_object, test_data_object, val_data_object, batch_size)

num_labels = 5

print("Yelp dataset loaded succesfully\n")

model = RobertaForSequenceClassification.from_pretrained(model_type, 
                                                    num_labels=num_labels,
                                                    output_attentions=False,
                                                    output_hidden_states=False)
                                                    

print("Roberta model for finetuning loaded successfully\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Loss function
criterion = nn.CrossEntropyLoss()


optimizer = AdamW(model.parameters(), lr = learning_rate, eps=1e-8)

# Defining LR Scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=len(dataloaders['Train'])*epochs/15, 
    num_training_steps=len(dataloaders['Train'])*epochs
)


config_train = {
    
    'dataset': dataset,
    'dataloaders':dataloaders, 
    'model': model, 
    'device': device, 
    'criterion':criterion, 
    'optimizer':optimizer, 
    'mode':mode, 
    'scheduler': scheduler,
    'epochs': epochs,
    'save_checkpoint': True,
    'checkpoint': None
}

train_model(config_train)

saved_model_path = os.path.join("saved_models", dataset + "_" + mode + '.pt')


if os.path.exists(saved_model_path):
    model.load_state_dict(torch.load(saved_model_path))
    print("Model checkpoint loaded succesfully from", saved_model_path)

else:
    print("Required model checkpoint not found!\n")
    raise FileNotFoundError

config_test = {

'model': model, 
'test_loader':dataloaders['Test'], 
'device': device, 
'criterion':criterion,
'mode':mode
}

test_model(config_test)
