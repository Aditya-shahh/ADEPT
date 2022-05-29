import os
import sys

import time
import random
import warnings
import numpy as np
import argparse
import pandas as pd

import torch
import torch.nn as nn

from torch.optim import lr_scheduler, AdamW

from dataset import create_dataset_object, load_agnews_dataset, load_imdb_dataset, load_topic_dataset, load_yelp_dataset
from dataloader import get_dataloaders

from prompt import PROMPTEmbedding
from model import APT
from utils import freeze_params_roberta, get_accuracy, count_parameters, freeze_params

from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from transformers import get_linear_schedule_with_warmup, logging



from train import test_model, train_model


warnings.filterwarnings('ignore', '.*do not.*', )
warnings.warn('Do not show this message')

logging.set_verbosity_warning()

parser = argparse.ArgumentParser(description='APT')

parser.add_argument('--epochs', default=15, type=int,
                    help='number of epochs to run')

parser.add_argument('--batch_size', default=16, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lr', default=8e-4, type=float,    #8e - 5  #4e - 4
                    metavar='LR', help='learning rate for the model')

parser.add_argument('--n_tokens', default=20, type=int,
                    help='number of soft prompt tokens')

parser.add_argument('--adapter_modules', default=1, type=int,
                    help='number of adapter modules')

parser.add_argument('--adapter_hidden_size', default=8, type=int,
                    help='hidden size of adapter module')



parser.add_argument('--model',
                    default='roberta-base',
                    const='roberta-base',
                    nargs='?',
                    choices=['roberta-base', 'roberta-large'],
                    help='select the model (default: %(default)s\)')

parser.add_argument('--dataset',
                    default='agnews',
                    const='agnews',
                    nargs='?',
                    choices=['imdb', 'topic', 'agnews', 'yelp'],
                    help='select dataset to test on (default: %(default)s\)')


parser.add_argument('--train_dir', type=str, default='train.csv',
                    help='path to train file')

parser.add_argument('--valid_dir', type=str, default='valid.csv',
                    help='path to validation file')

parser.add_argument('--test_dir', type=str, default='test.csv',
                    help='path to test file')


parser.add_argument('--mode',
                    default='apt',
                    const='apt',
                    nargs='?',
                    choices=['apt', 'prompt', 'finetune', 'prompthead', 'head'],
                    help='select one of the training mode (default: %(default)s\)')


parser.add_argument('--save_checkpoint', type=bool, default=True,
                    help='whether to save the model')

parser.add_argument('--checkpoint', type=str, default=None,
                    help='path to model checkpoint')

parser.add_argument('--train', dest='train', help='Train the model', action='store_true')
parser.add_argument('--no-train', dest='train', help='Train the model', action='store_false')

parser.set_defaults(train=True)
 
parser.add_argument('--test', type=bool, default=False,
                    help='Test the model')

 
parser.add_argument('--count_params', type=bool, default=False,
                    help='Count model parameters')

args = parser.parse_args()


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():

    RANDOM_SEED = 42
    set_all_seeds(RANDOM_SEED)

    dataset = args.dataset   #imdb

    model_type = args.model   #roberta

    number_of_tokens = args.n_tokens    #  20

    mode = args.mode   #prompt_head


    batch_size = args.batch_size   

    learning_rate = args.lr

    epochs = args.epochs


    # Model Choice
    if model_type == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(model_type)


    #Dataset choice
    if dataset == 'imdb':

        train_text, train_labels, test_text, test_labels, valid_text, valid_labels = load_imdb_dataset(dataset)

        train_data_object = create_dataset_object(train_text, train_labels, number_of_tokens, tokenizer, dataset,  mode)
        
        test_data_object  = create_dataset_object(test_text, test_labels, number_of_tokens, tokenizer, dataset, mode)
        
        val_data_object = create_dataset_object(valid_text, valid_labels, number_of_tokens, tokenizer, dataset, mode)

        dataloaders = get_dataloaders(train_data_object, test_data_object, val_data_object, batch_size)

        num_labels = 2

        print("IMDB dataset loaded succesfully\n")


    elif dataset == 'topic':

        train_text, train_labels, test_text, test_labels, valid_text, valid_labels = load_topic_dataset(dataset)

        train_data_object = create_dataset_object(train_text, train_labels, number_of_tokens, tokenizer, dataset, mode)
        
        test_data_object  = create_dataset_object(test_text, test_labels, number_of_tokens, tokenizer, dataset, mode)
        
        val_data_object = create_dataset_object(valid_text, valid_labels, number_of_tokens, tokenizer, dataset, mode)

        dataloaders = get_dataloaders(train_data_object, test_data_object, val_data_object, batch_size)

        num_labels = 10

        print("Topic dataset loaded succesfully\n")
    
    elif dataset == 'agnews':

        train_text, train_labels, test_text, test_labels, valid_text, valid_labels = load_agnews_dataset(dataset)

        train_data_object = create_dataset_object(train_text, train_labels, number_of_tokens, tokenizer, dataset, mode)
        
        test_data_object  = create_dataset_object(test_text, test_labels, number_of_tokens, tokenizer, dataset, mode)
        
        val_data_object = create_dataset_object(valid_text, valid_labels, number_of_tokens, tokenizer, dataset, mode)

        dataloaders = get_dataloaders(train_data_object, test_data_object, val_data_object, batch_size)

        num_labels = 4

        print("AGnews dataset loaded succesfully\n")


    elif dataset == 'yelp':

        train_text, train_labels, test_text, test_labels, valid_text, valid_labels = load_yelp_dataset(dataset)

        train_data_object = create_dataset_object(train_text, train_labels, number_of_tokens, tokenizer, dataset, mode)
        
        test_data_object  = create_dataset_object(test_text, test_labels, number_of_tokens, tokenizer, dataset, mode)
        
        val_data_object = create_dataset_object(valid_text, valid_labels, number_of_tokens, tokenizer, dataset, mode)

        dataloaders = get_dataloaders(train_data_object, test_data_object, val_data_object, batch_size)

        num_labels = 5

        print("Yelp dataset loaded succesfully\n")


    else:
        print('Select one of the datasets')
    if model_type == 'roberta-base':

        model = RobertaForSequenceClassification.from_pretrained(model_type, 
                                                         num_labels=num_labels,
                                                         output_attentions=False,
                                                         output_hidden_states=False)

    #Training modes
    if mode == 'finetune':

        print("Roberta model for finetuning loaded successfully\n")

    elif mode == 'prompt':

        model =  freeze_params(model)
    
        prompt_emb = PROMPTEmbedding(model.get_input_embeddings(), 
                                    n_tokens= number_of_tokens, 
                                    initialize_from_vocab=True)

        model.set_input_embeddings(prompt_emb)

        print("Prompt model loaded successfully\n")

    elif mode == 'apt':

        adapter_hidden_size = args.adapter_hidden_size
        adapter_modules = args.adapter_modules
        

        roberta = freeze_params(model)

        prompt_emb = PROMPTEmbedding(roberta.get_input_embeddings(), 
                    n_tokens= number_of_tokens, 
                    initialize_from_vocab=True)

        roberta.set_input_embeddings(prompt_emb)

        model = APT(roberta, adapter_hidden_size, adapter_modules)

        print("APT model loaded successfully\n")

    elif mode == 'prompthead':

        model = freeze_params_roberta(model)

        prompt_emb = PROMPTEmbedding(model.roberta.get_input_embeddings(), 
                      n_tokens= number_of_tokens, 
                      initialize_from_vocab=True)

        model.roberta.set_input_embeddings(prompt_emb)

        print("Prompt head model loaded successfully\n")

    elif mode == 'head':

        model = freeze_params_roberta(model)

        print("Roberta model for head finetuning loaded successfully\n")


    else:
        print('Select one of the given modes\n')

    if args.count_params:
        count_parameters(model)

    # Check GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #Loss function
    criterion = nn.CrossEntropyLoss()

    if args.train:


        #optimizer
        if mode == 'prompt':
            optimizer = AdamW([model.roberta.embeddings.word_embeddings.learned_embedding], lr = learning_rate, eps=1e-8)

        else:
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
            'save_checkpoint': args.save_checkpoint,
            'checkpoint': args.checkpoint
        }

        train_model(config_train)


    if args.test:

        if args.checkpoint == None:
            saved_model_path = os.path.join("saved_models", dataset + "_" + mode + '.pt')

        else:
            saved_model_path = os.path.join("saved_models", args.checkpoint)

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


if __name__ == "__main__":

  main()