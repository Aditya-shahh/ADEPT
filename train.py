import os
import sys

import time
import numpy as np
import argparse
import pandas as pd

import torch
import torch.nn as nn

from torch.optim import lr_scheduler

from dataset import IMDB_Dataset, get_imdb_dataset
from dataloader import get_dataloaders

from prompt import PROMPTEmbedding
from model import APT, Model_Prompt_Head
from utils import get_accuracy, count_parameters, freeze_params

from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='APT')

parser.add_argument('--epochs', default=15, type=int,
                    help='number of epochs to run')

parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lr', default=8e-5, type=float,
                    metavar='LR', help='learning rate for the model')

parser.add_argument('--n_tokens', default=20, type=int,
                    help='number of soft prompt tokens')

parser.add_argument('--adapter_modules', default=1, type=int,
                    help='number of adapter modules')

parser.add_argument('--adapter_hidden_size', default=16, type=int,
                    help='hidden size of adapter module')

parser.add_argument('--model',
                    default='roberta-base',
                    const='roberta-base',
                    nargs='?',
                    choices=['roberta-base', 'roberta-large'],
                    help='select the model (default: %(default)s\)')

parser.add_argument('--dataset',
                    default='topic',
                    const='topic',
                    nargs='?',
                    choices=['imdb', 'topic', 'emotion'],
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
                    choices=['apt', 'prompt', 'prompt_head'],
                    help='select one of the training mode (default: %(default)s\)')


parser.add_argument('--save_checkpoint', type=bool, default=True,
                    help='whether to save the model')

parser.add_argument('--train', dest='train', help='Train the model', action='store_true')
parser.add_argument('--no-train', dest='train', help='Train the model', action='store_false')

parser.set_defaults(train=True)
 
parser.add_argument('--test', type=bool, default=False,
                    help='Test the model')

 
parser.add_argument('--count_params', type=bool, default=False,
                    help='Count model parameters')

args = parser.parse_args()


def train_model(config_train):

    dataset = config_train['dataset']
    dataloaders = config_train['dataloaders']
    model = config_train['model']
    device = config_train['device']
    criterion = config_train['criterion']
    optimizer = config_train['optimizer']
    mode = config_train['mode']
    scheduler = config_train['scheduler']
    epochs = config_train['epochs']
    save_checkpoint = config_train['save_checkpoint']


    model = model.to(device)

    best_valid_f1 = 0.0

    if save_checkpoint:
        saved_model_path = dataset + "_" + mode + '.pt'

    for epoch in range(0, epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))

        for phase in ['Train', 'Val']:
            
            batch_loss = 0.0000   #live loss
            batch_acc = 0.0000   #live accuracy

            y_true = []
            y_pred = []

            if phase == 'Train':
                model.train()
            else:
                model.eval()
            
            with tqdm(dataloaders[phase], unit="batch", desc=phase) as tepoch:

                for idx, (data, labels) in enumerate(tepoch):
                    input_ids =  data['input_ids'].squeeze(1).to(device)
                    attention_mask = data['attention_mask'].squeeze(1).to(device)
                    
                    
                    labels = labels.to(device)

                    if mode == 'prompt':
                        output = model(input_ids = input_ids, attention_mask = attention_mask).logits

                    else:
                        output = model(input_ids = input_ids, attention_mask = attention_mask)

                    loss = criterion(output, labels)

                    if phase == 'Train':

                        #zero gradients
                        optimizer.zero_grad() 

                        # Backward pass  (calculates the gradients)
                        loss.backward()   

                        optimizer.step()             # Updates the weights
                        
                        scheduler.step()
                        
                        
                    batch_loss += loss.item()
                        
                    _, preds = output.data.max(1)
                    y_pred.extend(preds.tolist())
                    y_true.extend(labels.tolist())
                    
                    batch_acc = get_accuracy(y_pred, y_true)
                    
                    tepoch.set_postfix(loss = batch_loss/(idx+1), accuracy = batch_acc )

                pre = precision_score(y_true, y_pred, average='weighted')
                recall = recall_score(y_true, y_pred, average='weighted')
                f1 = f1_score(y_true, y_pred, average='weighted')
                

                print("F1: {:.4f}, Precision: {:.4f}, Recall : {:.4f}.".format(f1, pre, recall))



                if save_checkpoint:
                
                    if phase == 'Val':
                        if f1 > best_valid_f1:
                            best_valid_f1 = f1
                            torch.save(model.state_dict(), saved_model_path)
                            print('Model Saved!')
                
                print()


def test_model(config_test):

    model = config_test['model']
    test_loader = config_test['test_loader']
    device = config_test['device']
    criterion = config_test['criterion']
    mode = config_test['mode']

    model = model.to(device)


    batch_loss = 0.0   #batch loss
    batch_acc = 0.0   #batch accuracy

    y_true = []
    y_pred = []

    # set the model to evaluation mode            
    model.eval()

    phase = 'Test'

    with tqdm(test_loader, unit="batch", desc=phase) as tepoch:
        
        for idx, (data, labels) in enumerate(tepoch):
            
            input_ids =  data['input_ids'].squeeze(1).to(device)
            attention_mask = data['attention_mask'].squeeze(1).to(device)

            labels = labels.to(device)
            
            with torch.no_grad():


                if mode == 'prompt':
                    output = model(input_ids = input_ids, attention_mask = attention_mask).logits
                    
                else:
                    output = model(input_ids = input_ids, attention_mask = attention_mask)

                loss = criterion(output, labels)
                
                _, preds = output.data.max(1)
                y_pred.extend(preds.tolist())
                y_true.extend(labels.tolist())
                
                batch_acc = get_accuracy(y_pred, y_true)
                batch_loss += loss.item()
                
            tepoch.set_postfix(loss = batch_loss/(idx+1), accuracy = batch_acc )


    pre = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print("")

    print("F1: {:.6f}, Precision: {:.6f}, Recall : {:.6f}".format(f1, pre, recall))

def main():

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

        train_data_object, test_data_object, val_data_object = get_imdb_dataset(dataset, number_of_tokens, tokenizer)

        dataloaders = get_dataloaders(train_data_object, test_data_object, val_data_object, batch_size)

        num_labels = 2


    elif dataset == 'topic':
        pass
    
    elif dataset == 'emotion':

        pass

    else:
        print('Select one of the given datasets')



    #Training modes
    if mode == 'prompt_head':

        if model_type == 'roberta-base':

            roberta = RobertaModel.from_pretrained(model_type)

            roberta = freeze_params(roberta)
            
            prompt_emb = PROMPTEmbedding(roberta.get_input_embeddings(), 
                                        n_tokens= number_of_tokens, 
                                        initialize_from_vocab=True)

            roberta.set_input_embeddings(prompt_emb)

            model = Model_Prompt_Head(roberta, num_labels)

    elif mode == 'prompt':
        if model_type == 'roberta-base':

            model = RobertaForSequenceClassification.from_pretrained(model_type, 
                                                         num_labels=num_labels,
                                                         output_attentions=False,
                                                         output_hidden_states=False)

            model =  freeze_params(model)

        
            prompt_emb = PROMPTEmbedding(model.get_input_embeddings(), 
                                        n_tokens= number_of_tokens, 
                                        initialize_from_vocab=True)

            model.set_input_embeddings(prompt_emb)

    elif mode == 'apt':
        if model_type == 'roberta-base':

            adapter_hidden_size = args.adapter_hidden_size
            adapter_modules = args.adapter_modules
            
            
            roberta = RobertaForSequenceClassification.from_pretrained(model_type, 
                                                         num_labels=num_labels,
                                                         output_attentions=False,
                                                         output_hidden_states=False)

            roberta = freeze_params(roberta)

            prompt_emb = PROMPTEmbedding(roberta.get_input_embeddings(), 
                      n_tokens= number_of_tokens, 
                      initialize_from_vocab=True)

            roberta.set_input_embeddings(prompt_emb)

            model = APT(roberta, adapter_hidden_size, adapter_modules)

    else:
        print('Select one of the given modes')

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
            num_warmup_steps=len(dataloaders['Train'])*epochs/40, 
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
            'save_checkpoint': args.save_checkpoint
        }

        train_model(config_train)



    if args.test:

        saved_model_path = dataset + "_" + mode + '.pt'

        if os.path.exists(saved_model_path):
            model.load_state_dict(torch.load(saved_model_path))
            print("Model checkpoint loaded succesfully")

        else:
            print("Required model checkpoint not found!")

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