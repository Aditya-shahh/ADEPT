
import os
from pydoc import describe
import pandas as pd
from sklearn.model_selection import train_test_split

import json
import torch
import pickle
from torch.utils.data import Dataset 

import sys

class CustomDataset(torch.utils.data.Dataset):
    """
    This is our custom dataset class which will load the text and their corresponding labels into Pytorch tensors
    """
    def __init__(self, labels, text, n_tokens, tokenizer, dataset, mode):
        self.labels = labels
        self.text = text
        
        self.n_tokens = n_tokens
        self.tokenizer = tokenizer

        self.dataset = dataset

        self.mode = mode

    def __getitem__(self, idx):
        sample = {}
        text = self.text[idx]

        max_length = 512


        if self.mode == 'finetune':

            #Roberta Tokenizer to tokenize the text
            inputs = self.tokenizer.encode_plus(text, 
                                            add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                            max_length=max_length, 
                                            truncation=True, 
                                            return_tensors='pt',
                                            padding="max_length")

            return inputs, torch.tensor(self.labels[idx])

        else:


            #Roberta Tokenizer to tokenize the text
            inputs = self.tokenizer.encode_plus(text, 
                                            add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                            max_length=max_length-self.n_tokens, 
                                            truncation=True, 
                                            return_tensors='pt',
                                            padding="max_length")
            
            inputs['input_ids'] = torch.cat([torch.full((1,self.n_tokens), 5256), inputs['input_ids']], 1)
            inputs['attention_mask'] = torch.cat([torch.full((1, self.n_tokens), 1), inputs['attention_mask']], 1)

            return inputs, torch.tensor(self.labels[idx])

    
    def __len__(self):
        return len(self.labels)
        

class SuperGlueDataset(torch.utils.data.Dataset):
    """
    This is Boolq dataset class which will load the text and their corresponding labels into Pytorch tensors
    """
    def __init__(self, labels, questions, passages, n_tokens, tokenizer, dataset, mode):

        self.labels = labels

        self.questions = questions
        self.passages = passages
        

        self.n_tokens = n_tokens
        self.tokenizer = tokenizer

        self.dataset = dataset

        self.mode = mode

    def __getitem__(self, idx):

        question = self.questions[idx]
        passage = self.passages[idx]


        max_length = 512


        if self.mode == 'finetune':

            #Roberta Tokenizer to tokenize the text
            inputs = self.tokenizer.encode_plus(question, passage,
                                            add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                            max_length=max_length, 
                                            truncation=True, 
                                            return_tensors='pt',
                                            padding="max_length")

            if self.labels != None:
                
                return inputs, torch.tensor(self.labels[idx])

            else:

                return inputs, torch.tensor(0)

        else:


            #Roberta Tokenizer to tokenize the text
            inputs = self.tokenizer.encode_plus(question, passage,
                                            add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                            max_length=max_length-self.n_tokens, 
                                            truncation=True, 
                                            return_tensors='pt',
                                            padding="max_length")
            
            inputs['input_ids'] = torch.cat([torch.full((1,self.n_tokens), 5256), inputs['input_ids']], 1)
            inputs['attention_mask'] = torch.cat([torch.full((1, self.n_tokens), 1), inputs['attention_mask']], 1)

            if self.labels != None:
                
                return inputs, torch.tensor(self.labels[idx])

            else:

                return inputs, torch.tensor(0)

        
    
    def __len__(self):
        return len(self.questions)


# LOAD Datasets      

def load_imdb_dataset(dataset):



        if os.path.exists(os.path.join(dataset, "train.csv")):

            train_path = os.path.join(dataset, "train.csv")

        else:
            raise FileNotFoundError


        if os.path.exists(os.path.join(dataset, "test.csv")):

            test_path = os.path.join(dataset, "test.csv")

        else:
            raise FileNotFoundError

        if os.path.exists(os.path.join(dataset, "valid.csv")):

            valid_path = os.path.join(dataset, "valid.csv")

        else:
            raise FileNotFoundError
        

        # load data

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        valid_data = pd.read_csv(valid_path)


        train_text = train_data['text']
        train_labels = train_data['label']

        valid_text = valid_data['text']
        valid_labels = valid_data['label']

        test_text = test_data['text']
        test_labels = test_data['label']

        return train_text, train_labels, test_text, test_labels, valid_text, valid_labels



def load_topic_dataset(dataset):

        if os.path.exists(os.path.join(dataset, "train_v0.csv")):

            train_path_v0 = os.path.join(dataset, "train_v0.csv")

        else:
            raise FileNotFoundError


        if os.path.exists(os.path.join(dataset, "train_v1.csv")):

            train_path_v1 = os.path.join(dataset, "train_v1.csv")

        else:
            raise FileNotFoundError


        if os.path.exists(os.path.join(dataset, "test.csv")):

            test_path = os.path.join(dataset, "test.csv")

        else:
            raise FileNotFoundError

        if os.path.exists(os.path.join(dataset, "valid.csv")):

            valid_path = os.path.join(dataset, "valid.csv")

        else:
            raise FileNotFoundError

        train_data_v0 = pd.read_csv(train_path_v0)
        train_data_v1 = pd.read_csv(train_path_v1)

        test_data = pd.read_csv(test_path)
        valid_data = pd.read_csv(valid_path)

        train_text_v0 = train_data_v0['input_string'].tolist()
        train_labels_v0 = train_data_v0['label'].tolist()

        train_text_v1 = train_data_v1['input_string'].tolist()
        train_labels_v1 = train_data_v1['label'].tolist()

        valid_text = valid_data['input_string'].tolist()
        valid_labels = valid_data['label'].tolist()

        test_text = test_data['input_string'].tolist()
        test_labels = test_data['label'].tolist()

        train_text_v0.extend(train_text_v1)
        train_labels_v0.extend(train_labels_v1)

        train_text = train_text_v0
        train_labels = train_labels_v0

        del train_text_v0
        del train_labels_v0

        del train_text_v1
        del train_labels_v1

        #sample the dataset


        return train_text, train_labels, test_text, test_labels, valid_text, valid_labels


def load_yelp_dataset(dataset):

    
    with open(os.path.join(dataset, 'train_text.pkl'), 'rb') as f:
        train_text = pickle.load(f)

    with open(os.path.join(dataset, 'train_labels.pkl'), 'rb') as f:
        train_labels = pickle.load(f)

    with open(os.path.join(dataset, 'test_text.pkl'), 'rb') as f:
        test_text = pickle.load(f)

    with open(os.path.join(dataset, 'test_labels.pkl'), 'rb') as f:
        test_labels = pickle.load(f)
        
    # train_text, _, train_labels, _ = train_test_split(train_text, train_labels, train_size=0.001, random_state = 42)


    train_text, valid_text, train_labels, valid_labels = train_test_split(train_text, train_labels, train_size=0.8, random_state = 42)



    return train_text, train_labels, test_text, test_labels, valid_text, valid_labels


def load_agnews_dataset(dataset):

        if os.path.exists(os.path.join(dataset, "train.csv")):

            train_path = os.path.join(dataset, "train.csv")

        else:
            raise FileNotFoundError


        if os.path.exists(os.path.join(dataset, "test.csv")):

            test_path = os.path.join(dataset, "test.csv")

        else:
            raise FileNotFoundError

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        train_title = train_data['Title']
        train_description = train_data['Description']

        train_text = [title +" "+ description for title, description in zip(train_title, train_description)]

        train_labels = train_data['Class Index'].values


        test_title = test_data['Title']
        test_description = test_data['Description']

        test_text = [title +" "+ description for title, description in zip(test_title, test_description)]

        test_labels = test_data['Class Index'].values


        train_text, valid_text, train_labels, valid_labels = train_test_split(train_text, train_labels, train_size=0.85, random_state = 42)

        #labels start from 0
        train_labels = [label-1 for label in train_labels]

        valid_labels = [label-1 for label in valid_labels]

        test_labels = [label-1 for label in test_labels]

        # #validate
        # train_text = train_text[:50]
        # train_labels = train_labels[:50]

        # valid_text = valid_text[:10]
        # valid_labels = valid_labels[:10]

        # test_text = test_text[:10]
        # test_labels = test_labels[:10]

        return train_text, train_labels, test_text, test_labels, valid_text, valid_labels

def load_boolq_dataset(dataset):

    if os.path.exists(os.path.join(dataset, "train.jsonl")):

        train_path = os.path.join(dataset, "train.jsonl")

    else:
        raise FileNotFoundError


    if os.path.exists(os.path.join(dataset, "test.jsonl")):

        test_path = os.path.join(dataset, "test.jsonl")

    else:
        raise FileNotFoundError

    if os.path.exists(os.path.join(dataset, "val.jsonl")):

        val_path = os.path.join(dataset, "val.jsonl")

    else:
        raise FileNotFoundError

    with open(train_path, 'r') as train_json_file:
        train_json_list = list(train_json_file)

    with open(val_path, 'r') as val_json_file:
        val_json_list = list(val_json_file)

    with open(test_path, 'r') as test_json_file:
        test_json_list = list(test_json_file)

    train_ques = []
    train_pass = []
    train_labels = []

    val_ques = []
    val_pass = []
    val_labels = []

    test_ques = []
    test_pass = []


    for train_data in train_json_list:
        train_result = json.loads(train_data)
        train_ques.append(train_result['question'])
        train_pass.append(train_result['passage'])

        if train_result["label"] == True:
            train_labels.append(0)
        else:
            train_labels.append(1)


    for val_data in val_json_list:
        val_result = json.loads(val_data)
        val_ques.append(val_result['question'])
        val_pass.append(val_result['passage'])

        if val_result['label'] == True:
            val_labels.append(0)
        else:
            val_labels.append(1)


    for test_data in test_json_list:
        result = json.loads(test_data)
        test_ques.append(result['question'])
        test_pass.append(result['passage'])


    return train_ques, train_pass, train_labels, val_ques, val_pass, val_labels, test_ques, test_pass



def load_cb_dataset(dataset):

    if os.path.exists(os.path.join(dataset, "train.jsonl")):

        train_path = os.path.join(dataset, "train.jsonl")

    else:
        raise FileNotFoundError


    if os.path.exists(os.path.join(dataset, "test.jsonl")):

        test_path = os.path.join(dataset, "test.jsonl")

    else:
        raise FileNotFoundError

    if os.path.exists(os.path.join(dataset, "val.jsonl")):

        val_path = os.path.join(dataset, "val.jsonl")

    else:
        raise FileNotFoundError

    with open(train_path, 'r') as train_json_file:
        train_json_list = list(train_json_file)

    with open(val_path, 'r') as val_json_file:
        val_json_list = list(val_json_file)

    with open(test_path, 'r') as test_json_file:
        test_json_list = list(test_json_file)

    train_prem = []
    train_hypo = []
    train_labels = []

    val_prem = []
    val_hypo = []
    val_labels = []

    test_prem = []
    test_hypo = []


    for train_data in train_json_list:
        train_result = json.loads(train_data)
        train_prem.append(train_result['premise'])
        train_hypo.append(train_result['hypothesis'])

        if train_result["label"] == 'contradiction':
            train_labels.append(0)
        elif train_result["label"] == 'entailment':
            train_labels.append(1)
        else:
            train_labels.append(2)



    for val_data in val_json_list:
        val_result = json.loads(val_data)
        val_prem.append(val_result['premise'])
        val_hypo.append(val_result['hypothesis'])

        if val_result["label"] == 'contradiction':
            val_labels.append(0)

        elif val_result["label"] == 'entailment':
            val_labels.append(1)

        else:
            val_labels.append(2)


    for test_data in test_json_list:
        result = json.loads(test_data)
        test_prem.append(result['premise'])
        test_hypo.append(result['hypothesis'])


    return train_prem, train_hypo, train_labels, val_prem, val_hypo, val_labels, test_prem, test_hypo
 

def load_rte_dataset(dataset):

    if os.path.exists(os.path.join(dataset, "train.jsonl")):

        train_path = os.path.join(dataset, "train.jsonl")

    else:
        raise FileNotFoundError


    if os.path.exists(os.path.join(dataset, "test.jsonl")):

        test_path = os.path.join(dataset, "test.jsonl")

    else:
        raise FileNotFoundError

    if os.path.exists(os.path.join(dataset, "val.jsonl")):

        val_path = os.path.join(dataset, "val.jsonl")

    else:
        raise FileNotFoundError

    with open(train_path, 'r') as train_json_file:
        train_json_list = list(train_json_file)

    with open(val_path, 'r') as val_json_file:
        val_json_list = list(val_json_file)

    with open(test_path, 'r') as test_json_file:
        test_json_list = list(test_json_file)

    train_prem = []
    train_hypo = []
    train_labels = []

    val_prem = []
    val_hypo = []
    val_labels = []

    test_prem = []
    test_hypo = []


    for train_data in train_json_list:
        train_result = json.loads(train_data)
        train_prem.append(train_result['premise'])
        train_hypo.append(train_result['hypothesis'])

        if train_result["label"] == 'entailment':
            train_labels.append(0)
        elif train_result["label"] == 'not_entailment':
            train_labels.append(1)
        else:
            print("WOW")



    for val_data in val_json_list:
        val_result = json.loads(val_data)
        val_prem.append(val_result['premise'])
        val_hypo.append(val_result['hypothesis'])

        if val_result["label"] == 'entailment':
            val_labels.append(0)

        elif val_result["label"] == 'not_entailment':
            val_labels.append(1)

        else:
            print("WOW")


    for test_data in test_json_list:
        result = json.loads(test_data)
        test_prem.append(result['premise'])
        test_hypo.append(result['hypothesis'])


    return train_prem, train_hypo, train_labels, val_prem, val_hypo, val_labels, test_prem, test_hypo

def create_dataset_object(text, labels, n_tokens, tokenizer, dataset, mode):    
    
    data_object = CustomDataset(
            labels = labels,
            text = text,
            n_tokens=n_tokens,
            tokenizer=tokenizer,
            dataset = dataset,
            mode = mode
        )

    return data_object



def create_superglue_dataset_object(question, passage, labels, n_tokens, tokenizer, dataset, mode):    
    
    data_object = SuperGlueDataset(
            labels = labels,
            questions = question,
            passages = passage,
            n_tokens=n_tokens,
            tokenizer=tokenizer,
            dataset = dataset,
            mode = mode
        )

    return data_object

