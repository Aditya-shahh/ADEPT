
import os
import pandas as pd

import torch
from torch.utils.data import Dataset 



class CustomDataset(torch.utils.data.Dataset):
    """
    This is our custom dataset class which will load the text and their corresponding labels into Pytorch tensors
    """
    def __init__(self, labels, text, n_tokens, tokenizer, dataset):
        self.labels = labels
        self.text = text
        
        self.n_tokens = n_tokens
        self.tokenizer = tokenizer

        self.dataset = dataset

    def __getitem__(self, idx):
        sample = {}
        text = self.text[idx]

        if self.dataset == 'imdb':
            max_length = 512
        elif self.dataset == 'topic':
            max_length = 256
        elif self.dataset == 'emotion':
            max_length = 256
        else:
            print('Enter valid dataset')

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

        # train_data_v0 = train_data_v0.head(200)
        # train_data_v1 = train_data_v1.head(200)

        # test_data = test_data.head(100)
        # valid_data = valid_data.head(200)


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

        return train_text, train_labels, test_text, test_labels, valid_text, valid_labels

 

def create_dataset_object(text, labels, n_tokens, tokenizer, dataset):
    
    
    data_object = CustomDataset(
            labels = labels,
            text = text,
            n_tokens=n_tokens,
            tokenizer=tokenizer,
            dataset = dataset
        )

    return data_object


