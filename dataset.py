
import os
import pandas as pd

import torch
from torch.utils.data import Dataset 



class IMDB_Dataset(torch.utils.data.Dataset):
    """
    This is our custom dataset class which will load the text and their corresponding labels into Pytorch tensors
    """
    def __init__(self, labels, text, n_tokens, tokenizer):
        self.labels = labels
        self.text = text
        
        self.n_tokens = n_tokens
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sample = {}
        text = self.text[idx]

        #Roberta Tokenizer to tokenize the text
        inputs = self.tokenizer.encode_plus(text, 
                                        add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                        max_length=512-self.n_tokens, 
                                        truncation=True, 
                                        return_tensors='pt',
                                        padding="max_length")
        
        inputs['input_ids'] = torch.cat([torch.full((1,self.n_tokens), 5256), inputs['input_ids']], 1)
        inputs['attention_mask'] = torch.cat([torch.full((1, self.n_tokens), 1), inputs['attention_mask']], 1)

        
        return inputs, torch.tensor(self.labels[idx])
    
    def __len__(self):
        return len(self.labels)
        



def get_imdb_dataset(dataset, number_of_tokens, tokenizer):

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



        train_data_object = IMDB_Dataset(
            labels = train_labels,
            text = train_text,
            n_tokens=number_of_tokens,
            tokenizer=tokenizer
        )

        test_data_object = IMDB_Dataset(
            labels = test_labels,
            text = test_text,
            n_tokens=number_of_tokens,
            tokenizer=tokenizer
        )

        val_data_object = IMDB_Dataset(
            labels = valid_labels,
            text = valid_text,
            n_tokens=number_of_tokens,
            tokenizer=tokenizer
        )

        return train_data_object, test_data_object, val_data_object