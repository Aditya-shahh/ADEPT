
import torch
import torch.nn as nn
import torch.nn.functional as F


class APT(torch.nn.Module):
    
    def __init__(self, base_model, adapter_hidden, adapter_modules, model_type):

        super(APT, self).__init__()

        self.model_type = model_type
        
        self.base_model = base_model

        self.adapter_modules = adapter_modules

        if self.adapter_modules ==1:
        
            self.adapter_1_inp = nn.Linear(768, adapter_hidden)
            
            self.adapter_1_out = nn.Linear(adapter_hidden, 768)

        else:
            print("Only 1 adapter module permitted")



    def forward(self, input_ids, attention_mask):
        
        # get output embeddings

        if self.model_type == 'roberta-base':

            output_embed = self.base_model.roberta.embeddings(input_ids = input_ids)
            
            roberta_text = output_embed


            if self.adapter_modules == 1:
            
                # pass the output of embeddings into first 8 encoder layers
                for i in range(9):
                    
                    roberta_text = self.base_model.roberta.encoder.layer[i](roberta_text)[0]
                    
                    
                # pass the ouput of 8th encoder layer to adapter module
                roberta_text = self.adapter_1_inp(roberta_text)
                
                roberta_text = self.adapter_1_out(roberta_text)
                
                
                # output of adapter layer to 9th encoder layer and so on
                    
                for i in range(9, 12):
                    roberta_text = self.base_model.roberta.encoder.layer[i](roberta_text)[0]
            
            
                # final output to classifier head
                output = self.base_model.classifier(roberta_text)
                
                # return final oitput
                return output

            else:
                print("Only 1 adapter module permitted")

        elif self.model_type == 'bert-base-cased':

            output_embed = self.base_model.bert.embeddings(input_ids = input_ids)
            
            bert_text = output_embed
            
            # pass the output of embeddings into first 8 encoder layers
            for i in range(9):
                
                bert_text = self.base_model.bert.encoder.layer[i](bert_text)[0]
                
                
            # pass the ouput of 8th encoder layer to adapter module
            bert_text = self.adapter_1_inp(bert_text)
            
            bert_text = self.adapter_1_out(bert_text)
        
            
            # output of adapter layer to 9th encoder layer and so on
                
            for i in range(9, 12):
                bert_text = self.base_model.bert.encoder.layer[i](bert_text)[0]
        
            # final output to classifier head
            output = self.base_model.classifier(bert_text)
            
            # return final oitput
            return output[:, 0, :]

        else:
            print("Choose correct model")


