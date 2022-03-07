
import torch
import torch.nn as nn
import torch.nn.functional as F



class Prompt_Head(torch.nn.Module):
    
    def __init__(self, base_model, num_classes):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Prompt_Head, self).__init__()
        
        self.base_model = base_model

        self.fcbert1 = nn.Linear(768, 128)
        self.fcbert2 = nn.Linear(128, 16)
        self.fcbert3 = nn.Linear(16, num_classes)

        self.dropout = nn.Dropout(0.3)
        


    def forward(self, input_ids, attention_mask):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # 1, 768
        output = self.base_model(input_ids = input_ids, attention_mask = attention_mask).pooler_output
        
        
        output = self.dropout(F.leaky_relu(self.fcbert1(output), 0.1))    #1, 128
         
        output = self.dropout(F.leaky_relu(self.fcbert2(output), 0.1))     #1, 16
        
        output = self.fcbert3(output)    #1, 3
         
        return output



class APT(torch.nn.Module):
    
    def __init__(self, base_model, adapter_hidden, adapter_modules):

        super(APT, self).__init__()
        
        
        self.base_model = base_model

        self.adapter_modules = adapter_modules
        
        self.adapter_1 = nn.Linear(768, adapter_hidden)
        
        self.adapter_2 = nn.Linear(adapter_hidden, 768)


    def forward(self, input_ids, attention_mask):
        
        # get output embeddings

        output_embed = self.base_model.roberta.embeddings(input_ids = input_ids)
        
        roberta_text = output_embed


        if self.adapter_modules == 1:
        
        
            # pass the output of embeddings into first 8 encoder layers
            for i in range(9):
                
                roberta_text = self.base_model.roberta.encoder.layer[i](roberta_text)[0]
                
                
            # pass the ouput of 8th encoder layer to adapter module
            roberta_text = self.adapter_1(roberta_text)
            
            roberta_text = self.adapter_2(roberta_text)
            
            
            # output of adapter layer to 9th encoder layer and so on
                
            for i in range(9, 12):
                roberta_text = self.base_model.roberta.encoder.layer[i](roberta_text)[0]
        
        
            # final output to classifier head
            output = self.base_model.classifier(roberta_text)
            
            # return final oitput
            return output

        else:
            print("Upto 3 adapter modules permitted")