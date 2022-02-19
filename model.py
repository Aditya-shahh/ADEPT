
import torch
import torch.nn as nn
import torch.nn.functional as F



class Model_Prompt_Head(torch.nn.Module):
    
    def __init__(self, bert, num_classes):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Model_Prompt_Head, self).__init__()
        
        self.bert = bert
        
        
        
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
        output_bert = self.bert(input_ids = input_ids, attention_mask = attention_mask).pooler_output
        
        
        output = self.dropout(F.leaky_relu(self.fcbert1(output_bert), .1))    #1, 128
         
        output = self.dropout(F.leaky_relu(self.fcbert2(output), 0.1))     #1, 16
        
        output = self.fcbert3(output)    #1, 3
         
        return output