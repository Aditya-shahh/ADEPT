
from prettytable import PrettyTable


#to calculate accuracy

def get_accuracy(preds, labels):
  total_acc = 0.0
  
  for i in range(len(labels)):
    if labels[i] == preds[i]:
      total_acc+=1.0
  
  return total_acc / len(labels)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    print()



def freeze_params(model):
  
  for param in model.parameters():
    param.requires_grad = False

  return model


def freeze_params_encoder(model, model_type):

  if model_type == 'roberta-base':
    
    for param in model.roberta.parameters():
      param.requires_grad = False

    return model

  if model_type == 'bert-base-cased':

    for name, param in model.named_parameters():
      if 'classifier' not in name: 
        if "pooler" not in name:   
          param.requires_grad = False

    return model