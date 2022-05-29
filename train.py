import os
import torch
from tqdm import tqdm

from utils import get_accuracy

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from torch.optim import lr_scheduler


#TEST MODEL
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

                if mode =='apt':
                    output = model(input_ids = input_ids, attention_mask = attention_mask)

                else:
                    output = model(input_ids = input_ids, attention_mask = attention_mask).logits

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



# TRAIN MODEL
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

    checkpoint = config_train['checkpoint']


    model = model.to(device)

    best_valid_f1 = 0.0

    if save_checkpoint:
        if checkpoint != None:
            saved_model_path = os.path.join("saved_models", checkpoint)
        
        else:
            saved_model_path = os.path.join("saved_models", dataset + "_" + mode + '.pt')

    print("Model will be saved at", saved_model_path)

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


                    if mode =='apt':
                        output = model(input_ids = input_ids, attention_mask = attention_mask)

                    else:
                        output = model(input_ids = input_ids, attention_mask = attention_mask).logits


                    # elif mode == 'finetune' or mode == 'head':
                    #      output = model(input_ids = input_ids, attention_mask = attention_mask).logits

                    # else:  # prompt and prompt head
                    #     output = model(input_ids = input_ids, attention_mask = attention_mask).logits

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