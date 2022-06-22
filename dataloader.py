import torch
from torch.utils.data import  DataLoader

def get_dataloaders(train_data_object, test_data_object, val_data_object, batch_size):

    train_loader = DataLoader(
        train_data_object ,
        batch_size=batch_size,
        pin_memory=True,
        num_workers= 64,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_data_object,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=64,
        shuffle=False,
        drop_last=False
    )

    val_loader = DataLoader(
        val_data_object,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=64,
        shuffle=True,
        drop_last=True
    )

    dataloaders = {'Train': train_loader, 'Test': test_loader, 'Val': val_loader}

    return dataloaders