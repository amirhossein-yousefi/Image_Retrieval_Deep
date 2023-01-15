from dataset_svhn import SVHNDataset
from utils import EarlyStopping, get_default_device, to_device, DeviceDataLoader
import torch
from torch.utils.data import random_split
from autoencoder_second_app import Autoencoder
import torchvision.transforms as transforms
from dataset_svhn import transform


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, model, train_loader, val_loader, opt_func=None):
    history = []
    optimizer = opt_func
    valid_losses = []
    early_stopping = EarlyStopping(patience=20, verbose=True,
                                   path='model_autoenc_app2_activation_relu_tanhenc_modif.pt')
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        early_stopping(result['val_loss'], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return history


def train():
    trans = transform()
    full_dataset = SVHNDataset("./data", 'unlabelled_task3', transform=trans)  # Create folder dataset.

    val_size = 9000
    train_size = len(full_dataset) - 9000

    # Split data to train and test
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create the train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    # Create the validation dataloader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32 * 2, drop_last=True)
    device = get_default_device()

    train_dl = DeviceDataLoader(train_loader, device)
    val_dl = DeviceDataLoader(val_loader, device)
    full_loader = DeviceDataLoader(full_dataset, device)
    # Create the full dataloader
    model = Autoencoder()
    encoder = model.encoder  # Our encoder model
    decoder = model.decoder  # Our decoder model

    to_device(encoder, device)  # GPU device
    to_device(decoder, device)
    to_device(model, device)
    num_epochs = 250
    opt_func = torch.optim.AdamW(model.parameters(), lr=0.005)
    history = fit(num_epochs, model, train_dl, val_dl, opt_func)
    return history


if __name__ == '__main__':
    train()
