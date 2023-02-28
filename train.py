import torch
import time
import srcnn
import torch.optim as optim
import torch.nn as nn
import os
import argparse
from tqdm import tqdm
from datasets import get_datasets, get_dataloaders
from utils import (
    psnr, save_model, save_model_state, 
    save_plot, save_validation_results
)
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', default=100, type=int,
    help='number of epochs to train for'
)
parser.add_argument(
    '-w', '--weights', default=None,
    help='weights/checkpoint path to resume training'
)
parser.add_argument(
    '-hr', '--hr-path', default=None, dest='hr_path',
    help='path to the high-res images training set'
)
parser.add_argument(
    '-lr', '--lr-path', default=None, dest='lr_path',
    help='path to the low-res images training set'
)
parser.add_argument(
    '-vh', '--hr-validation-path', default=None, dest='hr_val_path',
    help='path to the high-res images validation set'
)
parser.add_argument(
    '-vl', '--lr-validation-path', default=None, dest='lr_val_path',
    help='path to the low-res images validation set'
)
parser.add_argument(
    '-s', '--save-path', default=None, dest='save_path',
    help='path to the directory to save models and figures'
)
args = vars(parser.parse_args())
# Learning parameters.
epochs = args['epochs'] # Number of epochs to train the SRCNN model for.
lr = 0.001 # Learning rate.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Constants
TRAIN_LABEL_PATHS = args['hr_path']
TRAIN_IMAGE_PATHS = args['lr_path']
VALID_LABEL_PATHS = args['hr_val_path']
VALID_IMAGE_PATHS = args['lr_val_path']
save_path = args['save_path']
SAVE_VALIDATION_RESULTS = True
os.makedirs(save_path+'/valid_results', exist_ok=True)

# Initialize the model.
print('Computation device: ', device)
model = srcnn.SRCNN().to(device)
if args['weights'] is not None:
    print('Loading weights to resume training...')
    checkpoint = torch.load(args['weights'])
    model.load_state_dict(checkpoint['model_state_dict'])
print(model)
# Optimizer.
optimizer = optim.Adam(model.parameters(), lr=lr)
# Loss function. 
criterion = nn.MSELoss()
dataset_train, dataset_valid = get_datasets(
    TRAIN_IMAGE_PATHS, TRAIN_LABEL_PATHS,
    VALID_IMAGE_PATHS, VALID_LABEL_PATHS
)
train_loader, valid_loader = get_dataloaders(dataset_train, dataset_valid)
print(f"Training samples: {len(dataset_train)}")
print(f"Validation samples: {len(dataset_valid)}")

def train(model, dataloader):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        image_data = data[0].to(device)
        label = data[1].to(device)
        # Zero grad the optimizer.
        optimizer.zero_grad()
        outputs = model(image_data)
        loss = criterion(outputs, label)
        # Backpropagation.
        loss.backward()
        # Update the parameters.
        optimizer.step()
        # Add loss of each item (total items in a batch = batch size).
        running_loss += loss.item()
        # Calculate batch psnr (once every `batch_size` iterations).
        batch_psnr =  psnr(label, outputs)
        running_psnr += batch_psnr
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/len(dataloader)
    return final_loss, final_psnr
def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image_data = data[0].to(device)
            label = data[1].to(device)
            outputs = model(image_data)
            loss = criterion(outputs, label)
            # Add loss of each item (total items in a batch = batch size) .
            running_loss += loss.item()
            # Calculate batch psnr (once every `batch_size` iterations).
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr
            # For saving the batch samples for the validation results
            # every 500 epochs.
            if SAVE_VALIDATION_RESULTS and ((epoch-1) % 1) == 0:
                save_validation_results(image_data, outputs, epoch, bi, save_path)
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/len(dataloader)
    return final_loss, final_psnr
train_loss, val_loss = [], []
train_psnr, val_psnr = [], []
start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")
    train_epoch_loss, train_epoch_psnr = train(model, train_loader)
    val_epoch_loss, val_epoch_psnr = validate(model, valid_loader, epoch+1)
    print(f"Train loss: {train_epoch_loss:.5f}")
    print(f"Val loss: {val_epoch_loss:.5f}")
    print(f"Train PSNR: {train_epoch_psnr:.3f}")
    print(f"Val PSNR: {val_epoch_psnr:.3f}")
    train_loss.append(train_epoch_loss)
    train_psnr.append(train_epoch_psnr)
    val_loss.append(val_epoch_loss)
    val_psnr.append(val_epoch_psnr)
    
    # Save model with all information every 100 epochs. Can be used 
    # resuming training.
    if (epoch+1) % 100 == 0:
        save_model(epoch, model, optimizer, criterion,save_path)
    # Save the model state dictionary only every epoch. Small size, 
    # can be used for inference.
    save_model_state(model,save_path)
    # Save the PSNR and loss plots every epoch.
    save_plot(train_loss, val_loss, train_psnr, val_psnr,save_path)
end = time.time()
print(f"Finished training in: {((end-start)/60):.3f} minutes") 