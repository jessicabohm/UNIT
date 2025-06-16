from torch.utils.data import Dataset
import torch
import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from networks_update import Encoder, Decoder
import csv
import time

class Segmentation3DDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform  # Optional (e.g., normalization, crop)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.image_paths[idx])
        image = sitk.GetArrayFromImage(image)

        # Add channel dimension if needed (C x D x H x W)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        if self.transform:
            image = self.transform(image)

        return torch.from_numpy(image).to(torch.float)
    

# Dataset path
train_folder = "../3D-CycleGan-Pytorch-MedImaging/Data_folder_train_2/train/labels/"
train_paths = [train_folder + file_name for file_name in os.listdir(train_folder)]

train_paths = train_paths

val_folder = "../3D-CycleGan-Pytorch-MedImaging/Data_folder_train_2/test/labels/"
val_paths = [val_folder + file_name for file_name in os.listdir(val_folder)]
val_paths.sort()

# folder to save model checkpoints
train_save_folder = "./save_models/human_train_5/"

# file to save losses
csv_file = 'loss_log.csv'

# epoch to load from
start_epoch = 0

# num epoch to train for
num_epochs = 500

# how often to save model checkpoints and images
save_imgs = True
save_freq = 20

###################################################################################################################
################################################################################################################### finish setting some params

# Create dataset and dataloader
train_dataset = Segmentation3DDataset(image_paths=train_paths)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = Segmentation3DDataset(image_paths=val_paths)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize model
encoder = Encoder(n_downsample=2, n_res=4, input_dim=1, dim=4, norm='in', activ='relu', pad_type='zero') # encodes to 32 dim??
decoder = Decoder(n_upsample=2, n_res=4, dim=encoder.output_dim, output_dim=4)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder.to(device)
decoder.to(device)

# if not training from 0 load a pretrained model from saved checkpoint
if start_epoch != 0:
    encoder.load_state_dict(torch.load(train_save_folder + "checkpoint_epoch_" + str(start_epoch))['encoder_state_dict'])
    decoder.load_state_dict(torch.load(train_save_folder + "checkpoint_epoch_" + str(start_epoch))['decoder_state_dict'])

# Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

save_loss = []

print("Start training!!")
for epoch in range(start_epoch, start_epoch + num_epochs + 1):
    epoch_start_time = time.time()
    encoder.train()
    decoder.train()
    train_loss = 0

    for batch in train_loader:
        batch = batch.to(device)  # (B, C, D, H, W)

        # Forward pass
        z = encoder(batch)
        recon = decoder(z)

        # Compute loss
        loss = criterion(recon, batch.squeeze().to(torch.long))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    encoder.eval()
    decoder.eval()
    val_loss = 0

    with torch.no_grad():
        for i, val_batch in enumerate(val_loader):
            z = encoder(val_batch.float().to(device))
            recon = decoder(z)
            loss = criterion(recon, val_batch.squeeze(1).long().to(device))
            val_loss += loss.item()

            # save a few test images
            if save_imgs and (epoch % save_freq == 0) and i < 3:
                recon = torch.argmax(recon, dim=1)
                recon_img = sitk.GetImageFromArray(np.array(recon.detach().cpu()).squeeze())
                sitk.WriteImage(recon_img, train_save_folder + "test_images/recon_" + val_paths[i].split("/")[-1].split(".")[0] + "_epoch_" + str(epoch) + ".nii")
    
    train_loss = train_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    elapsed_time = time.time() - epoch_start_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f} (time: {elapsed_time:.4f})")
    
    # write loss to a csv every epoch
    with open(train_save_folder + 'loss_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, val_loss])  # Writes a single row with two values

    # save model checkpoint
    if epoch % save_freq == 0:
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': val_loss,
        }

        torch.save(checkpoint, train_save_folder + "checkpoint_epoch_" + str(epoch))

