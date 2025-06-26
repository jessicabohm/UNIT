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
from networks_update import *
import csv
import time

# Dataset path
train_folder = "../3D-CycleGan-Pytorch-MedImaging/Data_folder_train_2/train/images/"
train_paths = [train_folder + file_name for file_name in os.listdir(train_folder)]

train_paths = train_paths

val_folder = "../3D-CycleGan-Pytorch-MedImaging/Data_folder_train_2/test/images/"
val_paths = [val_folder + file_name for file_name in os.listdir(val_folder)]
val_paths.sort()

# folder to save model checkpoints
train_save_folder = "./VAE_train/train_1/"

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
    
class Encoder_VAE(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(Encoder_VAE, self).__init__()
        self.model = []
        self.model += [Conv3dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv3dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]

        latent_dim = 1024
        flattened_dim = 16*30*30*32

        self.model += [nn.Flatten(), nn.Linear(flattened_dim, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU()]

        self.output_dim = dim

        # NOTE: dim might be incorrect here??
        self.inplace = nn.Linear(latent_dim, latent_dim)

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        out = self.model(x)
        means = self.inplace(out)
        log_vars = self.inplace(out) # why was it called log vars? Probs cuz of how it's used in KL divergence
        return means, log_vars
    
class Reshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (16, 32, 30, 30)  # e.g., (-1, 512) or (batch_size, channels, height, width)

    def forward(self, x):
        return x.reshape(x.size(0), *self.shape)  # keeps batch dim intact

class Decoder_VAE(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='in', activ='relu', pad_type='zero'): # NOTE: updated normalization to in to not have to compute weight and bias externally
        super(Decoder_VAE, self).__init__()

        self.model = []

        latent_dim = 1024
        flattened_dim = 16*30*30*32

        self.model += [nn.Linear(latent_dim, flattened_dim), nn.LayerNorm(flattened_dim), nn.ReLU(), Reshape()]
        
        # AdaIN residual blocks # NOTE: changed!!
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv3dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)] # NOTE: could update to instance norm since only a batch size of 2 -> don't want to normalize over full layer??
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv3dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='none', pad_type=pad_type)] 
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
    
# Loss function for VAE
def loss_func(imgs, recons, means, log_vars):
    criterion = nn.CrossEntropyLoss()
    recon = criterion(recons, imgs) # computes average per voxel (in CVAE they use this instead to sum over all voxels)

    BS = 4
    num_voxels = 1843200
    beta = 10
    KLD = (-0.5*torch.sum(1 + log_vars - means.pow(2) - log_vars.exp())) / num_voxels*BS

    return recon + beta*KLD

def reparameterization(means, log_vars):
    # move random vars sampled from a normal dist to size log_vars to device
    epsilon = torch.randn_like(log_vars).to(device) 
    z = means + log_vars*epsilon
    return z
    
# Create dataset and dataloader
train_dataset = Segmentation3DDataset(image_paths=train_paths)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = Segmentation3DDataset(image_paths=val_paths)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize model
encoder = Encoder_VAE(n_downsample=2, n_res=4, input_dim=1, dim=4, norm='in', activ='relu', pad_type='zero') # encodes to 32 dim??
decoder = Decoder_VAE(n_upsample=2, n_res=4, dim=encoder.output_dim, output_dim=4)

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
        means, log_vars = encoder(batch)

        # get latent vectors - sampled from learned dists
        z = reparameterization(means, log_vars)

        recon = decoder(z)

        # Compute loss
        loss = loss_func(batch.squeeze(1).long(), recon, means, log_vars)


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

            means, log_vars = encoder(batch)
            # get latent vectors - sampled from learned dists
            z = reparameterization(means, log_vars)
            recon = decoder(z)
            # Compute loss
            loss = loss_func(batch.squeeze(1).long(), recon, means, log_vars)

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

