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

        # # NOTE: just for a quick train test - pad the image to 144^3
        # pad_x = 8
        # pad_y = 12
        # pad_z = 12
        # image = np.pad(image, ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)))

        # Add channel dimension if needed (C x D x H x W)
        #if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
        
        if self.transform:
            image = self.transform(image)

        return torch.from_numpy(image).to(torch.float)
    

# Dataset path

test_folder = "./datasets/anat/mouse_test/"
test_files = os.listdir(test_folder)
test_paths = [test_folder + file_name for file_name in test_files]

# folder to save model checkpoints
train_save_folder = "./save_anat_e_d/mouse_train_1/"

# checkpoint to load
epoch = 300

# folder to save reconstructed images to
recon_folder = "recon_images_epoch_" + str(epoch)
os.makedirs(train_save_folder + recon_folder, exist_ok=True)

# where to eval (NOTE: wayyyy faster with gpu - 6 sec vs 2 min 30 sec)
gpu = True

###################################################################################################################
################################################################################################################### finish setting some params

# Create dataset and dataloader
test_dataset = Segmentation3DDataset(image_paths=test_paths)
test_loader = DataLoader(test_dataset, batch_size=1)

# Initialize model

# NOTE: current anat train:
encoder = Encoder(n_downsample=3, n_res=1, input_dim=1, dim=2, norm='in', activ='relu', pad_type='zero') # encodes to 32 dim??
decoder = Decoder(n_upsample=3, n_res=1, dim=encoder.output_dim, output_dim=271)

# load from checkpoint to test
encoder.load_state_dict(torch.load(train_save_folder + "checkpoint_epoch_" + str(epoch))['encoder_state_dict'])
decoder.load_state_dict(torch.load(train_save_folder + "checkpoint_epoch_" + str(epoch))['decoder_state_dict'])

# Move to GPU if available
if gpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

encoder.to(device)
decoder.to(device)

# Loss function (for example, reconstruction loss)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

encoder.eval()
decoder.eval()


test_loss = 0
test_start_time = time.time()

with torch.no_grad():
    for i, test_batch in enumerate(test_loader):
        z = encoder(test_batch.float().to(device))
        recon = decoder(z)
        loss = criterion(recon, test_batch.squeeze(1).long().to(device))
        test_loss += loss.item()
        
        recon = torch.argmax(recon, dim=1)
        recon_img = sitk.GetImageFromArray(np.array(recon.detach().cpu()).squeeze())
        sitk.WriteImage(recon_img, train_save_folder + recon_folder + "/recon_" + test_files[i] + ".nii")

test_loss = test_loss / len(test_loader)
elapsed_time = time.time() - test_start_time
print(f"Test loss: {test_loss:.4f} (time: {elapsed_time/60:.4f} min)")


