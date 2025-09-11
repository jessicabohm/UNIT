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
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# Dataset path
train_folder = "./datasets/3d_anat_align_64/human_train/"
train_paths = [train_folder + file_name for file_name in os.listdir(train_folder)]

train_paths = train_paths

val_folder = "./datasets/3d_anat_align_64/human_test/"
val_paths = [val_folder + file_name for file_name in os.listdir(val_folder)]
val_paths.sort()

# folder to save model checkpoints
train_save_folder = "./VAE_train/3d_anat/human_train_64_2/"

os.makedirs(train_save_folder + "/test_images", exist_ok=True)

# file to save losses
csv_file = 'loss_log.csv'

# epoch to load from
start_epoch = 0

# num epoch to train for
num_epochs = 1000

# how often to save model checkpoints and images
save_imgs = True
save_imgs_freq = 5
save_model_freq = 5

lr=1e-4
batch_size= 4

###################################################################################################################
################################################################################################################### finish setting some params

# Loss function for VAE
def loss_func(imgs, recons, means, log_vars):
    criterion = nn.CrossEntropyLoss()
    recon = criterion(recons, imgs) # computes average per voxel (in CVAE they use this instead to sum over all voxels)

    BS = batch_size
    #num_voxels = 120*120*128 # NOTE: update for 3D
    num_voxels = 64*64*64 # NOTE: update for 2D
    beta = 10
    KLD = (-0.5 * torch.sum(1 + log_vars - means.pow(2) - log_vars.exp())) / (num_voxels * BS)

    return recon + beta*KLD

def reparameterization(means, log_vars):
    # move random vars sampled from a normal dist to size log_vars to device
    epsilon = torch.randn_like(log_vars).to(device) 
    std = torch.exp(0.5 * log_vars)
    z = means + std * epsilon
    return z

class Segmentation3DDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform  # Optional (e.g., normalization, crop)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.image_paths[idx])
        image = sitk.GetArrayFromImage(image)

        image = np.expand_dims(image, axis=0)
        if self.transform:
            image = self.transform(image)

        return torch.from_numpy(image).to(torch.float)

two_d = False

# ===== Your consistent color setup =====
base_colors = plt.cm.get_cmap('tab20').colors  # 20 RGBA colors
n_labels = 300
repeated_colors = np.tile(base_colors, (n_labels // 20 + 1, 1))[:n_labels]
cmap = ListedColormap(repeated_colors)
n_labels = 270
norm = BoundaryNorm(np.arange(n_labels + 1), cmap.N)

# Manually set label 0 to white
colors_with_white_bg = cmap.colors
colors_with_white_bg[0] = (1.0, 1.0, 1.0)  # RGB white
cmap = ListedColormap(colors_with_white_bg)

def __write_images(image_outputs, display_image_num, file_name):
    imgs, recons = image_outputs

    # If 3D volumes: [B, D, H, W] → take middle slice
    if imgs.ndim == 4:
        slice_idx = imgs.shape[1] // 2
        imgs = torch.stack([img[slice_idx] for img in imgs[:display_image_num]])
        recons = torch.stack([img[slice_idx] for img in recons[:display_image_num]])
    else:
        imgs = imgs[:display_image_num]
        recons = recons[:display_image_num]

    # Apply discrete colormap and return RGB tensors
    def apply_cmap_rgb(tensor):
        arr = tensor.cpu().numpy().astype(np.int32)
        rgb_list = []
        for img in arr:
            rgb_img = cmap(norm(img))[..., :3]  # drop alpha
            rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1)  # [3, H, W]
            rgb_list.append(rgb_tensor)
        return torch.stack(rgb_list)

    imgs_rgb = apply_cmap_rgb(imgs)
    recons_rgb = apply_cmap_rgb(recons)

    # Create grids
    grid_in = vutils.make_grid(imgs_rgb, nrow=display_image_num, padding=2)
    grid_rec = vutils.make_grid(recons_rgb, nrow=display_image_num, padding=2)

    # Stack vertically
    full_grid = torch.cat([grid_in, grid_rec], dim=1)

    vutils.save_image(full_grid, file_name)

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, save_path="best_model.pth"):
        """
        patience: how many epochs to wait after last improvement
        verbose: print updates
        save_path: where to save the best model
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.verbose = verbose
        self.save_path = save_path

    def __call__(self, val_loss, model_dict):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model_dict)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model_dict):
        if self.verbose:
            print(f"Validation loss improved → {self.best_loss:.4f}. Saving model...")
        torch.save(model_dict, self.save_path)


# Create dataset and dataloader
train_dataset = Segmentation3DDataset(image_paths=train_paths)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = Segmentation3DDataset(image_paths=val_paths)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize model
encoder = Encoder_VAE(n_downsample=2, n_res=1, input_dim=1, dim=8, norm='in', activ='relu', pad_type='zero') # encodes to 32 dim??
decoder = Decoder_VAE(n_upsample=2, n_res=1, dim=encoder.output_dim, output_dim=271)

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
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

save_loss = []

# for saving images during training
display_size = 16 # num images to display

early_stopping = EarlyStopping(patience=5, save_path=train_save_folder + "best_model.pth")


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


        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    encoder.eval()
    decoder.eval()
    val_loss = 0

    img_to_save = []
    recon_to_save = []

    with torch.no_grad():
        for i, val_batch in enumerate(val_loader):
            val_batch = val_batch.to(device)

            means, log_vars = encoder(val_batch)
            # get latent vectors - sampled from learned dists
            z = reparameterization(means, log_vars)
            recon = decoder(z)
            # Compute loss
            loss = loss_func(val_batch.squeeze(1).long(), recon, means, log_vars)

            val_loss += loss.item()

            # save a few test images
            if save_imgs and (epoch % save_imgs_freq == 0) and i < 3:
                recon = torch.argmax(recon, dim=1).squeeze().detach().cpu()
                recon_to_save.append(recon) # might just be a shallow copy

                #recon_img = sitk.GetImageFromArray(np.array(recon))
                # save a few nifty image reconstructions - use for analysis
                #sitk.WriteImage(recon_img, train_save_folder + "test_images/recon_" + val_paths[i].split("/")[-1].split(".")[0] + "_epoch_" + str(epoch) + ".nii")

                # to display
                img_to_save.append(val_batch)

            if save_imgs and i >= 3 and i < display_size and (epoch % save_imgs_freq == 0):
                recon = torch.argmax(recon, dim=1).squeeze().detach().cpu()
                recon_to_save.append(recon)
                img_to_save.append(val_batch)

        # save a png of some reconstructions - to observe during training
        if save_imgs and epoch % save_imgs_freq == 0:
            img_to_save = torch.stack(img_to_save).squeeze()
            recon_to_save = torch.stack(recon_to_save)

            print("save images...")
            __write_images([img_to_save, recon_to_save], display_size, train_save_folder + "test_images/recons_epoch_" + str(epoch) + ".png")


    train_loss = train_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    elapsed_time = time.time() - epoch_start_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f} (time: {elapsed_time:.4f})")
    
    # write loss to a csv every epoch
    with open(train_save_folder + 'loss_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, val_loss])  # Writes a single row with two values
   
    if epoch % save_model_freq == 0:
        # model dict for saving
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': val_loss,
        }

        # check early stopping
        early_stopping(val_loss, checkpoint)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

#    # save model checkpoint
#    if epoch % save_model_freq == 0:
#        checkpoint = {
#            'epoch': epoch,
#            'encoder_state_dict': encoder.state_dict(),
#            'decoder_state_dict': decoder.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'train_loss': train_loss,
#            'test_loss': val_loss,
#        }
#
#        torch.save(checkpoint, train_save_folder + "checkpoint_epoch_" + str(epoch))

