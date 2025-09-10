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
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


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
        # pad_x = 4
        # pad_y = 4
        # pad_z = 12
        # image = np.pad(image, ((pad_x, pad_x), (pad_y, pad_y)))

        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        
        if self.transform:
            image = self.transform(image)

        return torch.from_numpy(image).to(torch.float)
    

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


# Dataset path
train_folder = "./datasets/3d_anat/mouse_train/"
train_paths = [train_folder + file_name for file_name in os.listdir(train_folder)]

train_paths = train_paths

val_folder = "./datasets/3d_anat/mouse_test/"
val_paths = [val_folder + file_name for file_name in os.listdir(val_folder)]
val_paths.sort()

# folder to save model checkpoints
train_save_folder = "./save_anat_e_d/mouse_train_3/"
os.makedirs(train_save_folder, exist_ok=True)


# file to save losses
csv_file = 'loss_log.csv'

# epoch to load from
start_epoch = 0

# num epoch to train for
num_epochs = 1000

# how often to save model checkpoints and images
save_imgs = True
save_imgs_freq = 25
save_model_freq = 25

###################################################################################################################
################################################################################################################### finish setting some params

# Create dataset and dataloader
train_dataset = Segmentation3DDataset(image_paths=train_paths)
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4, pin_memory=True)

val_dataset = Segmentation3DDataset(image_paths=val_paths)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# for saving images during training
display_size = 16 # num images to display

# Initialize model
encoder = Encoder(n_downsample=3, n_res=4, input_dim=1, dim=4, norm='in', activ='relu', pad_type='zero') # encodes to 32 dim??
decoder = Decoder(n_upsample=3, n_res=4, dim=encoder.output_dim, output_dim=271) # output_dim=# channels in seg (4/271 - since include background)

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

os.makedirs(train_save_folder + "test_images/" , exist_ok=True)

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
        loss = criterion(recon, batch.squeeze(1).long())

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
            z = encoder(val_batch.float().to(device))
            recon = decoder(z)
            loss = criterion(recon, val_batch.squeeze(1).long().to(device))
            val_loss += loss.item()

            # save a few test images
            if save_imgs and (epoch % save_imgs_freq == 0) and i < 3:
                recon = torch.argmax(recon, dim=1).squeeze().detach().cpu()
                recon_to_save.append(recon) # might just be a shallow copy

                recon_img = sitk.GetImageFromArray(np.array(recon))
                # save a few nifty image reconstructions - use for analysis
                sitk.WriteImage(recon_img, train_save_folder + "test_images/recon_" + val_paths[i].split("/")[-1].split(".")[0] + "_epoch_" + str(epoch) + ".nii")

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

    # save model checkpoint
    if epoch % save_model_freq == 0:
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': val_loss,
        }

        torch.save(checkpoint, train_save_folder + "checkpoint_epoch_" + str(epoch))

