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
import scipy.stats
from sklearn.linear_model import LinearRegression


# Dataset path
train_folder = "./datasets/comb/human_train/"
train_paths = [train_folder + file_name for file_name in os.listdir(train_folder)]

train_paths = train_paths

val_folder = "./datasets/comb/human_test/"
val_paths = [val_folder + file_name for file_name in os.listdir(val_folder)]
val_paths.sort()

# folder to save model checkpoints
train_save_folder = "./VAE_train/comb/human_1/"

os.makedirs(train_save_folder + "/test_images", exist_ok=True)
os.makedirs(train_save_folder + "/vol_plots", exist_ok=True)

# file to save losses
csv_file = 'loss_log.csv'

# epoch to load from
start_epoch = 0

# num epoch to train for
num_epochs = 1000

# how often to save model checkpoints and images
save_imgs = True
save_imgs_freq = 5
save_model_freq = 10

lr = 1e-4
batch_size = 4
accum_steps = 1 # 4

###################################################################################################################
################################################################################################################### finish setting some params

# Loss function for VAE
def loss_func(imgs, recons, means, log_vars):
    criterion = nn.CrossEntropyLoss()
    recon = criterion(recons, imgs) # computes average per voxel (in CVAE they use this instead to sum over all voxels)

    BS = batch_size
    num_voxels = img_x*img_y*img_z
    KLD = (-0.5 * torch.sum(1 + log_vars - means.pow(2) - log_vars.exp())) / (num_voxels * BS)

    return recon, KLD

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
# make cmap
# make cmap
base_colors = list(plt.cm.get_cmap('tab20').colors)
set_2 = list(plt.cm.get_cmap('Set2').colors)
base_colors.pop(14)
base_colors.pop(14)

n_labels = 20
base_colors = base_colors + set_2[:2]

get_greys = plt.cm.get_cmap('tab20c').colors

def rotate(l, n):
    return l[n:] + l[:n]

base_colors = rotate(base_colors, 11)

base_colors.insert(0, (1.0, 1.0, 1.0))
base_colors.insert(0, get_greys[17])
base_colors.insert(0, get_greys[16])
base_colors.insert(0, (1.0, 1.0, 1.0))


cmap = ListedColormap(base_colors)


print(len(base_colors))

n_labels = 23
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
            rgb_img = cmap(img)[..., :3]  # drop alpha
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

# Save vol ratios plot
def get_vol_ratio(img_arr):
    vol_ratios = []
    total_count = np.sum(img_arr != 0)

    for i in range(0, 24):
        vol_ratios.append(0 if total_count == 0 else np.sum(img_arr == i) / total_count)

    return vol_ratios


def plot_vol_ratios(real_vol_ratios, synth_vol_ratios, struct_names):
    common_struct_colors = base_colors

    fig2, axs2 = plt.subplots(2, 13, figsize=(40, 6))

    for plot_index in range(2):  # Two sets: 0–9 and 10–19
        
        for i in range(13):
            struct_i = plot_index * 13 + i
            if struct_i >= len(struct_names):
                break
            
            color = common_struct_colors[struct_i + 1]

            real_data = real_vol_ratios[:, struct_i]
            synth_data = synth_vol_ratios[:, struct_i]
            combined_min = min(np.min(real_data), np.min(synth_data))
            combined_max = max(np.max(real_data), np.max(synth_data))
            if combined_min == combined_max:
                continue
                
            # analyze matter preservation
            r, p = scipy.stats.pearsonr(real_data, synth_data)

            real_reshape = np.array(real_data).reshape(-1, 1)
            synth_reshape = np.array(synth_data).reshape(-1, 1)
            model = LinearRegression().fit(real_reshape, synth_reshape)

            # Predict y values for the regressed line
            y_pred = model.predict(real_reshape)

            axs2[plot_index, i].scatter(real_data, synth_data, color=color)
            axs2[plot_index, i].plot(real_data, y_pred, color="black")
            axs2[plot_index, i].set_xlim(combined_min*0.90, combined_max*1.03)
            axs2[plot_index, i].set_ylim(combined_min*0.90, combined_max*1.03)
            axs2[plot_index, i].set_title(struct_names[struct_i] + "(pearsons r=" + str(round(r, 2)) + ")", fontsize=8)


    fig2.tight_layout()
    return fig2


struct_names = ['lateral ventricle', 'basal forebrain', 'hippocampus', 'amygdala', 'fourth ventricle', 'thalamus', 'third ventricle', 'arbor vita of cerebellum', 'nucleus accumbens', 'globus pallidus']
struct_names = [[struct_name + " R", struct_name + " L"] for struct_name in struct_names]
struct_names = [n for pair in struct_names for n in pair]
struct_names = ['GM', 'WM', 'CSF'] + struct_names


# Create dataset and dataloader
train_dataset = Segmentation3DDataset(image_paths=train_paths)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = Segmentation3DDataset(image_paths=val_paths)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize model
encoder = Encoder_VAE(n_downsample=n_downsample, n_res=1, input_dim=1, dim=dim, norm='in', activ='relu', pad_type='zero') # encodes to 32 dim??
decoder = Decoder_VAE(n_upsample=n_downsample, n_res=1, dim=encoder.output_dim, output_dim=24)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder.to(device)
decoder.to(device)

# if not training from 0 load a pretrained model from saved checkpoint
if start_epoch != 0:
    encoder.load_state_dict(torch.load(train_save_folder + "checkpoint_epoch_" + str(start_epoch))['encoder_state_dict'])
    decoder.load_state_dict(torch.load(train_save_folder + "checkpoint_epoch_" + str(start_epoch))['decoder_state_dict'])

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

save_loss = []

# for saving images during training
display_size = 16 # num images to display

early_stopping = EarlyStopping(patience=20, save_path=train_save_folder + "best_model.pth")

beta = 10

print("Start training!!")
optimizer.zero_grad()
for i, epoch in enumerate(range(start_epoch, start_epoch + num_epochs + 1)):
    epoch_start_time = time.time()
    encoder.train()
    decoder.train()

    train_loss = 0
    train_recon_loss = 0
    train_KLD_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)  # (B, C, D, H, W)

        # Forward pass
        means, log_vars = encoder(batch)

        # get latent vectors - sampled from learned dists
        z = reparameterization(means, log_vars)

        recon = decoder(z)

        # Compute loss
        recon_loss, KLD_loss = loss_func(batch.squeeze(1).long(), recon, means, log_vars)
        loss = (recon_loss + beta*KLD_loss) / accum_steps

        # Backprop
        loss.backward()
        if (batch_idx+1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_KLD_loss += KLD_loss.item()

    # if there are still losses to accumlate
    if (batch_idx + 1) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    encoder.eval()
    decoder.eval()
    val_loss = 0
    val_recon_loss = 0
    val_KLD_loss = 0

    img_to_save = []
    recon_to_save = []

    img_vol_ratios = []
    recon_vol_ratios = []

    with torch.no_grad():
        for i, val_batch in enumerate(val_loader):
            val_batch = val_batch.to(device)

            means, log_vars = encoder(val_batch)
            # get latent vectors - sampled from learned dists
            z = reparameterization(means, log_vars)
            recon = decoder(z)
            # Compute loss
            recon_loss, KLD_loss = loss_func(val_batch.squeeze(1).long(), recon, means, log_vars)
            loss = (recon_loss + beta*KLD_loss)  / accum_steps

            val_loss += loss.item()
            val_recon_loss += recon_loss.item()
            val_KLD_loss += KLD_loss.item()

            recon = torch.argmax(recon, dim=1).squeeze().detach().cpu()

            img_vol_ratios.append(get_vol_ratio(val_batch.squeeze().detach().cpu().numpy()))
            recon_vol_ratios.append(get_vol_ratio(recon.numpy()))

            # save a few test images
            if save_imgs and (epoch % save_imgs_freq == 0) and i < 3:
                recon_to_save.append(recon) # might just be a shallow copy

                # to display
                img_to_save.append(val_batch)

            if save_imgs and i >= 3 and i < display_size and (epoch % save_imgs_freq == 0):
                recon_to_save.append(recon)
                img_to_save.append(val_batch)

        # save a png of some reconstructions - to observe during training
        if save_imgs and epoch % save_imgs_freq == 0:
            img_to_save = torch.stack(img_to_save).squeeze()
            recon_to_save = torch.stack(recon_to_save)

            print("save images...")
            __write_images([img_to_save, recon_to_save], display_size, train_save_folder + "test_images/recons_epoch_" + str(epoch) + ".png")

            # save vol ratios plot
            fig = plot_vol_ratios(np.array(img_vol_ratios), np.array(recon_vol_ratios), struct_names)
            fig.savefig(train_save_folder + "vol_plots/epoch_"+ str(epoch) + ".png")


    train_loss = train_loss / len(train_loader)
    train_recon_loss = train_recon_loss / len(train_loader)
    train_KLD_loss = train_KLD_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    val_recon_loss = val_recon_loss / len(val_loader)
    val_KLD_loss = val_KLD_loss / len(val_loader)

    elapsed_time = time.time() - epoch_start_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f} (time: {elapsed_time:.4f})")
    
    # write loss to a csv every epoch
    with open(train_save_folder + 'loss_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, val_loss, train_recon_loss, val_recon_loss, train_KLD_loss, val_KLD_loss])  # Writes a single row with two values
   
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


