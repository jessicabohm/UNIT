import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import csv
import pandas as pd

import scipy.stats
from sklearn.linear_model import LinearRegression

import torch
from networks_update import Encoder, Decoder, Encoder_VAE, Decoder_VAE
import math
import random
import pickle



from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

#latent_shape = [16, 32, 30, 30] 
latent_shape = [2000]
latent_len = math.prod(latent_shape)
middle_feat = 256
do=0.5

MLP = nn.Sequential(
    nn.Linear(2*latent_len, middle_feat),  # compress
    nn.ReLU(),
    nn.LayerNorm(middle_feat),
    #nn.Dropout(do),

    # nn.Linear(middle_feat, middle_feat),    # map in latent space
    # nn.ReLU(),

    nn.Linear(middle_feat, middle_feat),    # map in latent space
    nn.ReLU(),
    
    nn.LayerNorm(middle_feat),
    #nn.Dropout(do),

    nn.Linear(middle_feat, 2*latent_len)   # reconstruct
)


class LatentVectorsVolRatios(Dataset):
    def __init__(self, mouse_encodings_folder, human_encodings_folder, common_vols):
        self.mouse_encodings_folder = mouse_encodings_folder
        self.human_encodings_folder = human_encodings_folder
        
        self.mouse_encodings_files = os.listdir(mouse_encodings_folder)
        self.human_encodings_files = os.listdir(human_encodings_folder)

        num_samples = min(len(self.mouse_encodings_files), len(self.human_encodings_files))
        self.mouse_encodings_files = self.mouse_encodings_files[:num_samples]
        self.human_encodings_files = self.human_encodings_files[:num_samples]

        self.paths = ([mouse_encodings_folder + file for file in self.mouse_encodings_files]) + ([human_encodings_folder + file for file in self.human_encodings_files])


        # compute vol ratios
        self.features = []
        for path in self.paths:
            split_path = path.split("/")
            is_mouse = False
            is_test = False

            if split_path[3].split("_")[1] == "mouse":
                is_mouse = True

            is_test = (split_path[4].split("_")[0] == "test")

            img_path = "./datasets/3d_anat_align/" + ("mouse_" if is_mouse else "human_") + ("test/" if is_test else "train/") + split_path[-1][8:-3]
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_path))

            vol_ratios = []
            total_count = np.sum(img_arr != 0)

            common_struct_idxs = np.arange(251, 270 + 1)

            for i in common_struct_idxs[common_vols]:
                vol_ratios.append(np.sum(img_arr == i) / total_count)

            self.features.append(vol_ratios)
        
        self.mouse_features = np.array(self.features[:num_samples])
        self.human_features = np.array(self.features[num_samples:])

        self.human_features_non_norm = self.human_features
        self.mouse_features_non_norm = self.mouse_features
        
        # calc mean and std to z-norm for loss term
        self.mouse_features_mean = np.mean(self.mouse_features, axis=0)
        self.mouse_features_std = np.std(self.mouse_features, axis=0)

        self.mouse_features = (self.mouse_features - self.mouse_features_mean) / self.mouse_features_std

        # calc mean and std to z-norm for loss term
        self.human_features_mean = np.mean(self.human_features, axis=0)
        self.human_features_std = np.std(self.human_features, axis=0)

        self.human_features = (self.human_features - self.human_features_mean) / self.human_features_std


        # from path, features, species arrays - split by species and reorder by gm low to high
        self.mouse_encodings_paths = np.array(self.paths[:num_samples])
        self.human_encodings_paths = np.array(self.paths[num_samples:])


    def get_human_features_min_maxes(self):
        return np.min(self.human_features_non_norm, axis=0), np.max(self.human_features_non_norm, axis=0)


    def get_human_features_non_norm(self, num):
        return self.human_features_non_norm[:, :num]
    
    def get_mouse_features_non_norm(self, num):
        return self.mouse_features_non_norm[:, :num]

    def get_mouse_mean_and_std(self):
        return self.mouse_features_mean, self.mouse_features_std
    
    def get_human_mean_and_std(self):
        return self.human_features_mean, self.human_features_std

    def get_features(self):
        return self.human_features, self.mouse_features
    
    def __getitem__(self, idx):
        # mouse_latent_encoding = torch.from_numpy(np.load(self.mouse_encodings_paths[idx]))
        # human_latent_encoding = torch.from_numpy(np.load(self.human_encodings_paths[idx]))
        
        mouse_latent_encoding = torch.load(self.mouse_encodings_paths[idx])
        mouse_latent_encoding = [mouse_latent_encoding[0].detach(), mouse_latent_encoding[1].detach()]

        human_latent_encoding = torch.load(self.human_encodings_paths[idx])
        human_latent_encoding = [human_latent_encoding[0].detach(), human_latent_encoding[1].detach()]

        
        #return [mouse_latent_encoding, human_latent_encoding]
        return [mouse_latent_encoding, human_latent_encoding, self.mouse_features[idx]]
    
    def __len__(self):
        return len(self.human_encodings_paths)
    
# HELPERS

# Ensures consistent colouring of values even with missing labels
from matplotlib.colors import ListedColormap, BoundaryNorm

# Get the 20 distinct colors from tab20
base_colors = plt.cm.get_cmap('tab20').colors  # tuple of 20 RGBA colors

# Repeat them to cover 300 labels
n_labels = 271
repeated_colors = np.tile(base_colors, (n_labels // 20 + 1, 1))[:n_labels]

# Create a ListedColormap
cmap = ListedColormap(repeated_colors)

def plot_seg(img_arr, slice_idx, size, title):
    plt.figure(figsize=(size, size))
    #plt.imshow(np.ma.masked_where(img_arr[slice_idx] == 0, img_arr[slice_idx]), cmap=cmap, norm=norm, alpha=1.0, interpolation='nearest')
    plt.imshow(np.ma.masked_where(img_arr[slice_idx] == 0, img_arr[slice_idx]), cmap=cmap, vmin=0, vmax=270, alpha=1.0, interpolation='nearest')
    plt.title(title)
    plt.show()

def reparameterization(means, log_vars, deterministic=False):
    if deterministic:
        return means
    
    # move random vars sampled from a normal dist to size log_vars to device
    epsilon = torch.randn_like(log_vars).to("cuda") 
    std = torch.exp(0.5 * log_vars)
    z = means + std * epsilon
    return z

def get_vol_ratios_diff(imgs, common_vols):
    #print(imgs.shape)  # e.g. [BS, 271, 120, 120]

    # Flatten spatial dimensions
    imgs = imgs.view(imgs.shape[0], imgs.shape[1], -1)  # [BS, num_channels, voxels]

    # Select structure indices
    common_struct_idxs = torch.arange(251, 271, device=imgs.device)
    struct_idxs = common_struct_idxs[common_vols]

    background_val = 0
    imgs = torch.softmax(imgs, dim=1)
    back_count = torch.sum(imgs[:,background_val,:], dim=1)
    total_count = imgs.shape[-1] - back_count

    vol_ratios_list = []
    for struct_idx in struct_idxs:
        struct_count = torch.sum(imgs[:,struct_idx,:], dim=1)
        ratio_per_vol = struct_count / total_count
        vol_ratios_list.append(ratio_per_vol)

    # Stack and permute to [BS, num_structs]
    vol_ratios = torch.stack(vol_ratios_list, dim=1)

    return vol_ratios


def get_vol_ratios_non_diff(imgs, common_vols):
    imgs = torch.argmax(imgs, dim=1).reshape(-1, 128*120*120)

    total_counts = torch.sum(imgs != 0, dim=1)

    # Structure labels to check (251 → 270 inclusive)
    common_struct_idxs = torch.arange(251, 271)
    
    # Only keep the selected structures
    struct_idxs = common_struct_idxs[common_vols]

    vol_ratios = []
    for struct_idx in struct_idxs:
        struct_counts = (imgs == struct_idx).sum(dim=1).float()
        vol_ratios.append(torch.where(total_counts == 0, torch.zeros_like(struct_counts), struct_counts / total_counts))
    
    vol_ratios = torch.stack(vol_ratios, dim=1).cuda()
    return vol_ratios

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


# TRAIN
# Test out different models to learn the mapping between latent vectors
import csv
import pickle

# mouse_encodings_folder = "./save_anat_e_d/mouse_train_3/train_latent_encodings/"
# human_encodings_folder = "./save_anat_e_d/human_train_3/train_latent_encodings/"

# mouse_test_encodings_folder = "./save_anat_e_d/mouse_train_3/test_latent_encodings/"
# human_test_encodings_folder = "./save_anat_e_d/human_train_3/test_latent_encodings/"4

common_vols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] # [10, 11] # thalumus


mouse_encodings_folder = "./VAE_train/3d_anat/arc_mouse_train_7/train_encodings/"
human_encodings_folder = "./VAE_train/3d_anat/arc_human_train_7/train_encodings/"

mouse_test_encodings_folder = "./VAE_train/3d_anat/arc_mouse_train_7/test_encodings/"
human_test_encodings_folder = "./VAE_train/3d_anat/arc_human_train_7/test_encodings/"

dataset = LatentVectorsVolRatios(mouse_encodings_folder, human_encodings_folder, common_vols=common_vols)
human_ratios_means, human_ratios_stds = dataset.get_human_mean_and_std()

batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

valdataset = LatentVectorsVolRatios(mouse_test_encodings_folder, human_test_encodings_folder, common_vols=common_vols)
valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = MLP.cuda() #ElementWiseLinear(latent_len).cuda()  # move to GPU
#model.load_state_dict(torch.load('./pca/gm_loss_mod_500.pth')['model_state_dict'])

criterion = nn.MSELoss()

def calc_KLD(means, log_vars):
    num_voxels = 120*120*128 # NOTE: update for 3D
    #num_voxels = 120*120 # NOTE: update for 3D
    KLD = (-0.5 * torch.sum(1 + log_vars - means.pow(2) - log_vars.exp())) / (num_voxels * batch_size)
    return KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)#, weight_decay=1e-5)
#optimizer.load_state_dict(torch.load('./pca/gm_loss_mod_500.pth')['optimizer_state_dict'])
train_save_folder = "./map_3d_anat_align/all_common/"
csv_path = train_save_folder + "loss.csv"
os.makedirs(train_save_folder, exist_ok=True)

mode = 'w' # 'w' -> overwrite, 'a' -> append


if mode == 'w': # new file
    with open(csv_path, mode=mode, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "val_loss", "train_mse_loss", "val_mse_loss", "train_loss_s1_diff", "val_loss_s1_diff", "train_loss_s1_non_diff", "val_loss_s1_non_diff"])


# Human decoder to translate the mapped latent encoding - to evaluate objective
human_model_folder = "./VAE_train/3d_anat/arc_human_train_7/"
decoder_trans = Decoder_VAE(n_upsample=2, n_res=1, dim=16, output_dim=271)
decoder_trans.load_state_dict(torch.load(human_model_folder + "best_model.pth")['decoder_state_dict'])

# FREEEEZE parameters so it's not trained!!
for param in decoder_trans.parameters():
    param.requires_grad = False
decoder_trans.eval().cuda()


# Training loop
start_epoch = 0
save_model_freq = 10
epochs = 2000
early_stop_patience = 10
save_loss = []
epsilon = 1e-6
alpha = 1 # ratio preservation
beta = 1 # MSE 
gamma = 0
pca_encodings = []

early_stopping = EarlyStopping(patience=early_stop_patience, save_path=train_save_folder + "best_model.pth")


print("start training!!")
for epoch in range(start_epoch, epochs + start_epoch):
    model.train()

    train_loss = 0.0
    train_loss_s1_non_diff = 0.0
    train_loss_s1_diff = 0.0
    train_mse_loss = 0.0
    train_KLD = 0.0
    for mouse_batch, human_batch, mouse_features_batch in dataloader:

        mouse_batch = torch.cat([mouse_batch[0].cuda(), mouse_batch[1].cuda()], dim=2).squeeze(1)
        human_batch = [human_batch[0].cuda(), human_batch[1].cuda()]
        mouse_features_batch = mouse_features_batch.cuda()

        #print("mouse_batch:", mouse_batch.shape)
        human_out = model(mouse_batch)
        #print("human_out:", human_out.shape)

        u = human_out[:, :latent_len]
        std = human_out[:, latent_len:]

        #print("std:", std.shape)

        human_z = reparameterization(u, std)
        #print("human_z:", human_z.shape)


        optimizer.zero_grad()

        recon_imgs = decoder_trans(human_z)
        #print("recon_imgs:", recon_imgs.shape)


        vols_diff = get_vol_ratios_diff(recon_imgs, common_vols)
        vols_non_diff = get_vol_ratios_non_diff(recon_imgs, common_vols)

        vols_norm_diff = (vols_diff - torch.asarray(human_ratios_means).cuda()) / (torch.asarray(human_ratios_stds).cuda() + 1e-6)
        vols_norm_non_diff = (vols_non_diff - torch.asarray(human_ratios_means).cuda()) / (torch.asarray(human_ratios_stds).cuda() + 1e-6)
    
        # print(torch.asarray(human_ratios_means).cuda().shape)
        # print(vols_norm_diff.shape)
        mse_loss = criterion(u, human_batch[0].squeeze()) +  criterion(std, human_batch[1].squeeze()) # mse(means) + mse(vars)
        #mse_loss = criterion(human_out, human_batch)

        # loses dim here
        vols_loss_diff = torch.mean(torch.abs(mouse_features_batch - vols_norm_diff), dim=0)

        vols_loss_non_diff = torch.mean(torch.abs(mouse_features_batch - vols_norm_non_diff), dim=0) # non diff

        # print(vols_loss_diff.shape)
        # print(vols_loss_non_diff.shape)

        loss_s1_diff = vols_loss_diff[0]
        loss_s1_non_diff = vols_loss_non_diff[0]

        loss_s2_diff = vols_loss_diff[1]
        loss_s2_non_diff = vols_loss_non_diff[1]

        tot_vol_loss = torch.sum(vols_loss_diff)

        KLD = calc_KLD(u, std)
        loss = beta*mse_loss + alpha*(tot_vol_loss)/len(common_vols) + gamma*KLD

        loss.backward()
        optimizer.step()

        train_mse_loss += mse_loss.item() * mouse_batch.size(0)
        train_loss_s1_diff += loss_s1_diff.item() * mouse_batch.size(0)
        train_loss_s1_non_diff += loss_s1_non_diff.item() * mouse_batch.size(0)
        train_loss += loss.item() * mouse_batch.size(0)
        train_KLD += KLD.item() * mouse_batch.size(0)
        

    train_mse_loss = train_mse_loss / len(dataset)
    train_loss_s1_diff = train_loss_s1_diff / len(dataset)
    train_loss_s1_non_diff = train_loss_s1_non_diff / len(dataset)
    train_loss = train_loss / len(dataset)
    train_KLD = train_KLD / len(dataset)

    model.eval()
    val_loss = 0.0
    val_mse_loss = 0.0
    val_loss_s1_non_diff = 0.0
    val_loss_s1_diff = 0.0
    val_KLD = 0.0
    
    for mouse_batch, human_batch, mouse_features_batch in valloader:
        mouse_batch = torch.cat([mouse_batch[0].cuda(), mouse_batch[1].cuda()], dim=2).squeeze(1)
        human_batch = [human_batch[0].cuda(), human_batch[1].cuda()]
        mouse_features_batch = mouse_features_batch.cuda()

        human_out = model(mouse_batch)

        u = human_out[:, :latent_len]
        std = human_out[:, latent_len:]

        human_z = reparameterization(u, std)

        optimizer.zero_grad()

        recon_imgs = decoder_trans(human_z)


        vols_diff = get_vol_ratios_diff(recon_imgs, common_vols)
        vols_non_diff = get_vol_ratios_non_diff(recon_imgs, common_vols)

        vols_norm_diff = (vols_diff - torch.asarray(human_ratios_means).cuda()) / (torch.asarray(human_ratios_stds).cuda() + 1e-6)
        vols_norm_non_diff = (vols_non_diff - torch.asarray(human_ratios_means).cuda()) / (torch.asarray(human_ratios_stds).cuda() + 1e-6)
    
        vols_loss_diff = torch.mean(torch.abs(mouse_features_batch - vols_norm_diff), dim=0)

        vols_loss_non_diff = torch.mean(torch.abs(mouse_features_batch - vols_norm_non_diff), dim=0) # non diff

        # print("u:", u.shape)
        # print("human_batch[0]:", human_batch[0].squeeze().shape)

        mse_loss = criterion(u, human_batch[0].squeeze()) +  criterion(std, human_batch[1].squeeze()) # mse(means) + mse(vars)

        loss_s1_diff = vols_loss_diff[0]
        loss_s1_non_diff = vols_loss_non_diff[0]

        loss_s2_diff = vols_loss_diff[1]
        loss_s2_non_diff = vols_loss_non_diff[1]

        tot_vol_loss = torch.sum(vols_loss_diff)

        KLD = calc_KLD(u, std)
        loss = beta*mse_loss + alpha*(tot_vol_loss)/len(common_vols)  + gamma*KLD

        val_mse_loss += mse_loss.item() * mouse_batch.size(0)
        val_loss_s1_diff += loss_s1_diff.item() * mouse_batch.size(0)
        val_loss_s1_non_diff += loss_s1_non_diff.item() * mouse_batch.size(0)
        val_loss += loss.item() * mouse_batch.size(0)
        val_KLD += KLD.item() * mouse_batch.size(0)
        

    val_mse_loss = val_mse_loss / len(valdataset)
    val_loss_s1_diff = val_loss_s1_diff / len(valdataset)
    val_loss_s1_non_diff = val_loss_s1_non_diff / len(valdataset)
    val_loss = val_loss / len(valdataset)
    val_KLD = val_KLD / len(valdataset)

    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Train mse: {train_mse_loss:.6f}, Val mse: {val_mse_loss:.6f}, Train s1: {train_loss_s1_diff:.6f}, Val s1: {val_loss_s1_diff:.6f}, Train KLD: {train_KLD:.6f}")

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss, train_mse_loss, val_mse_loss, train_loss_s1_diff, val_loss_s1_diff, train_loss_s1_non_diff, val_loss_s1_non_diff])

    if epoch % save_model_freq == 0:
        # model dict for saving
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        # check early stopping
        early_stopping(val_loss, checkpoint)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
