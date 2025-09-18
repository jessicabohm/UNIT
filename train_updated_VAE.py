import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import SimpleITK as sitk
import numpy as np
import csv
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import scipy.stats
from sklearn.linear_model import LinearRegression
import warnings
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)


latent_dim = 2000
pool_val = 8

#####################################
# Minimal 3D VQ-VAE
#####################################

class Encoder3D(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, z_dim=64):
        super().__init__()
        
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, z_dim, 3, stride=1, padding=1)
        )

        self.pool = nn.AdaptiveAvgPool3d((pool_val, pool_val, pool_val))  # shrink to fixed spatial size

        self.flat_map = nn.Sequential(
            nn.Flatten(),
            nn.Linear(z_dim*pool_val**3, (z_dim*pool_val**3)//2),
            nn.LayerNorm((z_dim*pool_val**3)//2),
            nn.ReLU(),

            nn.Linear((z_dim*pool_val**3)//2, (z_dim*pool_val**3)//4),
            nn.LayerNorm((z_dim*pool_val**3)//4),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.enc(x)         # (B, z_dim, D, H, W)
        out = self.pool(out)      # (B, z_dim, 8, 8, 8)
        out = self.flat_map(out)

        return out


class Decoder3D(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=64, out_channels=271, output_shape=(16,15,15)):
        super().__init__()
        self.z_dim = z_dim
        self.output_shape = output_shape  # store expected pre-convtranspose size
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(z_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, out_channels, 3, stride=1, padding=1)
        )

        self.unflatten_map = nn.Sequential(
            nn.Linear(latent_dim, (z_dim*pool_val**3)//4),
            nn.LayerNorm((z_dim*pool_val**3)//4),
            nn.ReLU(),
            nn.Linear((z_dim*pool_val**3)//4, (z_dim*pool_val**3)//2),
            nn.LayerNorm((z_dim*pool_val**3)//2),
            nn.ReLU(),
            nn.Linear((z_dim*pool_val**3)//2, z_dim*pool_val**3),
            nn.LayerNorm(z_dim*pool_val**3),
            nn.ReLU(),
        )

    def forward(self, z):
        z = self.unflatten_map(z)
        z = z.view(z.size(0), self.z_dim, pool_val, pool_val, pool_val)
        z = F.interpolate(z, size=self.output_shape, mode="trilinear", align_corners=False)
        return self.dec(z)


class VAE(nn.Module):
    def __init__(self, in_channels=1, num_classes=271, embedding_dim=64):
        super().__init__()
        self.encoder = Encoder3D(in_channels, hidden_dim=64, z_dim=embedding_dim)
        self.decoder = Decoder3D(z_dim=embedding_dim, hidden_dim=64, out_channels=num_classes)
        
        self.fc_mu = nn.Linear((embedding_dim*pool_val**3)//4, latent_dim)
        self.fc_logvar = nn.Linear((embedding_dim*pool_val**3)//4, latent_dim)

    def forward(self, x):
        out = self.encoder(x)

        means = self.fc_mu(out)
        log_vars = torch.clamp(self.fc_logvar(out), -10, 10)

        z = self.reparameterization(means, log_vars)
        recon = self.decoder(z)

        KLD = self.KLD(means, log_vars)

        return recon, KLD


    def reparameterization(self, means, log_vars):
        # move random vars sampled from a normal dist to size log_vars to device
        epsilon = torch.randn_like(log_vars).to(device) 
        std = torch.exp(0.5 * log_vars)
        z = means + std * epsilon

        return z
    
    def KLD(self, means, log_vars):
        B, C = means.shape
        KLD = (-0.5 * torch.sum(1 + log_vars - means.pow(2) - log_vars.exp())) / B

        return KLD


#####################################
# Dataset
#####################################
class Segmentation3DDataset(Dataset):
    def __init__(self, image_folder, num_classes=271, n_samp=100000, common_vols=[], return_vols=False):
        self.num_classes = num_classes
        self.image_paths = glob.glob(image_folder + "*")[:n_samp]
        self.return_vols = return_vols
        self.vol_ratios = []

        # compute volume ratios:
        for path in self.image_paths:
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(path))

            img_vol_ratios = []
            total_count = np.sum(img_arr != 0)

            common_struct_idxs = np.arange(251, 270 + 1)

            for i in common_struct_idxs[common_vols]:
                img_vol_ratios.append(np.sum(img_arr == i) / total_count)

            self.vol_ratios.append(img_vol_ratios)
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load 3D mask
        image = sitk.ReadImage(self.image_paths[idx])
        image = sitk.GetArrayFromImage(image).astype(np.int64)  # [D, H, W]

        if self.return_vols:
            return torch.from_numpy(image), self.vol_ratios[idx]
        
        else:
            return torch.from_numpy(image)
    

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


# Save vol ratios plot
def get_vol_ratio(img_arr):
    vol_ratios = []
    total_count = np.sum(img_arr != 0)

    for i in range(251, 270 + 1):
        vol_ratios.append(0 if total_count == 0 else np.sum(img_arr == i) / total_count)

    return vol_ratios


def plot_vol_ratios(real_vol_ratios, synth_vol_ratios, struct_names):
    common_struct_colors = repeated_colors[251:]

    fig2, axs2 = plt.subplots(2, 10, figsize=(35, 6))

    for plot_index in range(2):  # Two sets: 0–9 and 10–19
        fig, axs = plt.subplots(2, 10, figsize=(30, 5))
        
        for i in range(10):
            struct_i = plot_index * 10 + i
            if struct_i >= len(struct_names):
                break
            
            color = common_struct_colors[struct_i]

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


#####################################
# Training setup - VAE
#####################################
train_save_folder = "./VAE_train/3d_anat/train_human_updated_1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = VAE(in_channels=1, num_classes=271, embedding_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

kld_beta = 1

# Data
train_dataset = Segmentation3DDataset("./datasets/3d_anat_align/human_train/", num_classes=271)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = Segmentation3DDataset("./datasets/3d_anat_align/human_test/", num_classes=271)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# how often to save model checkpoints and images
save_imgs = True
save_imgs_freq = 10
save_model_freq = 10
num_epochs = 1000
display_size = 16 # num images to display
early_stop_patience = 10

os.makedirs(train_save_folder + "/test_images", exist_ok=True)
os.makedirs(train_save_folder + "/vol_plots", exist_ok=True)

# ===== Compute class weights from atlas =====
atlas_path = "../../MDSC689.03-Final-Project/spring term/data/anat_atlases/human_anat_seg_common.nii"  # adjust path
atlas_img = sitk.GetArrayFromImage(sitk.ReadImage(atlas_path))  # [D,H,W]

num_classes = 271
counts = np.bincount(atlas_img.astype(np.int64).flatten(), minlength=num_classes)
freq = counts / counts.sum()

# Inverse frequency weighting (with smoothing)
weights = 1.0 / (freq + 1e-6)       # avoid div by 0
weights = np.sqrt(weights)          # dampen extreme rare-class weights
weights = weights / weights.mean()  # normalize (so average weight = 1.0)

class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
recon_loss_fn = nn.CrossEntropyLoss(weight=class_weights)


#####################################
# Training loop
#####################################

early_stopping = EarlyStopping(patience=early_stop_patience, save_path=train_save_folder + "best_model.pth")

print("Start training!")
optimizer.zero_grad()
for epoch in range(num_epochs):
    train_loss = train_recon = train_kld = 0
    for i, batch in enumerate(train_loader):
        model.train()
        batch = batch.to(device)  # [B, D, H, W]
        x_in = batch.unsqueeze(1).float()  # [B, 1, D, H, W]

        accum_steps = 4
        logits, kld_loss = model(x_in)
        loss_recon = recon_loss_fn(logits, batch)
        loss = loss_recon + kld_beta*kld_loss

        loss.backward()
        if (i+1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item()
        train_recon += loss_recon.item()
        train_kld += kld_loss.item()


    img_to_save = []
    recon_to_save = []
    
    img_vol_ratios = []
    recon_vol_ratios = []
    
    test_loss = test_recon = test_kld = 0
    for i, batch in enumerate(test_loader):
        model.eval()
        batch = batch.to(device)
        x_in = batch.unsqueeze(1).float()
        with torch.no_grad():
            logits, kld_loss = model(x_in)
            loss_recon = recon_loss_fn(logits, batch)
            loss = loss_recon + kld_beta*kld_loss
            test_loss += loss.item()
            test_recon += loss_recon.item()
            test_kld += kld_loss.item()

            recon = torch.argmax(logits, dim=1).squeeze().detach().cpu()

            img_vol_ratios.append(get_vol_ratio(batch.squeeze().detach().cpu().numpy()))
            recon_vol_ratios.append(get_vol_ratio(recon.numpy()))

        #save a few test images
        if save_imgs and i < display_size and (epoch % save_imgs_freq == 0):
            recon = torch.argmax(logits, dim=1).squeeze().detach().cpu()
            recon_to_save.append(recon)
            img_to_save.append(batch)
    
    train_loss /= len(train_loader)
    train_recon /= len(train_loader)
    train_kld /= len(train_loader)

    test_loss /= len(test_loader)
    test_recon /= len(test_loader)
    test_kld /= len(test_loader)

    # save a png of some reconstructions - to observe during training
    if save_imgs and epoch % save_imgs_freq == 0:
        img_to_save = torch.stack(img_to_save).squeeze()
        recon_to_save = torch.stack(recon_to_save).squeeze()

        print("save images...")
        __write_images([img_to_save, recon_to_save], display_size, train_save_folder + "test_images/recons_epoch_" + str(epoch) + ".png")

        # save vol ratios plot
        fig = plot_vol_ratios(np.array(img_vol_ratios), np.array(recon_vol_ratios), struct_names)
        fig.savefig(train_save_folder + "vol_plots/epoch_"+ str(epoch) + ".png")


    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train: Loss {train_loss:.4f} Recon {train_recon:.4f} KLD {train_kld:.4f} | "
          f"Test: Loss {test_loss:.4f} Recon {test_recon:.4f} KLD {test_kld:.4f}")

    
    # Save CSV log
    with open(train_save_folder + "loss_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_recon, train_kld, test_loss, test_recon, test_kld])

    if epoch % save_model_freq == 0:
        # model dict for saving
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        # check early stopping
        early_stopping(test_loss, checkpoint)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

# takes ~2 min / epoch