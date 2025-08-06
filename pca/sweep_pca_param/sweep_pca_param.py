from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pickle
import torch
import torchio as tio
import scipy.stats
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

# parameters
data_type = "human" # of "mouse", "human"
elastic_transf_type = "a" # of "a", "b"
save_pearsons_r = "./save_pearsons_r.csv"

print("Data type:", data_type)
print("Elastic transf type:", elastic_transf_type)

# create file to save pearsons r if not exist yet
if not os.path.exists(save_pearsons_r):
    with open(save_pearsons_r, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["data_type", "elastic_transf_type", "num_aug", "num_comp", "gm_r", "wm_r", "csf_r"])

# helper functions
def load_imgs(folder_path):
    images = []
    file_names = []

    for file_name in os.listdir(folder_path):
        img = sitk.ReadImage(folder_path + file_name)
        img_array = sitk.GetArrayFromImage(img)  # Normalize to [0, 1]
        images.append(img_array)

        file_names.append(file_name)

    # Convert to NumPy array: shape (n_images, height, width)
    images = np.stack(images)
    
    return images, file_names

def get_matter_ratio(img_arr):
    gm_count = np.sum(img_arr == 1)
    wm_count = np.sum(img_arr == 2)
    csf_count = np.sum(img_arr == 3)
    total = gm_count + wm_count + csf_count

    gm_ratio = gm_count/total
    wm_ratio = wm_count/total
    csf_ratio = csf_count/total

    return gm_ratio, wm_ratio, csf_ratio

def compute_matter_ratios(imgs):
    gm_ratios = []
    wm_ratios = []
    csf_ratios = []

    for img in imgs:
        gm_ratio, wm_ratio, csf_ratio = get_matter_ratio(img)

        gm_ratios.append(gm_ratio)
        wm_ratios.append(wm_ratio)
        csf_ratios.append(csf_ratio)

    return gm_ratios, wm_ratios, csf_ratios

# 1. load non augmented data
folder_path = "../../../3D-CycleGan-Pytorch-MedImaging/data/"

if data_type == "mouse":
    train_imgs, _ = load_imgs(folder_path + "train/images/") # mouse
    test_imgs, _ = load_imgs(folder_path + "test/images/")
else:
    train_imgs, _ = load_imgs(folder_path + "train/labels/") # human
    test_imgs, _ = load_imgs(folder_path + "test/labels/")

train_imgs_aug = train_imgs # include train images in augmented set
print("1. Images loaded:", train_imgs_aug.shape)


# don't change for each set of param
np.random.shuffle(test_imgs)
test_imgs = np.ascontiguousarray(test_imgs)  # optional, ensures C-order
test_imgs_flat = test_imgs.reshape(test_imgs.shape[0], -1)

# Define transform
if elastic_transf_type == "a":
    num_control_points = 7
    max_displacement = 4.0
    locked_borders = 2

else:
    num_control_points = 5
    max_displacement = 2.0
    locked_borders = 5

elastic_transform = tio.RandomElasticDeformation(
    num_control_points=num_control_points,  # grid resolution, higher = more local warping
    max_displacement=max_displacement,  # max displacement in mm or voxels
    locked_borders=locked_borders,      # keep border fixed (avoid artifacts)
    image_interpolation='nearest'  # preserve label integrity
)

# NOTE: start with 6 augs

print("Start augmenting imgs")
aug_list = [train_imgs]  # Keep original

for step in range(4):
    this_step_aug = []  # Fresh list each step

    for img in train_imgs:
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        subject = tio.Subject(mask=tio.LabelMap(tensor=tensor))

        for i in range(2):
            transformed = elastic_transform(subject)
            transformed_np = transformed['mask'].data.squeeze(0).numpy()
            this_step_aug.append(transformed_np[None, ...])

    aug_list.append(np.concatenate(this_step_aug, axis=0))  # Only once per step
    print("Add set", step)

# Final combination (only once):
train_imgs_aug = np.concatenate(aug_list, axis=0)

# param sweeping loop
for num_aug in [6, 10, 14]:
    print("Num aug:", num_aug)

    this_step_aug = []  # Fresh list each step

    for img in train_imgs:
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        subject = tio.Subject(mask=tio.LabelMap(tensor=tensor))

        for i in range(2):
            transformed = elastic_transform(subject)
            transformed_np = transformed['mask'].data.squeeze(0).numpy()
            this_step_aug.append(transformed_np[None, ...])

    aug_list.append(np.concatenate(this_step_aug, axis=0))  # Only once per step
    train_imgs_aug = np.concatenate(aug_list, axis=0)

    print("2. Data augmented (shape:", train_imgs_aug.shape, ")")
    # takes ~3min

    # shuffle and flatten train data (to mix up augmentations)
    np.random.shuffle(train_imgs_aug)
    train_imgs_aug = np.ascontiguousarray(train_imgs_aug)  # optional, ensures C-order
    train_imgs_flat = train_imgs_aug.reshape(train_imgs_aug.shape[0], -1)

    # sweep over differnt # of pca componenets
    for num_comp in np.arange(1000, len(train_imgs_aug), 500):
        
        print("Num comp:", num_comp)
        # 3. train scaler and pca on num_comp
        # ii) fit scaler
        eps = 1e-8  # small constant to avoid divide by zero
        num_sub_samp = 500

        # Compute mean and std across the dataset per pixel
        mean = np.mean(train_imgs_flat[:num_sub_samp], axis=0)   # shape: (num_pixels,)
        print("mean calced") # 20 s

        std = np.std(train_imgs_flat[:num_sub_samp], axis=0)     # shape: (num_pixels,)
        print("var calced") # 10 s

        # Avoid division by zero
        std[std < eps] = 1.0  # or keep std = 1 where it's too small

        # In-place standardization (broadcasting-safe)
        train_imgs_flat -= mean[None, :]
        train_imgs_flat /= (std[None, :] + eps)

        # ii) fit pca
        pca = PCA(n_components=num_comp, svd_solver='randomized') # new pca
        train_imgs_pca = pca.fit_transform(train_imgs_flat)


        # for 660 samples and 300 pca comps - takes ~3min 30s
        # 4. save models

        with open("./save_models/pca_" + data_type + "_" + elastic_transf_type + "_" + str(num_aug) + "_" + str(num_comp) + ".pkl", "wb") as f:
            pickle.dump(pca, f)

        print("3/4. PCA fit and saved")

        # 5. apply to test
        test_scaled = (test_imgs_flat - mean) / (std + eps) #scaler.transform(test_imgs_flat)
        test_pca = pca.transform(test_scaled)
        test_recon = pca.inverse_transform(test_pca)
        test_recon_unscale = test_recon*std + mean #scaler.inverse_transform(test_recon)
        imgs_test_recon = np.clip(np.round(test_recon_unscale.reshape(test_imgs.shape)), 0, 3) # reshape to 2D imgs

        # 6. evaluate quality of recon (pearsons r, save a line of recons)
        # pearsons r comparing img and recon matter ratios
        imgs_gm_matter_ratio, imgs_wm_matter_ratio, imgs_csf_matter_ratio = compute_matter_ratios(test_imgs)
        recon_gm_matter_ratio, recon_wm_matter_ratio, recon_csf_matter_ratio = compute_matter_ratios(imgs_test_recon)

        gm_r, _ = scipy.stats.pearsonr(imgs_gm_matter_ratio, recon_gm_matter_ratio)
        wm_r, _ = scipy.stats.pearsonr(imgs_wm_matter_ratio, recon_wm_matter_ratio)
        csf_r, _ = scipy.stats.pearsonr(imgs_csf_matter_ratio, recon_csf_matter_ratio)

        with open(save_pearsons_r, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([data_type, elastic_transf_type, num_aug, num_comp, gm_r, wm_r, csf_r])

        # save 10 imgs and their reconstruction as a png
        num_save_imgs = 10
        slice_idx = 70

        fig, axes = plt.subplots(2, 10, figsize=(20, 4))  # 2 rows (orig, recon), 10 columns (samples)
        cmap = plt.get_cmap('tab10', 4)

        for i in range(num_save_imgs):
            axes[0, i].imshow(test_imgs[i][slice_idx], cmap="gray", vmin=0, vmax=3)
            axes[0, i].set_title("Original")
            axes[0, i].axis('off')

            axes[1, i].imshow(imgs_test_recon[i][slice_idx], cmap="gray", vmin=0, vmax=3)
            axes[1, i].set_title("Reconstruction")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig("./save_imgs/recons_" + data_type + "_" + elastic_transf_type + "_" + str(num_aug) + "_" + str(num_comp) + ".png", dpi=300)
        plt.close()
        print("5/6. Test scaled and evaluated")

    print()

    