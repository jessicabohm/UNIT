import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pickle
import torch
import torchio as tio
import scipy.stats
import csv
import pandas as pd
import glob


read_folder = "./3d_anat_align/human_train/"
write_folder = "./3d_anat_align/human_train/"
num_aug = 5

os.makedirs(write_folder, exist_ok=True)

num_control_points = 7
max_displacement = 4.0
locked_borders = 2

elastic_transform = tio.RandomElasticDeformation(
    num_control_points=num_control_points,  # grid resolution, higher = more local warping
    max_displacement=max_displacement,  # max displacement in mm or voxels
    locked_borders=locked_borders,      # keep border fixed (avoid artifacts)
    image_interpolation='nearest'  # preserve label integrity
)

this_step_aug = []
for path in glob.glob(read_folder + "*.nii"):
    file_name = path.split("/")[-1]
    img_arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    tensor = torch.from_numpy(img_arr).unsqueeze(0).float()
    subject = tio.Subject(mask=tio.LabelMap(tensor=tensor))

    for i in range(num_aug):
        non_lin_transformed = elastic_transform(subject)
        non_lin_transformed = non_lin_transformed['mask'].data.squeeze(0).numpy()

        aug_img = sitk.GetImageFromArray(non_lin_transformed)
        sitk.WriteImage(aug_img, write_folder + "a_" + str(i) + "_" + file_name)