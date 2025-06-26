"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, write_loss_to_csv, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer_3d import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import SimpleITK as sitk
import numpy as np


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

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/unit_mouse2human_train_3_CE_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='UNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")
trainer.cuda()

# Dataset path
train_folder_mouse = "../3D-CycleGan-Pytorch-MedImaging/Data_folder_train_2/train/images/"
train_paths_mouse = [train_folder_mouse + file_name for file_name in os.listdir(train_folder_mouse)]

val_folder_mouse = "../3D-CycleGan-Pytorch-MedImaging/Data_folder_train_2/test/images/"
val_paths_mouse = [val_folder_mouse + file_name for file_name in os.listdir(val_folder_mouse)]
val_paths_mouse.sort()

train_folder_human = "../3D-CycleGan-Pytorch-MedImaging/Data_folder_train_2/train/labels/"
train_paths_human = [train_folder_human + file_name for file_name in os.listdir(train_folder_human)]

val_folder_human = "../3D-CycleGan-Pytorch-MedImaging/Data_folder_train_2/test/labels/"
val_paths_human = [val_folder_human + file_name for file_name in os.listdir(val_folder_human)]
val_paths_human.sort()

train_dataset_mouse = Segmentation3DDataset(image_paths=train_paths_mouse)
test_dataset_mouse = Segmentation3DDataset(image_paths=val_paths_mouse)
train_dataset_human = Segmentation3DDataset(image_paths=train_paths_human)
test_dataset_human = Segmentation3DDataset(image_paths=val_paths_human)

train_loader_a = DataLoader(train_dataset_mouse, batch_size=4, shuffle=True)
train_loader_b = DataLoader(train_dataset_human, batch_size=4, shuffle=True)
test_loader_a = DataLoader(test_dataset_mouse, batch_size=1, shuffle=False)
test_loader_b = DataLoader(test_dataset_human, batch_size=1, shuffle=False)

train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)
            write_loss_to_csv(iterations, trainer, output_directory + "/loss.csv")

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            #write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        #if (iterations + 1) % config['image_display_iter'] == 0:
        #    with torch.no_grad():
        #        image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
        #    write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

