import os

import numpy as np
import SimpleITK as sitk
import torch

from biasfield_interpolate_cchen.adv_bias import AdvBias
from biasfield_interpolate_cchen.utils import rescale_intensity
from dataloaders import niftiio as nio
from models.imagefilter3d import GINGroupConv3D


input_folder = "./images/mial"
output_folder = "./output_ginipa/mial"
experiment = "B"
experiment_folder = os.path.join(output_folder, experiment)
os.makedirs(experiment_folder, exist_ok=True)

conv3D_config = {  # GIN config
    'n_layer': 6,
    'interm_channel': 2,
    'out_norm': 'frob',
}

blender_cofig = {  # bias field interpolation config for IPA
    'epsilon': 0.3,
    'xi': 1e-6,
    'control_point_spacing': [32, 32],
    'downscale': 1, #
    'data_size': [1, 3, 256, 256, 256],
    'interpolation_order': 2,
    'init_mode': 'gaussian',
    'space': 'log',
}

HIST_CUT_TOP = 0.5
files = sorted(os.listdir(input_folder))

for i, file in enumerate(files):
    if i < 4:
        for j in range(1):
            img, info = nio.read_nii_bysitk(
                os.path.join(input_folder, file), peel_info=True
            )

            img = np.float32(img)
            hir = float(np.percentile(img, 100.0 - HIST_CUT_TOP))
            img[img > hir] = hir
            img = np.transpose(img, (1, 2, 0))
            img = torch.from_numpy(img).cuda()
            print(img.shape)
            img = img.unsqueeze(0).unsqueeze(0)
            img = torch.cat((img, img, img), dim = 1)
            print(img.shape)

            augmenter1 = GINGroupConv3D(
                out_channel = 1,
                in_channel = 3,
                n_layer = conv3D_config['n_layer'],
                interm_channel = conv3D_config['interm_channel'],
                out_norm = conv3D_config['out_norm'],
            ).cuda()
            augmenter2 = GINGroupConv3D(
                out_channel = 1,
                in_channel = 3,
                n_layer = conv3D_config['n_layer'],
                interm_channel = conv3D_config['interm_channel'],
                out_norm = conv3D_config['out_norm'],
            ).cuda()
            augmenter3 = GINGroupConv3D( ####
                out_channel = 1,
                in_channel = 3,
                n_layer = conv3D_config['n_layer'],
                interm_channel = conv3D_config['interm_channel'],
                out_norm = conv3D_config['out_norm'],
            ).cuda()

            gin1 = augmenter1(img)
            gin2 = augmenter2(img)
            gin3 = augmenter3(img) ####
            
            gin = torch.cat((gin1, gin2, gin3), dim = 0) ####

            blender_node = AdvBias(blender_cofig)
            blender_node.init_parameters()
            blend_mask = rescale_intensity(blender_node.bias_field
                ).repeat(1, 3, 1, 1, 1)

            ginipa1 = gin[:1].clone().detach() * (1.0 - blend_mask) + gin[1:2].clone().detach() * blend_mask
            ginipa2 = gin[:1] * blend_mask + gin[1:2] * (1.0 - blend_mask)

            """ginipa1 = np.transpose(ginipa1[0, 0, ...].cpu().numpy(), (2, 0, 1))
            ginipa1_itk = nio.convert_to_sitk(ginipa1, info)
            sitk.WriteImage(ginipa1_itk, f"{experiment_folder}/ginipa_{j}-{i}.nii.gz", True)"""

            ginipa2 = np.transpose(ginipa2[0, 0, ...].cpu().numpy(), (2, 0, 1))
            ginipa2_itk = nio.convert_to_sitk(ginipa2, info)
            sitk.WriteImage(ginipa2_itk, f"{experiment_folder}/ginipa_{j}-{i}.nii.gz", True)

            gin1_itk = np.transpose(gin1[0, 0, ...].cpu().numpy(), (2, 0, 1))
            gin1_itk = nio.convert_to_sitk(gin1_itk, info)
            sitk.WriteImage(gin1_itk, f"{experiment_folder}/gin1_{j}-{i}.nii.gz", True)

            gin2_itk = np.transpose(gin2[0, 0, ...].cpu().numpy(), (2, 0, 1))
            gin2_itk = nio.convert_to_sitk(gin2_itk, info)
            sitk.WriteImage(gin2_itk, f"{experiment_folder}/gin2_{j}-{i}.nii.gz", True)

            blend_mask_itk = sitk.GetImageFromArray(
                blend_mask[0, 0, ...].cpu().numpy()
            )
            blend_mask_itk.SetOrigin(info['origin'])
            blend_mask_itk.SetDirection(info['direction'])
            blend_mask_itk.SetSpacing(info['spacing'])
            sitk.WriteImage(
                blend_mask_itk, f"{experiment_folder}/blend_mask_{j}-{i}.nii.gz", True
            )

config_file_path = os.path.join(experiment_folder, f"config_{experiment}.txt")

with open(config_file_path, 'w') as config_file:
    config_file.write("conv3D_config:\n")
    for key, value in conv3D_config.items():
        config_file.write(f"{key}: {value}\n")

    config_file.write("\nblender_config:\n")
    for key, value in blender_cofig.items():
        config_file.write(f"{key}: {value}\n")
