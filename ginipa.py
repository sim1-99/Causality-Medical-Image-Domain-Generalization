import os

import numpy as np
import SimpleITK as sitk
import torch

from biasfield_interpolate_cchen.adv_bias import AdvBias
from biasfield_interpolate_cchen.utils import rescale_intensity
from dataloaders import niftiio as nio
from models.exp_trainer import to01, t2n
from models.imagefilter3d import GINGroupConv3D


input_folder = "./images/irtk"
output_folder = "./output_ginipa/irtk"
experiment = "D"
experiment_folder = os.path.join(output_folder, experiment)
os.makedirs(experiment_folder, exist_ok=True)

conv3D_config = {  # GIN config
    'n_layer': 4,
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
    for j in range(1):
        img, info = nio.read_nii_bysitk(
            os.path.join(input_folder, file), peel_info=True
        )

        img = np.float32(img)
        mean = np.mean(img)
        std = np.std(img)
        img = (img - mean) / std
        #hir = float(np.percentile(img, 100.0 - HIST_CUT_TOP))
        #img[img > hir] = hir
        img = torch.from_numpy(img).cuda()
        img = img.unsqueeze(0).unsqueeze(0)
        img = torch.cat((img, img, img), dim = 1)

        augmenter = GINGroupConv3D(
            out_channel = 1,
            in_channel = 3,
            n_layer = conv3D_config['n_layer'],
            interm_channel = conv3D_config['interm_channel'],
            out_norm = conv3D_config['out_norm'],
        ).cuda()

        gin = torch.cat([augmenter(img) for _ in range(3)], dim = 0)

        blender_node = AdvBias(blender_cofig)
        blender_node.init_parameters()
        blend_mask = rescale_intensity(blender_node.bias_field
            ).repeat(1, 3, 1, 1, 1)
        print('bias field', blender_node.bias_field.shape)
        print('blend mask', blend_mask.shape)

        input_cp1 = gin[:1].clone().detach() * (1.0 - blend_mask) + gin[1:2].clone().detach() * blend_mask
        input_cp2 = gin[:1] * blend_mask + gin[1:2] * (1.0 - blend_mask)

        gin[:1] = input_cp1
        gin[1:2] = input_cp2
        input_img_3copy = gin

        ginipa = t2n(to01(input_img_3copy[:1], True))

        ginipa = nio.convert_to_sitk(ginipa[0], info)
        sitk.WriteImage(
            ginipa, f"{experiment_folder}/ginipa_{i}.nii.gz", True)  # ginipa_{j}-{i}.nii.gz

        """gin1_itk = gin1[0, 0, ...].cpu().numpy() # np.transpose(gin1[0, 0, ...].cpu().numpy(), (2, 0, 1))
        gin1_itk = nio.convert_to_sitk(gin1_itk, info)
        sitk.WriteImage(gin1_itk, f"{experiment_folder}/gin1_{j}-{i}.nii.gz", True)

        gin2_itk = gin2[0, 0, ...].cpu().numpy() # np.transpose(gin2[0, 0, ...].cpu().numpy(), (2, 0, 1))
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
        """

config_file_path = os.path.join(experiment_folder, f"config_{experiment}.txt")

with open(config_file_path, 'w') as config_file:
    config_file.write("conv3D_config:\n")
    for key, value in conv3D_config.items():
        config_file.write(f"{key}: {value}\n")

    config_file.write("\nblender_config:\n")
    for key, value in blender_cofig.items():
        config_file.write(f"{key}: {value}\n")
