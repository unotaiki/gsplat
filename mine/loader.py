# This code is from https://github.com/yzslab/gaussian-splatting-lightning/blob/main/internal/utils/gaussian_model_loader.py#L154

import torch
import numpy as np
from plyfile import PlyData
from mine.gaussian_utils import GaussianPlyUtils


def initialize_model_from_ply_file(ply_file_path: str, device, pre_activate: bool = True):
    gaussian_ply_utils = GaussianPlyUtils.load_from_ply(ply_file_path).to_parameter_structure()
    model_state_dict = {
        "_active_sh_degree": torch.tensor(gaussian_ply_utils.sh_degrees, dtype=torch.int, device=device),
        "gaussians.means": gaussian_ply_utils.xyz.to(device),
        "gaussians.opacities": gaussian_ply_utils.opacities.to(device),
        "gaussians.shs_dc": gaussian_ply_utils.features_dc.to(device),
        "gaussians.shs_rest": gaussian_ply_utils.features_rest.to(device),
        "gaussians.scales": gaussian_ply_utils.scales.to(device),
        "gaussians.rotations": gaussian_ply_utils.rotations.to(device),
    }
    if model_state_dict["gaussians.scales"].shape[-1] == 2:
        from internal.models.gaussian_2d import Gaussian2D
        model = Gaussian2D(sh_degree=gaussian_ply_utils.sh_degrees).instantiate()
    else:
        from internal.models.vanilla_gaussian import VanillaGaussian
        model = VanillaGaussian(sh_degree=gaussian_ply_utils.sh_degrees).instantiate()
    model.setup_from_number(gaussian_ply_utils.xyz.shape[0])
    model.to(device)
    model.load_state_dict(model_state_dict, strict=False)
    if pre_activate:
        model.pre_activate_all_properties()
    return model
