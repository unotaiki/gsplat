import os
import sys
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from gsplat import rasterization
from mine.loader import initialize_model_from_ply_file
from examples.datasets.blender_nerf_synthetic import Parser, Dataset

home_dir = os.path.expanduser("~")
device = torch.device("cuda:0")

class LoadDataset:
    def __init__(self, 
        wo_ref_dataset_path: str=os.path.abspath(os.path.join(home_dir, "dataset/river2/wo_refraction")),
        w_ref_dataset_path: str=os.path.abspath(os.path.join(home_dir, "dataset/river2/refraction")),
        wo_ref_model_path: str=os.path.abspath(os.path.join(home_dir, "dataset/river2/wo_refraction_wo_plane/cut_sphere.ply")),                                                
        ):
        
        # set the dataset path
        self.wo_ref_dataset_path = wo_ref_dataset_path
        self.w_ref_dataset_path = w_ref_dataset_path
        self.wo_ref_model_path = wo_ref_model_path
        
        # load the non-refractive dataset
        self.dataloader_wo_ref = self.load_dataset(self.wo_ref_dataset_path)
        self.dataloader_w_ref = self.load_dataset(self.w_ref_dataset_path)
        
        # Load training 3DGS model
        self.model = self.load_trained_model(self.wo_ref_model_path)
        self.means     = self.model.gaussians["means"] 
        self.scales    = self.model.gaussians["scales"]
        self.quats     = self.model.gaussians["rotations"]    
        self.opacities = self.model.gaussians["opacities"].squeeze(-1)
        self.colors    = self.model.gaussians["shs"]
        
    # Load the dataset
    def load_dataset(self,
                     dataset_path: str):
        parser = Parser(
            data_dir=dataset_path
        )
        dataset = Dataset(
            parser,
            split="train",
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        return dataloader
        
    # Load trained 3DGS model
    def load_trained_model(self,
                           model_path):
        model = initialize_model_from_ply_file(model_path, device=device)
        return model
    
    def set_iamge(self,
                  idx: int):
        self.data_wo = self.dataloader_wo_ref.dataset[idx]
        self.data_w = self.dataloader_w_ref.dataset[idx]
        
        self.camtoworld = self.data_wo["camtoworld"].view(1,4,4).to(device)
        self.worldtocam = self.camtoworld.inverse()
        self.cam_center = self.camtoworld[0, :3, 3].detach().clone()
        self.K = self.data_wo["K"].view(1,3,3).to(device)
        self.image_id = self.data_wo["image_id"]
        
        self.pixels_wo = self.data_wo["image"].unsqueeze(0).to(device) / 255.0
        self.pixels_w = self.data_w["image"].unsqueeze(0).to(device) / 255.0
        self.height, self.width = self.data_wo["image"].shape[0], self.data_wo["image"].shape[1]
        
        self.image_path_wo = os.path.join(self.wo_ref_dataset_path, "train", f"{self.image_id:04d}" +  ".png")
        self.image_path_w = os.path.join(self.w_ref_dataset_path, "train", f"{self.image_id:04d}" +  ".png")
        
    def get_model_param(self):
        return self.means, self.scales, self.quats, self.opacities, self.colors
    
    def get_camera_param(self):
        return self.pixels_wo, self.pixels_w, self.camtoworld, self.worldtocam, self.cam_center, self.K, self.height, self.width, self.image_id
    
    

def combert_into_colormap(x, colormap="viridis", device=device):
    """
    Convert a tensor of shape (N, ) with values in [0, 1] to a colormap (N, 3) RGB
    colormap is a string, e.g. "viridis", "plasma", "inferno", etc.
    """
    colors = plt.get_cmap("viridis")(x.detach().cpu().numpy())[:, :3]  # remove alpha channel
    colors = torch.from_numpy(colors).to(device)
    # convert to float32
    colors = colors.float()
    return colors.unsqueeze(1)  # shape (N, 1, 3), type: float32

def compare_rasterization_variants(
    base_data: dict,
    transformed_data: dict,
    opacities,
    colors,
    worldtocam,
    Ks,
    width,
    height,
    rasterization_fn=rasterization,
    comparison_keys=("means", "quats", "scales"),
    titles=("M", "Q", "S"),
    transformed_titles=("tM", "tQ", "tS"),
    rasterize_mode="antialiased"
):
    """
    Compare all 2^N combinations of using base vs transformed data.
    
    Args:
        rasterization_fn: Function for rasterization (returns RGB image).
        base_data: dict with keys like "means", "quats", "scales" (untransformed).
        transformed_data: dict with same keys but transformed values.
        opacities, colors, worldtocam, Ks, width, height: rendering params.
        comparison_keys: Which data keys to vary (default = means, quats, scales).
        titles: Short name for original key (default = M, Q, S).
        transformed_titles: Short name for transformed version.
        rasterize_mode: Rasterization mode (default: "antialiased").
    """
    num_variants = 2 ** len(comparison_keys)
    fig_cols = min(num_variants, 4)
    fig_rows = (num_variants + fig_cols - 1) // fig_cols
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(5 * fig_cols, 5 * fig_rows))
    axes = axes.flatten()

    combinations = list(itertools.product([False, True], repeat=len(comparison_keys)))

    for idx, use_transformed in enumerate(combinations):
        data_inputs = {
            key: (transformed_data[key] if use_t else base_data[key])
            for key, use_t in zip(comparison_keys, use_transformed)
        }

        rgb, _, _ = rasterization_fn(
            data_inputs["means"],
            data_inputs["quats"],
            data_inputs["scales"],
            opacities, colors,
            worldtocam, Ks,
            width, height,
            sh_degree=0,
            rasterize_mode=rasterize_mode
        )
        img = rgb.squeeze().detach().cpu().numpy()

        title_parts = [
            (transformed_titles[i] if use_t else titles[i])
            for i, use_t in enumerate(use_transformed)
        ]
        axes[idx].imshow(img)
        axes[idx].set_title(" ".join(title_parts))
        axes[idx].axis("off")

    for i in range(len(combinations), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def np_rgb_render(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    worldtocam: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int = 0,
    rasterize_mode: str = "antialiased"
):
    rgb, _, _ = rasterization(
        means,
        quats,
        scales,
        opacities, colors,
        worldtocam, Ks,
        width, height,
        sh_degree=sh_degree,
        rasterize_mode=rasterize_mode
    )
    return rgb.squeeze().detach().cpu().numpy()


def colormap_ray_angle_and_ratio(
    rays: torch.Tensor,
    ratio: torch.Tensor,
):
    """
    convert rays into a colormap based on angle and refrection ratio.

    Args:
        rays (torch.Tensor): _description_
    """
    
    if rays.shape[-1] == 3:
        cos_theta = torch.clamp(-rays[..., 2], -1.0, 1.0)
    
    theta = torch.acos(cos_theta) * 180.0 / np.pi  # convert to degrees
    theta = theta.detach().cpu().numpy()    
    ratio = ratio.squeeze().detach().cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(theta, cmap="plasma")
    plt.axis("off")
    plt.title("Ray Angle (degrees)")
    plt.colorbar(label="Angle (degrees)")
    plt.subplot(1, 2, 2)
    plt.imshow(ratio, cmap="plasma")
    plt.axis("off")
    plt.title("Ratio")
    plt.colorbar(label="Ratio")
    plt.tight_layout()
    plt.show()
    
    