import os
import torch
from examples.datasets.blender_nerf_synthetic import Parser, Dataset
from mine.loader import initialize_model_from_ply_file

home_dir = os.path.expanduser("~")
device = torch.device("cuda:0")

class Dataset:
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
        self.means     = self.gaussians["means"] 
        self.scales    = self.gaussians["scales"]
        self.quats     = self.gaussians["quats"]    
        self.opacities = self.gaussians["opacities"].squeeze(-1)
        self.colors    = self.gaussians["shs"]
        
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
    def load_trained_model(model_path):
        model = initialize_model_from_ply_file(model_path)
        return model
    
    def set_iamge(self,
                  idx: int):
        self.data_wo = self.dataloader_wo_ref.dataset[idx]
        self.data_w = self.dataloader_w_ref.dataset[idx]
        
        self.camtoworld = self.data_wo["camtoworld"].view(1,4,4).to(device)
        self.worldtocam = self.camtoworld.inverse()
        self.cam_center = self.camtoworld[0, :3, 3].detach().clone()
        self.Ks = self.data_wo["K"].view(1,3,3).to(device)
        self.image_id = self.data_wo["image_id"]
        
        self.pixels_wo = self.data_wo["image"].unsqueeze(0).to(device) / 255.0
        self.pixels_w = self.data_w["image"].unsqueeze(0).to(device) / 255.0
        self.height, self.width = self.data_wo["image"].shape[0], self.data_wo["image"].shape[1]
        
        self.image_path_wo = os.path.join(self.wo_ref_dataset_path, "train", f"{self.image_id:04d}" +  ".png")
        self.image_path_w = os.path.join(self.w_ref_dataset_path, "train", f"{self.image_id:04d}" +  ".png")
        
        
    