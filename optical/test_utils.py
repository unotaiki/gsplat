import os
import torch
from gsplat import rasterization
import matplotlib.pyplot as plt
import sys


from mine.loader import initialize_model_from_ply_file
from examples.datasets.blender_nerf_synthetic import Parser, Dataset

device = torch.device("cuda:0")

def load_dataset()