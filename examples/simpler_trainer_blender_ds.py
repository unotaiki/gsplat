#!/usr/bin/env python3
import os
import time
import math
import tyro
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

# 必要なモジュールのみインポート
from datasets.blender_nerf_synthetic import Parser, Dataset
from gsplat.rendering import rasterization
from utils import knn, rgb_to_sh, set_random_seed

########################################
# Config: 最低限必要なパラメータのみ
########################################

@dataclass
class Config:
    # データセット関連
    data_dir: str = "data/transformers"   # Blender NeRF Synthetic のルートディレクトリ
    result_dir: str = "results/transformers"  # 結果出力先
    max_steps: int = 1000          # 学習ステップ数
    batch_size: int = 1            # バッチサイズ
    patch_size: int = None         # クロップサイズ（指定なければ全体）

    # GS（Gaussian Splatting）の初期化パラメータ
    init_type: str = "sfm"         # "sfm" または "random"
    init_num_pts: int = 100_000    # 初期点数（"random" の場合のみ使用）
    init_extent: float = 3.0       # 初期分布の広がり
    init_opa: float = 0.1          # 初期透明度
    init_scale: float = 1.0        # 初期スケール
    sh_degree: int = 3             # 球面調和関数の次数

    # シーンスケール（Blender 生成データは正規化不要）
    global_scale: float = 1.0

########################################
# create_splats_with_optimizers: 最低限の実装
########################################

def create_splats_with_optimizers(parser: Parser, cfg: Config, device: str = "cuda") -> tuple:
    # "sfm"の場合、Parserから読み込んだ点群を使用（存在しない場合はランダム）
    if cfg.init_type == "sfm" and parser.points.size > 0:
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    else:
        points = cfg.init_extent * (torch.rand((cfg.init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((cfg.init_num_pts, 3))
    
    # 近傍点との距離から初期サイズを計算
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)
    N = points.shape[0]
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), cfg.init_opa))
    
    params = [
        ("means", torch.nn.Parameter(points), 1.6e-4),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]
    # 色は、球面調和関数の係数として初期化
    colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = rgb_to_sh(rgbs)
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
    params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    
    splats = torch.nn.ParameterDict({name: param for name, param, lr in params}).to(device)
    
    optimizers = {}
    BS = cfg.batch_size
    for name, _, lr in params:
        optimizers[name] = torch.optim.Adam([splats[name]], lr=lr * math.sqrt(BS))
    
    return splats, optimizers

########################################
# Simple Trainer
########################################

def main(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed(42)
    os.makedirs(cfg.result_dir, exist_ok=True)

    # データセットの読み込み（Parser, Dataset）
    print("Loading dataset...")
    parser_obj = Parser(data_dir=os.path.abspath(cfg.data_dir))
    dataset = Dataset(parser_obj, split="train", patch_size=cfg.patch_size, load_depths=False)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    
    # モデル初期化：GSパラメータと最適化器
    splats, optimizers = create_splats_with_optimizers(parser_obj, cfg, device=device)
    
    # カメラ内部パラメータは全画像共通（カメラID 0）
    K = parser_obj.Ks_dict[0]
    # シーンスケールは Blender 用なので 1.0 * global_scale
    scene_scale = 1.0 * cfg.global_scale

    print("Starting training loop...")
    start_time = time.time()
    global_step = 0
    # 学習ループ：シンプルに各バッチごとにレンダリングし、L1 Loss を最小化する
    for step in range(cfg.max_steps):
        for batch in dataloader:
            camtoworld = batch["camtoworld"].to(device)  # (B, 4, 4)
            image = batch["image"].to(device) / 255.0      # (B, H, W, 3)
            B, H, W, _ = image.shape
            
            # 内部パラメータをバッチサイズ分拡張
            Ks = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)
            # viewmats はカメラ座標系に変換するため、逆行列を計算
            viewmats = torch.linalg.inv(camtoworld)  # (B, 4, 4)
            
            # レンダリング: rasterization 関数を用いる（必須の引数のみ）
            render_colors, _, _ = rasterization(
                means=splats["means"],
                quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=torch.cat([splats["sh0"], splats["shN"]], 1),
                viewmats=viewmats,
                Ks=Ks,
                width=W,
                height=H,
                packed=False,
                rasterize_mode="classic",
            )
            # ここではレンダリング画像と元画像の L1 Loss を計算
            loss = F.l1_loss(render_colors, image)
            for opt in optimizers.values():
                opt.zero_grad()
            loss.backward()
            for opt in optimizers.values():
                opt.step()
            
            global_step += 1
            if global_step % 100 == 0:
                print(f"Step {global_step}: Loss = {loss.item():.4f}")
                
            if global_step >= cfg.max_steps:
                break
        if global_step >= cfg.max_steps:
            break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
