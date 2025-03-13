import os
import json
import numpy as np
import imageio.v2 as imageio
import torch
import cv2
from PIL import Image
import random

# 3D点群の読み込み用（オプション）
try:
    import open3d as o3d
except ImportError:
    o3d = None

########################################
# Parser クラス
########################################

class Parser:
    """
    Blender NeRF Synthetic データセット用の Parser クラス
    
    読み込むファイル:
     - transforms_train.json : カメラ姿勢と FOV（camera_angle_x）の情報
     - 画像ファイル            : JSON 内の file_path に基づく各画像
     - points3d.ply           : (オプション) 3D点群（存在する場合のみ）
     
    保持する内部データ構造は以下の通り:
      - self.image_names  : List[str]   JSON 内の各 frame の file_path (必要に応じて拡張子付与)
      - self.image_paths  : List[str]   画像ファイルの絶対パス
      - self.camtoworlds  : np.ndarray  (N, 4, 4) 各画像のカメラ-to-world 行列（transform_matrix）
      - self.Ks_dict      : Dict[int, np.ndarray] 内部パラメータ行列 K (3x3)；全画像同一カメラ (ID=0)
      - self.camera_ids   : List[int]   各画像のカメラID（すべて 0 でよい）
      - self.imsize_dict  : Dict[int, Tuple[int,int]]  カメラIDごとの画像サイズ (W,H)
      - self.params_dict  : Dict[int, np.ndarray] 歪みパラメータ（全て空配列）
      - self.points       : np.ndarray  (M, 3) 3D点群座標（存在する場合）
      - self.points_rgb   : np.ndarray  (M, 3) 3D点のRGB色（存在する場合）
      - self.point_indices: Dict[str, np.ndarray] 画像ごとに参照する点群のインデックス（今回は全点 or 空）
      - self.transform    : np.ndarray  (4, 4) 変換行列。Blenderでは単位行列 (np.eye(4))
    """
    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)
        transforms_path = os.path.join(data_dir, "transforms_train.json")
        if not os.path.exists(transforms_path):
            raise ValueError(f"transforms_train.json not found in {data_dir}")
        
        with open(transforms_path, "r") as f:
            transforms = json.load(f)
        
        # --- グローバルパラメータの取得 ---
        self.camera_angle_x = transforms.get("camera_angle_x", None)
        if self.camera_angle_x is None:
            raise ValueError("camera_angle_x not found in JSON.")
        
        frames = transforms.get("frames", [])
        if not frames:
            raise ValueError("No frames found in JSON.")
        
        # --- 各フレームから画像ファイルパスとカメラ姿勢の取得 ---
        self.image_names = []  # JSON内の file_path（拡張子付与後）
        self.image_paths = []  # 絶対パス
        camtoworld_list = []  # 各画像のカメラ-to-world行列
        self.camera_ids = []   # すべて 0 を格納
        
        for frame in frames:
            file_path = frame.get("file_path", "")
            # 拡張子が無い場合、.png を補完
            if not os.path.splitext(file_path)[1]:
                file_path = file_path + ".png"
            self.image_names.append(file_path)
            full_path = os.path.join(data_dir, file_path)
            self.image_paths.append(full_path)
            
            # transform_matrix を numpy array 化
            transform_matrix = np.array(frame.get("transform_matrix"), dtype=np.float32)
            camtoworld_list.append(transform_matrix)
            self.camera_ids.append(0)
        
        self.camtoworlds = np.stack(camtoworld_list, axis=0)  # (N, 4, 4)
        
        # --- 内部パラメータ K の計算 ---
        # 画像サイズは、最初の画像から取得
        if not os.path.exists(self.image_paths[0]):
            raise ValueError(f"Image file {self.image_paths[0]} not found.")
        img0 = imageio.imread(self.image_paths[0])[..., :3]
        H, W = img0.shape[:2]
        
        # 焦点距離 fx の計算 (camera_angle_x は radians)
        fx = (0.5 * W) / np.tan(0.5 * self.camera_angle_x)
        fy = fx  # ここでは fx=fy と仮定
        K = np.array([[fx, 0, W/2],
                      [0, fy, H/2],
                      [0,  0,   1]], dtype=np.float32)
        self.Ks_dict = {0: K}
        self.imsize_dict = {0: (W, H)}
        # Blender の場合、歪みは無いので空配列でよい
        self.params_dict = {0: np.empty(0, dtype=np.float32)}
        
        # --- 変換行列は正規化不要のため単位行列 ---
        self.transform = np.eye(4, dtype=np.float32)
        
        # --- オプション: 3D 点群の読み込み ---
        ply_path = os.path.join(data_dir, "points3d.ply")
        if os.path.exists(ply_path) and o3d is not None:
            print("Loading 3D point cloud from:", ply_path)
            pcd = o3d.io.read_point_cloud(ply_path)
            pts = np.asarray(pcd.points, dtype=np.float32)
            colors = np.asarray(pcd.colors, dtype=np.uint8)
            self.points = pts
            self.points_rgb = colors
            # 簡易的に、各画像は全点を参照する（実際は精度を上げるための工夫が必要）
            self.point_indices = {name: np.arange(pts.shape[0], dtype=np.int32)
                                  for name in self.image_names}
        else:
            self.points = np.empty((0, 3), dtype=np.float32)
            self.points_rgb = np.empty((0, 3), dtype=np.uint8)
            self.point_indices = {name: np.empty((0,), dtype=np.int32)
                                  for name in self.image_names}
            
        # size of the scene measured by cameras
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)
        
        print(f"[Parser] Loaded {len(self.image_names)} images from {data_dir}.")


########################################
# Dataset クラス
########################################

class Dataset:
    """
    Blender NeRF Synthetic データセット用 Dataset クラス
    
    Parser クラスで抽出した情報から、各サンプル（画像、内部パラメータ K、カメラ姿勢など）を
    PyTorch Tensor に変換して返す。
    
    オプション:
      - patch_size: 画像からランダムクロップする場合のサイズ
      - load_depths: 3D 点群から Depth 情報を生成（points3d.ply が存在する場合）
    """
    def __init__(self, parser: Parser, 
                 split: str = "train", 
                 patch_size: int = None, 
                 load_depths: bool = False):
        self.parser = parser
        self.split = split  # Blender では "train" のみ対応
        self.patch_size = patch_size
        self.load_depths = load_depths
        # train/test 分割は不要なので、全画像を対象とする
        self.indices = np.arange(len(self.parser.image_names))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, item: int) -> dict:
        index = self.indices[item]
        # 画像の読み込み
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        # 画像サイズはそのまま
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()
        camtoworld = self.parser.camtoworlds[index]
        
        # --- パッチクロップ (オプション) ---
        if self.patch_size is not None:
            H, W = image.shape[:2]
            if H > self.patch_size and W > self.patch_size:
                x = random.randint(0, W - self.patch_size)
                y = random.randint(0, H - self.patch_size)
                image = image[y:y+self.patch_size, x:x+self.patch_size, :]
                # K の主点（cx,cy）を調整
                K[0, 2] -= x
                K[1, 2] -= y
        
        data = {
            "image": torch.from_numpy(image).float(),
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image_id": int(item)
        }
        
        # --- オプション: Depth 情報の生成 ---
        if self.load_depths and self.parser.points.shape[0] > 0:
            # カメラの world-to-camera 行列
            world_to_cam = np.linalg.inv(camtoworld)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices.get(image_name, np.empty((0,), dtype=np.int32))
            if point_indices.size > 0:
                points_world = self.parser.points[point_indices]
                # カメラ座標に変換
                R = world_to_cam[:3, :3]
                t = world_to_cam[:3, 3]
                points_cam = (R @ points_world.T + t.reshape(3, 1)).T  # (M,3)
                # 内部パラメータ行列 K を用いて投影
                pts_proj = (K @ points_cam.T).T  # (M,3)
                pts_pixel = pts_proj[:, :2] / pts_proj[:, 2:3]
                depths = points_cam[:, 2]
                # 画像内に収まるかつ、正の深度を持つ点のみ採用
                H_img, W_img = image.shape[:2]
                valid = (pts_pixel[:, 0] >= 0) & (pts_pixel[:, 0] < W_img) & \
                        (pts_pixel[:, 1] >= 0) & (pts_pixel[:, 1] < H_img) & (depths > 0)
                pts_pixel = pts_pixel[valid]
                depths = depths[valid]
                data["points"] = torch.from_numpy(pts_pixel).float()
                data["depths"] = torch.from_numpy(depths).float()
        
        return data

########################################
# 開発用テストコード
########################################

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("--data_dir", type=str, default="dataset",
                            help="Blender NeRF Synthetic データセットのルートディレクトリ")
    parser_arg.add_argument("--patch_size", type=int, default=None,
                            help="ランダムクロップするパッチサイズ（指定しない場合はフル画像）")
    parser_arg.add_argument("--load_depths", action="store_true",
                            help="3D点群から Depth 情報を生成する場合のフラグ")
    args = parser_arg.parse_args()
    
    print("=== Blender NeRF Synthetic Parser のテスト ===")
    dataset_parser = Parser(args.data_dir)
    print(f"Loaded {len(dataset_parser.image_names)} images.")
    W, H = dataset_parser.imsize_dict[0]
    print(f"Image size: {W}x{H}")
    print("Camera angle x (radian):", dataset_parser.camera_angle_x)
    
    ds = Dataset(dataset_parser, patch_size=args.patch_size, load_depths=args.load_depths)
    print(f"Dataset length: {len(ds)}")
    
    # サンプルの取得と内容の表示
    sample = ds[0]
    print("Sample keys:", sample.keys())
    print("Image tensor shape:", sample["image"].shape)
    print("Internal matrix K:\n", sample["K"])
    print("Camera-to-world matrix:\n", sample["camtoworld"])
    
    if args.load_depths and "points" in sample:
        print("Depth points shape:", sample["points"].shape)
        print("Depth values shape:", sample["depths"].shape)
    
    # matplotlib による画像表示
    plt.figure()
    plt.imshow(sample["image"].numpy().astype(np.uint8))
    plt.title("Sample Image")
    plt.axis("off")
    plt.show()

