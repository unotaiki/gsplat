import os
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import NamedTuple

##############################################
# 既存コードで定義されている補助関数の例
##############################################

def focal2fov(focal_length, resolution):
    """
    例: `fov = 2 * arctan(0.5 * resolution / focal_length)` のような変換。
    """
    # 実装は utils.graphics_utils など参照
    # ここでは簡易サンプル (近似) を書きます:
    return 2.0 * np.arctan2(resolution * 0.5, focal_length)

def fov2focal(fov, resolution):
    """
    fov(ラジアン) から focal_length への変換。
    """
    return 0.5 * resolution / np.tan(0.5 * fov)

##############################################
# CameraInfo などの定義(元コードと同じ)
##############################################
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: float
    FovX: float
    image: Image.Image
    image_path: str
    image_name: str
    width: int
    height: int

##############################################
# ここから本題: NeRF Synthetic のカメラを読む関数
##############################################
def readCamerasFromTransforms(path, transforms_json, white_background, extension=".png"):
    """
    1つの transforms_*.json を読み込み、CameraInfo のリストを返す。
    """
    cam_infos = []

    json_path = os.path.join(path, transforms_json)
    with open(json_path, "r") as f:
        contents = json.load(f)

    # 横方向の視野角 (radians)
    fovx = contents["camera_angle_x"]

    frames = contents["frames"]
    for idx, frame in enumerate(frames):
        # 画像ファイル名を組み立てる
        # 例: "file_path": "./r_0" などが入っている
        cam_name = os.path.join(path, frame["file_path"] + extension)

        # NeRF 'transform_matrix' は カメラ→ワールド(c2w) 変換
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)

        # OpenGL/Blender座標(Y up, Z back) → COLMAP座標(Y down, Z forward) へ
        # y軸とz軸を反転する
        #   c2w[:3,1:3] *= -1  と同義
        c2w[0:3, 1] *= -1
        c2w[0:3, 2] *= -1

        # world→camera = その逆行列
        w2c = np.linalg.inv(c2w)
        # R, T の取り出し
        #   R は転置が必要(行列を転置して保存する実装の場合の互換性)
        R = w2c[:3, :3].T
        T = w2c[:3, 3]

        # 画像を読み込む (PIL)
        image_path = cam_name
        image_name = Path(image_path).stem
        image = Image.open(image_path)

        # 白背景指定なら、アルファチャンネル部分を白埋めにする
        if image.mode in ("RGBA", "LA") and white_background:
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1], dtype=np.float32)
            norm_data = im_data.astype(np.float32) / 255.0
            rgb = norm_data[:,:,:3]
            alpha = norm_data[:,:,3:4]
            # アルファブレンド
            arr = rgb * alpha + bg * (1.0 - alpha)
            arr_255 = np.uint8(np.clip(arr * 255.0, 0, 255))
            image = Image.fromarray(arr_255, "RGB")
        else:
            # RGBAでもwhite_backgroundを使わなければそのままRGBへ
            image = image.convert("RGB")

        # 焦点距離を計算しつつ、FovY を計算
        width, height = image.size
        # fovy = focal2fov(fov2focal(fovx, width), height)
        focal_x = fov2focal(fovx, width)   # 画像の横幅に対応する焦点距離
        fovy = focal2fov(focal_x, height) # それを使って縦方向FOVを計算

        FovY = fovy
        FovX = fovx

        cam_infos.append(CameraInfo(
            uid = idx,
            R   = R,
            T   = T,
            FovY = FovY,
            FovX = FovX,
            image = image,
            image_path = image_path,
            image_name = image_name,
            width = width,
            height= height
        ))

    return cam_infos

def readNerfSyntheticCameras(path, white_background=True, extension=".png", eval_mode=False):
    """
    transforms_train.json / transforms_test.json からカメラを読み込み、
    (train_cams, test_cams) の2リストを返す、簡易ヘルパー関数。

    eval_mode=True なら「train/test を分割したまま」返す。
    eval_mode=False なら「train と test をまとめて train 扱い」にし、testは空。
    """
    # train
    train_json = "transforms_train.json"
    if os.path.exists(os.path.join(path, train_json)):
        print("Reading Training Transforms...")
        train_cams = readCamerasFromTransforms(path, train_json, white_background, extension)
    else:
        train_cams = []

    # test
    test_json = "transforms_test.json"
    if os.path.exists(os.path.join(path, test_json)):
        print("Reading Test Transforms...")
        test_cams = readCamerasFromTransforms(path, test_json, white_background, extension)
    else:
        test_cams = []

    if not eval_mode:
        # train と test をまとめる
        train_cams.extend(test_cams)
        test_cams = []

    return train_cams, test_cams


# Gaussian Splatting 用のデータセットクラス
class GaussianSplattingDataset:
    """
    NeRF Synthetic の transforms_json を用いてカメラ情報と画像を読み込み、
    Gaussian Splatting 用のデータセットを作成するクラスです。
    ※ depth 情報は読み込みません。
    """
    def __init__(self, root_dir, split="train", patch_size=None, white_background=True):
        """
        Args:
            root_dir (str): transforms_*.json ファイルがあるディレクトリのパス
            split (str): "train" または "test"
            patch_size (int, optional): 画像のランダムクロップサイズ。None の場合はクロップしません。
            white_background (bool): RGBA画像の場合、白背景に変換するかどうか
        """
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size

        train_cams, test_cams = readNerfSyntheticCameras(root_dir, white_background=white_background)
        self.cameras = train_cams if split == "train" else test_cams

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, index):
        cam = self.cameras[index]
        # 画像は既に PIL.Image で読み込まれているので、NumPy配列に変換
        image = np.array(cam.image)
        
        # カメラ内部パラメータ行列 (K) の計算
        fx = 0.5 * cam.width / np.tan(0.5 * cam.FovX)
        fy = 0.5 * cam.height / np.tan(0.5 * cam.FovY)
        cx = cam.width / 2.0
        cy = cam.height / 2.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float32)
        
        # カメラ→ワールド変換行列の作成（4×4）
        camtoworld = np.eye(4, dtype=np.float32)
        camtoworld[:3, :3] = cam.R
        camtoworld[:3, 3] = cam.T

        # patch_size が指定されている場合はランダムクロップを行う
        if self.patch_size is not None:
            h, w = image.shape[0], image.shape[1]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y:y+self.patch_size, x:x+self.patch_size]
            # 内部パラメータの主点も調整
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "image": torch.from_numpy(image).float(),       # 画像データ
            "K": torch.from_numpy(K).float(),               # カメラ内部パラメータ行列
            "camtoworld": torch.from_numpy(camtoworld).float(),  # カメラ→ワールド変換行列
            "image_id": index,
        }
        return data