import numpy as np

def transform_gaussian_point(point, cam_center, n=1.33, plane=0, atol=1e-8):
    """
    1点のGaussian中心 (world座標) を、カメラ中心および水面 z=plane を基準に屈折補正する。
    point: (x,y,z)
    cam_center: (x0,y0,H) カメラ位置（world座標）
    n: 屈折率（例: 水の場合 1.33）
    """
    # カメラ中心とのオフセット
    cam_center = np.squeeze(cam_center)
    x0, y0, _ = cam_center
    H = cam_center[2]
    dx = point[0] - x0
    dy = point[1] - y0
    # 水面 z=plane に対する深さ。水中であれば h は負になる想定
    h = point[2] - plane  
    angle = np.arctan2(dy, dx)
    r = np.sqrt(dx**2 + dy**2)
    
    n2 = n**2
    H2 = H**2
    h2 = h**2
    r2 = r**2
    
    # 四次方程式の係数 (sに関する方程式)
    a4 = 1 - n2
    a3 = 2 * (n2 - 1) * r 
    a2 = (1 - n2) * (h2 + r2)
    a1 = 2 * n2 * r * H2
    a0 = - n2 * H2 * r2
    
    coeffs = [a4, a3, a2, a1, a0]
    s_roots = np.roots(coeffs)
    rs = 0  # デフォルト値
    # 実数解かつ s < r となる解を採用
    for s in s_roots:
        if np.abs(s.imag) < atol and s.real < r:
            rs = s.real
            break

    # 入射角 theta0, 屈折角 theta1 の計算
    theta0 = np.arcsin(rs / np.hypot(H, rs))
    theta1 = np.arcsin((r - rs) / np.hypot(-h, r - rs))
    
    # 補正量の算出
    dr = h * (n2 - 1) * (np.tan(theta1)**3)  # r の補正
    ra = r - dr
    A = (1 - n2 * (np.sin(theta1)**2))**1.5
    za = h * A / (n * (np.cos(theta1)**3))
    
    # 水平方向の補正 (カメラ中心からのずれ)
    dx_app = ra * np.cos(angle)
    dy_app = ra * np.sin(angle)
    
    # 見かけ上の位置 (world座標)
    new_x = x0 + dx_app
    new_y = y0 + dy_app
    new_z = plane + za
    
    return np.array([new_x, new_y, new_z])


def transform_all_gaussians(means_np, cam_center, n=1.33, plane=0, atol=1e-8):
    """
    全てのGaussian中心 (numpy 配列, shape: [N,3]) に対して屈折変換を適用する。
    """
    new_means = []
    for i in range(means_np.shape[0]):
        new_point = transform_gaussian_point(means_np[i], cam_center, n, plane, atol)
        new_means.append(new_point)
        
    if len(new_means) == 0:
    # Return an empty array with shape (0, number of dimensions), here assuming 3.
        return np.empty((0, means_np.shape[1]))
    
    return np.stack(new_means, axis=0)



# Culling
def culling_points(points, W2C, K, width, height):
    """
    点群(points: [N,3])を、カメラの視錐台内にあるかで判定する。
    1. 点を同次座標に拡張して W2C でカメラ座標系に変換
    2. カメラ座標での深度 (z) > 0 であることを確認
    3. K を用いて画像平面に射影し、(u,v) が画像サイズ内にあるか判定
    """
    
    N = points.shape[0]
    points_h = np.concatenate([points, np.ones((N,1))], axis=1) # [N, 4]
    
    # convert to camera coordinate
    points_cam = (W2C @ points_h.T).T
    points_cam = points_cam[:, :3]
    
    # get points in front of camera
    valid_depth = points_cam[:, 2] > 0
    
    # project to image plane
    proj = (K @ points_cam.T).T
    proj = proj / proj[:, 2:3] # normalize by z
    u = proj[:, 0]
    v = proj[:, 1]
    
    valid_u = (u >= 0) & (u < width)
    valid_v = (v >= 0) & (v < height)
    
    mask = valid_depth & valid_u & valid_v
    
    return mask.squeeze()