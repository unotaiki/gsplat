import torch
from internal.utils.gaussian_utils import GaussianTransformUtils

###############################################################################
# 1. GPU上で動作する屈折変換のための補助関数群
###############################################################################

def newton_solve_quartic_torch(r: torch.Tensor, h: torch.Tensor, H: float, n: float, num_iters: int = 10, tol: float = 1e-2) -> torch.Tensor:
    """
    各サンプルについて、Newton法により s に関する4次方程式の根を解く関数
    4次方程式は次の係数で表される:
      a4 = 1 - n^2  
      a3 = 2*(n^2 - 1)*r  
      a2 = (1 - n^2)*(h^2 + r^2)  
      a1 = 2*n^2*r*H^2  
      a0 = - n^2 * H^2 * r^2  
    """
    n2 = n ** 2
    H2 = H ** 2
    r2 = r ** 2
    h2 = h ** 2

    a4 = 1 - n2
    a3 = 2 * (n2 - 1) * r
    a2 = (1 - n2) * (h2 + r2)
    a1 = 2 * n2 * r * H2
    a0 = - n2 * H2 * r2

    eps = 1e-8
    s = r / 1.7 + eps  # 初期値

    for _ in range(num_iters):
        f = a4 * s**4 + a3 * s**3 + a2 * s**2 + a1 * s + a0
        f_prime = 4 * a4 * s**3 + 3 * a3 * s**2 + 2 * a2 * s + a1
        # 微小な勾配を防ぐため tol で補正
        f_prime_safe = torch.where(torch.abs(f_prime) < tol, torch.full_like(f_prime, tol), f_prime)
        s_new = s - f / f_prime_safe
        if torch.max(torch.abs(s_new - s)) < tol:
            s = s_new
            break
        s = s_new
    return s

def transform_gaussians_torch(means: torch.Tensor, quats: torch.Tensor, cam_center: torch.Tensor, n: float = 1.33, plane: float = 0, num_iters: int = 10, tol: float = 1e-6) -> torch.Tensor:
    """
    GPU上の torch.Tensor (shape: [N, 3]) に対して、
    カメラ中心 cam_center ([3]) および水面 z = plane を基準に屈折補正を適用する関数。
    """
    x0, y0, H = cam_center[0], cam_center[1], cam_center[2]
    
    dx = means[:, 0] - x0
    dy = means[:, 1] - y0
    h = means[:, 2] - plane  # 水面に対する深さ (< 0)
    r = torch.sqrt(dx**2 + dy**2)
    angle = torch.atan2(dy, dx)
    
    # Newton法で s を求める
    s = newton_solve_quartic_torch(r, h, H, n, num_iters=num_iters, tol=tol)
    # 物理的制約として s < r となるように clamping（必要に応じて調整）
    s = torch.where(s < r, s, r * 0.99)

    # 入射角・屈折角の計算
    theta0 = torch.atan(s / H)
    h_safe = h.clone()
    h_safe[h.abs() < tol] = tol * torch.sign(h[h.abs() < tol])
    theta1 = torch.atan((s - r) / h_safe)
    
    # 補正量の計算
    dr = - h * (n**2 - 1) * (torch.tan(theta1)**3)  # (> 0)
    r_app = r - dr
    A = (1 - n**2 * (torch.sin(theta1)**2)).clamp(min=1e-8) ** 1.5
    z_app = h * A / (n * (torch.cos(theta1)**3))
    
    dx_app = r_app * torch.cos(angle)
    dy_app = r_app * torch.sin(angle)
    
    # 見かけの位置に座標変換
    new_x = x0 + dx_app
    new_y = y0 + dy_app
    new_z = plane + z_app
    new_means = torch.stack([new_x, new_y, new_z], dim=1)  # [N, 3]
    
    
    # ======= Rotation =========
    flag_rotation = True
    # flag_rotation = False
    
    
    # set the rotation axis
    dir = torch.stack([dx_app, dy_app, z_app-H],dim=1 )  # [N, 3]
    z_axis = torch.tensor([0, 0, 1], device=means.device, dtype=means.dtype).expand_as(dir)  # [N, 3]
    
    # rotation axis is the cross product of dir and z_axis
    rot_axis = torch.cross(dir, z_axis, dim=1)  # [N, 3]
    rot_axis = torch.nn.functional.normalize(rot_axis, dim=1)  # [N, 3]
    
    # rotation angle
    d_theta = theta0 - theta1
    
    # rotation quaternion
    half_d_theta = d_theta / 2
    sin_half_d_theta = torch.sin(half_d_theta)
    d_qw = torch.cos(half_d_theta).unsqueeze(1)     # [N, 1]
    d_q = rot_axis * sin_half_d_theta.unsqueeze(1)  # [N, 3]
    d_quat = torch.cat([d_qw, d_q], dim=1)          # [N, 4]
    
    # rotate the quaternion
    if flag_rotation:
        new_quats = GaussianTransformUtils.quat_multiply(d_quat, quats) # 正解? # [N, 4]
        # new_quats = GaussianTransformUtils.quat_multiply(quats, d_quat) # 反対 # [N, 4]
    else:
        new_quats = quats.clone()

    return new_means, new_quats

def culling_points_torch(
    points: torch.Tensor, 
    W2C: torch.Tensor, 
    K: torch.Tensor, 
    width: int, 
    height: int, 
    mergin_factor: float=0.0
    ) -> torch.Tensor:
    """
    点群 points ([N, 3]) について、カメラの視錐台内にあるかを判定する関数。
      1. 同次座標に拡張し W2C でカメラ座標系へ変換
      2. カメラ座標で z > 0 かどうか確認
      3. K による射影後、(u, v) が画像内かどうか判定
    """
    N = points.shape[0]
    ones = torch.ones((N, 1), device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=1)  # [N, 4]
    
    points_cam = (W2C @ points_h.T).T[:, :3]
    valid_depth = points_cam[:, 2] > 0

    proj = (K @ points_cam.T).T
    proj = proj / proj[:, 2:3]
    u = proj[:, 0]
    v = proj[:, 1]
    
    w_mergin = int(width * mergin_factor)
    h_mergin = int(height * mergin_factor)
    valid_u = (u >= -w_mergin) & (u < width + w_mergin)
    valid_v = (v >= -h_mergin) & (v < height + h_mergin)
    
    mask = valid_depth & valid_u & valid_v
    return mask.squeeze()  # boolean Tensor [N]

###############################################################################
# 2. STE を用いた屈折補正の実装
###############################################################################

# アプローチ 1: カスタムAutograd Function を用いる方法
# Straight Through Estimator (STE)
class RefractionSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, quats, cam_center, n, plane, num_iters, tol):
        # forward: transform means by refraction model
        transformed_means, transformed_quats = transform_gaussians_torch(means, quats, cam_center, n, plane, num_iters, tol)
        ctx.save_for_backward(means, quats)  # 元の値を保存
        return transformed_means, transformed_quats

    @staticmethod
    def backward(ctx, grad_means, grad_quats):
        # backward: 元の勾配を流す
        d_means = grad_means
        d_quats = grad_quats
        return d_means, d_quats, None, None, None, None, None, None
    
# class RefractionSTE(torch.autograd.Function):
#     @staticmethod
    


