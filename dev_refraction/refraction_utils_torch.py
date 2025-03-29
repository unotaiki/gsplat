import torch

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

def transform_all_gaussians_torch(means: torch.Tensor, cam_center: torch.Tensor, n: float = 1.33, plane: float = 0, num_iters: int = 10, tol: float = 1e-6) -> torch.Tensor:
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
    h_safe = torch.where(torch.abs(h) < tol, torch.full_like(h, tol), h)
    theta1 = torch.atan((s - r) / h_safe)
    
    # 補正量の計算
    dr = - h * (n**2 - 1) * (torch.tan(theta1)**3)  # (> 0)
    ra = r - dr
    A = (1 - n**2 * (torch.sin(theta1)**2)).clamp(min=1e-8) ** 1.5
    za = h * A / (n * (torch.cos(theta1)**3))
    
    dx_app = ra * torch.cos(angle)
    dy_app = ra * torch.sin(angle)
    
    new_x = x0 + dx_app
    new_y = y0 + dy_app
    new_z = plane + za
    
    return torch.stack([new_x, new_y, new_z], dim=1)  # [N, 3]

def culling_points_torch(points: torch.Tensor, W2C: torch.Tensor, K: torch.Tensor, width: int, height: int) -> torch.Tensor:
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
    valid_u = (u >= 0) & (u < width)
    valid_v = (v >= 0) & (v < height)
    
    mask = valid_depth & valid_u & valid_v
    return mask.squeeze()  # boolean Tensor [N]

###############################################################################
# 2. STE を用いた屈折補正の実装
###############################################################################

# アプローチ 1: カスタムAutograd Function を用いる方法
class RefractionSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original_means, cam_center, n, plane, num_iters, tol):
        # forward では、torchのみの屈折変換関数を呼ぶ
        transformed = transform_all_gaussians_torch(original_means, cam_center, n, plane, num_iters, tol)
        return transformed

    @staticmethod
    def backward(ctx, grad_output):
        # backward では、変換をバイパスして入力（original_means）に対して勾配をそのまま返す
        return grad_output, None, None, None, None, None

def transform_with_ste_custom(means: torch.Tensor, cam_center: torch.Tensor, n: float = 1.33, plane: float = 0, num_iters: int = 10, tol: float = 1e-6) -> torch.Tensor:
    """
    カスタムAutograd Function を用いた STE のラッパー関数。
    forward は屈折補正後の値を返すが、backward では元の means に対して勾配が流れる。
    """
    return RefractionSTE.apply(means, cam_center, n, plane, num_iters, tol)

# アプローチ 2: Detach & Identity Gradient Trick を用いる方法
def transform_with_detach_identity(means: torch.Tensor, cam_center: torch.Tensor, n: float = 1.33, plane: float = 0, num_iters: int = 10, tol: float = 1e-6) -> torch.Tensor:
    """
    Detach と identity gradient trick を用いた STE のラッパー関数。
    forward は屈折補正後の値を返すが、backward では元の means に対して勾配が流れる。
    """
    transformed_detached = transform_all_gaussians_torch(means.detach(), cam_center, n, plane, num_iters, tol)
    return transformed_detached + (means - means.detach())
