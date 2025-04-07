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
    a2 = (1 - n2)*r2 + h2 - n2 * H2
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

def transform_gaussians_torch(
    means: torch.Tensor, 
    quats: torch.Tensor, 
    cam_center: torch.Tensor, 
    n: float = 1.33, 
    plane: float = 0, 
    num_iters: int = 10, 
    tol: float = 1e-6) -> torch.Tensor:
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
    offset_r = - h * (n**2 - 1) * (torch.tan(theta1)**3)  # (> 0)
    r_app = r - offset_r
    A = (1 - n**2 * (torch.sin(theta1)**2)).clamp(min=1e-8) ** 1.5
    z_app = h * A / (n * (torch.cos(theta1)**3))
    
    x_app = r_app * torch.cos(angle)
    y_app = r_app * torch.sin(angle)
    
    # 見かけの位置に座標変換
    new_x = x0 + x_app
    new_y = y0 + y_app
    new_z = plane + z_app
    new_means = torch.stack([new_x, new_y, new_z], dim=1)  # [N, 3]
    
    
    # ======= Rotation =========
    flag_rotation = True
    # flag_rotation = False
    
    
    # set the rotation axis
    dir = torch.stack([x_app, y_app, z_app-H],dim=1 )  # [N, 3]
    z_axis = torch.tensor([0, 0, 1], device=means.device, dtype=means.dtype).expand_as(dir)  # [N, 3]
    
    # rotation axis is the cross product of dir and z_axis
    rot_axis = torch.cross(dir, z_axis, dim=1)  # [N, 3]
    rot_axis = torch.nn.functional.normalize(rot_axis, dim=1)  # [N, 3]
    
    # rotation angle
    delta_theta = theta0 - theta1
    
    # rotation quaternion
    half_d_theta = delta_theta / 2
    sin_half_delta_theta = torch.sin(half_d_theta)
    delta_qw = torch.cos(half_d_theta).unsqueeze(1)     # [N, 1]
    delta_qxqyqz = rot_axis * sin_half_delta_theta.unsqueeze(1)  # [N, 3]
    delta_quat = torch.cat([delta_qw, delta_qxqyqz], dim=1)          # [N, 4]
    
    # rotate the quaternion
    if flag_rotation:
        new_quats = GaussianTransformUtils.quat_multiply(delta_quat, quats) # 正解? # [N, 4]
        # new_quats = GaussianTransformUtils.quat_multiply(quats, d_quat) # 反対 # [N, 4]
    else:
        new_quats = quats.clone()

    # return new_means, new_quats, theta0, theta1, s, r_app, offset_r
    return new_means, new_quats


class Refraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                means, 
                quats, 
                cam_center, 
                n, 
                plane, 
                num_iters, 
                tol
        ):
        device = means.device
        
        RT = RefractionTransform(
            device=device,
            n=n,
            plane=plane
        )
        
        
        transformed_means, jacobian = RT.transform_to_appearance(
            means, quats, cam_center, n, plane, num_iters, tol
        )
        
        ctx.save_for_backward(jacobian)  
        return transformed_means
    
    @staticmethod
    def backward(ctx, grad_means, grad_quats):
        jacobian, = ctx.saved_tensors
        grad_input = torch.einsum('nij,nj->ni', jacobian, grad_means)
        print("grad_input", grad_input.shape)
        print("grad_means", grad_means.shape)
        print("grad_quats", grad_quats.shape)
        
        return grad_input, grad_quats, None, None, None, None, None, None
    
    
        
            
        



class RefractionTransform(torch.autograd.Function):
    def __init__(self, 
                 device: str = "cuda",
                 means: torch.Tensor = None,
                 quats: torch.Tensor = None,
                 cam_center: torch.Tensor = None, 
                 n: float = 1.33, 
                 plane: float = 0
    ):
        self.n = torch.tensor(n, dtype=torch.float32, device=device, 
                              requires_grad=False)
        self.plane = torch.tensor(plane, dtype=torch.float32, device=device, 
                                   requires_grad=False)
        self.device = device
        
        self.x0, self.y0, self.H = cam_center[0], cam_center[1], cam_center[2]
        
        self.means = means
        self.quats = quats
        self.num_g = means.shape[0]
        self.x = means[:, 0] - self.x0
        self.y = means[:, 1] - self.y0
        self.z = means[:, 2] - self.plane
        self.r = torch.sqrt(self.x**2 + self.y**2)
        self.phi = torch.atan2(self.y, self.x)
        
        self.n2 = self.n ** 2
        self.n2m1 = self.n2 - 1
        self.H2 = self.H ** 2
        self.r2 = self.r ** 2
        self.x2 = self.x ** 2
        self.y2 = self.y ** 2
        self.z2 = self.z ** 2
        
    # def get_camera_center(self,
    #     cam_center: torch.Tensor,
    # ):
    #     self.x0, self.y0, self.H = cam_center[0], cam_center[1], cam_center[2]
    #     return self.x0, self.y0, self.H
    
    # def get_gaussian_params(self,
    #     means: torch.Tensor,
    #     quats: torch.Tensor,
    # ):
    #     self.means = means
    #     self.quats = quats
    #     self.num_g = means.shape[0]
    #     self.x = means[:, 0] - self.x0
    #     self.y = means[:, 1] - self.y0
    #     self.z = means[:, 2] - self.plane
    #     self.r = torch.sqrt(self.x**2 + self.y**2)
    #     self.phi = torch.atan2(self.y, self.x)
    
    # def set_quadratic(self
    #     ):
    #     self.n2 = self.n ** 2
    #     self.n2m1 = self.n2 - 1
    #     self.H2 = self.H ** 2
    #     self.r2 = self.r ** 2
    #     self.x2 = self.x ** 2
    #     self.y2 = self.y ** 2
    #     self.z2 = self.z ** 2
        
    # [TODO] i dont implement this yet
    def calc_s(self,
        num_iters: int = 10,
        tol: float = 1e-2
    ):
        # Newton法で s を求める
        s = newton_solve_quartic_torch(self.r, self.z, self.H, self.n, num_iters=num_iters, tol=tol)
        # 物理的制約として s < r となるように clamping（必要に応じて調整）
        self.s = torch.where(s < self.r, s, self.r * 0.99)
        self.s2 = self.s ** 2

        
    
    def calc_theta(self,
    ):
        self.theta0 = torch.atan(self.s / self.H)
        self.theta1 = torch.atan((self.s - self.r) / (-self.z))
    
    def calc_appearance(self,
    ):
        self.offset_r =  self.n2m1 * self.z * (torch.tan(self.theta1)**3)    
        self.ra = self.r + self.offset_r
        self.za = 1/self.n * self.z * (torch.cos(self.theta0) / torch.cos(self.theta1))**3 
        
        
    def dtheta1_dtheta0(self,
    ):
        return torch.cos(self.theta0) / (self.n * torch.cos(self.theta1))
    
    def dtheta0_ds(self,
    ):
        return self.n * torch.cos(self.theta1)**3 / (self.z * torch.cos(self.theta0))
    
    def dtheta1_ds(self,
    ):
        return self.dtheta1_dtheta0 * self.dtheta0_ds 

    def ds_dr(self,
    ):
        num = (self.n2m1*self.s2 + self.n2*self.H2) * (self.s-self.r) # 分子
        denom = (self.n2m1*(2*self.s-self.r)*self.s + self.n2*self.H2) * (self.s-self.r) - self.z2*self.s # 分母
        return num / denom
    
    def ds_dz(self,
    ):
        num = self.z*self.s2 - (self.n2m1*self.r2 + self.n2*self.H2) * self.s + self.n2*self.H2*self.r
        demon = self.n2m1 * (2*self.s - 3*self.r) * self.s2 - self.z2*self.s
        return num / demon
    
    def dra_dr(self,
    ):
        return 1 + 3*self.n2m1 * self.z * torch.tan(self.theta1)**2 / torch.cos(self.theta1)**2 * self.dtheta1_ds * self.ds_dr
    
    def dra_dz(self,  
    ):
        return self.n2m1 * torch.tan(self.theta1)**3 + 3*self.n2m1 * self.z * torch.tan(self.theta1)**2 / torch.cos(self.theta1)**2 * self.dtheta1_ds * self.ds_dz

    def dza_dr(self,
    ):
        return 3*self.n2m1/self.n * torch.cos(self.theta0) * torch.sin(self.theta1) /  torch.cos(self.theta1)**4 * self.dtheta1_ds * self.ds_dr
    
    def dza_dz(self,
    ):
        return 1/self.n * (torch.cos(self.theta1) / torch.cos(self.theta0))**3 - 3*self.n2m1/self.n * self.z * torch.cos(self.theta0) * torch.sin(self.theta1) /  torch.cos(self.theta1)**4 * self.dtheta1_ds * self.ds_dz
    
    def dPa_dP(self,
    ):
        jacobian = torch.zeros((self.num_g, 3, 3), device=self.device, dtype=self.means.dtype).expand_as(self.means)
        jacobian[:, 0, 0] = self.dra_dr * self.x2 / self.r2 + self.ra + self.y2 / self.r**3
        jacobian[:, 0, 1] = (self.dra_dr - self.ra / self.r) * self.x * self.y / self.r2
        jacobian[:, 0, 2] = self.dra_dz * self.x / self.r
        jacobian[:, 1, 0] = jacobian[0, 1]
        jacobian[:, 1, 1] = self.dra_dr * self.y2 / self.r2 + self.ra + self.x2 / self.r**3
        jacobian[:, 1, 2] = self.dra_dz * self.y / self.r
        jacobian[:, 2, 0] = jacobian[0, 2]
        jacobian[:, 2, 1] = jacobian[1, 2]
        jacobian[:, 2, 2] = self.dza_dz 
        
        return jacobian
    
    def transform_to_appearance(self,
        means: torch.Tensor,
        quats: torch.Tensor,
        cam_center: torch.Tensor,
        n: float = 1.33,
        plane: float = 0,
        num_iters: int = 10,
        tol: float = 1e-6
    ):
        self.get_camera_center(cam_center)
        self.get_gaussian_params(means, quats)
        self.set_quadratic()
        
        self.calc_s(num_iters=num_iters, tol=tol)
        self.calc_theta()
        self.calc_appearance()
        
        # 見かけの位置に座標変換
        new_x = self.x0 + self.ra * torch.cos(self.phi)
        new_y = self.y0 + self.ra * torch.sin(self.phi)
        new_z = self.plane + self.za
        
        new_means = torch.stack([new_x, new_y, new_z], dim=1)
        
        return new_means, self.dPa_dP()
        
    
    # @staticmethod
    # def forward(ctx, means, quats, cam_center, n, plane, num_iters, tol):
    #     transformed_means, transformed_quats,jacobian = RefractionTransform().transform_to_appearance(
    #         means, quats, cam_center, n, plane, num_iters, tol
    #     )
    #     ctx.save_for_backward(means, quats, jacobian)  # 元の値を保存
    #     return transformed_means, transformed_quats
    
    # @staticmethod
    # def backward(ctx, grad_means, grad_quats):
    #     # backward: 元の勾配を流す
    #     means= ctx.saved_tensors
    #     d_means = ctx.jacobian 
    #     return d_means, grad_quats, None, None, None, None, None, None
        
        
# class aaa(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, means, quats, cam_center, n, plane, num_iters, tol):
#         # 計算に必要なパラメータを取得
#         device = means.device
#         n_tensor = torch.tensor(n, dtype=torch.float32, device=device)
#         plane_tensor = torch.tensor(plane, dtype=torch.float32, device=device)
        
#         # 変換処理を実行
#         transformed_means, jacobian = aaa.transform_to_appearance(
#             means, cam_center, n_tensor, plane_tensor, num_iters, tol
#         )
        
#         # 逆伝播に必要な情報を保存
#         ctx.save_for_backward(jacobian)
#         return transformed_means, quats  # quatsはとりあえず変更なしで返す

#     @staticmethod
#     def transform_to_appearance(means, cam_center, n, plane, num_iters, tol):
#         device = means.device
#         N = means.shape[0]
        
#         # カメラ中心を取得
#         x0, y0, H = cam_center[0], cam_center[1], cam_center[2]
        
#         # Gaussianパラメータを計算
#         x = means[:, 0] - x0
#         y = means[:, 1] - y0
#         z = means[:, 2] - plane
#         r = torch.sqrt(x**2 + y**2)
#         phi = torch.atan2(y, x)
        
#         # Newton法でsを計算
#         s = newton_solve_quartic_torch(r, z, H, n, num_iters=num_iters, tol=tol)
#         s = torch.where(s < r, s, r * 0.99)
        
#         # 角度計算
#         theta0 = torch.atan(s / H)
#         theta1 = torch.atan((s - r) / (-z))
        
#         # 見かけの位置計算
#         n2 = n**2
#         n2m1 = n2 - 1
#         offset_r = n2m1 * z * torch.tan(theta1)**3
#         ra = r + offset_r
#         za = (z / n) * (torch.cos(theta0) / torch.cos(theta1))**3
        
#         # Jacobian計算
#         jacobian = aaa.compute_jacobian(
#             x, y, z, r, s, H, n, theta0, theta1, device
#         )
        
#         # 新しい座標を計算
#         new_x = x0 + ra * torch.cos(phi)
#         new_y = y0 + ra * torch.sin(phi)
#         new_z = plane + za
        
#         return torch.stack([new_x, new_y, new_z], dim=1), jacobian

#     @staticmethod
#     def compute_jacobian(x, y, z, r, s, H, n, theta0, theta1, device):
#         N = x.shape[0]
#         jacobian = torch.zeros((N, 3, 3), device=device)
        
#         # 中間変数の計算
#         n2 = n**2
#         n2m1 = n2 - 1
#         cos_theta0 = torch.cos(theta0)
#         cos_theta1 = torch.cos(theta1)
#         tan_theta1 = torch.tan(theta1)
        
#         # 偏微分計算
#         dtheta1_dtheta0 = cos_theta0 / (n * cos_theta1)
#         dtheta0_ds = n * cos_theta1**3 / (z * cos_theta0)
#         dtheta1_ds = dtheta1_dtheta0 * dtheta0_ds
        
#         # ds/drとds/dzの計算（仮実装、実際の物理モデルに合わせて修正が必要）
#         ds_dr = torch.ones_like(r)
#         ds_dz = torch.ones_like(z)
        
#         # dra/drとdra/dz
#         dra_dr = 1 + 3 * n2m1 * z * tan_theta1**2 / cos_theta1**2 * dtheta1_ds * ds_dr
#         dra_dz = n2m1 * tan_theta1**3 + 3 * n2m1 * z * tan_theta1**2 / cos_theta1**2 * dtheta1_ds * ds_dz
        
#         # dza/drとdza/dz
#         dza_dr = 3 * n2m1 / n * cos_theta0 * torch.sin(theta1) / cos_theta1**4 * dtheta1_ds * ds_dr
#         dza_dz = (1/n) * (cos_theta1 / cos_theta0)**3 - 3 * n2m1/n * z * cos_theta0 * torch.sin(theta1) / cos_theta1**4 * dtheta1_ds * ds_dz
        
#         # Jacobian行列を構築
#         x2 = x**2
#         y2 = y**2
#         r2 = r**2
#         r3 = r**3
        
#         jacobian[:, 0, 0] = dra_dr * x2 / r2 + (dra_dr * y2) / r3
#         jacobian[:, 0, 1] = (dra_dr - dra_dr * x2 / r2) * x * y / r2
#         jacobian[:, 0, 2] = dra_dz * x / r
        
#         jacobian[:, 1, 0] = jacobian[:, 0, 1]
#         jacobian[:, 1, 1] = dra_dr * y2 / r2 + (dra_dr * x2) / r3
#         jacobian[:, 1, 2] = dra_dz * y / r
        
#         jacobian[:, 2, 0] = dza_dr * x / r
#         jacobian[:, 2, 1] = dza_dr * y / r
#         jacobian[:, 2, 2] = dza_dz
        
#         return jacobian

#     @staticmethod
#     def backward(ctx, grad_means, grad_quats):
#         jacobian, = ctx.saved_tensors
#         grad_input = torch.einsum('nij,nj->ni', jacobian, grad_means)
#         return grad_input, None, None, None, None, None, None, None
        

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
    


