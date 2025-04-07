import torch
from internal.utils.gaussian_utils import GaussianTransformUtils


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
            means=means,
            quats=quats,
            cam_center=cam_center,
            n=n,
            plane=plane
        )
        
        transformed_means = RT.transform_to_appearance()
        
        jacobian = RT.dPa_dP()
        
        ctx.save_for_backward(jacobian)  
        return transformed_means, quats
    
    @staticmethod
    def backward(ctx, grad_means, grad_quats):
        jacobian, = ctx.saved_tensors
        grad_input = torch.einsum('nij,nj->ni', jacobian, grad_means) # must be [N, 3]
        
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
        
        
        # define gaussian params
        self.means = means
        self.quats = quats
        self.num_g = means.shape[0]
        
        # define the coordinate system
        self.x0, self.y0, self.H = cam_center[0], cam_center[1], cam_center[2]
        self.x = means[:, 0] - self.x0
        self.y = means[:, 1] - self.y0
        self.z = means[:, 2] - self.plane
        self.r = torch.sqrt(self.x**2 + self.y**2)
        self.phi = torch.atan2(self.y, self.x)
        
        # define quadratic
        self.n2 = self.n ** 2
        self.n2m1 = self.n2 - 1
        self.H2 = self.H ** 2
        self.r2 = self.r ** 2
        self.x2 = self.x ** 2
        self.y2 = self.y ** 2
        self.z2 = self.z ** 2

    def calc_s(self,
        num_iters: int = 10,
        tol: float = 1e-2
    ):
        # Newton法で s を求める
        self.solver_quartic()
        # 物理的制約として s < r となるように clamping（必要に応じて調整）
        # self.s = torch.where(self.s < self.r, self.s, self.r * 0.99)
        self.s2 = self.s ** 2
        
    def solver_quartic(
        self,
        num_iters: int = 10,
        tol: float = 1e-2
    ):
        a4 = 1 - self.n2
        a3 = 2 * (self.n2 - 1) * self.r
        a2 = (1 - self.n2) * self.r2 + self.z2 - self.n2 * self.H2
        a1 = 2 * self.n2 * self.r * self.H2
        a0 = - self.n2 * self.H2 * self.r2
        
        eps = 1e-6
        tol = 1e-2
        
        s = self.r / 1.7 + eps  # Initial guess
        for _ in range(8):
            f = a4 * s**4 + a3 * s**3 + a2 * s**2 + a1 * s + a0
            f_prime = 4 * a4 * s**3 + 3 * a3 * s**2 + 2 * a2 * s + a1
            # Avoid small gradient
            f_prime_safe = torch.where(torch.abs(f_prime) < tol, torch.full_like(f_prime, tol), f_prime)
            s_new = s - f / f_prime_safe
            if torch.max(torch.abs(s_new - s)) < tol:
                break
            s = s_new

        self.s = s
        self.s2 = s ** 2
        
    
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
        return self.dtheta1_dtheta0() * self.dtheta0_ds ()

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
        return 1 + 3*self.n2m1 * self.z * torch.tan(self.theta1)**2 / torch.cos(self.theta1)**2 * self.dtheta1_ds() * self.ds_dr()
    
    def dra_dz(self,  
    ):
        return self.n2m1 * torch.tan(self.theta1)**3 + 3*self.n2m1 * self.z * torch.tan(self.theta1)**2 / torch.cos(self.theta1)**2 * self.dtheta1_ds() * self.ds_dz()

    def dza_dr(self,
    ):
        return 3*self.n2m1/self.n * torch.cos(self.theta0) * torch.sin(self.theta1) /  torch.cos(self.theta1)**4 * self.dtheta1_ds() * self.ds_dr()
    
    def dza_dz(self,
    ):
        return 1/self.n * (torch.cos(self.theta1) / torch.cos(self.theta0))**3 - 3*self.n2m1/self.n * self.z * torch.cos(self.theta0) * torch.sin(self.theta1) /  torch.cos(self.theta1)**4 * self.dtheta1_ds() * self.ds_dz()
    
    def dPa_dP(self,
    ):
        jacobian = torch.zeros((self.num_g, 3, 3), device=self.device, dtype=self.means.dtype)
        jacobian[:, 0, 0] = self.dra_dr() * self.x2 / self.r2 + self.ra + self.y2 / self.r**3
        jacobian[:, 0, 1] = (self.dra_dr() - self.ra / self.r) * self.x * self.y / self.r2
        jacobian[:, 0, 2] = self.dra_dz() * self.x / self.r
        jacobian[:, 1, 0] = jacobian[:, 0, 1]
        jacobian[:, 1, 1] = self.dra_dr() * self.y2 / self.r2 + self.ra + self.x2 / self.r**3
        jacobian[:, 1, 2] = self.dra_dz() * self.y / self.r
        jacobian[:, 2, 0] = jacobian[:, 0, 2]
        jacobian[:, 2, 1] = jacobian[:, 1, 2]
        jacobian[:, 2, 2] = self.dza_dz()
        return jacobian
    
    def transform_to_appearance(self,
        num_iters: int = 10,
        tol: float = 1e-6
    ):
        
        self.calc_s(num_iters=num_iters, tol=tol)
        self.calc_theta()
        self.calc_appearance()
        
        # 見かけの位置に座標変換
        new_x = self.x0 + self.ra * torch.cos(self.phi)
        new_y = self.y0 + self.ra * torch.sin(self.phi)
        new_z = self.plane + self.za
        
        new_means = torch.stack([new_x, new_y, new_z], dim=1)
        return new_means


        
    

        

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


