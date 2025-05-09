import torch
from torch import Tensor
from optical.quartic_solver.feralli import solve_quartic_ferrari
from optical.quartic_solver.newton_method import solve_quartic_newton

from internal.utils.gaussian_utils import GaussianTransformUtils
from optical.utils.rotation_utils import quat_from_2dirs


class WaterSruface(torch.autograd.Function):
    def __init__(self, 
                 device: str = "cuda",
                 means: torch.Tensor = None,
                 quats: torch.Tensor = None,
                 cam_center: torch.Tensor = None, 
                 n: torch.Tensor = 1.33, 
                 plane: torch.Tensor = 0,
                 flag_solve_quartic_by_newton: bool = True,
    ):
        self.n = torch.tensor(n, dtype=torch.float32, device=device, requires_grad=False)
        self.plane = torch.tensor(plane, dtype=torch.float32, device=device, requires_grad=False)
        self.device = device
        self.flag_solve_quartic_by_newton = flag_solve_quartic_by_newton
        
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

    ### ------------------------------
    ###        Calcurate Intersection of the Ray, 
    ###        r value that light path on the surface between camera center and gaussians center
    ### ------------------------------

    def calculate_quartic_terms(self):
        # 4次方程式の係数を計算
        a4 = - self.n2m1
        a3 = 2 * self.n2m1 * self.r
        a2 = - self.n2m1 * self.r2 + self.z2 - self.n2 * self.H2
        a1 = 2 * self.n2 * self.r * self.H2
        a0 = -self.n2 * self.H2 * self.r2

        return a4, a3, a2, a1, a0

    def calc_intersection(self,
        num_iters: int = 10,
        tol: float = 1e-2
    ):
        a4, a3, a2, a1, a0 = self.calculate_quartic_terms()  
        if self.flag_solve_quartic_by_newton:
            self.s = solve_quartic_newton(a4, a3, a2, a1, a0, self.r, num_iters=num_iters, tol=tol)
        else:
            s = solve_quartic_ferrari(a4, a3, a2, a1, a0)
            # s is like roots = torch.stack([r1 - z0, r2 - z0, r3 - z0, r4 - z0], dim=-1) of dtype = torch.complex128
            # Extract real roots (imaginary part is close to zero)
            real_mask = torch.abs(s.imag) < 1e-5
            s_real = s.real.clone()
            s_real[~real_mask] = float('inf')  # Mark non-real roots as infinity
            # Find the smallest real root for each Gaussian
            self.s = torch.min(s_real, dim=-1).values
        
        # self.s = torch.where(self.s < self.r, self.s, self.r * 0.99) this is not needed
        self.s2 = self.s ** 2
        
        # compute intersection point
        self.xs = self.x0 + self.s * torch.cos(self.phi)
        self.ys = self.y0 + self.s * torch.sin(self.phi)
        plane_vec = torch.full_like(self.xs, (self.plane - self.H), device=self.device, dtype=self.xs.dtype)
        self.intersection = torch.stack([self.xs, self.ys, plane_vec], dim=1)
        
        # compute Ray direction from camera center to intersection point
        # this means the direction from camera center to the apparent position of the Gaussian
        self.ray_to_apparent = torch.stack(
            [self.xs - self.x0, 
             self.ys - self.y0, 
             plane_vec], dim=1
        )
        self.ray_to_apparent = self.ray_to_apparent / torch.norm(self.ray_to_apparent, dim=1, keepdim=True).clamp(min=1e-8)
        
        # compute Ray direction from intersection point to real Gaussian center
        self.ray_intersec2gaussian = torch.stack(
            [self.x - self.xs, 
             self.y - self.ys, 
             self.z - plane_vec], dim=1
        )
        self.ray_intersec2gaussian = self.ray_intersec2gaussian / torch.norm(self.ray_intersec2gaussian, dim=1, keepdim=True).clamp(min=1e-8)
        
    def calc_theta(self,
    ):
        self.theta0 = torch.atan(self.s / self.H)
        self.theta1 = torch.atan((self.r - self.s) / (-self.z))
        self.d_theta = self.theta0 - self.theta1
        
        

    ### ------------------------------
    ###        Calcurate apparent position of Gaussian centers
    ### ------------------------------    
    def calc_appearance_position(self,
    ):
        self.offset_r =  self.n2m1 * self.z * (torch.tan(self.theta1)**3)    
        self.ra = self.r + self.offset_r
        self.za = 1/self.n * self.z * (torch.cos(self.theta0) / torch.cos(self.theta1))**3 
            
    
    def transform_to_appearance(self,
    ):        
        self.calc_intersection()
        self.calc_theta()
        self.calc_appearance_position()
        
        # 見かけの位置に座標変換
        new_x = self.x0 + self.ra * torch.cos(self.phi)
        new_y = self.y0 + self.ra * torch.sin(self.phi)
        new_z = self.plane + self.za
        
        self.new_means = torch.stack([new_x, new_y, new_z], dim=1)
        self.new_quats = self.calc_apparent_quaternion()
        
        return self.new_means, self.new_quats


    ### ------------------------------
    ###        Calcurate ROTATION corrected by quaternion
    ### ------------------------------
    def calc_apparent_quaternion(self,
    ):
        d_q = quat_from_2dirs(self.ray_intersec2gaussian, self.ray_to_apparent)
        # d_q = quat_from_2dirs(self.ray_to_apparent, self.ray_intersec2gaussian) 
        new_quats = GaussianTransformUtils.quat_multiply(self.quats, d_q)
        return new_quats