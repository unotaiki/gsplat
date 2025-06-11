
import torch
from torch import Tensor
from optical.quartic_solver.feralli import solve_quartic_ferrari
from optical.quartic_solver.newton_method import solve_quartic_newton

from internal.utils.gaussian_utils import GaussianTransformUtils
from optical.utils.rotation_utils import quat_from_2dirs, quaternion_from_axis_angle


class TransformWaterSurface(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                means: Tensor,
                quats: Tensor,
                scales: Tensor,
                camtoworld: Tensor = None, # (4, 4) matrix
                n: float = 1.33, 
                plane: float = 0,
                
                method_solve_quartic: str = "newton", # "newton" or "ferrari"
                newton_iters: int = 10,
                newton_tol: float = 1e-6,
                delta_numerical_jacobian: float = 1e-4,
                both_sides: bool = True,
                method_transform_quats: str = "dPa_dP", # "dPa_dP" or "ray_angle"
                method_transform_scales: str = "volume", # "volume" or "edges" or "ray_length"
                coeff_transform_scales: float = 1/3, # "1/2" or "1/3"
                scale_correct_space: str = "real",   # "real" or "log"
    ):
        
        device = means.device
                
        WS = WaterSurface(
            device=device,
            means=means,
            quats=quats,
            scales=scales,
            camtoworld=camtoworld,
            
            n=n,
            plane=plane,
            
            method_solve_quartic=method_solve_quartic, 
            newton_iters=newton_iters, 
            newton_tol=newton_tol,
            
            delta_numerical_jacobian=delta_numerical_jacobian,
            both_sides=both_sides,
            method_transform_quats=method_transform_quats,
            
            method_transform_scales=method_transform_scales,
            coeff_transform_scales=coeff_transform_scales,
            scale_correct_space=scale_correct_space,
        )
        
        t_means = WS.transform_means()  # Get the transformed means
        dPa_dP = WS.dPa_dP()  # Jacobian of means transformation # numerical or theoretical
        
        t_quats = WS.transform_quats(method="dPa_dP")  # dPa_dP, ray_angle
        t_scales = WS.transform_scales(method_transform_scales=method_transform_scales,  # Scale correction
                                       coeff_transform_scales=coeff_transform_scales,
                                       scale_correct_space=scale_correct_space)         
        
        dSa_dS = WS.dSa_dS()  # Jacobian of scales transformation
        
        ctx.save_for_backward(dPa_dP, dSa_dS)
        return t_means, t_quats, t_scales
    
    @staticmethod
    def backward(ctx, grad_means, grad_quats, grad_scales):
        dPa_dP, dSa_dS = ctx.saved_tensors
        
        # Calculate gradients with respect to means, quats, and scales
        # grad_means = torch.bmm(grad_means.unsqueeze(1), dPa_dP).squeeze(1) 
        grad_means = (grad_means.unsqueeze(1)* dPa_dP).sum(1) 
        # grad_scales = (grad_scales.unsqueeze(1) * dSa_dS).sum(1)
        
        return grad_means, grad_quats, grad_scales, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    

class WaterSurface():
    def __init__(self, 
                 device: str = "cuda",
                 means: torch.Tensor = None,
                 quats: torch.Tensor = None,
                 scales: torch.Tensor = None,
                 
                 camtoworld: torch.Tensor = None, # (4, 4) matrix
                 cam_center: torch.Tensor = None, # (3,) vector
                 K: torch.Tensor = None, # (3, 3) matrix
                 width: int = None,
                 height: int = None, 
                 
                 n: torch.Tensor = 1.33, 
                 plane: torch.Tensor = 0,
                 
                 # t_means
                 method_solve_quartic: str = "newton", # "newton" or "ferrari"
                 newton_iters: int = 10,
                 newton_tol: float = 1e-6,
                 
                 # dPa_dP
                 delta_numerical_jacobian: float = 1e-4,
                 both_sides: bool = True, # If True, calculate both plus and minus perturbations [TODO]
                 
                 # t_quats
                 method_transform_quats: str = "dPa_dP", # "dPa_dP" or "difference_ray_angle"
                 # t_scales
                 method_transform_scales: str = "edges", # "volume" or "edges" or "ray_length"
                 coeff_transform_scales: float = 1/2,    # "1/2" or "1/3"
                 scale_correct_space: str = "real", # "real" or "log"
                 
    ):
        self.n = torch.tensor(n, dtype=torch.float32, device=device, requires_grad=False)
        self.plane = torch.tensor(plane, dtype=torch.float32, device=device, requires_grad=False)
        self.device = device
        
        self.method_solve_quartic = method_solve_quartic.lower()
        self.newton_iters = newton_iters
        self.newton_tol = newton_tol
        self.delta_numerical_jacobian = delta_numerical_jacobian
        self.both_sides = both_sides
        self.method_transform_quats = method_transform_quats
        self.method_transform_scales = method_transform_scales
        self.method_coeff_transform_scales = coeff_transform_scales
        self.scale_correct_space = scale_correct_space
        
        self.camtoworld = camtoworld
        self.cam_center = self.camtoworld[:3, 3] if camtoworld is not None else cam_center
        self.K = K
        self.width = width
        self.height = height
        
        self.x0 = self.cam_center[0]
        self.y0 = self.cam_center[1]
        self.H = self.cam_center[2] - self.plane
        
        self.means = means
        self.quats = quats
        self.scales = scales
        # self.opacities = opacities
        self.num_g = means.shape[0]
        
        self.x = means[:, 0] - self.x0
        self.y = means[:, 1] - self.y0
        self.z = means[:, 2] - self.plane
        self.r = torch.sqrt(self.x**2 + self.y**2)
        self.phi = torch.atan2(self.y, self.x)
        
        self.n2 = self.n ** 2
        self.n2m1 = self.n2 - 1
        self.reci_n = 1 / self.n # reciprocal of n 逆数
        self.reci_n2 = 1 / self.n2 
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
    ):
        a4, a3, a2, a1, a0 = self.calculate_quartic_terms()  
        if self.method_solve_quartic == "newton":
            self.s = solve_quartic_newton(a4, a3, a2, a1, a0, self.r, num_iters=self.newton_iters, tol=self.newton_tol)
        elif self.method_solve_quartic == "ferrari":
            s = solve_quartic_ferrari(a4, a3, a2, a1, a0)
            # s is like roots = torch.stack([r1 - z0, r2 - z0, r3 - z0, r4 - z0], dim=-1) of dtype = torch.complex128
            # Extract real roots (imaginary part is close to zero)
            real_mask = torch.abs(s.imag) < 1e-5
            s_real = s.real.clone()
            s_real[~real_mask] = float('inf')  # Mark non-real roots as infinity
            # Find the smallest real root for each Gaussian
            self.s = torch.min(s_real, dim=-1).values
        else:
            raise ValueError(f"Unknown method for solving quartic equation: {self.method_solve_quartic}")
              
    def calc_theta(self,
    ):
        self.theta0 = torch.atan(self.s / self.H)
        self.theta1 = torch.atan((self.r - self.s) / (-self.z))
        self.d_theta = self.theta0 - self.theta1

    ### ------------------------------
    ###        Calcurate apparent position of Gaussian centers
    ### ------------------------------    
    def transform_means(self,
    ):
        """
        Calculate the apparent position of Gaussian centers.
        """
        if not hasattr(self, 's'):
            self.calc_intersection()
        if not hasattr(self, 'theta0'):
            self.calc_theta()

        # calculate the offset in the radial and vertical directions
        self.offset_r =  self.n2m1 * self.z * (torch.tan(self.theta1)**3)  # < 0
        self.ra = self.r + self.offset_r
        self.za = 1/self.n * self.z * (torch.cos(self.theta0) / torch.cos(self.theta1))**3         
        
        # apparent position in relative coordinates
        self.xa = self.ra * torch.cos(self.phi)
        self.ya = self.ra * torch.sin(self.phi)
        
        # apparent position in absolute coordinates
        new_x = self.x0 + self.xa
        new_y = self.y0 + self.ya
        new_z = self.plane + self.za
        self.new_means = torch.stack([new_x, new_y, new_z], dim=1)
        
        return self.new_means
        

    ### ------------------------------
    ###        Calcurate ROTATION corrected by quaternion
    ### ------------------------------
    
    def transform_quats(self,
                        method: str = "dPa_dP", # "dPa_dP" or "ray_angle"
                        quat_multiply_order: str = "d_q->quats" # "d_q->quats" or "quats->d_q"
    ):
        """
        Calculate the apparent quaternion based on the method specified.
        """
        self.quat_multiply_order = quat_multiply_order
        if method == "dPa_dP":
            return self.calc_apparent_quaternion_by_dPa_dP()
        elif method == "ray_angle":
            return self.calc_apparent_quaternion_by_ray_angle()
        else:
            raise ValueError(f"Unknown method for calculating apparent quaternion: {method}")
    
    
    def calc_apparent_quaternion_by_dPa_dP(self,
    ):
        """
        Calculate the apparent quaternion based on dPa/dP.
        """
        if getattr(self, 'jacobian', None) is None:
            _ = self.dPa_dP()  # Ensure jacobian is computed before calculating new quaternions

        d_xa = self.jacobian[:, :, 0]  # ∂x'/∂x, ∂y'/∂x, ∂z'/∂x
        d_ya = self.jacobian[:, :, 1]

        # Calculate radius and vertical components of the apparent direction                
        r_safe = torch.where(self.r < 1e-6, 1e-6, self.r)
        dPa_dr = d_xa * (self.x / r_safe).unsqueeze(1) + d_ya * (self.y / r_safe).unsqueeze(1)
        
        d_vertical = dPa_dr[:, 2]  # Vertical component of the apparent direction
        d_horizontal = torch.sqrt(dPa_dr[:, 0]**2 + dPa_dr[:, 1]**2)  # Horizontal component of the apparent direction
        
        theta = torch.atan2(d_vertical, d_horizontal)  # Calculate the angle of rotation
        z_axis = torch.tensor([0, 0, 1], device=self.device, dtype=self.x.dtype).expand(self.num_g, 3)  # Z-axis for rotation
        
        horizontal_axis = torch.stack(
            [self.x, self.y, torch.zeros_like(self.x)],
            dim=1
        )
        # numerical stability
        horizontal_axis *= 100.0
        
        # cross product to find the axis of rotation
        axis = torch.cross(horizontal_axis, z_axis, dim=1)
        
        # Calculate the quaternion from axis and angle
        d_q = quaternion_from_axis_angle(axis, theta) 
        
        # Multiply the original quaternion with the delta quaternion
        if self.quat_multiply_order == "d_q->quats":
            new_quats = GaussianTransformUtils.quat_multiply(d_q, self.quats)
        elif self.quat_multiply_order == "quats->d_q":
            new_quats = GaussianTransformUtils.quat_multiply(self.quats, d_q)
        else:
            new_quats = GaussianTransformUtils.quat_multiply(d_q, self.quats)
        return new_quats
    
    # [TODO]
    def calc_apparent_quaternion_by_ray_angle(self,
    ):
        """
        Calculate the apparent quaternion based on the angle of the ray.
        """
        raise NotImplementedError("Method 'calc_apparent_quaternion_by_ray_angle' is not implemented yet.")

        
    ### ------------------------------
    ###        Calcurate Jacobian of refractive transformation
    ### ------------------------------    
    def dPa_dP(self,
    ):
        self.jacobian = torch.zeros((self.num_g, 3, 3), device=self.device, dtype=self.x.dtype)
        for i in range(3):
            delta = torch.zeros((self.num_g, 3), device=self.device, dtype=self.x.dtype)
            delta[:, i] = self.delta_numerical_jacobian
            means_plus = self.means + delta
            means_minus = self.means - delta
            
            WS_plus = WaterSurface(
                means=means_plus,
                quats=self.quats,
                scales=self.scales,
                cam_center=self.cam_center,
                n=self.n,
                plane=self.plane,
                method_solve_quartic=self.method_solve_quartic,
                newton_iters=self.newton_iters,
                newton_tol=self.newton_tol,
                delta_numerical_jacobian=self.delta_numerical_jacobian
            )
            
            if self.both_sides: # ここで分岐があととめっゃゃ遅くなるかも、、、そんなことないか? [TODO]
                WS_minus = WaterSurface(
                    means=means_minus,
                    quats=self.quats,
                    scales=self.scales,
                    cam_center=self.cam_center,
                    n=self.n,
                    plane=self.plane,
                    method_solve_quartic=self.method_solve_quartic,
                    newton_iters=self.newton_iters,
                    newton_tol=self.newton_tol,
                    delta_numerical_jacobian=self.delta_numerical_jacobian
                )

                t_means_plus = WS_plus.transform_means()
                t_means_minus = WS_minus.transform_means()
                self.jacobian[:, :, i] = (t_means_plus - t_means_minus) / (2 * self.delta_numerical_jacobian)
                
            else:
                t_means_plus = WS_plus.transform_means()
                self.jacobian[:, :, i] = (t_means_plus - self.means) / self.delta_numerical_jacobian
            
        return self.jacobian
    
    
    
    
    
    
    ### ------------------------------
    ###        Calcurate SCALE corrected by quaternion
    ### ------------------------------
    def transform_scales(self,
                         method_transform_scales: str = None, # "volume" or "edges" or "ray_length"
                         coeff_transform_scales: float = None,   # "1/3" seems theoretical, "1/2" seems empirical
                         scale_correct_space: str = None # "real" or "log"
    ):
        """
        Calculate the apparent scales based on the method specified.
        """
        # if parameter is not specified, use the default value
        if method_transform_scales is None:
            method_transform_scales = self.method_transform_scales
        if coeff_transform_scales is None:
            coeff_transform_scales = self.method_coeff_transform_scales
        if scale_correct_space is None:
            scale_correct_space = self.scale_correct_space
        
        if method_transform_scales == "volume":
            self.volume_correction_factor = self.calc_spatial_compression_by_volume()
        elif method_transform_scales == "edges":
            self.volume_correction_factor = self.calc_spatial_compression_by_edges()
        elif method_transform_scales == "ray_length":
            raise NotImplementedError("Method 'ray_length' is not implemented yet.")
        else:
            raise ValueError(f"Unknown method for calculating apparent scales: {method_transform_scales}")  
        
        self.scale_correction_factor = torch.pow(self.volume_correction_factor, coeff_transform_scales)  # (N,)
        
        if scale_correct_space == "real":
            new_scales = self.scale_correction_factor.unsqueeze(-1) * self.scales  
        elif scale_correct_space == "log":
            logK = torch.log(self.scale_correction_factor).unsqueeze(-1) # (N, 1)
            new_scales = logK + self.scales 
        else:
            raise ValueError(f"Unknown space for scale correction: {scale_correct_space}")
        
        return new_scales
    
    def calc_spatial_compression_by_volume(self,
    ):
        """
        ヤコビアンを構成する3つのベクトルで構成される六面体の体積
        """
        # Ensure spatial compression have been computed
        if getattr(self, 'jacobian', None) is None:
            _ = self.dPa_dP()        
        d_xa = self.jacobian[:, :, 0] # ∂x'/∂x, ∂y'/∂x, ∂z'/∂x
        d_ya = self.jacobian[:, :, 1] # ∂x'/∂y, ∂y'/∂y, ∂z'/∂y
        d_za = self.jacobian[:, :, 2] # ∂x'/∂z, ∂y'/∂z, ∂z'/∂z
        
        # Calculate the volume of the parallelepiped formed by the three vectors
        cross_xy = torch.cross(d_xa, d_ya, dim=1)
        volume_correction_factor = torch.abs(torch.sum(cross_xy * d_za, dim=1))  # (N,)
        
        return volume_correction_factor
    
    def calc_spatial_compression_by_edges(self,
    ):
        """
        ヤコビアンを構成する3つのベクトルで構成される六面体の辺の長さの変化率の積
        """
        # Ensure spatial compression have been computed
        if getattr(self, 'jacobian', None) is None:
            _ = self.dPa_dP()       
            
        d_xa = self.jacobian[:, :, 0]
        d_ya = self.jacobian[:, :, 1]
        d_za = self.jacobian[:, :, 2]
               
        edge_x = torch.norm(d_xa, dim=1) # (N,)
        edge_y = torch.norm(d_ya, dim=1)
        edge_z = torch.norm(d_za, dim=1) 
        
        volume_correction_factor = edge_x * edge_y * edge_z # (N,)
             
        return volume_correction_factor
    
    def calc_spatial_compression_by_ray_length(self,
    ):
        """
        変化率 Ray Lenght between camera center and apparent position of Gaussian center
        """
        raise NotImplementedError("Method 'calc_spatial_compression_by_ray_length' is not implemented yet.")
        
    def dSa_dS(self,
    ):
        """
        Calculate the Jacobian of the scale transformation.
        """
        if getattr(self, 'scale_correction_factor', None) is None:
            _ = self.transform_scales()
        
        # (N, 3, 3) tensor, 
        # diagonal elements are the scale correction factor, and off-diagonal elements are 0
        if self.scale_correct_space == "real":
            dSa_dS = torch.diag_embed(self.scale_correction_factor.unsqueeze(-1).expand(-1, 3))
        elif self.scale_correct_space == "log":
            # dSa_dS = torch.diag_embed(((1.0/self.scale_correction_factor).unsqueeze(-1)).expand(-1, 3))
            dSa_dS = torch.diag_embed(self.scale_correction_factor.unsqueeze(-1).expand(-1, 3))
            # dSa_dS = torch.diag_embed(torch.ones(self.num_g, 3))
        
        return dSa_dS
        
    