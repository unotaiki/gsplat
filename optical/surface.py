import torch
from torch import Tensor
from optical.quartic_solver.feralli import solve_quartic_ferrari
from optical.quartic_solver.newton_method import solve_quartic_newton

from internal.utils.gaussian_utils import GaussianTransformUtils
from optical.utils.rotation_utils import quat_from_2dirs

# environment map
import imageio
import numpy as np
import torch.nn.functional as F

class WaterSurface():
    def __init__(self, 
                 device: str = "cuda",
                 means: torch.Tensor = None,
                 quats: torch.Tensor = None,
                 scales: torch.Tensor = None,
                 opacities: torch.Tensor = None,
                 
                 camtoworld: torch.Tensor = None, # (4, 4) matrix
                 cam_center: torch.Tensor = None,
                 K: torch.Tensor = None, # (3, 3) matrix
                 width: int = None,
                 height: int = None, 
                 
                 n: torch.Tensor = 1.33, 
                 plane: torch.Tensor = 0,
                 flag_solve_quartic_by_newton: bool = True,
                 newton_iters: int = 10,
                 newton_tol: float = 1e-6,
                 numercial_jacobians_delta: float = 1e-4,
                 
    ):
        self.n = torch.tensor(n, dtype=torch.float32, device=device, requires_grad=False)
        self.plane = torch.tensor(plane, dtype=torch.float32, device=device, requires_grad=False)
        self.device = device
        self.flag_solve_quartic_by_newton = flag_solve_quartic_by_newton
        self.newton_iters = newton_iters
        self.newton_tol = newton_tol
        self.numerical_jacobians_delta = numercial_jacobians_delta
        
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
        self.opacities = opacities
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
        if self.flag_solve_quartic_by_newton:
            self.s = solve_quartic_newton(a4, a3, a2, a1, a0, self.r, num_iters=self.newton_iters, tol=self.newton_tol)
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
        self.xs = self.s * torch.cos(self.phi)
        self.ys = self.s * torch.sin(self.phi)
        mH_vec = torch.full_like(self.xs, -self.H, device=self.device, dtype=self.xs.dtype)
        
        # compute Ray direction from camera center to intersection point
        # this means the direction from camera center to the apparent position of the Gaussian
        self.ray_cam2intersec = torch.stack(
            [self.xs, 
             self.ys, 
             mH_vec], dim=1
        )
        self.unit_dir_to_apparent = self.ray_cam2intersec / torch.norm(self.ray_cam2intersec, dim=1, keepdim=True).clamp(min=1e-8)
        
        # compute Ray direction from intersection point to real Gaussian center
        self.ray_intersec2gaussian = torch.stack(
            [self.x - self.xs, 
             self.y - self.ys, 
             self.z], dim=1
        )
        self.unit_dir_intersec2gaussian = self.ray_intersec2gaussian / torch.norm(self.ray_intersec2gaussian, dim=1, keepdim=True).clamp(min=1e-8)
        
    def calc_theta(self,
    ):
        self.theta0 = torch.atan(self.s / self.H)
        self.theta1 = torch.atan((self.r - self.s) / (-self.z))
        self.d_theta = self.theta0 - self.theta1
        
    def calc_ray_length(self,
    ):
        self.len_cam2intersec = torch.norm(self.ray_cam2intersec, dim=1)
        self.len_intersec2gaussian = torch.norm(self.ray_intersec2gaussian, dim=1)
        
        self.ray_intersec2apparent = torch.stack(
            [self.x_app - self.xs, 
             self.y_app - self.ys, 
             self.z_app ], dim=1
        )
        self.len_intersec2apparent = torch.norm(self.ray_intersec2apparent, dim=1)

    ### ------------------------------
    ###        Calcurate apparent position of Gaussian centers
    ### ------------------------------    
    def calc_appearance_position(self,
    ):
        self.offset_r =  self.n2m1 * self.z * (torch.tan(self.theta1)**3)    
        self.ra = self.r + self.offset_r
        self.z_app = 1/self.n * self.z * (torch.cos(self.theta0) / torch.cos(self.theta1))**3 
            
    
    def transform_to_appearance(self,
    ):        
        self.calc_intersection()
        self.calc_theta()
        self.calc_appearance_position()
        
        # 相対座標
        self.x_app = self.ra * torch.cos(self.phi)
        self.y_app = self.ra * torch.sin(self.phi)
    
        # 見かけの位置に座標変換 (絶対座標)
        new_x = self.x0 + self.x_app    
        new_y = self.y0 + self.y_app    
        new_z = self.plane + self.z_app
        
        self.new_means = torch.stack([new_x, new_y, new_z], dim=1)
        self.new_quats = self.calc_apparent_quaternion()
        
        return self.new_means, self.new_quats


    ### ------------------------------
    ###        Calcurate ROTATION corrected by quaternion
    ### ------------------------------
    def calc_apparent_quaternion(self,
    ):
        d_q = quat_from_2dirs(self.unit_dir_intersec2gaussian, self.unit_dir_to_apparent) 
        new_quats = GaussianTransformUtils.quat_multiply(self.quats, d_q)
        return new_quats
    
    ### ------------------------------
    ###        Calcurate Jacobian of refractive transformation
    ### ------------------------------    
    def dPa_dP(self,
    ):
        self.jacobian = torch.zeros((self.num_g, 3, 3), device=self.device, dtype=self.x.dtype)
        for i in range(3):
            delta = torch.zeros((self.num_g, 3), device=self.device, dtype=self.x.dtype)
            delta[:, i] = self.numerical_jacobians_delta
            means_plus = self.means + delta
            means_minus = self.means - delta
            WS_plus = WaterSurface(
                means=means_plus,
                quats=self.quats,
                scales=self.scales,
                opacities=self.opacities,
                cam_center=self.cam_center,
                n=self.n,
                plane=self.plane,
                flag_solve_quartic_by_newton=self.flag_solve_quartic_by_newton,
                newton_iters=self.newton_iters,
                newton_tol=self.newton_tol,
                numercial_jacobians_delta=self.numerical_jacobians_delta
            )
            WS_minus = WaterSurface(
                means=means_minus,
                quats=self.quats,
                scales=self.scales,
                opacities=self.opacities,
                cam_center=self.cam_center,
                n=self.n,
                plane=self.plane,
                flag_solve_quartic_by_newton=self.flag_solve_quartic_by_newton,
                newton_iters=self.newton_iters,
                newton_tol=self.newton_tol,
                numercial_jacobians_delta=self.numerical_jacobians_delta
            )
            t_means_plus, _ = WS_plus.transform_to_appearance()
            t_means_minus, _ = WS_minus.transform_to_appearance()
            self.jacobian[:, :, i] = (t_means_plus - t_means_minus) / (2 * self.numerical_jacobians_delta)
            
        return self.jacobian
    
    def calc_spatial_compression_by_volume(self,
    ):
        """
        ヤコビアンを構成する3つのベクトルで構成される六面体の体積
        """
        # Ensure spatial compression have been computed
        if getattr(self, 'jacobian', None) is None:
            _ = self.dPa_dP()        
        d_xa = self.jacobian[:, 0, :]
        d_ya = self.jacobian[:, 1, :]
        d_za = self.jacobian[:, 2, :]
        
        cross_xy = torch.cross(d_xa, d_ya, dim=1)
        self.volume_compression_ratio = torch.abs(torch.sum(d_za * cross_xy, dim=1)) # (N,)    
    
    def calc_spatial_compression_by_edges(self,
    ):
        """
        ヤコビアンを構成する3つのベクトルで構成される六面体の辺の長さの変化率の積
        """
        # Ensure spatial compression have been computed
        if getattr(self, 'jacobian', None) is None:
            _ = self.dPa_dP()       
             
        d_xa = self.jacobian[:, 0, :]
        d_ya = self.jacobian[:, 1, :]
        d_za = self.jacobian[:, 2, :]
        
        edge_x = torch.norm(d_xa, dim=1) # (N,)
        edge_y = torch.norm(d_ya, dim=1)
        edge_z = torch.norm(d_za, dim=1) 
        
        self.edge_compression_ratio = edge_x * edge_y * edge_z # (N,)
             
        
    ### ------------------------------
    ###        Calcurate SCALE corrected by quaternion
    ### ------------------------------
    
    def scale_correction_as_log(self,
    ):
        # Ensure spatial compression have been computed
        if getattr(self, 'jacobian', None) is None:
            _ = self.dPa_dP()
        if getattr(self, 'volume_compression_ratio', None) is None:
            self.calc_spatial_compression_by_volume()
        
        # calculate scale correction factor from volume compression rario
        self.scale_correction_factor_by_volume = self.volume_compression_ratio**(1/3) # (N,)
        logK = torch.log(self.scale_correction_factor_by_volume).unsqueeze(-1) # (N, 1)
        self.new_scales = logK + self.scales 
        return self.new_scales
    
    def scale_correction_as_real(self,
                                 comp_by: str = "volume" # "volume" or "edges"
    ):
        # Ensure spatial compression have been computed
        if getattr(self, 'jacobian', None) is None:
            _ = self.dPa_dP()
            
        if comp_by == "volume":
            if getattr(self, 'volume_compression_ratio', None) is None:
                self.calc_spatial_compression_by_volume()

            # calculate scale correction factor from volume compression rario
            self.scale_correction_factor_by_volume = self.volume_compression_ratio**(1/3) # (N,)
            # self.scale_correction_factor = self.volume_compression_ratio**1 # (N,)
            K = self.scale_correction_factor_by_volume.unsqueeze(-1) # (N, 1)
            self.new_scales = K * self.scales 
            return self.new_scales
        
        elif comp_by == "edges":
            if getattr(self, 'edge_compression_ratio', None) is None:
                self.calc_spatial_compression_by_edges()

            # calculate scale correction factor from edge compression ratio
            self.scale_correction_factor_by_edges = self.edge_compression_ratio**(1/3)
            K = self.scale_correction_factor_by_edges.unsqueeze(-1) # (N, 1)
            self.new_scales = K * self.scales
            return self.new_scales
        else:
            raise ValueError("comp_by must be 'volume' or 'edges'.")
        
    
    # Rayの距離による補間 ← 3D空間が歪むことが原因のため、不適
    # def scale_correction(self,
    # ):
    #     # Ensure ray lengths have been computed
    #     if getattr(self, 'len_cam2intersec', None) is None:
    #         self.calc_ray_length()
    #     self.scale_correction_factor = \
    #         (self.len_cam2intersec + self.len_intersec2apparent) / (self.len_cam2intersec + self.len_intersec2gaussian).clamp(min=1e-4) # (N,)
    #     logK = torch.log(self.scale_correction_factor).unsqueeze(-1) # (N, 1)
    #     self.new_scales = logK + self.scales 
    #     return self.new_scales
        
    
    # ### ------------------------------
    # ###        Calcurate OPACITY corrected by quaternion
    # ### ------------------------------
    # def opacity_correction(self,
    # ):
    #     # Ensure scale correction factor have been computed
    #     if getattr(self, 'scale_correction_factor', None) is None:
    #         self.scale_correction()
    #     volume_ratio = self.scale_correction_factor 
    #     opacities_abs = torch.sigmoid(self.opacities) # parameter -> real opacity
    #     new_opacities_abs = (opacities_abs / volume_ratio).clamp(min=1e-4, max=1-1e-4) # (N,)
    #     self.new_opacities = torch.logit(new_opacities_abs)
    #     return self.new_opacities
    
    ### -----------------------------------
    ###        Calcurate ray of each pixel
    ### -----------------------------------
    def calc_ray_of_each_pixel(self, 
    ):
        
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
    
        # (H, W) indexing
        v, u = torch.meshgrid(
            torch.arange(0, self.height, device=self.device),
            torch.arange(0, self.width, device=self.device),
            indexing='ij'
        )
    
        self.rays = torch.stack([
            (u - cx) / fx,
            (v - cy) / fy,
            torch.ones_like(u) 
        ], dim=-1)  # (H, W, 3)
    
        # Rotate ray directions to world coordinates
        self.rays = torch.einsum('ij,hwj->hwi', self.camtoworld[:3, :3], self.rays) # (3, 3) @ (H, W, 3) -> (H, W, 3)
        self.rays = self.rays / torch.norm(self.rays, dim=-1, keepdim=True)
    
        # Expand camera origin
        self.ray_origins = self.cam_center.view(1, 1, 3).expand(self.height, self.width, 3)
        
        return self.rays
    
    def calc_incidence_angle_of_rays(self,
    ):
        self.incidence_angle = torch.acos(torch.clamp(self.rays[:, :, 2], -1, 1))  # (H, W) tensor with angle in radians
        
    
    ### -----------------------------------
    ###        Environment map
    ### -----------------------------------
    def load_environment_map(self, 
                             envmap_path: str = None,
    ):
        """
        Load environment map for rendering.
        """
        format = envmap_path.split('.')[-1].lower() # lower() -> 小文字に
        env_np = imageio.imread(envmap_path, format=format)
        
        if env_np.dtype == np.uint8:
            # 8-bit LDR image: map [0,255] → [0,1]
            env_np = env_np.astype(np.float32) / 255.0 
        else:
            # exposure/gamma header you want to apply
            exposure = 1.0
            gammna = 2.2
            
            hdr_exp = env_np * (2.0**exposure )
            tonemap = hdr_exp ** (1.0 / gammna)
            env_np = tonemap.astype(np.float32)
                        
        
        self.env = torch.from_numpy(env_np).permute(2,0,1).unsqueeze(0).to(self.device) # (1, C, H, W)
    
    def get_colors_from_envmap(self, 
                               rays: torch.Tensor = None,
    ):
        """
        Get colors from environment map.
        """
        if not hasattr(self, 'env'):
            raise ValueError("Environment map is not loaded. Please load it using `load_environment_map` method.")
        
        x, y, z = rays.unbind(dim=-1)
        phi = torch.atan2(y, x)                           # [-pi, pi] # longitude
        self.incidence_angle = torch.acos(torch.clamp(z, -1,1))          # [0, pi] # latitude
        u = phi / (2*torch.pi) + 0.5                      # [0, 1]4
        u = u % 1.0                                       # [0, 1]
        v = self.incidence_angle / torch.pi               # [0, 1]
        grid = torch.stack([2*u-1, 2*v-1], dim=-1)        # [−1,1]^2

        N = self.width * self.height
        grid = grid.view(1, N, 1, 2)
        
        C = self.env.shape[1]  # Number of channels in the environment map
        sampled = F.grid_sample(self.env, grid, align_corners=True, mode="bilinear")
        self.env_colors = sampled.view(C, N).permute(1,0).reshape(self.height, self.width, C)  # (H, W, C)
        
        return self.env_colors[:,:,:3] if C == 4 else self.env_colors # (N, 3) tensor with RGB colors
    
    
    ### -----------------------------------
    ###        Refrection model
    ### -----------------------------------
    
    def calc_refrected_ray(self, 
    ):
        """
        Calculate the refracted ray direction.
        """
        self.refrected_rays = self.rays.clone()
        self.refrected_rays[:, :, 2] = - self.rays[:, :, 2]
        return self.refrected_rays
    
    # def calc_refraction_angle_of_each_pixel(self,
    # ):
    #     """
    #     Calculate the refraction angle of each pixel.
    #     """
    #     self.refraction_angle = torch.asin(torch.clamp(torch.sin(self.incidence_angle) / self.n, -1, 1))
        
    def schlick_fresnel_reflectance(self, 
    ):
        """
        Calculate Fresnel reflectance using Schlick's approximation.
        https://www.optics-words.com/kogaku_kiso/Frenel-equations.html
        """
        # specular reflectance, when the angle of incidence is 0
        self.spec_refle_ratio = ((self.n - 1) / (self.n + 1)) ** 2
        self.cos_theta_i = torch.clamp(-self.rays[:,:,2], -1, 1)  # cos(theta_i) for incidence angle
        print(f"cos_theta_i:\n {self.cos_theta_i}")
        self.spec_schlick = self.spec_refle_ratio + (1 - self.spec_refle_ratio) * (1 - self.cos_theta_i) ** 5
        self.spec_schlick = torch.clamp(self.spec_schlick, min=0, max=1).unsqueeze(-1)  # (H, W, 1)
        self.trans_schlick = 1 - self.spec_schlick
        
        return self.spec_refle_ratio, self.spec_schlick, self.trans_schlick
        
    def fresnel_reflectance(self, 
    ):
        """
        Calculate Fresnel reflectance using the Fresnel equations.
        http://marupeke296.com/DXPS_PS_No7_FresnelReflection.html
        """
        A = self.reci_n
        B = self.cos_theta_i = torch.clamp(-self.rays[:, :, 2], -1, 1)  # cos(theta_i) for incidence angle
        C = torch.sqrt(1 - (self.reci_n2 * (1 - self.cos_theta_i ** 2))) 
        
        Rs = ((A*B - C) / (A*B + C))**2
        Rp = ((A*C - B) / (A*C + B))**2
        self.spec_ave = (Rs + Rp) / 2.0
        self.spec_ave = torch.clamp(self.spec_ave, min=0, max=1).unsqueeze(-1)  # (H, W, 1)
        
        self.trans_ave = 1 - self.spec_ave
        
        return self.spec_ave, self.trans_ave

    
    
    def fresnel(self,
    ):
        n_i = self.n
        n_t = 1.0
        
        self.cos_theta_air = torch.clamp(-self.rays[:, :, 2], -1, 1)
        print(f"cos_theta_air:\n {self.cos_theta_air}")
        theta_air = torch.acos(self.cos_theta_air)  # angle in radians
        print(f"theta_air:\n {theta_air}")
        sin_theta_t = torch.sin(theta_air) * n_i / n_t  # if this > 1, the ray must be
        print(f"sin_theta_t:\n {sin_theta_t}")
        cos_theta_t = torch.sqrt(torch.clip(1 - sin_theta_t**2, 0, 1))
        
        rs = ((n_t*self.cos_theta_air - n_i*cos_theta_t)/(n_t*self.cos_theta_air + n_i*cos_theta_t))**2
        rp = ((n_i*self.cos_theta_air - n_t*cos_theta_t)/(n_i*self.cos_theta_air + n_t*cos_theta_t))**2
        reflectance = (rs + rp) / 2       
        
        self.spec = torch.where(sin_theta_t > 1, 1.0, reflectance)  # Use Rs for incidence and Rp for transmission
        self.spec = torch.clamp(self.spec, min=0, max=1).unsqueeze(-1)  # (H, W, 1)
        self.trans = 1.0 - self.spec
        return self.spec, self.trans
        

        
        