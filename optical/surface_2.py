
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
                
                flag_transform_quats: bool = True,
                flag_transform_scales: bool = True, 
                
                method_solve_quartic: str = "newton", # "newton" or "ferrari"
                newton_iters: int = 10,
                newton_tol: float = 1e-6,
                delta_numerical_jacobian: float = 1e-4,
                both_sides: bool = True,
                method_transform_quats: str = "dPa_dP", # "dPa_dP" or "ray_angle"
                method_transform_scales: str = "edges", # "volume" or "edges" or "ray_length"
                coeff_transform_scales: float = 1/3, # "1/2" or "1/3"
                scale_correct_space: str = "log",   # "real" or "log" 
    ):
        
        device = means.device
        
        with torch.no_grad():
            WS = WaterSurface(
                device=device,
                means=means.detach().clone(),
                quats=quats.detach().clone(),
                scales=scales.detach().clone(),
                camtoworld=camtoworld.detach().clone(),
                
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
            # dPa_dP = WS.calc_dPa_dP_with_minibatches()  # Jacobian of means transformation # numerical or theoretical
            dPa_dP = WS.calc_dPa_dP()  # Jacobian of means transformation # numerical or theoretical
            
            if flag_transform_quats:
                t_quats = WS.transform_quats(method="dPa_dP")  # dPa_dP, ray_angle
            else: 
                t_quats = quats.clone()

            if flag_transform_scales:
                t_scales = WS.transform_scales(method_transform_scales=method_transform_scales,  # Scale correction
                                            coeff_transform_scales=coeff_transform_scales,
                                            scale_correct_space=scale_correct_space)         
                dSa_dS = WS.calc_dSa_dS()  # Jacobian of scales transformation
            else:
                t_scales = scales.clone()
                dSa_dS = None
                
        # 勾配計算用にテンソルを保存
        ctx.save_for_backward(dPa_dP, dSa_dS)
        
        # 出力テンソルの勾配追跡を設定
        t_means = t_means.clone().requires_grad_(means.requires_grad)
        if t_quats is not None:
            t_quats = t_quats.clone().requires_grad_(quats.requires_grad)
        if t_scales is not None:
            t_scales = t_scales.clone().requires_grad_(scales.requires_grad)
        
        # 変換フラグを保存（後方伝播で使用）
        ctx.flag_transform_quats = flag_transform_quats
        ctx.flag_transform_scales = flag_transform_scales
        
        return t_means, t_quats, t_scales
    
    @staticmethod
    def backward(ctx, grad_means, grad_quats, grad_scales):
        # 保存したテンソルを取得
        dPa_dP, dSa_dS = ctx.saved_tensors
        
        # 入力勾配の初期化
        grad_means_input = None
        grad_quats_input = None
        grad_scales_input = None
        
        # 平均の勾配計算
        if grad_means is not None:
            # [N,1,3] @ [N,3,3] = [N,1,3] → [N,3]
            grad_means_input = (grad_means.unsqueeze(1) @ dPa_dP).squeeze(1)
        
        # クォータニオンの勾配計算
        if grad_quats is not None and ctx.flag_transform_quats:
            # クォータニオン変換のヤコビアンは未実装のため、単位行列を仮定
            grad_quats_input = grad_quats
        elif grad_quats is not None:
            # 変換しない場合はそのまま
            grad_quats_input = grad_quats
        
        # スケールの勾配計算
        if grad_scales is not None and ctx.flag_transform_scales and dSa_dS is not None:
            # [N,1,3] @ [N,3,3] = [N,1,3] → [N,3]
            grad_scales_input = (grad_scales.unsqueeze(1) @ dSa_dS).squeeze(1)
        elif grad_scales is not None:
            # 変換しない場合はそのまま
            grad_scales_input = grad_scales
            
        return grad_means_input, grad_quats_input, grad_scales_input, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    

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
        
        self.camtoworld = camtoworld.detach().clone() if camtoworld is not None else None
        if camtoworld is not None:
            self.cam_center = camtoworld[:3, 3].squeeze()
        elif cam_center is not None:
            self.cam_center = cam_center.detach().clone()
        else:
            raise ValueError("Either camtoworld or cam_center must be provided.")
        
        self.x0 = self.cam_center[0]
        self.y0 = self.cam_center[1]
        self.H = self.cam_center[2] - self.plane
        
        self.means = means.detach().clone() if means is not None else None
        self.quats = quats.detach().clone() if quats is not None else None
        self.scales = scales.detach().clone() if scales is not None else None
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
        with torch.no_grad():
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
        with torch.no_grad():
            self.theta0 = torch.atan(self.s / self.H)
            self.theta1 = torch.atan((self.r - self.s) / (-self.z))
            self.d_theta = self.theta0 - self.theta1
            
    def calc_ray_length(self,
    ):
        """
        Calculate the ray length from camera center to Gaussian center.
        """
        with torch.no_grad():
            if not hasattr(self, 's'):
                self.calc_intersection()
            if not hasattr(self, 'theta0'):
                self.calc_theta()
                
            # ray length cam to intersection point
            self.ray_length_cam_to_intersec = torch.sqrt(self.s**2 + self.H**2)
            
            # ray length cam to apparent position of Gaussian center
            self.ray_Length_cam_to_app = torch.sqrt((self.r + self.offset_r)**2 + \
                                                  (self.H - self.za)**2)
            
            # ray length intersection point to real Gaussian center
            self.ray_length_intersec_to_gaussian = torch.sqrt(self.za**2 + \
                                                        (self.r - self.s)**2)


    ### ------------------------------
    ###        Calcurate apparent position of Gaussian centers
    ### ------------------------------    
    def transform_means(self,
    ):
        """
        Calculate the apparent position of Gaussian centers.
        """
        with torch.no_grad():
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
        with torch.no_grad():
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
        with torch.no_grad():
            if getattr(self, 'jacobian', None) is None:
                _ = self.calc_dPa_dP()  # Ensure jacobian is computed before calculating new quaternions

            d_xa = self.dPa_dP[:, :, 0]  # ∂x'/∂x, ∂y'/∂x, ∂z'/∂x
            d_ya = self.dPa_dP[:, :, 1]

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
    def calc_dPa_dP(self):
        self.dPa_dP = torch.zeros((self.num_g, 3, 3), device=self.device, dtype=self.x.dtype)
        
        # 元のmeansを保存（勾配情報を保持）
        original_means = self.means
        
        for i in range(3):
            # 勾配追跡なしで差分計算
            with torch.no_grad():
                delta = torch.zeros((self.num_g, 3), device=self.device, dtype=self.x.dtype)
                delta[:, i] = self.delta_numerical_jacobian
                
                # 差分位置を計算（勾配追跡なし）
                means_plus = original_means + delta
                means_minus = original_means - delta
            
            # WaterSurfaceインスタンス作成と変換は勾配追跡なしで実行
            with torch.no_grad():
                # プラス側の計算
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
                t_means_plus = WS_plus.transform_means()
                
                # マイナス側の計算（both_sidesがTrueの場合）
                if self.both_sides:
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
                    t_means_minus = WS_minus.transform_means()
                    self.dPa_dP[:, :, i] = (t_means_plus - t_means_minus) / (2 * self.delta_numerical_jacobian)
                else:
                    self.dPa_dP[:, :, i] = (t_means_plus - self.means) / self.delta_numerical_jacobian
        
        # 元のmeansを復元
        self.means = original_means
        return self.dPa_dP
    
    def _batch_transform_means(self, means_batch):
        """バッチ処理された点群に対する見かけの位置を計算（メモリ最適化版）"""
        # カメラパラメータ
        x0 = self.x0
        y0 = self.y0
        plane = self.plane
        H = self.H
        n = self.n
        
        # 相対座標計算 (メモリ効率のためビューを使用)
        x = means_batch[:, 0] - x0
        y = means_batch[:, 1] - y0
        z = means_batch[:, 2] - plane
        
        # 半径と角度 (インプレース操作)
        r = torch.sqrt(x.square_() + y.square_())
        phi = torch.atan2(y, x)
        
        # 屈折率関連の定数
        n2 = n.square()
        n2m1 = n2 - 1
        H2 = H.square()
        
        # 4次方程式の係数をバッチ計算 (メモリ効率のため中間変数再利用)
        r2 = r.square()
        z2 = z.square()
        
        a4 = -n2m1
        a3 = 2 * n2m1 * r
        a2 = -n2m1 * r2 + z2 - n2 * H2
        a1 = 2 * n2 * r * H2
        a0 = -n2 * H2 * r2
        
        # 4次方程式を解く
        if self.method_solve_quartic == "newton":
            s = solve_quartic_newton(a4, a3, a2, a1, a0, r, 
                                    num_iters=self.newton_iters, 
                                    tol=self.newton_tol)
        else:  # ferrari
            s_complex = solve_quartic_ferrari(a4, a3, a2, a1, a0)
            real_mask = torch.abs(s_complex.imag) < 1e-5
            s_real = s_complex.real.clone()
            s_real[~real_mask] = float('inf')
            s = torch.min(s_real, dim=-1).values
        
        # # CHECK: s is satisfied as intersection point
        # # → SATISFIED !
        # print(f"check s is satisfied:")
        # mask = (s >= 0) & (s <= r)
        # if not mask.all():
        #     print("Warning: Some s values are out of bounds (s < 0 or s > r).")
        #     print(f"ratio {mask.sum().item() / mask.numel():.2f} of s are valid.")
        # else:
        #     print("All s values are valid (0 <= s <= r).")       
        
        # 角度計算 (ゼロ除算防止)
        safe_H = torch.where(H > 0, H, 1e-6)
        safe_z = torch.where(z < 0, z, -1e-6)  # zは負であるべき
        
        theta0 = torch.atan(s / safe_H)
        theta1 = torch.atan((r - s) / (-safe_z))
        
        # 見かけの位置計算 (メモリ効率のため中間変数最小化)
        tan_theta1 = torch.tan(theta1)
        tan3_theta1 = tan_theta1.square() * tan_theta1  # tan^3(θ1)
        
        offset_r = n2m1 * z * tan3_theta1
        ra = r + offset_r
        
        cos_ratio = torch.cos(theta0) / torch.cos(theta1)
        za = (1/n) * z * cos_ratio.pow(3)
        
        # 直交座標変換
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        xa = ra * cos_phi
        ya = ra * sin_phi
        
        # 絶対座標に戻す (インプレース加算)
        new_x = x0 + xa
        new_y = y0 + ya
        new_z = plane + za
        
        return torch.stack([new_x, new_y, new_z], dim=1)    

    def calc_dPa_dP_with_minibatches(self, batch_size=5000):
        n_gaussians = self.num_g
        # Ensure self.dPa_dP is initialized for the full set of Gaussians
        self.dPa_dP = torch.empty((n_gaussians, 3, 3), device=self.device, dtype=self.means.dtype)
        
        if n_gaussians == 0:
            return self.dPa_dP
        
        for start in range(0, n_gaussians, batch_size):
            end = min(start + batch_size, n_gaussians)
            if start == end:
                continue
            
            batch_indices = slice(start, end)
            current_means_batch = self.means[batch_indices]
            
            # Calculate Jacobian for the current batch
            batch_dPa_dP_values = self._calculate_jacobian_for_batch(current_means_batch)
            
            # 結果を格納
            self.dPa_dP[batch_indices] = batch_dPa_dP_values
            
            del current_means_batch, batch_dPa_dP_values
        
        return self.dPa_dP

    def _calculate_jacobian_for_batch(self, means_batch: Tensor) -> Tensor:
        """
        Calculates the Jacobian d(P_apparent)/d(P_real) for a given batch of means.
        P_real are the input means_batch.
        P_apparent are the transformed means.
        The Jacobian is computed using numerical differentiation (central differences).
        """
        n_gaussians_in_batch = means_batch.shape[0]
        delta_perturb = self.delta_numerical_jacobian # Use the configured delta
        device = means_batch.device # Use device of input batch
        dtype = means_batch.dtype

        jacobian_batch = torch.empty((n_gaussians_in_batch, 3, 3), device=device, dtype=dtype)
        
        # Perform calculations without gradient tracking for numerical differentiation
        with torch.no_grad():
            for i in range(3): # Iterate over x, y, z dimensions for perturbation
                # Perturbation vector for the i-th dimension
                perturb_vector = torch.zeros_like(means_batch)
                perturb_vector[:, i] = delta_perturb

                means_plus = means_batch + perturb_vector
                means_minus = means_batch - perturb_vector
                
                # Transform the perturbed means
                # _batch_transform_means calculates apparent positions for a batch of real positions
                t_means_plus = self._batch_transform_means(means_plus)
                t_means_minus = self._batch_transform_means(means_minus)
                
                # Central difference formula for the i-th column of the Jacobian
                jacobian_batch[:, :, i] = (t_means_plus - t_means_minus) / (2 * delta_perturb)
                
                # Explicitly delete intermediate tensors if memory is a concern, but avoid empty_cache in tight loops.
                del means_plus, means_minus, t_means_plus, t_means_minus, perturb_vector
        
        return jacobian_batch
    
    
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
        with torch.no_grad():
            # if parameter is not specified, use the default value
            if method_transform_scales is None:
                method_transform_scales = self.method_transform_scales
            if coeff_transform_scales is None:
                coeff_transform_scales = self.method_coeff_transform_scales
            if scale_correct_space is None:
                scale_correct_space = self.scale_correct_space
            
            if method_transform_scales == "volume":
                self.volume_correction_factor = self.calc_spatial_compression_by_volume()
                self.scale_correction_factor = torch.pow(self.volume_correction_factor, coeff_transform_scales)  # (N,)
            elif method_transform_scales == "edges":
                self.volume_correction_factor = self.calc_spatial_compression_by_edges()
                self.scale_correction_factor = torch.pow(self.volume_correction_factor, coeff_transform_scales)  # (N,)
            elif method_transform_scales == "ray_length":
                self.scale_correction_factor = self.calc_spatial_compression_by_ray_length()
            else:
                raise ValueError(f"Unknown method for calculating apparent scales: {method_transform_scales}")  
            
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
        with torch.no_grad():
            # Ensure spatial compression have been computed
            if getattr(self, 'jacobian', None) is None:
                _ = self.calc_dPa_dP()        
            d_xa = self.dPa_dP[:, :, 0] # ∂x'/∂x, ∂y'/∂x, ∂z'/∂x
            d_ya = self.dPa_dP[:, :, 1] # ∂x'/∂y, ∂y'/∂y, ∂z'/∂y
            d_za = self.dPa_dP[:, :, 2] # ∂x'/∂z, ∂y'/∂z, ∂z'/∂z
            
            # Calculate the volume of the parallelepiped formed by the three vectors
            cross_xy = torch.cross(d_xa, d_ya, dim=1)
            volume_correction_factor = torch.abs(torch.sum(cross_xy * d_za, dim=1))  # (N,)
            
            return volume_correction_factor
    
    def calc_spatial_compression_by_edges(self,
    ):
        """
        ヤコビアンを構成する3つのベクトルで構成される六面体の辺の長さの変化率の積
        """
        with torch.no_grad():
            # Ensure spatial compression have been computed
            if getattr(self, 'jacobian', None) is None:
                _ = self.calc_dPa_dP()       
                
            d_xa = self.dPa_dP[:, :, 0]
            d_ya = self.dPa_dP[:, :, 1]
            d_za = self.dPa_dP[:, :, 2]
                
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
        with torch.no_grad():
            if not hasattr(self, 'ray_length_cam_to_app'):
                self.calc_ray_length()
            
            # Calculate the ratio of the ray length from camera center to apparent position of Gaussian center
            scale_correction_factor = self.ray_Length_cam_to_app / (self.ray_length_cam_to_intersec + self.ray_length_intersec_to_gaussian)
            
            return scale_correction_factor
        
    def calc_dSa_dS(self,
    ):
        """
        Calculate the Jacobian of the scale transformation.
        """
        with torch.no_grad():
            if getattr(self, 'scale_correction_factor', None) is None:
                _ = self.transform_scales()
            
            # (N, 3, 3) tensor, 
            # diagonal elements are the scale correction factor, and off-diagonal elements are 0
            if self.scale_correct_space == "real":
                dSa_dS = torch.diag_embed(self.scale_correction_factor.unsqueeze(-1).expand(-1, 3))
            elif self.scale_correct_space == "log":
                # dSa_dS = torch.diag_embed(((1.0/self.scale_correction_factor).unsqueeze(-1)).expand(-1, 3))
                # dSa_dS = torch.diag_embed(self.scale_correction_factor.unsqueeze(-1).expand(-1, 3))
                dSa_dS = torch.diag_embed(torch.ones(self.num_g, 3, device=self.device, dtype=self.x.dtype))
            
            return dSa_dS
        
    