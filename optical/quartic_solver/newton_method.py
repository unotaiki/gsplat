import torch


def solve_quartic_newton(
    a4: torch.Tensor,
    a3: torch.Tensor,
    a2: torch.Tensor,
    a1: torch.Tensor,
    a0: torch.Tensor,
    r: torch.Tensor, # for initial guess
    num_iters: int = 10, 
    tol: float = 1e-2) -> torch.Tensor:
    """
    各サンプルについて、Newton法により s に関する4次方程式の根を解く関数
    4次方程式は次の係数で表される:
      a4 = 1 - n^2  
      a3 = 2*(n^2 - 1)*r  
      a2 = (1 - n2)*r2 + h2 - n2 * H2
      a1 = 2*n^2*r*mH^2  
      a0 = - n^2 * H^2 * r^2  
    """

    eps = 1e-6
    s = r / 1.7 + eps  # 初期値

    for _ in range(num_iters):
        f = a4 * s**4 + a3 * s**3 + a2 * s**2 + a1 * s + a0
        f_prime = 4 * a4 * s**3 + 3 * a3 * s**2 + 2 * a2 * s + a1
        # 微小な勾配を防ぐため tol で補正
        f_prime_safe = torch.where(torch.abs(f_prime) < tol, torch.full_like(f_prime, tol), f_prime)
        s_new = s - f / f_prime_safe
        if torch.max(torch.abs(s_new - s)) < tol:
            break
        s = s_new
    
    return s