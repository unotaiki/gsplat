import torch

def _roots_quadratic(a, b, c):
    """
    二次方程式 a x^2 + b x + c = 0 の解を返す
    戻り値: (root1, root2)
    """
    # Δ = b^2 - 4ac
    delta = b * b - 4 * a * c
    sqrt_delta = torch.sqrt(delta)
    root1 = (-b + sqrt_delta) / (2 * a)
    root2 = (-b - sqrt_delta) / (2 * a)
    return root1, root2

def _cardano_cubic(A, B, C, D):
    """
    Solve A x^3 + B x^2 + C x + D = 0 via Cardano’s method.
    Returns a tensor[..., 3] of roots (possibly complex).
    """
    dtype = A.dtype   # e.g. torch.complex128
    device = A.device

    # 1) Normalize and depress cubic: x = t - B/(3A)
    b = B / A; c = C / A; d = D / A
    shift = b / 3

    # 2) Depressed form t^3 + p t + q = 0
    p = c / 3 - b * b / 9
    q = b * (2 * b * b - 9 * c) / 54 + d / 2

    # 3) Discriminant
    disc = q * q + p * p * p
    sqrt_disc = torch.sqrt(disc)

    # 4) Cardano’s U/V
    S = torch.pow(-q + sqrt_disc, 1/3)
    T = torch.pow(-q - sqrt_disc, 1/3)
    t0 = S + T

    # 5) Complex cube‐root of unity ω = -½ + i·(√3/2)
    omega = torch.tensor(-0.5 + 0.5j * 1.732050807,  # 1.732050807 = √3
                         dtype=dtype, device=device)

    # 6) The three t‐solutions
    t1 = -t0 / 2 + (S - T) * omega
    t2 = -t0 / 2 + (S - T) * torch.conj(omega)

    # 7) Shift back to x
    x0 = t0 - shift
    x1 = t1 - shift
    x2 = t2 - shift

    return torch.stack([x0, x1, x2], dim=-1)


def solve_quartic_ferrari(a, b, c, d, e):
    """
    Ferrariの公式で四次方程式を解く
    入力: a,b,c,d,e はいずれも torch.Tensor  (shape はブロードキャスト可能)
    出力: roots は各方程式につき4つの解を持つ tensor[...,4]
    """
    # 複素計算用に cast
    dtype = torch.complex64  # must be float
    device = a.device

    a = a.to(dtype)
    b = b.to(dtype)
    c = c.to(dtype)
    d = d.to(dtype)
    e = e.to(dtype)

    # 平行移動 x = z - b/(4a)
    z0 = b / (4 * a)

    # Ferrari公式用の p,q,r の計算
    a2 = a * a
    b2 = b * b
    p = -3 * b2 / (8 * a2) + c / a
    q =  b * b2 / (8 * a2 * a) - b * c / (2 * a2) + d / a
    r = -3 * b2 * b2 / (256 * a2 * a2) + b2 * c / (16 * a2 * a) - b * d / (4 * a2) + e / a

    # 解くべき3次方程式の係数 (StackOverflow準拠)
    #   8 y^3 -4 p y^2 -8 r y + (4 r p - q^2) = 0
    A3 = torch.tensor(8, dtype=dtype, device=device)
    B3 = -4 * p
    C3 = -8 * r
    D3 = 4 * r * p - q * q

    # Cardanoで3解を求め、最も虚部が小さい解 y0 を選択
    ys = _cardano_cubic(A3, B3, C3, D3)  # shape[...,3]
    # |Im| が最小のインデックス
    idx = torch.argmin(ys.imag.abs(), dim=-1, keepdim=True)
    # shape[...,1]
    y0 = torch.gather(ys, dim=-1, index=idx).squeeze(-1)

    # 次に a0, b0 を計算
    a0 = torch.sqrt(-p + 2 * y0)
    # a0==0 の場合は別処理
    b0 = torch.where(a0 == 0,
                     y0 * y0 - r,
                     -q / (2 * a0))

    # 2つの二次式を解く
    r1, r2 = _roots_quadratic(torch.ones_like(a0),  a0,        y0 + b0)
    r3, r4 = _roots_quadratic(torch.ones_like(a0), -a0,        y0 - b0)

    # 元の変数に戻してまとめる
    roots = torch.stack([r1 - z0, r2 - z0, r3 - z0, r4 - z0], dim=-1)
    return roots

# ===========================
# 使い方例
if __name__ == "__main__":
    # GPUを使う場合
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 例: 2x^4 - 3x^3 + x^2 - 5x + 2 = 0
    a = torch.tensor([2.0], device=device)
    b = torch.tensor([-3.0], device=device)
    c = torch.tensor([1.0], device=device)
    d = torch.tensor([-5.0], device=device)
    e = torch.tensor([2.0], device=device)

    roots = solve_quartic_ferrari(a, b, c, d, e)
    print(roots)  # shape [1,4] に4解が出力される
