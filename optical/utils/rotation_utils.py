import torch

def quaternion_from_axis_angle(
    axis: torch.Tensor, 
    angle: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    正規化済み軸(axis)と角度(angle)からクォータニオン (w, x, y, z) を生成する。

    Args:
        axis: Tensor[N, 3]  各回転軸ベクトル（任意長でも可）
        angle: Tensor[N]    各回転角（ラジアン）、axis.shape[:-1] にブロードキャスト可能
        eps: ノルム除算時の下限クリップ値

    Returns:
        Tensor[N, 4]  クォータニオン [w, x, y, z]
    """
    
    
    # 軸ベクトルを正規化
    norm = axis.norm(dim=-1, keepdim=True).clamp(min=eps)
    axis_unit = axis / norm

    half = angle.unsqueeze(-1) * 0.5
    w = torch.cos(half)
    s = torch.sin(half)

    xyz = axis_unit * s
    return torch.cat([w, xyz], dim=-1)


def quat_from_2dirs(
    a: torch.Tensor, 
    b: torch.Tensor,
    eps: float = 1e-6,
    flag_need_normalize: bool = False
) -> torch.Tensor:
    """
    ベクトル a からベクトル b への回転を表すクォータニオンを計算する。

    Args:
        a: Tensor[N, 3]  回転前のベクトル
        b: Tensor[N, 3]  回転後のベクトル
        eps: ノルム除算時の下限クリップ値

    Returns:
        Tensor[N, 4]  クォータニオン [w, x, y, z]
    """
    
    if flag_need_normalize:
        # ベクトルを正規化
        a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # compute cross product and dot product
    cross = torch.cross(a, b, dim=-1)
    dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

    s = dot + 1.0
    w = torch.sqrt(s*0.5)
    xyz = cross / (2.0 * w.clamp(min=eps))
    q = torch.cat([w, xyz], dim=-1)
    
    return q / q.norm(dim=-1, keepdim=True) 


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    クォータニオン q = [w, x, y, z] から 3x3 回転行列を返す（PyTorch版）。

    Args:
        q: shape (..., 4) のテンソル。最後の次元が [w, x, y, z]

    Returns:
        R: shape (..., 3, 3) の回転行列テンソル
    """
    # 正規化
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(dim=-1)

    ww = w*w; xx = x*x; yy = y*y; zz = z*z
    wx = w*x; wy = w*y; wz = w*z
    xy = x*y; xz = x*z; yz = y*z

    # バッチ対応でスタック
    row0 = torch.stack([ww + xx - yy - zz, 2*(xy - wz),     2*(xz + wy)], dim=-1)
    row1 = torch.stack([2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)], dim=-1)
    row2 = torch.stack([2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz], dim=-1)

    R = torch.stack([row0, row1, row2], dim=-2)
    return R