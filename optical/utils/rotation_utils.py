import torch

# def quaternion_from_axis_angle(
#     axis: torch.Tensor, 
#     angle: torch.Tensor,
#     eps: float = 1e-8
# ) -> torch.Tensor:
#     """
#     正規化済み軸(axis)と角度(angle)からクォータニオン (w, x, y, z) を生成する。

#     Args:
#         axis: Tensor[N, 3]  各回転軸ベクトル（任意長でも可）
#         angle: Tensor[N]    各回転角（ラジアン）、axis.shape[:-1] にブロードキャスト可能
#         eps: ノルム除算時の下限クリップ値

#     Returns:
#         Tensor[N, 4]  クォータニオン [w, x, y, z]
#     """
    
    
#     # 軸ベクトルを正規化
#     norm = axis.norm(dim=-1, keepdim=True).clamp(min=eps)
#     axis_unit = axis / norm

#     half = angle.unsqueeze(-1) * 0.5
#     w = torch.cos(half)
#     s = torch.sin(half)

#     xyz = axis_unit * s
#     return torch.cat([w, xyz], dim=-1)


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
        a = a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        b = b.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # compute cross product and dot product
    cross = torch.cross(a, b, dim=-1)
    dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

    s = dot + 1.0
    w = torch.sqrt(s*0.5)
    xyz = cross / (2.0 * w.clamp(min=eps))
    q = torch.cat([w, xyz], dim=-1)
    
    return q / q.norm(dim=-1, keepdim=True) 