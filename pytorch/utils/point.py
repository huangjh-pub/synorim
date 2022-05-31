import torch


def index_points_group(points, knn_idx, t=False):
    """
    Input:
        points: input points data, [B, N', C], or [B, C, N'](transposed)
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C] or [B, C, N, K](transposed)
    """
    B, Np, C = points.size()
    if t:
        Np, C = C, Np

    _, N, K = knn_idx.size()
    knn_idx = knn_idx.reshape(B, -1)
    if not t:
        new_points = torch.gather(points, dim=1, index=knn_idx.unsqueeze(-1).expand(-1, -1, points.size(-1)))
        new_points = new_points.reshape(B, N, K, C)
    else:
        new_points = torch.gather(points, dim=-1, index=knn_idx.unsqueeze(1).expand(-1, points.size(1), -1))
        new_points = new_points.reshape(B, C, N, K)

    return new_points


def propagate_features(source_pc: torch.Tensor, target_pc: torch.Tensor,
                       source_feat: torch.Tensor, nk: int = 3, batched: bool = True):
    """
    Propagate features from the domain of source to the domain of target.
    :param source_pc: (B, N, 3) point coordinates
    :param target_pc: (B, M, 3) point coordinates
    :param source_feat: (B, N, F) source features
    :param nk: propagate k number
    :param batched: whether dimension B is present or not.
    :return: (B, M, F) target feature
    """
    if not batched:
        source_pc = source_pc.unsqueeze(0)
        target_pc = target_pc.unsqueeze(0)
        source_feat = source_feat.unsqueeze(0)

    dist = torch.cdist(target_pc, source_pc)  # (B, N, M)
    dist, group_idx = torch.topk(dist, nk, dim=-1, largest=False, sorted=False)     # (B, N, K)

    # Shifted reciprocal function.
    w_func = 1 / (dist + 1.0e-6)
    weight = (w_func / torch.sum(w_func, dim=-1, keepdim=True)).unsqueeze(-1)  # (B, N, k, 1)
    sparse_feature = index_points_group(source_feat, group_idx)
    full_flow = (sparse_feature * weight).sum(-2)  # (B, N, C)

    if not batched:
        full_flow = full_flow[0]

    return full_flow
