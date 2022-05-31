import jittor


def cdist(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * jittor.matmul(src, dst.permute(0, 2, 1))
    dist += jittor.sum(src ** 2, -1).view(B, N, 1)
    dist += jittor.sum(dst ** 2, -1).view(B, 1, M)
    return jittor.sqrt(dist)


def cdist_single(src, dst):
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * jittor.matmul(src, dst.permute(0, 1))
    dist += jittor.sum(src ** 2, -1).view(N, 1)
    dist += jittor.sum(dst ** 2, -1).view(1, M)
    return jittor.sqrt(dist)
