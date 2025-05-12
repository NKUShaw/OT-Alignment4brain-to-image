import torch

# def gaussian_kernel_matrix(x, y, sigma=1.0):
#     """
#     Compute the Gaussian RBF kernel matrix between two batches.
#     x: (N, D)
#     y: (M, D)
#     return: (N, M)
#     """
#     x = x.unsqueeze(1)  # (N, 1, D)
#     y = y.unsqueeze(0)  # (1, M, D)
#     return torch.exp(-((x - y) ** 2).sum(2) / (2 * sigma ** 2))

# def compute_mmd_loss(emb_voxel, emb_image, sigma=1.0):
#     """
#     emb_voxel: (B, N, D)
#     emb_image: (B, M, D)
#     Returns: scalar MMD loss between voxel and image distributions
#     """
#     batch_size = emb_voxel.size(0)
#     losses = []

#     for b in range(batch_size):
#         x = emb_voxel[b]  # (N, D)
#         y = emb_image[b]  # (M, D)

#         K_xx = gaussian_kernel_matrix(x, x, sigma)
#         K_yy = gaussian_kernel_matrix(y, y, sigma)
#         K_xy = gaussian_kernel_matrix(x, y, sigma)

#         mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
#         losses.append(mmd)

#     loss = torch.stack(losses).mean()
#     return loss

def compute_mmd_loss(emb_voxel, emb_image):
    """
    Linear-time MMD estimate using unbiased formulation.
    emb_voxel: (B, N, D)
    emb_image: (B, M, D)
    Returns: scalar MMD loss between voxel and image distributions
    """
    batch_size = emb_voxel.size(0)
    losses = []

    for b in range(batch_size):
        x = emb_voxel[b]  # (N, D)
        y = emb_image[b]  # (M, D)

        # Match sample size for fair comparison
        n = min(x.size(0), y.size(0))
        x = x[:n]
        y = y[:n]

        # Linear MMD (unbiased): MMD^2 = (1/n) sum (x_i - y_i)^2
        diff = x - y
        mmd = (diff ** 2).sum(1).mean()
        losses.append(mmd)

    return torch.stack(losses).mean()