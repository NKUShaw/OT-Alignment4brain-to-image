import torch
import torch.nn as nn
import torch.nn.functional as F

class FDivDiscriminator(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,) scalar score
        

def compute_f_divergence_loss(emb_voxel, emb_image, discriminator, device='cuda'):
    """
    Use a discriminator to estimate f-divergence between emb_voxel and emb_image.
    Returns the JS-like divergence loss.
    """
    batch_size, dim, emb_dim = emb_voxel.size()
    losses = []

    for b in range(batch_size):
        x = emb_voxel[b].to(device)  # (N, D)
        y = emb_image[b].to(device)  # (N, D)

        # Discriminator scores
        D_x = discriminator(x)         # (N,)
        D_y = discriminator(y)         # (N,)

        # JS-based f-divergence loss
        loss = - (F.logsigmoid(D_x).mean() + F.logsigmoid(-D_y).mean())
        losses.append(loss)

    return torch.stack(losses).mean()
