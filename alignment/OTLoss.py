import torch
import ot
import numpy as np

def compute_partial_wasserstein_loss(emb_voxel, emb_image, m_ratio=0.5, device='cuda'):
    batch_size, dim_a, emb_dim = emb_voxel.size()
    _, dim_b, _ = emb_image.size()
    losses = []

    for b in range(batch_size):
        voxel_sample = emb_voxel[b]  # (256, 1024)
        image_sample = emb_image[b]  # (256, 1024)

        cost_matrix = torch.cdist(voxel_sample, image_sample, p=2)  # Shape: (256, 256)

        a = torch.ones(dim_a, device=device) / dim_a  
        b = torch.ones(dim_b, device=device) / dim_b 

        total_mass = min(a.sum(), b.sum()) 
        m = m_ratio * total_mass.item()  

        gamma = ot.partial.partial_wasserstein(
            a.detach().cpu().numpy(), 
            b.detach().cpu().numpy(), 
            cost_matrix.detach().cpu().numpy(), 
            m=m
        )
        
        gamma_torch = torch.tensor(gamma, device=device)
        partial_w_dist = (cost_matrix * gamma_torch).sum()
        losses.append(partial_w_dist)

    loss = torch.stack(losses).mean()
    return loss
