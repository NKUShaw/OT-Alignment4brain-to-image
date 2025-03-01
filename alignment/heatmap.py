import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def plot_heatmap(voxel2emb, image2emb, test_dl, save_fig, batch_idx=0):
    voxel2emb.eval()
    image2emb.eval()
    
    for test_i, (voxel, image) in enumerate(test_dl): 
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                voxel = torch.mean(voxel, axis=1).float() # (300, 3, 15724) → (300, 1, 15724)
                indices = torch.randperm(voxel.shape[0])[:100]  
                voxel_sampled = voxel[indices]  
                image_sampled = image[indices]
                emb_voxel = voxel2emb(voxel_sampled)  # (300, 1, 15724) → (300, 256, 1024)
                emb_image = image2emb.encode_image(image_sampled, 'image')  # (300, 3, 256, 256) → (300, 256, 1024)

                emb_voxel = emb_voxel.view(emb_voxel.shape[0], -1)  # (300, 256, 1024) -> (300, 256*1024)
                emb_image = emb_image.view(emb_image.shape[0], -1)  
                emb_voxel_norm = emb_voxel / emb_voxel.norm(dim=1, keepdim=True)
                emb_image_norm = emb_image / emb_image.norm(dim=1, keepdim=True)

                similarity_matrix = torch.mm(emb_voxel_norm, emb_image_norm.T)
                similarity_matrix = similarity_matrix.cpu().numpy()
                similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(similarity_matrix, cmap="hot", cbar=True, 
                            annot=False, xticklabels=False, yticklabels=False)
                # plt.title("Embedding Similarity Heatmap", fontsize=18)
                plt.xlabel("Image Embeddings", fontsize=18)
                plt.ylabel("Voxel Embeddings", fontsize=18)
                
                # Save as PDF
                plt.savefig(f"{save_fig}.pdf", bbox_inches='tight')
                plt.close()

                print(f"Heatmap saved to {save_fig}.pdf")
                
                break