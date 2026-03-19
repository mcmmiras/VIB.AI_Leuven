#!usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import sys, os
from collections import defaultdict, Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import io
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(f"runs/test")

batch_size = 64
learning_rate = 1e-3
num_epochs = 10
input_dim = 4
latent_dim = 2
num_embeddings = 20  # Number of vectors in the codebook
num_classes = num_embeddings
commitment_cost = 0.25  # Beta, the commitment loss weight
hidden_dims = [32, 64]

class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):
        batch_size_z, emb_dim_z = z.shape
        z_flattened = z

        # Calculate distances between z and the codebook embeddings |a-b|²
        distances = (
            torch.sum(z_flattened ** 2, dim=-1, keepdim=True)                 # a²
            + torch.sum(self.embedding.weight.t() ** 2, dim=0, keepdim=True)  # b²
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())        # -2ab
        )

        # Get the index with the smallest distance
        encoding_indices = torch.argmin(distances, dim=-1)

        # Get the quantized vector
        z_q = self.embedding(encoding_indices)

        # Calculate the commitment loss
        loss = F.mse_loss(z_q, z.detach()) + commitment_cost * F.mse_loss(z_q.detach(), z)

        # Straight-through estimator trick for gradient backpropagation
        z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices # (batch, dim), scalar, (batch,)


class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], latent_dim)
        )
        # Vector Quantization
        self.vq_layer = VQEmbedding(num_embeddings, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, _ = self.vq_layer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss


def vqvae_loss(recon_x, x, vq_loss):
    recon_loss = F.mse_loss(recon_x, x)
    return recon_loss + vq_loss



errors = open("errors.txt","w")
predictions = open("predictions.txt","w")
results_fs = list()
results_aleph = list()
results_both = list()
embeddings = list()
labels = list()
labelsFS = list()
residues = list()
limit = 0
for file in os.listdir(sys.argv[1]):
    if "ai.txt" in file:
        if limit == 100:
            break
        try:
            data = pd.read_csv(file,sep="\t",header=0)
            if len(data.index) < 1:
                continue
            for i in data.index:
                modi = data.at[i,"module_CVi"]
                modj = data.at[i,"module_CVj"]
                dist = round(float(data.at[i,"dist_ij"])/11,3)
                if dist > 1:
                    continue
                angle = float(data.at[i,"angle_ij"])
                if float(angle)  >90:
                    add = "anti"
                else:
                    add = "para"
                angle = round(float(data.at[i,"angle_ij"])/180,3)
                emb = np.array((modi,modj,dist,angle))
                embeddings.append(emb)
                ssij = data.at[i,"ss_ij"].split("-")
                ssij = sorted(list(ssij))
                labels.append(f"{('-').join(ssij)}_{add}")
                labelsFS.append(f"{data.at[i, "FS_token"]}_{add}")
            limit+=1
            print(f"Processed {limit} lines.")
        except Exception as e:
            errors.write(f"{file} - Error: {e}\n")

embeddings = [torch.tensor(emb,dtype=torch.float32) for emb in embeddings]

train_loader = DataLoader(dataset=embeddings, batch_size=batch_size, shuffle=True, drop_last=True)

model = VQVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Training Loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    # Use tqdm for the progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')
    for batch_idx, data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, vq_loss = model(data)
        loss = vqvae_loss(recon_batch, data, vq_loss)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # Update tqdm description with current loss
        pbar.set_postfix({'Loss': loss.item()})
    avg_loss = train_loss / len(train_loader.dataset)
    writer.add_scalar("Training Loss", avg_loss, epoch+1)
    print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}')
    # Save and display a sample of the reconstructed images
    if (epoch + 1) % 2 == 0:
        with torch.no_grad():
            recon_batch, _ = model(data)
            # Get encoder outputs (z_e) and discrete codes for this batch
            z_e = model.encoder(data).cpu()  # (10, 2) - continuous latents
            _, _, discrete_codes = model.vq_layer(model.encoder(data))  # discrete indices
            # Extract codebook
            codebook_vectors = model.vq_layer.embedding.weight.data.cpu().numpy()
            fig = plt.figure(figsize=(10, 8))
            # 1. Plot CODEBOOK VECTORS (big black circles, colored by index)
            plt.scatter(codebook_vectors[:, 0], codebook_vectors[:, 1],
                        c=range(20), cmap='tab20', s=200, edgecolors='black', linewidth=1,
                        label='Codebook (20 discrete states)', zorder=3)
            # 2. Plot ENCODER OUTPUTS (small gray dots) + color by assigned code
            scatter = plt.scatter(z_e[:, 0], z_e[:, 1],
                                  c=discrete_codes.cpu(), cmap='tab20', s=60, alpha=0.7,
                                  edgecolors='gray', linewidth=0.5,
                                  label='Encoder outputs (your batch)', zorder=2)
            # 3. Lines connecting each z_e to its assigned codebook vector
            for i in range(len(z_e)):
                code_idx = discrete_codes[i].item()
                code_pos = codebook_vectors[code_idx]
                z_pos = z_e[i]
                plt.plot([z_pos[0], code_pos[0]], [z_pos[1], code_pos[1]],
                         'k-', alpha=0.3, linewidth=1, zorder=1)
            plt.colorbar(label='Code index (0-19)')
            plt.grid(True, alpha=0.3)
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.title(f'Codebook + Batch {len(z_e)} Encoder Outputs (Epoch {epoch + 1})')
            plt.legend()
            # Label codebook points
            for i, (x, y) in enumerate(codebook_vectors):
                plt.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points',
                             fontsize=10, fontweight='bold')
            plt.tight_layout()

            # Convert figure to image and add to TensorBoard
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            img = np.array(img)
            writer.add_image(f"Codebook + Batch {len(z_e)} Encoder Outputs", img, epoch+1, dataformats='HWC')
            #plt.show()
            buf.close()
