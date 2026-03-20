#!usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import sys, os
import math
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
from sklearn.model_selection import train_test_split

name = sys.argv[2]
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda")
writer = SummaryWriter(f"runs/{name}")
os.makedirs(f"results/{name}", exist_ok=True)
batch_size = 64
learning_rate = 1e-3
num_epochs = 100
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

def scalar_product(mod1,mod2,ang, eps=1e-8):
    ang = math.radians(angle)
    cos = math.cos(ang)
    dot = mod1*mod2*cos
    if dot > eps:
        return "para"
    elif dot < -eps:
        return "anti"
    else:
        return "neutral"

errors = open(f"errors_{name}.txt","w")
predictions = open(f"predictions_{name}.txt","w")
results_fs = list()
results_aleph = list()
results_both = list()
embeddings = list()
labels = list()
labelsFS = list()
residues = list()

limit = 0
allfiles = list()
for file in os.listdir(sys.argv[1]):
    if "ai.txt" in file:
        if limit == 1000000000000000000000000000000000000:
            break
        allfiles.append(file)
        limit+=1

train_df, test_df = train_test_split(
    allfiles,
    test_size=0.3,
    random_state=312,  # for reproducibility
)
with open(f"{name}_train_set.txt","w") as df:
    for ele in train_df:
        df.write(f"{ele}\n")

with open(f"{name}_test_set.txt","w") as df:
    for ele in test_df:
        df.write(f"{ele}\n")

embeddings_train = list()
embeddings_test = list()
emb_test_ids = list()

limit = 0
for file in os.listdir(sys.argv[1]):
    if "ai.txt" in file:
        if limit == 1000000000000000000000000000000000000:
            break
        try:
            data = pd.read_csv(file,sep="\t",header=0)
            if len(data.index) < 1:
                continue
            for i in data.index:
                resi = data.at[i,"i"]
                fs = data.at[i,"FS_token"]
                modi = data.at[i,"module_CVi"]
                modj = data.at[i,"module_CVj"]
                dist = round(float(data.at[i,"dist_ij"])/11,3)
                if dist > 1:
                    continue
                angle = float(data.at[i,"angle_ij"])
                add = scalar_product(float(modi), float(modj), angle)
                angle = round(float(data.at[i,"angle_ij"])/180,3)
                ssij = data.at[i,"ss_ij"].split("-")
                emb = np.array((modi,modj,dist,angle))
                if file in train_df:
                    embeddings_train.append(emb)
                elif file in test_df:
                    embeddings_test.append(emb)
                    emb_test_ids.append((file.replace("_ai.txt",""),resi,f"{fs}_{add}",ssij[0]))
                ssij = sorted(list(ssij))
                labels.append(f"{('-').join(ssij)}_{add}")
                labelsFS.append(f"{data.at[i, "FS_token"]}_{add}")
            limit+=1
            print(f"Processed {limit} lines.")
        except Exception as e:
            errors.write(f"{file} - Error: {e}\n")

embeddings_train = [torch.tensor(emb,dtype=torch.float32) for emb in embeddings_train]
embeddings_test = [torch.tensor(emb,dtype=torch.float32) for emb in embeddings_test]

train_loader = DataLoader(dataset=embeddings_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=embeddings_test, batch_size=batch_size, shuffle=False, drop_last=False)
print(f"Total entries in training set: {len(train_loader)}\nTotal entries in testing set: {len(test_loader)}.")


model = VQVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
emb_test_preds = list()
# Training Loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    test_loss = 0
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
    if (epoch + 1) % 10 == 0:
        #### CHECK TEST BATCHES ####
        with torch.no_grad():
            for batch_test, data_test in enumerate(test_loader):
                data_test = data_test.to(device)
                recon_batch_test, vq_loss_test = model(data_test)
                loss_test = vqvae_loss(recon_batch_test, data_test, vq_loss_test)
                test_loss += loss_test.item()
                # Get encoder outputs (z_e) and discrete codes for this batch
                z_e = model.encoder(data_test).cpu()  # (10, 2) - continuous latents
                _, _, discrete_codes = model.vq_layer(model.encoder(data_test))  # discrete indices
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
                    if epoch+1 == num_epochs:
                        emb_test_preds.append(code_idx)
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
                writer.add_image(f"Epoch {epoch+1}: Codebook + Test Batch {len(z_e)} Encoder Outputs", img, batch_test+1, dataformats='HWC')
                #plt.show()
                buf.close()
                plt.close(fig)
            avg_loss_test = test_loss / len(test_loader)
            writer.add_scalar("Test Loss", avg_loss_test, epoch + 1)
        """
        ##### CHECK TRAINING IMAGES ####
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
        """
PATH = f'./{name}_vqvae_net.pth'
torch.save(model.state_dict(), PATH)

if len(emb_test_ids) != len(emb_test_preds):
    print("Error: test embeddings and labels length do not match")
else:
    # emb_test_ids.append((file.replace("_ai.txt",""),resi,f"{fs}_{add}"))
    currentPDB = ""
    for emb,label in zip(emb_test_ids, emb_test_preds):
        pdb, resi, fs, ss = emb
        if currentPDB != pdb:
            currentPDB = pdb
            print(currentPDB)
            out = open(f"results/{name}/{currentPDB}.csv","w")
            out.write(f"Residue\tSS\tFoldSeek\tALEPH\n")
        print(f"- {resi}\tSS:{ss}\tFS:{fs}\tALEPH:{label}")
        out.write(f"{resi}\t{ss}\t{fs}\t{label}\n")