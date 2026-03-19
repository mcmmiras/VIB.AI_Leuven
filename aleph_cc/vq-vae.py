#!usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import sys, os
from collections import defaultdict, Counter

errors = open("errors.txt","w")
predictions = open("predictions.txt","w")
results_fs = list()
results_aleph = list()
results_both = list()
embeddings = list()
labels = list()
labelsFS = list()
residues = list()
for file in os.listdir(sys.argv[1]):
    if ".txt" in file:
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
        except Exception as e:
            errors.write(f"{file} - Error: {e}\n")


# kmeans
results_dict = defaultdict(dict)
results_dict_FS = defaultdict(dict)
#reduced_data = PCA(n_components=2).fit_transform(embeddings)
kmeans = KMeans(n_clusters=20, random_state=312, n_init="auto").fit(embeddings)
print(kmeans.labels_)
for lab in kmeans.labels_:
    results_dict[lab] = list()
    results_dict_FS[lab] = list()
predictions.write(f"ypred\tytrue\tyfs\n")
for ypred, ytrue, yfs in zip(kmeans.labels_, labels, labelsFS):
    print("ypred:",ypred,"ytrue:",ytrue, "yfs:",yfs)
    predictions.write(f"{ypred}\t{ytrue}\t{yfs}\n")
    results_dict[ypred].append(ytrue)
    results_dict_FS[ypred].append(yfs)
#kmeans.predict([[0, 0], [12, 3]])
print(kmeans.cluster_centers_)

"""
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering on the digits dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

"""

import plotly.express as px
from sklearn.decomposition import PCA


pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

fig = px.scatter(
    x=emb_2d[:,0], y=emb_2d[:,1],
    color=kmeans.labels_.astype(str),  # Just clusters
    title="KMeans 20 Clusters (PCA 2D)"
)
fig.show()

print("My labels")
for key, val in results_dict.items():
    counts = Counter(val)
    for key2, val in counts.items():
        print(key, key2, val)
        results_aleph.append((str(key), str(val), str(key2)))

print("My FS labels")
for key, val in results_dict_FS.items():
    counts = Counter(val)
    for key2, val in counts.items():
        print(key, key2, val)
        results_fs.append((str(key), str(val), str(key2)))

results_aleph.sort()
results_fs.sort()

with open("predictions_aleph.txt", "w") as f:
    for val in results_aleph:
        val = ("\t").join(val)
        f.write(val + "\n")

with open("predictions_fs.txt", "w") as f:
    for val in results_fs:
        val = ("\t").join(val)
        f.write(val + "\n")




import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import plotly.express as px

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=4, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Learnable codebook [num_embeddings, embedding_dim]
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):  # z = your 4D embeddings [batch, 4]
        # Find closest embedding
        z_flattened = z.view(-1, self.embedding_dim)  # [N, 4]
        distances = torch.cdist(z_flattened, self.embedding.weight)  # [N, K]
        encoding_indices = torch.argmin(distances, dim=1)  # [N]

        # Quantize
        z_q = self.embedding(encoding_indices).view(z.shape)  # [batch, 4]

        # Losses
        loss = nn.functional.mse_loss(z_q.detach(), z) + \
               self.commitment_cost * nn.functional.mse_loss(z_q, z.detach())

        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        return z_q, loss, encoding_indices

# Extract tokens from your embeddings
vq = VectorQuantizer(num_embeddings=20, embedding_dim=4)  # 128 tokens, 4D each
with torch.no_grad():
    embeddings_torch = torch.tensor(embeddings, dtype=torch.float32)
    quantized, _, token_ids = vq(embeddings_torch)  # token_ids = [0..127] per sample

# Now plot discrete tokens instead of continuous clusters!
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)
token_2d = pca.transform(quantized.numpy())

fig = px.scatter(
    x=emb_2d[:,0], y=emb_2d[:,1],
    color=token_ids.numpy().astype(str),  # Discrete tokens 0-127
    title="VQ-VAE Tokens (PCA 2D)"
)
#fig.show()