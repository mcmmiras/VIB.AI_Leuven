import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw, ImageFont
from Bio.PDB import PDBParser, MMCIFParser, DSSP
from collections import defaultdict, Counter
parser = PDBParser(QUIET=True)

# Global values
source = sys.argv[1]
name = sys.argv[2]
errors = open(f"{name}_errors.txt", "w")
writer = SummaryWriter(log_dir=f"{name}_run")

residue_types = {
    'ALA': 'hydrophobic',
    'ARG': 'basic',
    'ASN': 'polar',
    'ASP': 'acidic',
    'CYS': 'polar',
    'GLN': 'polar',
    'GLU': 'acidic',
    'GLY': 'disruptive',
    'HIS': 'basic',
    'ILE': 'hydrophobic',
    'LEU': 'hydrophobic',
    'LYS': 'basic',
    'MET': 'hydrophobic',
    'PHE': 'hydrophobic',
    'PRO': 'disruptive',
    'SER': 'polar',
    'THR': 'polar',
    'TRP': 'hydrophobic',
    'TYR': 'polar',
    'VAL': 'hydrophobic'
}


class Net(nn.Module): # Currently an autoencoder
    def __init__(self, input_channels=3, num_classes=3, image_size=(128,128),reconstruct = False):
        # Reconstruction parameter
        self.reconstruct = reconstruct
        # Encoder layers
        super(Net, self).__init__()
        # Convolution + pooling layers
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv2d(input_channels, 16, 4, stride=2, padding=1)
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        # Compute flattened feature size dynamically and feature map shape
        self._to_linear, self._enc_shape = self._compute_linear_input(image_size, input_channels)
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 256) # Latent: summary of the most relevant info of image
        self.fc3 = nn.Linear(256, num_classes) # Output to class logits, for classification tasks and CE loss

        # Decoder layers
        # Fully connected layers inverted
        self.ifc2 = nn.Linear(256, 512)
        self.ifc1 = nn.Linear(512, self._to_linear)
        # Upsampling instead of pooling
        #self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # Decoder convolutional layers
        self.iconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.iconv2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.iconv1 = nn.ConvTranspose2d(16, input_channels, 4, stride=2, padding=1)

    def encoder(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Drop neurons
        x = F.relu(self.fc2(x)) # Latent vector
        return x

    def decoder(self, x):
        x = F.relu(self.ifc2(x)) # Latent (x is not the last FC, which is only used in the last classification step)
        x = F.relu(self.ifc1(x))
        # reshape back to conv feature map
        x = x.view(-1, *self._enc_shape)
        #x = self.upsample(x) # For 'upscaling' the MaxPooling performed before
        x = F.relu(self.iconv3(x))
        x = F.relu(self.iconv2(x))
        #x = self.upsample(x)
        x = torch.sigmoid(self.iconv1(x))  # sigmoid for image output [0,1]
        return x

    def forward(self, x):
        latent = self.encoder(x)
        # Classification task (CrossEntropyLoss will be calculated)
        logits = self.fc3(latent)  # Logits
        # Reconstruction task (MSELoss will be calculated)
        recon = self.decoder(latent)

        return logits, recon

    def _compute_linear_input(self, image_size, input_channels):
        x = torch.zeros(1, input_channels, image_size[0], image_size[1])
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.numel(), x.shape[1:]  # save C,H,W



class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, classes, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=0, sep="\t")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        if "build" not in self.img_dir:
            img_path = os.path.join(self.img_dir, f"{self.img_labels.iloc[idx, 0].upper()}.png")
            label = self.classes[self.img_labels.iloc[idx, 1]]
            label = torch.tensor(label, dtype=torch.long)  # ✅ convert to tensor
        else:
            for img_file in os.listdir(self.img_dir):
                img_path = os.path.join(self.img_dir, img_file)
                label = img_file.split("_")[0]
                label = self.classes[label]
                self.samples.append((img_path, label))
        # Load image as PIL Image ("L" for greyscale, "RGB" for color)
        if "--color" in sys.argv:
            image = Image.open(img_path).convert("RGB")  # ✅ compatible with ToTensor() in the transformer
        else:
            image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class FragmentedImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, classes, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        # READ annotations
        self.img_labels = pd.read_csv(annotations_file, header=0, sep="\t")
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        # BUILD flattened list: each fragment = one dataset entry
        self.samples = []  # List of (img_path, label)
        if "build" not in img_dir:
            for idx in range(len(self.img_labels)):
                base_name = self.img_labels.iloc[idx, 0].upper()
                label_idx = self.img_labels.iloc[idx, 1]
                label = self.classes[label_idx]
                # Find ALL matching fragments
                matching_imgs = [f for f in self.img_list if base_name in f]
                # Add EACH fragment as separate sample with SAME label
                for img_file in matching_imgs:
                    img_path = os.path.join(img_dir, img_file)
                    self.samples.append((img_path, label))
        else:
            for img_file in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_file)
                label = img_file.split("_")[0]
                label = self.classes[label]
                self.samples.append((img_path, label))
    def __len__(self):
        return len(self.samples)  # Total fragments across ALL annotations
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Load single image
        if "--color" in sys.argv:
            image = Image.open(img_path).convert("RGB")  # ✅ compatible with ToTensor() in the transformer
        else:
            image = Image.open(img_path).convert("L")
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def generateImages(file, pdb_dir, classes, fragmented=False):
    global name
    if "--fragments" in sys.argv:
        fragmented = True
    data = pd.read_csv(file, header=0, sep="\t")
    if f"{name}_imgs" not in os.listdir():
        os.mkdir(f"{name}_imgs")
        os.mkdir(f"{name}_imgs/projected")
        os.mkdir(f"{name}_imgs/connected")
    for idx in data.index:
        code = data.iloc[idx, 0].upper()
        cc_chains = data.at[idx,"CC_chains"].split(",")
        label = data.iloc[idx, 1]
        projected = list()
        coordsCA = list()
        chainsRes = list()
        residues_list = list()
        #coordsLateral = list()
        try:
            structure = parser.get_structure(code, os.path.join(pdb_dir, f"{code}.pdb"))
            dssp = DSSP(structure[0], os.path.join(pdb_dir, f"{code}.pdb"))
        except:
            continue
        print("\npdb:", code, "residues:", len(dssp))
        for chain in structure[0]:
            if chain.id not in cc_chains:
                continue
            for residue in chain:
                # only standard residues (skip hetero/water)
                if residue.id[0] != " ":
                    continue
                key = (chain.id, residue.id)
                if key in dssp:
                    ss = dssp[key][2]
                    if ss is None or ss == " ":
                        ss = "-"
                else:
                    ss = "-"
                if ss == "H":
                    for atom in residue:
                        if atom.id == "CA":
                            coordsCA.append(atom.coord)
                            chainsRes.append(chain.id)
                            residues_list.append(residue.get_resname())
                        #elif atom.id not in ["CA", "C", "O", "N"]:
                        #    side_chain.append(atom.coord)
                    #if len(side_chain) == 0:
                    #    side_chain = coordsCA[-1]
                    #else:
                    #    side_chain = np.mean(side_chain)
                    #coordsLateral.append(side_chain)
        try:
            pca = PCA(n_components=3, random_state=312)
            pca.fit(coordsCA)
            projected_str = pca.transform(coordsCA)  # Already centered in centroid of protein
        except:
            print("Error")
            errors.write(f"{code}\tCould not extract coordinates/perform PCA\n")
            continue
        # projected_lateral = pca.transform(coordsLateral)
        TARGET_PIXELS = 128
        PIXELS_PER_ANGSTROM = 5.0
        ANGSTROM_SPAN = TARGET_PIXELS / PIXELS_PER_ANGSTROM  # 12.8 Å
        if fragmented:
            counter = 0
            window = 12
            while counter < len(projected_str):
                projected = np.array([p for p in projected_str if
                                      (projected_str[counter][0] + window) >= p[0] >= projected_str[counter][0]])
                proj_residues = np.array([r for p,r in zip(projected_str, residues_list) if
                                      (projected_str[counter][0] + window) >= p[0] >= projected_str[counter][0]])
                counter += len(projected)
                if len(projected) >= 7 * int(classes[label]):
                    projected_path = f"{name}_imgs/projected/{code}_{counter}.png"
                    connected_path = f"{name}_imgs/connected/{code}_{counter}.png"
                    # center 12.8 Å window on fragment
                    x_center = projected[:, 0].mean()
                    y_center = projected[:, 1].mean()
                    z_center = projected[:, 2].mean()
                    x_min = x_center - ANGSTROM_SPAN / 2.0
                    x_max = x_center + ANGSTROM_SPAN / 2.0
                    y_min = y_center - ANGSTROM_SPAN / 2.0
                    y_max = y_center + ANGSTROM_SPAN / 2.0
                    z_min = z_center - ANGSTROM_SPAN / 2.0
                    z_max = z_center + ANGSTROM_SPAN / 2.0
                    dpi = TARGET_PIXELS  # 1 inch = 128 px
                    fig = plt.figure(figsize=(1, 1), dpi=dpi)  # 128×128 px output [web:24][web:26]
                    # axes that fill the whole figure: [left, bottom, width, height] in 0–1
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.set_aspect("equal")
                    ax.axis("off")
                    ax.set_ylim(x_min, x_max)
                    #ax.set_xlim(z_min, z_max)
                    ax.set_xlim(z_min, z_max)  # 12.8 Å in each direction -> 5 px/Å
                    # Get residue names for this fragment (in same order as projected)
                    resnames = proj_residues
                    types_in_fragment = [residue_types.get(resname, 'unknown') for resname in resnames]
                    # Color map for types
                    color_map = {
                        'hydrophobic': '#1f77b4',  # blue
                        'polar': '#ff7f0e',  # orange
                        'acidic': '#d62728',  # red
                        'basic': '#2ca02c'  # green
                    }
                    color_map = {
                        'hydrophobic': '#d62728',  # red
                        'polar': '#1f77b4',  # blue
                        'acidic': '#1f77b4',  # blue
                        'basic': '#1f77b4',  # blue
                        'disruptive': '#2ca02c' # green
                    }
                    colors = [color_map.get(res_type, '#808080') for res_type in types_in_fragment]
                    # Now plot with per-point colors:
                    """
                    ax.scatter(projected[:, 2], projected[:, 0],
                               c=colors, s=20, marker='.')
                    fig.savefig(projected_path, transparent=True)
                    """

                    for i in range(len(projected) - 1):
                        ax.plot(projected[i:i + 2, 2], projected[i:i + 2, 0], color="black", linewidth=0.5)
                    fig.savefig(connected_path, transparent=True)
                    plt.close(fig)
        else:
            projected = np.array([p for p in projected_str if 6 >= p[0] >= -6])
            # Image paths:
            projected_path = f"{name}_imgs/projected/{code}.png"
            connected_path = f"{name}_imgs/connected/{code}.png"
            fig = plt.figure(figsize=(1.28, 1.28))
            ax = fig.add_subplot(111)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.scatter(projected[:, 1], projected[:, 2], c="#00000050", marker=".")
            fig.savefig(projected_path, transparent=True)
            # Single connected
            for i in range(len(projected) - 1):
                ax.plot(projected[i:i + 2, 1], projected[i:i + 2, 2], color="#00000050")
            fig.savefig(connected_path, transparent=True)
            plt.close(fig)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize [-1,1] → [0,1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # RGB image → NO cmap!
    plt.axis('off')
    plt.show()



def main():
    # Set device to transfer the Neural Net onto the GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)
    classes = defaultdict(dict)
    rootdir = os.getcwd()
    df = pd.read_csv(source, sep="\t",header=0)
    # Split 80% train, 20% test, stratified by 'orient'
    if f"{name}_train_set.csv" not in os.listdir(rootdir):
        train_df, temp_df = train_test_split(
            df,
            test_size=0.3,
            random_state=312,  # for reproducibility
            stratify=df['orient']  # ensures orient distribution is similar
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=312,
            stratify=temp_df['orient']
        )
        class_list = sorted(df["orient"].unique())
        idx_to_class = {i: c for i, c in enumerate(class_list)}
        class_to_idx = {c: i for i, c in idx_to_class.items()}
        # Check
        print("Train set class distribution:")
        print(train_df['orient'].value_counts(normalize=True))
        print("\nTest set class distribution:")
        print(test_df['orient'].value_counts(normalize=True))
        # Saving train/test datasets
        train_df.to_csv(f"{name}_train_set.csv", index=False, sep="\t")
        val_df.to_csv(f"{name}_val_set.csv", index=False, sep="\t")
        test_df.to_csv(f"{name}_test_set.csv", index=False,sep="\t")

    if "--embeddings" in sys.argv:
        generateImages(file=source,
                       pdb_dir="/media/mari/Data/vib_leuven/datasets/cc_sasa/biomols",
                       classes=class_to_idx,
                       fragmented=False)
    # Transform images to Tensors of normalized range [-1, 1].
    transformRGB = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transformL = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    if "--color" in sys.argv:
        transform = transformRGB
    else:
        transform = transformL

    batch_size = 16
    if "--fragments" in sys.argv:
        # Training
        trainset = FragmentedImageDataset(annotations_file=f"{name}_train_set.csv",
                                      img_dir=os.path.join(os.getcwd(),f"{name}_imgs/connected"),
                                      classes=class_to_idx,
                                      transform=transform)

        print("Classes:", class_to_idx)
        # Validation:
        valset = FragmentedImageDataset(annotations_file=f"{name}_val_set.csv",
                                     img_dir=os.path.join(os.getcwd(),f"{name}_imgs/connected"),
                                     classes=class_to_idx,
                                     transform=transform)
        # Testing
        testset = FragmentedImageDataset(annotations_file=f"{name}_test_set.csv",
                                     img_dir=os.path.join(os.getcwd(),f"{name}_imgs/connected"),
                                     classes=class_to_idx,
                                     transform=transform)
    else:
        # Training
        trainset = CustomImageDataset(annotations_file=f"{name}_train_set.csv",
                                      img_dir=os.path.join(os.getcwd(),f"{name}_imgs/connected"),
                                      classes=class_to_idx,
                                      transform=transform)

        print("Classes:", class_to_idx)
        # Validation
        valset = CustomImageDataset(annotations_file=f"{name}_val_set.csv",
                                     img_dir=os.path.join(os.getcwd(),f"{name}_imgs/connected"),
                                     classes=class_to_idx,
                                     transform=transform)
        # Testing
        testset = CustomImageDataset(annotations_file=f"{name}_test_set.csv",
                                     img_dir=os.path.join(os.getcwd(),f"{name}_imgs/connected"),
                                     classes=class_to_idx,
                                     transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("=" * 50)
    print("DATASET SIZES:")
    print(f"Train dataset: {len(trainset)} samples")
    print(f"Validation dataset: {len(valset)} samples")
    print(f"Test dataset:  {len(testset)} samples")
    print(f"Train batches: {len(trainloader)} batches")
    print(f"Validation batches: {len(valloader)} batches")
    print(f"Test batches:  {len(testloader)} batches")
    print("=" * 50)

    # Check first batch
    images, labels = next(iter(trainloader))
    print("FIRST BATCH SHAPES:")
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print("=" * 50)

    if "--train" in sys.argv:
        print("Starting Autoencoder Training...")
        # Showing some random training images
        dataiter = iter(trainloader)
        images, labels = next(dataiter)
        print(' '.join([str(label) for label in labels]))
        print(' '.join(idx_to_class[label.item()] for label in labels))
        imshow(torchvision.utils.make_grid(images))

        # Convolutional Neural Network
        if "--color" in sys.argv:
            channels_num = 3
        else:
            channels_num = 1
        net = Net(input_channels=channels_num, num_classes=2, image_size=(128,128), reconstruct=True)
        print(net)
        net = net.to(device)
        # Loss functions
        ce = nn.CrossEntropyLoss() # Classification task
        mse = nn.MSELoss() # Reconstruction task
        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=1e-3)  # Standard for deep nets
        # Training of CNN
        global_step = 0
        loss_list = list()

        # EARLY STOPPING BY LOSS
        best_val_loss = 1
        patience = 10  # Stop after 10 epochs without improvement
        patience_counter = 0

        for epoch in range(10):  # loop over the dataset multiple times
            global_step += 1
            net.train()
            running_loss = 0.0
            running_cls = 0
            running_rec = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # Inputs and labels are sent to the GPU at every step
                inputs, labels = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                logits, recon = net(inputs)
                loss_cls = ce(logits, labels)
                loss_rec = mse(recon, inputs)
                loss = loss_cls + 0.1 * loss_rec
                loss.backward()
                optimizer.step()
                # Statistics
                running_loss += loss.item()
                running_cls += loss_cls.item()
                running_rec += loss_rec.item()
                # Predictions in training epoch
                _, predicted = torch.max(logits, 1)
                correct_batch = (predicted == labels).sum().item()
                total_batch = labels.size(0)
                acc = correct_batch / total_batch  # Correct predictions in a training batch
                # ---- TensorBoard scalar logging ----
                writer.add_scalar("Autoencoder/Total_loss_train", loss.item(), global_step)
                writer.add_scalar("Autoencoder/Decoder_loss_train", loss_rec.item(), global_step)
                writer.add_scalar("Autoencoder/Encoder_loss_train", loss_cls.item(), global_step)
                writer.add_scalar("Autoencoder/Encoder_accuracy_val", acc, global_step)
                # Print loss after every 10 mini-batches
                if i % 10 == 9:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                    running_loss = 0.0
            # Log original vs reconstructed images in Tensorboard per epoch
            net.eval()
            loss_val_epoch = list()
            with torch.no_grad():
                for batch_idx, data in enumerate(valloader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    images = inputs
                    logits, recon = net(images)
                    loss_cls_val = ce(logits, labels)
                    loss_rec_val = mse(recon, inputs)
                    loss = loss_cls_val + 0.1 * loss_rec_val
                    loss_val_epoch.append(loss_rec_val.item()) # Using reconstruction loss as parameter to guide early-stopping
                    # Predictions on validation batches
                    _, predicted = torch.max(logits, 1)
                    correct_val = (predicted == labels).sum().item()
                    total_val = labels.size(0)
                    writer.add_scalar("Autoencoder/Decoder_loss_val", loss_rec_val.item(), global_step)
                    writer.add_scalar("Autoencoder/Encoder_loss_val", loss_cls_val.item(), global_step)
                    writer.add_scalar("Autoencoder/Total_loss_val", loss.item(), global_step)
                    acc = correct_val / total_val  # Correct predictions in a validation epoch
                    writer.add_scalar("Autoencoder/Encoder_accuracy_val", acc, global_step)
                    if batch_idx == 0:
                        batch_size = images.shape[0]  # e.g. 32
                        # Full batch: orig|recon pairs
                        orig_batch = images.cpu()
                        recon_batch = recon.cpu()
                        comparisons = torch.cat([orig_batch, recon_batch], dim=3)  # [32, 1, 128, 256]
                        # Dynamic grid layout (auto-fit batch size)
                        cols = int(np.ceil(np.sqrt(batch_size)))  # ~6x6 for batch=32
                        cols = batch_size // cols
                        grid = vutils.make_grid(comparisons, nrow=cols, normalize=True, scale_each=True)
                        # Get ALL predictions
                        pred_labels = torch.argmax(logits, dim=1).cpu().numpy()[0]
                        true_labels = labels.cpu()[0]
                        # Add labels to grid (PIL overlay)
                        grid_np = grid.permute(1, 2, 0).mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
                        grid_pil = Image.fromarray(grid_np)
                        draw = ImageDraw.Draw(grid_pil)
                        try:
                            font = ImageFont.truetype("arial", 20)
                        except:
                            font = ImageFont.load_default()
                        cell_w = grid_pil.width // cols
                        cell_h = grid_pil.height // cols
                        row, col = 1,1
                        x = col * cell_w + cell_w // 2 - 30
                        y = row * cell_h + cell_h - 20
                        label_text = f"T:{idx_to_class[int(true_labels)]} P:{idx_to_class[int(pred_labels)]}"
                        draw.text((x, y), label_text, fill='black', font=font, stroke_width=0.1, stroke_fill='black')
                        # Back to tensor
                        grid_with_labels = torch.from_numpy(np.array(grid_pil)).permute(2, 0, 1).float() / 255.0
                        # TensorBoard: full batch!
                        writer.add_image(f"Epoch: {epoch}", grid_with_labels, epoch)
                        # SAVE to directory
                        epoch_dir = f"{name}_recon_grids_val"
                        os.makedirs(epoch_dir, exist_ok=True)
                        save_path = f"{epoch_dir}/{epoch}.png"
                        grid_pil.save(save_path, "PNG", dpi=(150, 150))

            loss_val = np.mean(loss_val_epoch)
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                patience_counter = 0
                PATH = f'./{name}_autoencoder_net.pth'
                torch.save(net.state_dict(), PATH)
                print(f"New best validation loss: {best_val_loss:.3f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best validation loss: {best_val_loss:.3f})")
                break  # EXIT THE LOOP

        print('Finished Autoencoder Training')


    # Showing some random testing images
    print("Starting Autoencoder Testing...")
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    print('GroundTruth: ',' '.join(idx_to_class[label.item()] for label in labels))
    imshow(torchvision.utils.make_grid(images))
    # Loading trained model
    if "--color" in sys.argv:
        channels_num = 3
    else:
        channels_num = 1
    net = Net(input_channels=channels_num, num_classes=2, image_size=(128, 128), reconstruct=True)
    PATH = f'./{name}_autoencoder_net.pth'
    net.load_state_dict(torch.load(PATH, weights_only=True))
    net = net.to(device)

    # Predictions
    images = images.to(device)
    labels = labels.to(device)
    outputs, recons = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ',' '.join(idx_to_class[label.item()] for label in labels))
    print(net)

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in class_list}
    total_pred = {classname: 0 for classname in class_list}
    with torch.no_grad():
        for batch, data in enumerate(testloader):
            images, labels = data[0].to(device), data[1].to(device)
            total += len(images)
            outputs, recon = net(images)
            _, predictions = torch.max(outputs, 1)
            print("These predicted", predictions)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            total_batch = labels.size(0)
            correct_batch = (predictions == labels).sum().item()
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[idx_to_class[int(prediction)]] += 1
                total_pred[idx_to_class[int(prediction)]] += 1
            acc = correct_batch / total_batch # Accuracy over testing batch
            writer.add_scalar("Autoencoder/Encoder_accuracy_test", acc, batch+1)

            # Full batch: orig|recon pairs
            orig_batch = images.cpu()
            recon_batch = recon.cpu()
            true_labels = labels.cpu()
            print("These true labels", true_labels)
            comparisons = torch.cat([orig_batch, recon_batch], dim=3)  # [32, 1, 128, 256]
            # Dynamic grid layout (auto-fit batch size)
            cols = int(np.ceil(np.sqrt(batch_size)))  # ~6x6 for batch=32
            cols = batch_size // cols
            grid = vutils.make_grid(comparisons, nrow=cols, normalize=True, scale_each=True)
            # Get ALL predictions
            # Add labels to grid (PIL overlay)
            grid_np = grid.permute(1, 2, 0).mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
            grid_pil = Image.fromarray(grid_np)
            draw = ImageDraw.Draw(grid_pil)
            try:
                font = ImageFont.truetype("arial", 20)
            except:
                font = ImageFont.load_default()
            cell_w = grid_pil.width // cols
            cell_h = grid_pil.height // cols
            for j in range(total_batch):
                row, col = divmod(j, cols)
                x = col * cell_w + cell_w // 2 - 30
                y = row * cell_h + cell_h - 20
                label_text = f"T:{idx_to_class[int(true_labels[j])]} P:{idx_to_class[int(pred_labels[j])]}"
                draw.text((x, y), label_text, fill='black', font=font, stroke_width=0.1, stroke_fill='black')
            # Back to tensor
            grid_with_labels = torch.from_numpy(np.array(grid_pil)).permute(2, 0, 1).float() / 255.0
            # TensorBoard: full batch!
            writer.add_image(f"Batch: {batch+1}", grid_with_labels, batch)
            # SAVE to directory
            batch_dir = f"{name}_recon_grids_test"
            os.makedirs(batch_dir, exist_ok=True)
            save_path = f"{batch_dir}/{batch+1}.png"
            grid_pil.save(save_path, "PNG", dpi=(150, 150))

    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
    # Accuracy for each class
    for classname, correct_count in correct_pred.items():
        try:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        except:
            print(f"No representing entries for {classname:5s} in testset.")

    print(f"Finished Autoencoder Testing on {total} images.")


if __name__ == "__main__":
    main()
    for file in os.listdir(os.getcwd()):
        if ",19" in file:
            os.rename(file, f'results_{name}_model.txt')
        elif "log" in file:
            os.remove(file)