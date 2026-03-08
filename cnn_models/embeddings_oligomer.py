import sys
import torch
from torcheval.metrics.functional import binary_f1_score, multiclass_f1_score
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
import random
from sklearn.decomposition import PCA, SparsePCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image, ImageDraw, ImageFont
from imblearn.under_sampling import RandomUnderSampler
from Bio.PDB import PDBParser, MMCIFParser, DSSP
from collections import defaultdict, Counter
import itertools
parser = PDBParser(QUIET=True)

# Global values
source = sys.argv[1]
name = sys.argv[2]
errors = open(f"{name}_errors.txt", "w")

def calculateOrientation(coords):
    vecs = {}
    chains = list()
    for chain, data in coords.items():
        first = data["residues"][0]
        last = data["residues"][-1]
        vecs[chain] = last - first
        chains.append(chain)
    vA = vecs[chains[0]]
    vB = vecs[chains[1]]
    # scalar (dot) product
    scalar = np.dot(vA, vB)
    # optional: angle between chains in degrees
    angle = np.degrees(
        np.arccos(scalar / (np.linalg.norm(vA) * np.linalg.norm(vB)))
    )
    print("v1:", vA)
    print("v2:", vB)
    print("dot:", scalar)
    print("angle (deg):", angle)
    if angle >= 100:
        return "antiparallel"
    else:
        return "parallel"

def calculateDistance(coords, residues):
    coords_dict = defaultdict(dict)
    for i_r,res in enumerate(residues):
        cat = res.split("_")[1]
        if cat not in coords_dict.keys():
            coords_dict[cat]["coords"] = list()
        coords_dict[cat]["coords"].append(coords[i_r])
    # Compute centroid for each helix (exactly 2)
    centroids = {}
    for key, val in coords_dict.items():
        coord_array = np.array(val["coords"])
        centroid = np.mean(coord_array, axis=0)
        centroids[key] = centroid

    # Exactly 2 helices: compute single distance
    helix_keys = list(centroids.keys())
    c1, c2 = centroids[helix_keys[0]], centroids[helix_keys[1]]
    dist = np.linalg.norm(c1 - c2)

    return dist


def is_continuous_helix(chain_residues, max_gap=0):
    """Check if residue numbers are continuous within chain"""
    if not chain_residues:
        return False

    # Extract residue numbers for this chain
    res_numbers = [int(r.split("_")[0]) for r in chain_residues]
    res_numbers.sort()

    # Check consecutive (allow 1-residue gaps for missing density)
    for i in range(1, len(res_numbers)):
        if res_numbers[i] - res_numbers[i - 1] != 1:
            return False
    return True

def randomColor():
    # Generating a random number in between 0 and 2^24
    color = random.randrange(0, 2 ** 24)
    # Converting that number from base-10
    # (decimal) to base-16 (hexadecimal)
    hex_color = hex(color)
    if hex_color == "#000000" or hex_color == "#ffffff":
        again = randomColor()
    return hex_color
    return hex_color

def generateImages(file, pdb_dir, fragmented=False):
    global name
    positive = open(f"/media/mari/Data/vib_leuven/orient_tmb/testing/pdb_withCC.txt","r").read().splitlines()
    if "--fragments" in sys.argv:
        fragmented = True
    data = pd.read_csv(file, header=0, sep="\t")
    if f"{name}_imgs" not in os.listdir():
        os.mkdir(f"{name}_imgs")
        os.mkdir(f"{name}_imgs/projected")
        os.mkdir(f"{name}_imgs/connected")
        os.mkdir(f"{name}_pymol_sessions")
    for idx in data.index:
        code = data.iloc[idx, 0].upper()
        #if code not in positive:
        #    pass
        cc_chains = data.at[idx,"CC_chains"].split(",")
        if len(list(set(cc_chains))) != int(data.at[idx,"helixCount"]):
            continue
        label = data.at[idx,"helixCount"]
        if str(label) == "2":
            label = "dimer"
        elif str(label) == "3":
            label = "trimer"
        elif str(label) == "4":
            label = "tetramer"
        #label = data.iloc[idx, 1]
        projected = list()
        coordsCA = list()
        chainsRes = list()
        residues_list = list()
        #coordsLateral = list()
        pymol_session = open(f"{name}_pymol_sessions/{code}.pml", "w")
        pymol_session.write(f"cmd.load('{os.path.join(pdb_dir, f"{code}.pdb")}')\n")
        pymol_session.write("cmd.color('000000', 'all')\n")
        pymol_session.write("cmd.color('ffffff', 'ss h')\n")
        if "--parsehelices" in sys.argv:
            helicesDict = defaultdict(dict)
            helixCount = 1
            with open(f"{os.path.join(pdb_dir, f"{code}.pdb")}","r") as file:
                file = file.read().splitlines()
                for line in file:
                    if line.startswith("HELIX"):
                        helixChain = line[19]
                        helicesDict[helixCount]["residues"] = list()
                        start = int(line[21:25])
                        end = int(line[33:37])
                        for i in range(start,end+1):
                            helicesDict[helixCount]["residues"].append(f"{i}_{helixChain}")
                        helixCount+=1
        try:
            structure = parser.get_structure(code, os.path.join(pdb_dir, f"{code}.pdb"))
            dssp = DSSP(structure[0], os.path.join(pdb_dir, f"{code}.pdb"))
        except:
            errors.write(f"{code}\tCould not process file.\n")
            continue
        print("\npdb:", code, "residues:", len(dssp))
        for chain in structure[0]:
            #if chain.id not in cc_chains:
            #    continue
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
                            if "--parsehelices" in sys.argv:
                                for key,val in helicesDict.items():
                                    if f"{residue.id[1]}_{chain.id}" in val["residues"]:
                                        residues_list.append(f"{residue.id[1]}_{key}_{chain.id}")
                                        break
                            else:
                                residues_list.append(f"{residue.id[1]}_{chain.id}")
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

            #projected_str = coordsCA
        except:
            print("Error")
            errors.write(f"{code}\tCould not extract coordinates/perform PCA\n")
            continue
        # projected_lateral = pca.transform(coordsLateral)
        TARGET_PIXELS = 128
        PIXELS_PER_ANGSTROM = 5.0
        ANGSTROM_SPAN = TARGET_PIXELS / PIXELS_PER_ANGSTROM  # 12.8 Å
        if fragmented:
            window = 15
            fragments = 0
            used_residues_global = set()

            all_projected = list(projected_str)
            all_residues = list(residues_list)

            for index in range(len(all_projected)):
                center_x = all_projected[index][0]
                if all_residues[index] in used_residues_global:
                    continue

                hexcolor = randomColor()
                projected_list = []
                proj_residues_list = []
                used_chains = []
                chainresi = defaultdict(list)

                # grow fragment
                for p, r in zip(all_projected, all_residues):
                    if not ((center_x + window) >= p[0] >= (center_x - window)):
                        continue
                    if r in used_residues_global:
                        continue

                    chain = r.split("_")[1]
                    if len(chainresi[chain]) >= 8:
                        continue

                    projected_list.append(p)
                    proj_residues_list.append(r)
                    used_chains.append(chain)
                    chainresi[chain].append(r)

                if len(proj_residues_list) < 16:
                    continue

                counts = Counter(used_chains)
                if any(val < 8 for val in counts.values()):
                    continue

                # continuity per chain
                chain_residues = defaultdict(list)
                for r in proj_residues_list:
                    c = r.split("_")[1]
                    chain_residues[c].append(r)

                if not all(
                        is_continuous_helix(sorted(rs, key=lambda x: int(x.split("_")[0])))
                        for rs in chain_residues.values()
                ):
                    continue

                projected = np.array(projected_list)
                proj_residues = np.array(proj_residues_list)

                # re‑orient fragment
                #pca = PCA(n_components=3, random_state=312)
                #pca.fit(projected)
                #projected = pca.transform(projected)

                used_residues_global.update(proj_residues)
                fragments += 1

                # PyMOL commands as you had
                pymol_session.write(f"cmd.select('{fragments}', None)\n")
                for resi in proj_residues:
                    num = resi.split("_")[0]
                    chain = resi.split("_")[-1]
                    pymol_session.write(
                        f"cmd.select('{fragments}', '{fragments} or resi {num} and chain {chain}')\n"
                    )
                pymol_session.write(f"cmd.create('fragment{fragments}', '{fragments}')\n")
                pymol_session.write(f"cmd.color('{hexcolor}', 'fragment{fragments}')\n")
                pymol_session.write(f'cmd.set("cartoon_transparency", 0.7, "{code}")\n')

                # centered 2D projection (unchanged)
                x_center = projected[:, 0].mean()
                y_center = projected[:, 1].mean()
                z_center = projected[:, 2].mean()
                x_min = x_center - ANGSTROM_SPAN / 2
                x_max = x_center + ANGSTROM_SPAN / 2
                y_min = y_center - ANGSTROM_SPAN / 2
                y_max = y_center + ANGSTROM_SPAN / 2
                z_min = z_center - ANGSTROM_SPAN / 2
                z_max = z_center + ANGSTROM_SPAN / 2

                dpi = TARGET_PIXELS
                fig = plt.figure(figsize=(1, 1), dpi=dpi)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.set_aspect("equal")
                ax.axis("off")
                ax.set_xlim(y_min, y_max)
                ax.set_ylim(z_min, z_max)

                coords_dict = defaultdict(dict)
                for i in range(len(projected) - 1):
                    chain1 = proj_residues[i].split("_")[1]
                    chain2 = proj_residues[i + 1].split("_")[1]
                    num1 = int(proj_residues[i].split("_")[0])
                    num2 = int(proj_residues[i + 1].split("_")[0])
                    if chain1 not in coords_dict:
                        coords_dict[chain1]["residues"] = []
                    if chain2 not in coords_dict:
                        coords_dict[chain2]["residues"] = []
                    coords_dict[chain1]["residues"].append(projected[i])
                    if chain1 == chain2 and num1 + 1 == num2:
                        ax.plot(projected[i:i + 2, 1], projected[i:i + 2, 2],
                                color="black", linewidth=0.5)

                connected_path = f"{name}_imgs/connected/{code}_{fragments}_{label}.png"
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
            #ax.scatter(projected[:, 1], projected[:, 2], c="#00000050", marker=".")
            fig.savefig(projected_path, transparent=True)
            # Single connected
            for i in range(len(projected) - 1):
                ax.plot(projected[i:i + 2, 1], projected[i:i + 2, 2], color="#00000050")
            fig.savefig(connected_path, transparent=True)
            plt.close(fig)

if "--embeddings" in sys.argv:
    generateImages(file=source,
                   #pdb_dir="/media/mari/Data/vib_leuven/datasets/cc_membprot/pdb_files",
                   pdb_dir="/media/mari/Data/vib_leuven/datasets/cc_sasa/biomols",
                   fragmented=False)