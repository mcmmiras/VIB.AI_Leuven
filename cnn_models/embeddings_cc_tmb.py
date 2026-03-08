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
    print(hex_color)
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

            # get unique chains in order of appearance
            chains = []
            for r in residues_list:
                c = r.split("_")[1]
                if c not in chains:
                    chains.append(c)

            # loop over all chain pairs, but always in original order
            for chain1, chain2 in itertools.combinations(chains, 2):
                used_residues_pair = []  # only for this pair
                # pair-specific view, but keeping global order
                pair_projected_str = []
                pair_residues_list = []
                for p, r in zip(projected_str, residues_list):
                    c = r.split("_")[1]
                    if c in (chain1, chain2):
                        pair_projected_str.append(p)
                        pair_residues_list.append(r)
                # scan in order
                for index in range(len(pair_projected_str)):
                    chain1resi = list()
                    chain2resi = list()
                    counter = pair_projected_str[index][0]
                    if pair_residues_list[index] in used_residues_pair:
                        continue

                    hexcolor = randomColor()
                    projected_list = []
                    proj_residues_list = []
                    used_chains = []


                    # Grow fragment around this center
                    for p, r in zip(pair_projected_str, pair_residues_list):
                        if not ((counter + window) >= p[0] >= (counter - window)):
                            continue
                        if r in used_residues_pair:
                            continue

                        chain = r.split("_")[1]
                        # Skip if already 2 chains used
                        if chain not in used_chains and len(set(used_chains)) == 2:
                            continue

                        # STOP if this chain would exceed 8 residues
                        if chain == chain1 and len(chain1resi) >= 8:
                            continue
                        if chain == chain2 and len(chain2resi) >= 8:
                            continue

                        # Add residue
                        projected_list.append(p)
                        proj_residues_list.append(r)
                        used_chains.append(chain)

                        # Track per-chain residues
                        if chain == chain1:
                            chain1resi.append(r)  # Track residue ID, not coord
                        elif chain == chain2:
                            chain2resi.append(r)

                        # Stop if BOTH chains hit 8 (optional: early termination)
                        if len(chain1resi) >= 8 and len(chain2resi) >= 8:
                            break

                    # Filters (unchanged)
                    if len(proj_residues_list) < 16:
                        continue
                    if len(set(used_chains)) != 2:
                        continue
                    distance = calculateDistance(projected_list, proj_residues_list)
                    if distance > 12:
                        continue
                    counts = Counter(used_chains)
                    if any(val < 8 for val in counts.values()):
                        continue

                    # NEW: CONTINUITY CHECK
                    chain_residues = defaultdict(list)
                    for r in proj_residues_list:
                        chain = r.split("_")[1]
                        chain_residues[chain].append(r)

                    continuous = all(is_continuous_helix(residues) for residues in chain_residues.values())
                    if not continuous:
                        continue

                    # Fragment accepted - proceed with PCA, plotting...
                    # here proj_residues_list is still in original order
                    projected = np.array(projected_list)
                    proj_residues = np.array(proj_residues_list)
                    # Re-orient fragments
                    pca = PCA(n_components=3, random_state=312)
                    pca.fit(projected)
                    projected = pca.transform(projected)  # Already centered in centroid of protein
                    used_residues_pair.extend(proj_residues)
                    fragments += 1

                    print(f"Fragment from {code}_{fragments} ({chain1}-{chain2}):", proj_residues)
                    pymol_session.write(f"cmd.select('{fragments}', None)\n")
                    for resi in proj_residues:
                        num = resi.split("_")[0]
                        chain = resi.split("_")[-1]
                        #pymol_session.write(f"cmd.color('{hexcolor}', 'resi {num} and chain {chain}')\n")
                        pymol_session.write(f"cmd.select('{fragments}', '{fragments} or resi {num} and chain {chain}')\n")
                    pymol_session.write(f"cmd.create('fragment{fragments}', '{fragments}')\n")
                    pymol_session.write(f"cmd.color('{hexcolor}', 'fragment{fragments}')\n")
                    pymol_session.write(f'cmd.set("cartoon_transparency", 0.7, "{code}")\n')
                    #if len(projected) >= 7 * int(classes[label]):

                    # center 12.8 Å window on fragment
                    x_center = projected[:, 0].mean()
                    y_center = projected[:, 1].mean()
                    z_center = projected[:, 2].mean()
                    x_min = x_center - ANGSTROM_SPAN/2
                    x_max = x_center + ANGSTROM_SPAN/2
                    y_min = y_center - ANGSTROM_SPAN/2
                    y_max = y_center + ANGSTROM_SPAN/2
                    z_min = z_center - ANGSTROM_SPAN/2
                    z_max = z_center + ANGSTROM_SPAN/2
                    dpi = TARGET_PIXELS  # 1 inch = 128 px
                    fig = plt.figure(figsize=(1, 1), dpi=dpi)  # 128×128 px output [web:24][web:26]
                    # axes that fill the whole figure: [left, bottom, width, height] in 0–1
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.set_aspect("equal")
                    ax.axis("off")
                    ax.set_xlim(x_min, x_max)
                    #ax.set_xlim(y_min, y_max)
                    ax.set_ylim(y_min, y_max)  # 12.8 Å in each direction -> 5 px/Å
                    # Get residue names for this fragment (in same order as projected)
                    resnames = proj_residues
                    #types_in_fragment = [residue_types.get(resname, 'unknown') for resname in resnames]
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
                    #colors = [color_map.get(res_type, '#808080') for res_type in types_in_fragment]
                    # Now plot with per-point colors:
                    """
                    ax.scatter(projected[:, 2], projected[:, 0],
                               c=colors, s=20, marker='.')
                    fig.savefig(projected_path, transparent=True)
                    """
                    coords_dict = defaultdict(dict)
                    print(len(projected), len(proj_residues))
                    if len(proj_residues) >=0:
                        for i in range(len(projected) - 1):
                            if i+1 < len(projected):
                                chain1 = proj_residues[i].split("_")[1]
                                chain2 = proj_residues[i+1].split("_")[1]
                                num1 = proj_residues[i].split("_")[0]
                                num2 = proj_residues[i+1].split("_")[0]
                                if chain1 not in coords_dict.keys():
                                    coords_dict[chain1]["residues"] = list()
                                if chain2 not in coords_dict.keys():
                                    coords_dict[chain2]["residues"] = list()
                                coords_dict[chain1]["residues"].append(projected[i])
                                #ax.scatter(projected[:, 0], projected[:, 1], c="black", marker=".", s=3)
                                if chain1 == chain2:
                                    if int(num1)+1 == int(num2):
                                        ax.plot(projected[i:i + 2, 0], projected[i:i + 2, 1], color="black", linewidth=0.5)
                        #orientation = calculateOrientation(coords_dict)
                        #projected_path = f"{name}_imgs/projected/{code}_{fragments}_{orientation}.png"
                        #connected_path = f"{name}_imgs/connected/{code}_{fragments}_{orientation}.png"
                        connected_path = f"{name}_imgs/connected/{code.upper()}_{fragments}_cc.png"
                        fig.savefig(connected_path, transparent=True)
                        plt.close(fig)
                        print(coords_dict)
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