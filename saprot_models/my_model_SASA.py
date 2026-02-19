import os.path

from Bio.PDB import PDBParser, MMCIFParser, DSSP
from Bio.PDB.PDBIO import PDBIO
from foldseek_util import get_struc_seq
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from collections import Counter
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

sr = ShrakeRupley()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\n\n")
# Annotate secondary structure with DSSP
def dssp_labels(pdb_id, pdb_path):
    selected_chain=""
    numeric_label = []
    annotated = pd.read_csv(os.path.join(os.getcwd(),sys.argv[1]),header=0, sep="\t")
    chain_options = annotated.at[pdb_id,"CC_chains"].split(",")
    oligo = annotated.at[pdb_id,"oligo"]
    orient = annotated.at[pdb_id,"orient"]
    pdb_id = annotated.at[pdb_id,"pdb"]
    label_map = {"H":0,"B":1,"E":2,"G":3,"I":4,"T":5,"S":6,"P":7,"-":8}
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    sr.compute(structure, level="R")
    pdb_model = structure[0]
    print(pdb_model, pdb_path)
    dssp = DSSP(pdb_model, pdb_path)
    print(len(dssp))
    print("\npdb:", pdb_id,"residues:", len(dssp))
    # Obtain SS labels with DSSP
    for key in dssp.keys():
        chain_id, res_id = key
        if chain_id not in chain_options:
            continue
        if selected_chain == "":
            selected_chain = chain_id
        aa = dssp[key][1]          # Amino acid
        ss = dssp[key][2]          # Secondary structure
    print("\t- Selected chain", selected_chain)
    # DSSP mapping (from string to numerical processable format)
    residue_labels = []
    residue_keys = []
    res_lab_dssp =  []
    for chain in pdb_model:
        chain_id = chain.id
        if chain_id != selected_chain:
            continue
        for residue in chain:
            # only standard residues (skip hetero/water)
            if residue.id[0] != " ":
                continue
            key = (chain_id, residue.id)
            if key in dssp:
                ss = dssp[key][2]
                if ss is None or ss == " ":
                    ss = "-"
            else:
                ss = "-"
            residue_keys.append(key)
            try:
                sasa_res = round(residue.sasa, 2)
            except:
                continue
            if sasa_res <=50:
                residue_labels.append(0)
            else:
                residue_labels.append(1)
            res_lab_dssp.append(ss)

    # Numeric labels array
    labels_np = np.array(residue_labels)
    res_lab_dssp = np.array([label_map.get(ch, 8) for ch in res_lab_dssp], dtype=np.int64)
    print("\t- Built labels from DSSP. n_labels =", labels_np.shape[0])
    return labels_np, selected_chain, res_lab_dssp

# Generate foldseek annotation of structures
def foldseek_labels(pdb_path, selected_chain, labelnums, labels_dssp):
    # Extract the "A" chain from the pdb file and encode it into a struc_seq
    # pLDDT is used to mask low-confidence regions if "plddt_mask" is True. Please set it to True when
    # use AF2 structures for best performance.
    parsed_seqs = get_struc_seq("/home/mari/scripts/vib_leuven/foldseek", pdb_path, [selected_chain], plddt_mask=False)[selected_chain]
    seq, foldseek_seq, combined_seq = parsed_seqs
    combined_seq_masked = "".join([c if c.isupper() else "#" for c in combined_seq])
    print(f"\tInput lengths before selecting only helical residues:")
    print(f"\t- seq length: {len(seq)}")
    #print(f"\t- seq: {seq}")
    print(f"\t- foldseek_seq length: {len(foldseek_seq)}")
    #print(f"\t- foldseek_seq: {foldseek_seq}")
    print(f"\t- combined_seq length: {len(combined_seq)}")
    #print(f"\t- combined_seq: {combined_seq}")
    print(f"\t- combined_seq_masked length: {len(combined_seq_masked)}")
    #print(f"\t- combined_seq_masked: {combined_seq_masked}")

    # Selecting only helical residues
    newseq = str()
    newfoldseek_seq = str()
    newcombined_seq = str()
    newlabels = list()
    if len(seq) != len(labels_dssp):
        raise ValueError(f"Length mismatch: seq {len(seq)}, labels {len(labels_dssp)}")
    else:
        for i, (s, d) in enumerate(zip(seq, labels_dssp)):
            if d == 0:
                newseq+=s
                newfoldseek_seq+=foldseek_seq[i]
                newcombined_seq+=s
                newcombined_seq+=foldseek_seq[i].lower()
                newlabels.append(labelnums[i])
    seq = newseq
    foldseek_seq = newfoldseek_seq
    combined_seq = newcombined_seq
    combined_seq_masked = "".join([c if c.isupper() else "#" for c in combined_seq])
    print(f"\tInput lengths after selecting only helical residues:")
    print(f"\t- seq length: {len(seq)}")
    #print(f"\t- seq: {seq}")
    print(f"\t- foldseek_seq length: {len(foldseek_seq)}")
    #print(f"\t- foldseek_seq: {foldseek_seq}")
    print(f"\t- combined_seq length: {len(combined_seq)}")
    #print(f"\t- combined_seq: {combined_seq}")
    print(f"\t- combined_seq_masked length: {len(combined_seq_masked)}")
    #print(f"\t- combined_seq_masked: {combined_seq_masked}")

    """
    numeric_labels = list()
    for i in range(len(seq)):
        numeric_labels.append(labelnum)
    """
    return seq, foldseek_seq, combined_seq, combined_seq_masked, newlabels

# Load pre-trained SaProt models directly
# Load SaProt small model
def saprot_struc_model():
    tokenizer_struct = AutoTokenizer.from_pretrained("westlake-repl/SaProt_35M_AF2", force_download=False)
    saprot_struct = AutoModelForMaskedLM.from_pretrained("westlake-repl/SaProt_35M_AF2", force_download=False)
    return tokenizer_struct, saprot_struct

# Load SaProt small model (trained on sequence only)
def saprot_seq_model():
    tokenizer_seq = AutoTokenizer.from_pretrained("westlake-repl/SaProt_35M_AF2_seqOnly", force_download=False)
    saprot_seq = AutoModelForMaskedLM.from_pretrained("westlake-repl/SaProt_35M_AF2_seqOnly", force_download=False)
    return tokenizer_seq, saprot_seq

def residue_emb_seq(device, combined_seq_masked):
    tokenizer_seq, saprot_seq = saprot_seq_model()  # SaProt pre-trained model only seq-based
    saprot_seq.eval()
    saprot_seq.to(device)
    seq = combined_seq_masked
    tokens = tokenizer_seq.tokenize(seq)
    inputs = tokenizer_seq(seq, return_tensors="pt").to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = saprot_seq(**inputs, output_hidden_states=True)
    # outputs.hidden_states is a tuple of all layers, including embeddings
    # Shape of each layer: [batch_size, sequence_length, hidden_dim]
    all_hidden_states = outputs.hidden_states
    # Last layer hidden states
    last_hidden = all_hidden_states[-1]  # [1, seq_len, hidden_dim]
    per_residue_embeddings_seq_only_list = []
    for tok_embed, tok in zip(last_hidden[0], tokens):
        per_residue_embeddings_seq_only_list.append(tok_embed)
    seq_only_embeddings_tensor = torch.stack(per_residue_embeddings_seq_only_list)  # [num_residues, hidden_dim]
    print(f"\tShape of sequence-only embeddings: {seq_only_embeddings_tensor.shape}")
    emb_seq = seq_only_embeddings_tensor.detach().cpu().numpy()  # [N, 480]
    return emb_seq

def residue_emb_struct_aware(device, combined_seq):
    tokenizer_struct, saprot_struct = saprot_struc_model()  # SaProt pre-trained model including structural awareness
    saprot_struct.eval()
    saprot_struct.to(device)
    seq = combined_seq
    tokens = tokenizer_struct.tokenize(seq)
    inputs = tokenizer_struct(seq, return_tensors="pt").to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = saprot_struct(**inputs, output_hidden_states=True)
    # outputs.hidden_states is a tuple of all layers, including embeddings
    # Shape of each layer: [batch_size, sequence_length, hidden_dim]
    all_hidden_states = outputs.hidden_states
    # Last layer hidden states
    last_hidden = all_hidden_states[-1]  # [1, seq_len, hidden_dim]
    per_residue_embeddings_struct_aware_list = []
    for tok_embed, tok in zip(last_hidden[0], tokens):
        per_residue_embeddings_struct_aware_list.append(tok_embed)
    struct_aware_embeddings_tensor = torch.stack(per_residue_embeddings_struct_aware_list)  # [num_residues, hidden_dim]
    print(f"\tShape of structure-aware embeddings: {struct_aware_embeddings_tensor.shape}")
    emb_struct = struct_aware_embeddings_tensor.detach().cpu().numpy()  # [N, 480]
    return emb_struct

def train_test_subsets(emb, labels_np):
    num_proteins = len(emb)
    protein_indices = np.arange(num_proteins)
    train_proteins, test_proteins = train_test_split(protein_indices, test_size=0.2, random_state=42)

    # Concatenate residues
    X_train = np.vstack([emb[i] for i in train_proteins])
    y_train = np.hstack([labels_np[i] for i in train_proteins]).astype(int)
    X_test = np.vstack([emb[i] for i in test_proteins])
    y_test = np.hstack([labels_np[i] for i in test_proteins]).astype(int)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class weights:", class_weights)
    return X_train, X_test, y_train, y_test, class_weights

def pytorch_dataset_dataloader(emb_train, emb_test, y_train, y_test):
    class ResidueDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]

    train_dataset = ResidueDataset(emb_train, y_train)
    test_dataset = ResidueDataset(emb_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    return train_loader, test_loader

# MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], num_classes=2, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], num_classes)
        )
    def forward(self, x):
        return self.model(x)
# ---------------------------
# Training function
# ---------------------------
def train_mlp(model, train_loader, test_loader, class_weights, lr=1e-3, epochs=20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    best_preds = None
    best_labels = None
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        # Evaluate
        model.eval()
        preds = []
        labels_eval = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                pred = logits.argmax(dim=1).cpu().numpy()
                preds.extend(pred)
                labels_eval.extend(y_batch.cpu().numpy())
        acc = accuracy_score(labels_eval, preds)
        if acc > best_acc:
            best_acc = acc
            best_preds = preds.copy()
            best_labels = labels_eval.copy()
        print(f"\tEpoch {epoch + 1}/{epochs} - Test accuracy: {acc:.4f} (best: {best_acc:.4f})")
    return model, best_preds, best_labels

def plot_confusion(preds, labels, tag, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Buried","Exposed"], yticklabels=["Buried","Exposed"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.savefig(f"cm_{sys.argv[1].split('.')[0]}_{tag}.png")
    plt.show()

# PCA scatter plot
def PCA_embeddings(embeddings, labels, tag, title="Embedding PCA"):
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    plt.figure(figsize=(6,5))
    sns.scatterplot(
        x=emb_2d[:,0], y=emb_2d[:,1],
        hue=labels,
        palette="tab10",   # geschikt voor meerdere klassen
        s=40, alpha=0.8
    )
    plt.title(title)
    plt.legend(title="SS", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"pca_{sys.argv[1].split('.')[0]}_{tag}.png")
    plt.show()




# EXECUTION
#pdb_path = "/media/mari/Data/vib_leuven/colab_tutorial/example1.pdb"
errors =  open(f"errors_{sys.argv[1].split('.')[0]}.txt","w")
all_labels_num = list()
all_seq_embs = list()
all_struct_aware_embs = list()
#pdbs = open(sys.argv[1], "r").read().splitlines()
pdbs = pd.read_csv(sys.argv[1],sep="\t",header=0)
for i in pdbs.index:
    pdb_id = f"{pdbs.at[i,'pdb']}_{pdbs.at[i,'biomol']}"
    pdb = pdbs.at[i,"pdb"]
    biomol = pdbs.at[i,"biomol"]
    try:
        pdb_path = f"/media/mari/Data/vib_leuven/datasets/cc_sasa/pdb_files/{pdb.upper()}_{biomol}.pdb"
        label_nums, selected_chain, labels_dssp = dssp_labels(i,pdb_path)  # Get true labels with DSSP annotation, labels are converted into numerical
    except:
        errors.write(f"{pdb}\tCould not assign labels\n")
        continue
    try:
        seq, foldseek_seq, combined_seq, combined_seq_masked,label_nums = foldseek_labels(pdb_path, selected_chain, label_nums, labels_dssp)  # Obtain foldseek annotation of structural information
    except:
        errors.write(f"{pdb}\tCould not assign foldseek sequence\n")
        continue
    try:
        seq_emb = residue_emb_seq(device, combined_seq_masked)  # Masked foldseek labels as we want an only-seq approach
    except:
        errors.write(f"{pdb}\tCould not assign sequence embeddings\n")
        continue
    try:
        struct_aware_emb = residue_emb_struct_aware(device, combined_seq)  # Non-masked seq, we want foldseek structural labels
    except:
        errors.write(f"{pdb}\tCould not assign structure aware embeddings\n")
        continue
    """
    print(f"\tShape of labels: {len(labels_num)}")

    if len(labels_num) != seq_emb.shape[0]:
        raise ValueError(f"\tLength mismatch: emb_seq {seq_emb.shape[0]}, labels {len(labels_num)}")
    elif struct_aware_emb.shape[0] != len(labels_num):
        raise ValueError(
            f"\tLength mismatch: emb_struct {struct_aware_emb.shape[0]}, labels {len(labels_num)}")
    else:
        print(f"\t\u2713 Equal embeddings and DSSP labels shape ({struct_aware_emb.shape}, {len(labels_num)}).")
    """
    all_seq_embs.append(seq_emb)
    all_struct_aware_embs.append(struct_aware_emb)
    all_labels_num.append(label_nums)

# Consistency check
print("\nChecking all inputs:\n==========================================================================")
if len(all_seq_embs) != len(all_labels_num):
    raise ValueError(f"Length mismatch: emb_seq {len(all_seq_embs)}, labels {len(all_labels_num)}")
elif len(all_struct_aware_embs) != len(all_labels_num):
    raise ValueError(f"Length mismatch: emb_struct {len(all_struct_aware_embs)}, labels {len(all_labels_num)}")
else:
    print(f"\u2713 Equal embeddings and labels shape ({len(all_seq_embs)}, {len(all_labels_num)}).")

# Only seq-based model
train_seq_emb, test_seq_emb, y_train, y_test, class_weights_seq = train_test_subsets(all_seq_embs, all_labels_num)
train_loader_seq, test_loader_seq = pytorch_dataset_dataloader(train_seq_emb, test_seq_emb, y_train, y_test)
print("Seq-based embeddings shape",train_seq_emb.shape, test_seq_emb.shape)
print(f"Total training residues: {len(y_train)}\nTotal testing residues: {len(y_test)}")
print(f"Labels representation on training set:\n"
      f"\t- {Counter(y_train)}\n"
      f"Labels representation on test set:\n"
      f"\t - {Counter(y_test)}\n")

# Seq + 3Di token inputs model (structure-aware approach)
train_struct_aware_emb, test_struct_aware_emb, y_train, y_test, class_weights_struc = train_test_subsets(all_struct_aware_embs, all_labels_num)
train_loader_struct, test_loader_struct = pytorch_dataset_dataloader(train_struct_aware_emb, test_struct_aware_emb, y_train, y_test)
print("Seq+3Di embeddings shape", train_struct_aware_emb.shape,test_struct_aware_emb.shape)
print(f"Total training residues: {len(y_train)}\nTotal testing residues: {len(y_test)}")
print(f"Labels representation on training set:\n"
      f"\t- {Counter(y_train)}\n"
      f"Labels representation on test set:\n"
      f"\t - {Counter(y_test)}\n")

# MLP model training with own dataset
embedding_dim = all_struct_aware_embs[0].shape[1]

print("Training on sequence-only embeddings...")
mlp_seq = MLP(input_dim=embedding_dim)
mlp_seq, preds_seq, labels_seq = train_mlp(mlp_seq, train_loader_seq, test_loader_seq,class_weights_seq, lr=1e-3, epochs=20)

print("Training on sequence+3Di embeddings...")
mlp_struct = MLP(input_dim=embedding_dim)
mlp_struct, preds_struct, labels_struct = train_mlp(mlp_struct, train_loader_struct, test_loader_struct,class_weights_struc, lr=1e-3, epochs=20)
# MLP model evaluation

print("\nSequence-only classification report:")
print(classification_report(labels_seq, preds_seq, labels=list(range(2))))

print("\nSequence+3Di classification report:")
print(classification_report(labels_struct, preds_struct, labels=list(range(2))))

# Additional plots
# Confusion matrix (true vs predicted labels)
plot_confusion(preds_seq, labels_seq, "seq", title="Sequence-only MLP")
plot_confusion(preds_struct, labels_struct,"str", title="Sequence+3Di MLP")
# PCA scatter plot of test embeddings and DSSP SS labels
PCA_embeddings(test_seq_emb, y_test,"seq",title="Sequence-only embeddings (PCA)")
PCA_embeddings(test_struct_aware_emb, y_test,"str", title="Sequence+3Di embeddings (PCA)")

