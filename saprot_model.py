from Bio.PDB import PDBParser, MMCIFParser, DSSP
from foldseek_util import get_struc_seq
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Annotate secondary structure with DSSP
def dssp_labels(pdb_id, pdb_path):
    label_map = {"H":0,"B":1,"E":2,"G":3,"I":4,"T":5,"S":6,"P":7,"-":8}
    parser = PDBParser()
    structure = parser.get_structure("prot", pdb_path)
    pdb_model = structure[0]
    dssp = DSSP(pdb_model, pdb_path)
    print("pdb:", pdb_id,"residues:", len(dssp))
    # Obtain SS labels with DSSP
    for key in dssp.keys():
        chain_id, res_id = key
        aa = dssp[key][1]          # Amino acid
        ss = dssp[key][2]          # Secondary structure
        print(chain_id, res_id, aa, ss)
    # DSSP mapping (from string to numerical processable format)
    residue_labels = []
    residue_keys = []
    for chain in pdb_model:
        chain_id = chain.id
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
            residue_labels.append(ss)
    # Numeric labels array
    labels_np = np.array([label_map.get(ch, 8) for ch in residue_labels], dtype=np.int64)
    print("Built labels from DSSP. n_labels =", labels_np.shape[0])
    return labels_np

# Generate foldseek annotation of structures
def foldseek_labels(pdb_path):
    # Extract the "A" chain from the pdb file and encode it into a struc_seq
    # pLDDT is used to mask low-confidence regions if "plddt_mask" is True. Please set it to True when
    # use AF2 structures for best performance.
    parsed_seqs = get_struc_seq("/home/mari/repositories/vib_leuven/foldseek", pdb_path, ["A"], plddt_mask=False)["A"]
    seq, foldseek_seq, combined_seq = parsed_seqs
    combined_seq_masked = "".join([c if c.isupper() else "#" for c in combined_seq])
    print(f"seq length: {len(seq)}")
    print(f"seq: {seq}")
    print(f"foldseek_seq length: {len(foldseek_seq)}")
    print(f"foldseek_seq: {foldseek_seq}")
    print(f"combined_seq length: {len(combined_seq)}")
    print(f"combined_seq: {combined_seq}")
    print(f"combined_seq_masked length: {len(combined_seq_masked)}")
    print(f"combined_seq_masked: {combined_seq_masked}")
    return seq, foldseek_seq, combined_seq, combined_seq_masked

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
    print("Tokens:", tokens)
    inputs = tokenizer_seq(seq, return_tensors="pt").to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = saprot_seq(**inputs, output_hidden_states=True)
    # outputs.hidden_states is a tuple of all layers, including embeddings
    # Shape of each layer: [batch_size, sequence_length, hidden_dim]
    all_hidden_states = outputs.hidden_states
    # Last layer hidden states
    last_hidden = all_hidden_states[-1]  # [1, seq_len, hidden_dim]
    print(last_hidden.shape)
    print(last_hidden)
    per_residue_embeddings_seq_only_list = []
    for tok_embed, tok in zip(last_hidden[0], tokens):
        per_residue_embeddings_seq_only_list.append(tok_embed)
    seq_only_embeddings_tensor = torch.stack(per_residue_embeddings_seq_only_list)  # [num_residues, hidden_dim]
    print(f"Shape of sequence-only embeddings: {seq_only_embeddings_tensor.shape}")
    emb_seq = seq_only_embeddings_tensor.detach().cpu().numpy()  # [N, 480]
    return emb_seq

def residue_emb_struct_aware(device, combined_seq):
    tokenizer_struct, saprot_struct = saprot_struc_model()  # SaProt pre-trained model including structural awareness
    saprot_struct.eval()
    saprot_struct.to(device)
    seq = combined_seq
    tokens = tokenizer_struct.tokenize(seq)
    print(tokens)
    inputs = tokenizer_struct(seq, return_tensors="pt").to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = saprot_struct(**inputs, output_hidden_states=True)
    # outputs.hidden_states is a tuple of all layers, including embeddings
    # Shape of each layer: [batch_size, sequence_length, hidden_dim]
    all_hidden_states = outputs.hidden_states
    # Last layer hidden states
    last_hidden = all_hidden_states[-1]  # [1, seq_len, hidden_dim]
    print(last_hidden.shape)
    print(last_hidden)
    per_residue_embeddings_struct_aware_list = []
    for tok_embed, tok in zip(last_hidden[0], tokens):
        per_residue_embeddings_struct_aware_list.append(tok_embed)
    struct_aware_embeddings_tensor = torch.stack(per_residue_embeddings_struct_aware_list)  # [num_residues, hidden_dim]
    print(f"Shape of structure-aware embeddings: {struct_aware_embeddings_tensor.shape}")
    emb_struct = struct_aware_embeddings_tensor.detach().cpu().numpy()  # [N, 480]
    return emb_struct

def train_test_subsets(emb, labels_np):
    indices = np.arange(labels_np.shape[0])
    try:
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels_np)
    except:
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    y_train = labels_np[train_idx]
    y_test = labels_np[test_idx]
    emb_train = emb[train_idx]
    emb_test = emb[test_idx]
    print("Train/test sizes:", len(train_idx), len(test_idx))
    return emb_train, emb_test, y_train, y_test

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
    def __init__(self, input_dim, hidden_dims=[256, 128], num_classes=9, dropout=0.2):
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
def train_mlp(model, train_loader, test_loader, lr=1e-3, epochs=20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.CrossEntropyLoss()
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
        print(f"Epoch {epoch + 1}/{epochs} - Test accuracy: {acc:.4f} (best: {best_acc:.4f})")
    return model, best_preds, best_labels

def plot_confusion(preds, labels, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['H','B','E','G','I','T','S','P','-'],
                yticklabels=['H','B','E','G','I','T','S','P','-'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()

# PCA scatter plot
def PCA_embeddings(embeddings, labels, title="Embedding PCA"):
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
    plt.show()




pdb_path = "/media/mari/Data/vib_leuven/colab_tutorial/example1.pdb"
pdb_id = "prot"

dssp_annot_num = dssp_labels(pdb_id,pdb_path)  # Get true labels with DSSP annotation, labels are converted into numerical
print(dssp_annot_num)

seq, foldseek_seq, combined_seq, combined_seq_masked = foldseek_labels(pdb_path)  # Obtain foldseek annotation of structural information

seq_emb = residue_emb_seq(device, combined_seq_masked)  # Masked foldseek labels as we want an only-seq approach
struct_aware_emb = residue_emb_struct_aware(device, combined_seq)  # Non-masked seq, we want foldseek structural labels

# Consistency check
if seq_emb.shape[0] != dssp_annot_num.shape[0]:
    raise ValueError(f"Length mismatch: emb_seq {seq_emb.shape[0]}, labels {dssp_annot_num.shape[0]}")
elif struct_aware_emb.shape[0] != dssp_annot_num.shape[0]:
    raise ValueError(f"Length mismatch: emb_struct {struct_aware_emb.shape[0]}, labels {dssp_annot_num.shape[0]}")
else:
    print("\u2713 Equal embeddings and DSSP labels shape.")

train_seq_emb, test_seq_emb, y_train, y_test = train_test_subsets(seq_emb, dssp_annot_num)
train_struct_aware_emb, test_struct_aware_emb, y_train, y_test = train_test_subsets(struct_aware_emb, dssp_annot_num)

train_loader_seq, test_loader_seq = pytorch_dataset_dataloader(train_seq_emb, test_seq_emb, y_train, y_test)
train_loader_struct, test_loader_struct = pytorch_dataset_dataloader(train_struct_aware_emb, test_struct_aware_emb, y_train, y_test)

# MLP model training with own dataset
embedding_dim = seq_emb.shape[1]
print("Training on sequence-only embeddings...")
mlp_seq = MLP(input_dim=embedding_dim)
mlp_seq, preds_seq, labels_seq = train_mlp(mlp_seq, train_loader_seq, test_loader_seq, lr=1e-3, epochs=20)
print("Training on sequence+3Di embeddings...")
mlp_struct = MLP(input_dim=embedding_dim)
mlp_struct, preds_struct, labels_struct = train_mlp(mlp_struct, train_loader_struct, test_loader_struct, lr=1e-3, epochs=20)

# MLP model evaluation
print("\nSequence-only classification report:")
print(classification_report(labels_seq, preds_seq, labels=list(range(9))))
print("\nSequence+3Di classification report:")
print(classification_report(labels_struct, preds_struct, labels=list(range(9))))

# Additional plots
# Confusion matrix (true vs predicted labels)
plot_confusion(preds_seq, labels_seq, title="Sequence-only MLP")
plot_confusion(preds_struct, labels_struct, title="Sequence+3Di MLP")
# PCA scatter plot of test embeddings and DSSP SS labels
PCA_embeddings(test_seq_emb, y_test, title="Sequence-only embeddings (PCA)")
PCA_embeddings(test_struct_aware_emb, y_test, title="Sequence+3Di embeddings (PCA)")