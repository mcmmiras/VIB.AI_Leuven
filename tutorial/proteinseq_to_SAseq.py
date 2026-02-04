#!usr/bin/env python3.10

# Usage: run script from SaProt repository directory

import sys
from pathlib import Path

# Ruta absoluta al repo SaProt
sa_path = Path("/home/mari/repositories/SaProt")
sys.path.append(str(sa_path))  # ahora Python puede importar desde SaProt

from utils.foldseek_util import get_struc_seq

pdb_path = "example/8ac8.cif"

# Extract the "A" chain from the pdb file and encode it into a struc_seq
# pLDDT is used to mask low-confidence regions if "plddt_mask" is True. Please set it to True when
# use AF2 structures for best performance.
parsed_seqs = get_struc_seq("bin/foldseek", pdb_path, ["A"], plddt_mask=False)["A"]
seq, foldseek_seq, combined_seq = parsed_seqs

print(f"seq: {seq}")
print(f"foldseek_seq: {foldseek_seq}")
print(f"combined_seq: {combined_seq}")