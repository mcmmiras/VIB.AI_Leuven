import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
from transformers import EsmTokenizer, EsmForMaskedLM

#model_path = "/your/path/to/SaProt_650M_AF2" # Note this is the directory path of SaProt, not the ".pt" file
model_path = "westlake-repl/SaProt_35M_AF2"
tokenizer = EsmTokenizer.from_pretrained(model_path)
model = EsmForMaskedLM.from_pretrained(model_path)

#################### Example ####################
device = "cuda"
model.to(device)

seq = "M#EvVpQpL#VyQdYaKv" # Here "#" represents lower plDDT regions (plddt < 70)
seq = "NdSfPqRkIaVkQfSaNvDqLqTlElAkAaYwSeLaSaRpDlQlKvRlMvLvYsLvFqVlDsQcIlRaKvSvDvDqGqIkCdEkIdHaVlAvKvYsAcEvIvFlGvLhTdSsAvEvAsSvKvDsIvRvQvArLlKvSvFlApGpKtEwViVaFgYvYhEdSiFdPrWqFfIpKdPrAwHdSaPpSdRvGrLmYiSiVtHgIgNdPsYvLrIsPvFrFrIpGpLdQdNdRnFmTfQiFdRgLpSqEqTcKsEqIqTrNgPvYlAlMvRsLvYvErSvLqCsQsYaRaKdPpDqGqSwGgIkVdSkLdKwIlDvWsIsIcErRrYnQvLdPdQpSqYcQrRpMpPvDsFvRcRpRvFpLvQpVvCsVlNvErIsNcSvRrTgPqMkRnLkSdYwIdEwKdKdKdGpRpQdTtTtHiIiVmFmStFiRgDgIvTvSd"
tokens = tokenizer.tokenize(seq)
print(tokens)

inputs = tokenizer(seq, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model(**inputs)
print(outputs.logits.shape)

"""
['M#', 'Ev', 'Vp', 'Qp', 'L#', 'Vy', 'Qd', 'Ya', 'Kv']
torch.Size([1, 11, 446])
"""