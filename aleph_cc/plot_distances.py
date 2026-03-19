#!usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
from collections import defaultdict

ss_dict = defaultdict(dict)
x,y,labels = [], [], []
for file in os.listdir(sys.argv[1]):
    if ".txt" in file:
        data = pd.read_csv(file,sep="\t",header=0)
        for i in data.index:
            ssij = data.at[i, "ss_ij"].split("-")
            ssij = sorted(list(ssij))
            ssij = f"{('-').join(ssij)}"
            if ssij not in ss_dict.keys():
                ss_dict[ssij] = list()
            dist = float(data.at[i,"dist_ij"])/11
            if dist > 1:
                continue
            ss_dict[ssij].append(dist)
# plot
fig, ax = plt.subplots()
position = 1
for key,val in ss_dict.items():
    ax.boxplot(val,positions=[position],showfliers=False, labels=[key])
    position += 1
ax.legend()
plt.savefig("CCboxplot_distances.png")
plt.show()