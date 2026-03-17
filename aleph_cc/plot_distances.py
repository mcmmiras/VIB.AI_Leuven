#!usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
from collections import defaultdict

ss_dict = defaultdict(dict)
x,y,labels = [], [], []
for file in os.listdir(sys.argv[1]):
    data = pd.read_csv(file,sep="\t",header=0)
    for i in data.index:
        if data.at[i,"ss_ij"] not in ss_dict.keys():
            ss_dict[data.at[i,"ss_ij"]] = list()
        ss_dict[data.at[i,"ss_ij"]].append(data.at[i,"dist_ij"])
# plot
print(ss_dict)
fig, ax = plt.subplots()
position = 1
for key,val in ss_dict.items():
    ax.boxplot(val,positions=[position],showfliers=False, labels=[key])
    position += 1
ax.legend()
plt.show()