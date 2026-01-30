import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


df = pd.read_csv("class_f1.csv")

df["Class"] = df["Class"].str.replace("dot", ".").str.replace("forward_slash", "/").str.replace("minus", "-").str.replace("plus", "+")

df = df.sort_values("F1", ascending=True) 
y = np.arange(len(df))

plt.figure(figsize=(16, max(4, 0.7 * len(df))))
plt.rcParams.update({'font.size': 16})
plt.barh(y, df["F1"].values)
plt.yticks(y, df["Class"].values)
plt.xlabel("F1")
plt.title("F1 per class")

for i, v in enumerate(df["F1"].values):
    plt.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=16)

plt.xlim(0, 1)
plt.tight_layout()
plt.show()
