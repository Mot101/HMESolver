import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cm = pd.read_csv("confusion_matrix.csv", index_col=0)
#cm = pd.read_csv("confusion_validation_matrix.csv", index_col=0)

cm.index = cm.index.str.replace("dot", ".").str.replace("forward_slash", "/").str.replace("minus", "-").str.replace("plus", "+")
cm.columns = cm.columns.str.replace("dot", ".").str.replace("forward_slash", "/").str.replace("minus", "-").str.replace("plus", "+")

arr = cm.values

plt.figure(figsize=(14, 14))
plt.rcParams.update({'font.size': 16})
im = plt.imshow(arr, interpolation="nearest", cmap="Oranges")  
#im = plt.imshow(arr, interpolation="nearest", cmap="Blues")
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xticks(range(len(cm.columns)), cm.columns, rotation=60, ha="right")
plt.yticks(range(len(cm.index)), cm.index)
plt.xlabel("Predicted")
plt.ylabel("True")

plt.title("Confusion matrix")

for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        plt.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=16)

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


