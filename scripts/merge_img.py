import numpy as np
import matplotlib.pyplot as plt
import sys

image_files = sys.argv[1:]
all_images = [plt.imread(file) for file in image_files]
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_axis_off()


idx = [0, 12, 24, 37]

for i in range(3):
    images = all_images[idx[i] : idx[i + 1] + 1]
    for j, img in enumerate(images):
        f = j / (len(images) - 1)
        f = (0.6 + 0.4*f)**2
        print(f)
        ax.imshow(img, alpha=f)
    fig.tight_layout()
    fig.savefig(f"merge_ur5_{i}.pdf")
    fig.savefig(f"merge_ur5_{i}.png")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_axis_off()
