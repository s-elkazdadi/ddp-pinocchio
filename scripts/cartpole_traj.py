import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast
from PIL import Image
from PIL import ImageDraw
import itertools

with open("/tmp/affine_mults_cartpole_traj.dat") as f:
    lines = f.readlines()
    traj = [ast.literal_eval(l) for l in lines]

traj = traj[-1]

SCALE = 200
WIDTH = 5 * SCALE
HEIGHT = 2 * SCALE

LENGTH = 4 * SCALE

Y0 = 5 * SCALE
X0 = 15 * SCALE

alpha = 1
x = 1
theta = np.pi / 4


IMG_WIDTH = 20
IMG_HEIGHT = 10


def make_img(x, theta, cart):
    image = Image.fromarray(
        np.zeros((IMG_HEIGHT * SCALE, IMG_WIDTH * SCALE, 4), dtype=np.uint8)
    )
    draw = ImageDraw.Draw(image)
    if cart:
        draw.rectangle(
            (
                (X0 - WIDTH / 2 + SCALE * x, Y0 - HEIGHT / 2),
                (X0 + WIDTH / 2 + SCALE * x, Y0 + HEIGHT / 2),
            ),
            fill=(43, 140, 190, 255),
            outline="black",
        )
    else:
        xy = (
            X0 + SCALE * x,
            Y0,
            X0 + SCALE * x + LENGTH * np.sin(theta),
            Y0 + LENGTH * np.cos(theta),
        )
        draw.line(xy, fill="black", width=8)
        draw.ellipse(
            (
                X0 - SCALE / 4 + x * SCALE,
                Y0 - SCALE / 4,
                X0 + SCALE / 4 + x * SCALE,
                Y0 + SCALE / 4,
            ),
            fill="black",
        )
        draw.ellipse(
            (
                X0 + SCALE * x + LENGTH * np.sin(theta) - SCALE / 4,
                Y0 + LENGTH * np.cos(theta) - SCALE / 4,
                X0 + SCALE * x + LENGTH * np.sin(theta) + SCALE / 4,
                Y0 + LENGTH * np.cos(theta) + SCALE / 4,
            ),
            fill="black",
        )
    return np.array(image)


fig, ax = plt.subplots(figsize=(IMG_WIDTH, IMG_HEIGHT))
n = 15

a = -0.8
b = 0

imgs = []
for i in range(n):
    idx = max(0, min(len(traj) - 1, int(i / (n - 1) * (len(traj) - 1))))
    print(idx)
    imgs.append(make_img(traj[idx][0] + idx / n * a + b, traj[idx][1], True))

for i, img in enumerate(imgs):
    f = i / (len(imgs) - 1)
    f = f ** 2
    print(f)
    ax.imshow(img, alpha=(0.6 + f * 0.4) ** 2)

imgs = []
for i in range(n):
    idx = max(0, min(len(traj) - 1, int(i / (n - 1) * (len(traj) - 1))))
    print(idx)
    imgs.append(make_img(traj[idx][0] + idx / n * a + b, traj[idx][1], False))

for i, img in enumerate(imgs):
    f = i / (len(imgs) - 1)
    f = f ** 2
    print(f)
    ax.imshow(img, alpha=(0.6 + f * 0.4) ** 2)

ax.set_axis_off()
fig.tight_layout()
fig.savefig("a.png")
