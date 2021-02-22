import numpy as np
import matplotlib.pyplot as plt
import sys

COLORS = ["#67a9cf", "#ca0020"]

fig, ax = plt.subplots(figsize=(4, 2))
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)

with open("result_data/cart_const.dat") as f:
    err_c = eval(f.readline())
with open("result_data/cart_affine.dat") as f:
    err_a = eval(f.readline())

for eps in err_c.keys():
    err_c[eps] = np.array(err_c[eps]).mean()
for eps in err_a.keys():
    err_a[eps] = np.array(err_a[eps]).mean()

ax.loglog(
    list(err_c.keys()),
    list(err_c.values()),
    label="λ: constant",
    color=COLORS[0],
)
ax.loglog(
    list(err_a.keys()),
    list(err_a.values()),
    label="λ: affine",
    color=COLORS[1],
)
ax.legend()
ax.set_xlabel("Noise scale")
ax.set_ylabel("Constraint error")

fig.tight_layout()
fig.savefig("noise.pdf")


n_iter = 80

fig, ax = plt.subplots(1, figsize=(5, 3), sharex=True)
ax = [ax]
ax[0].spines["top"].set_visible(True)
ax[0].spines["right"].set_visible(True)
# ax[1].spines["top"].set_visible(True)
# ax[1].spines["right"].set_visible(True)


def plot(title, path_prefix, ls):
    primal = path_prefix + "_primal.dat"
    dual = path_prefix + "_dual.dat"
    mu = path_prefix + "_mu.dat"
    cost = path_prefix + "_cost.dat"

    with open(primal) as f:
        primal = list(map(float, f.readlines()))[:n_iter]
    with open(dual) as f:
        dual = list(map(float, f.readlines()))[:n_iter]
    with open(mu) as f:
        mu = list(map(float, f.readlines()))[:n_iter]
    with open(cost) as f:
        cost = list(map(float, f.readlines()))[:n_iter]

    ax[0].semilogy(
        primal, label=title + ": Primal error", color=COLORS[0], linestyle=ls
    )
    ax[0].semilogy(dual, label=title + ": Dual error", color=COLORS[1], linestyle=ls)

    # ax[1].plot(cost, label=title, color=COLORS[1], linestyle=ls)


prefix = sys.argv[1]
plot(
    "λ: constant",
    prefix + "/cartpole",
    ":",
)
plot(
    "λ: affine",
    prefix + "/affine_mults_cartpole",
    "-",
)

ax[0].legend()
# ax[0].set_ylabel("")
# ax[1].legend()
ax[0].set_xlabel("Iteration")
# ax[1].set_ylabel("Cost value")

fig.tight_layout()
fig.savefig(f"error_{sys.argv[2]}.pdf")
