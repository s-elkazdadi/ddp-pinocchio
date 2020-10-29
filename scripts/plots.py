import numpy as np
import matplotlib.pyplot as plt

COLORS = ["#67a9cf", "#ca0020"]

fig, ax = plt.subplots(figsize=(4, 3))
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
    label="Constant Lagrange multipliers",
    color=COLORS[0],
)
ax.loglog(
    list(err_a.keys()),
    list(err_a.values()),
    label="Affine Lagrange multipliers",
    color=COLORS[1],
)
ax.legend()
ax.set_xlabel("Noise scale")
ax.set_ylabel("Constraint error")

fig.tight_layout()
fig.savefig("noise.pdf")


n_iter = 80


def plot(primal, dual, mu, out):
    with open(primal) as f:
        primal = list(map(float, f.readlines()))[:n_iter]
    with open(dual) as f:
        dual = list(map(float, f.readlines()))[:n_iter]
    with open(mu) as f:
        mu = list(map(float, f.readlines()))[:n_iter]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.semilogy(primal, label="Primal error", color=COLORS[0])
    ax.semilogy(dual, label="Dual error", color=COLORS[1])

    ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("")
    ax.set_ylim( 1e-300, 1e10)

    fig.tight_layout()
    fig.savefig(out)


plot(
    "result_data/cartpole_primal.dat",
    "result_data/cartpole_dual.dat",
    "result_data/cartpole_mu.dat",
    "error_c.pdf",
)
plot(
    "result_data/affine_mults_cartpole_primal.dat",
    "result_data/affine_mults_cartpole_dual.dat",
    "result_data/affine_mults_cartpole_mu.dat",
    "error_a.pdf",
)
