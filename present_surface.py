# present_surface.py
# Plot a 3D critical-level surface: IA* = f(IB, r)
# Axes: X = IB, Y = r (periods remaining), Z = IA*(IB,r)
# One PNG per label, overwriting any previous figure for that label.

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def _clean_previous(outdir: str, tag: str):
    for fp in glob.glob(os.path.join(outdir, f"surface{tag}*.png")):
        try: os.remove(fp)
        except OSError: pass


def present_surface(inst,
                    solution,
                    outdir: str = "figures",
                    label: str = None,
                    include_r0: bool = False,
                    dpi: int = 180):
    """
    Plot the critical-level surface IA* = f(IB, r).
    - Just one figure per run label.
    - Shade disabled (avoids divide-by-zero warnings).
    """

    os.makedirs(outdir, exist_ok=True)
    tag = f"_{label}" if label else ""
    _clean_previous(outdir, tag)

    PI       = solution["PI"]        # list of arrays, r=0..N, shape (nI x nA)
    IB_vals  = np.array(solution["IB_vals"])
    IA_vals  = np.array(solution["bA_vals"])

    # r = N..1 unless include_r0
    if include_r0:
        r_list = list(range(inst.N, -1, -1))
    else:
        r_list = list(range(inst.N, 0, -1))

    # Build IA*(IB,r)
    IAstar = np.full((len(r_list), len(IB_vals)), np.nan)

    for idx_r, r in enumerate(r_list):
        M = PI[r]  # shape (nI x nA), index: [i_IB , j_IA]
        for i_IB, IB in enumerate(IB_vals):
            row = M[i_IB, :]
            # find smallest IA where q* > 0
            pos = np.where(row > 0)[0]
            if len(pos) > 0:
                IAstar[idx_r, i_IB] = IA_vals[pos[0]]
            else:
                IAstar[idx_r, i_IB] = np.nan

    # meshes for surface
    X, Y = np.meshgrid(IB_vals, np.array(r_list, dtype=float), indexing="xy")
    Z = np.ma.masked_invalid(IAstar.astype(float))

    # --- plot ---
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,      # IMPORTANT: avoids shading warnings
        color="#1f77b4",
        alpha=0.95
    )

    ax.set_xlabel("IB (inventory at B)")
    ax.set_ylabel("r (periods remaining)")
    ax.set_zlabel("IA* (critical A backlog level)")

    ax.set_xlim(left=max(0, IB_vals.min()))
    ax.set_ylim(bottom=0, top=inst.N)
    ax.set_zlim(bottom=0, top=max(IA_vals) if len(IA_vals) else 1)

    title = "Critical-Level Surface: IA*(IB, r)"
    if label:
        title += f" [{label}]"
    ax.set_title(title)

    fig.tight_layout()
    path = os.path.join(outdir, f"surface{tag}.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    print(f"Saved critical-level surface: {path}")
    return path
