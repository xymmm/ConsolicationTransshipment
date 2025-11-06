# present_surface_fig.py  (X=IB, Y=r, Z=IA threshold)
import os, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def _unique(path):
    if not os.path.exists(path): return path
    root, ext = os.path.splitext(path); k = 1
    while True:
        cand = f"{root}_{k}{ext}"
        if not os.path.exists(cand): return cand
        k += 1

def _compute_phi_surface(inst, solution, include_r0=False):
    """
    phi_r(IB) = min IA such that q*(IB, IA; r) > 0 ; NaN if never dispatch.
    Returns: IB_vals (nI,), r_vals (nr,), phi (nr x nI)
    """
    PI = solution["PI"]
    IB_vals = solution["IB_vals"]   # columns
    IA_vals = solution["bA_vals"]   # rows
    r_vals = list(range(inst.N, -1, -1)) if include_r0 else list(range(inst.N, 0, -1))

    phi = np.full((len(r_vals), len(IB_vals)), np.nan, float)
    for ri, r in enumerate(r_vals):
        M = PI[r]  # (nI x nA)
        for i in range(len(IB_vals)):             # fix IB column index
            # scan IA rows ascending to find first positive dispatch
            idx = next((j for j in range(len(IA_vals)) if int(M[i, j]) > 0), None)
            if idx is not None:
                phi[ri, i] = float(IA_vals[idx])
    return IB_vals, np.array(r_vals, int), phi

def present_critical_surface_figure(inst, solution, outdir="figures",
                                    label=None, include_r0=False, dpi=180):
    os.makedirs(outdir, exist_ok=True)
    tag = f"_{label}" if label else ""
    IB_vals, r_vals, phi = _compute_phi_surface(inst, solution, include_r0=include_r0)

    # 3D surface: X=IB, Y=r, Z=IA* (threshold)
    IB_grid, r_grid = np.meshgrid(IB_vals, r_vals)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(IB_grid, r_grid, phi, linewidth=0, antialiased=True)
    ax.set_xlabel("IB (inventory at B)")
    ax.set_ylabel("r (periods remaining)")
    ax.set_zlabel("IA threshold for dispatch")
    ttl = "Critical Surface: IA* vs (IB, r)"
    if label: ttl += f" [{label}]"
    ax.set_title(ttl)
    f1 = _unique(os.path.join(outdir, f"critical_surface_IBx_r_to_IA{tag}.png"))
    fig.tight_layout(); fig.savefig(f1, dpi=dpi); plt.close(fig)

    # 2D stacked lines: IA*(IB) per epoch r
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for ri, r in enumerate(r_vals):
        ax2.plot(IB_vals, phi[ri, :], label=f"r={int(r)}")
    ax2.set_xlabel("IB (inventory at B)")
    ax2.set_ylabel("IA threshold for dispatch")
    ttl2 = "Critical Frontiers per Epoch (IA* vs IB)"
    if label: ttl2 += f" [{label}]"
    ax2.set_title(ttl2)
    ax2.legend(ncol=2, fontsize=8)
    f2 = _unique(os.path.join(outdir, f"critical_frontiers_IB_to_IA{tag}.png"))
    fig2.tight_layout(); fig2.savefig(f2, dpi=dpi); plt.close(fig2)

    print(f"✅ Saved figures:\n - {f1}\n - {f2}")
    return f1, f2
