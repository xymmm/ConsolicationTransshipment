# verify_theory.py
# Full numerical check to resonate with the analytical condition:
#   (pA - pB) * (r*dt) > cu  => when q*>0, optimal policy should "full-clear":
#   q*(IB, IA, r) = min(IB, IA) (with feasibility 0..min(IB,IA))
#
# Outputs:
#   - fullclear_report.csv : per-epoch (r) conformity + overall summary
#   - critical_surface.csv : matrix with IB*(IA, r) threshold (smallest IB that triggers q*>0)
#   - critical_surface.png : 3D surface of the threshold (X=IA, Y=r, Z=IB*)
#
# Usage (standalone):
#   python verify_theory.py
#
# Usage (import):
#   from verify_theory import check_full_clear, build_critical_surface, save_critical_surface_figure
#   stats = check_full_clear(inst, solution)
#   surf = build_critical_surface(inst, solution)
#   save_critical_surface_figure(inst, surf, out_png="critical_surface.png")

import os
import csv
import numpy as np

# Optional: import your solver module if you run this standalone
try:
    from minimalSolver import Instance, solveDP_AMO_Bpriority_dynamic
except Exception:
    Instance = None
    solveDP_AMO_Bpriority_dynamic = None

import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _feasible_full_clear_q(IB: int, IA: int) -> int:
    """Feasible 'full-clear' quantity = max(0, min(IB, IA))."""
    return max(0, min(IB, IA))


def check_full_clear(inst, solution, include_r0: bool = False,
                     out_csv: str = "fullclear_report.csv") -> dict:
    """
    Check how often the DP policy matches 'full-clear' in the theoretical region:
        (pA - pB) * (r*dt) > cu
    Only count states with feasible_max > 0.
    For matching, if q*>0 then require q* == feasible_full_clear.

    Returns a dict with per-r stats and overall summary; also appends CSV.
    """
    dt = inst.dt()
    pA, pB, cu = inst.pA, inst.pB, inst.cu
    IB_vals = solution["IB_vals"]
    IA_vals = solution["bA_vals"]
    PI_list = solution["PI"]
    N = inst.N

    r_range = range(N, -1, -1) if include_r0 else range(N, 0, -1)

    rows = []
    overall_total = 0
    overall_in_region = 0
    overall_feasible = 0
    overall_qpos = 0
    overall_match = 0

    for r in r_range:
        tau = r * dt
        margin = (pA - pB) * tau - cu
        in_theory_region = margin > 0

        M = PI_list[r]  # shape (nI x nA)
        nI, nA = M.shape

        total_states = nI * nA
        feasible_states = 0          # states with feasible_max > 0
        in_region_count = 0          # among feasible, in theory region (same for all states at this r)
        qpos_states = 0              # among feasible, q*>0
        match_count = 0              # among q*>0 in region, q* == full-clear

        for i, IB in enumerate(IB_vals):
            for j, IA in enumerate(IA_vals):
                feasible_max = _feasible_full_clear_q(IB, IA)
                if feasible_max <= 0:
                    continue  # cannot dispatch anything here
                feasible_states += 1

                qstar = int(M[i, j])
                if in_theory_region:
                    in_region_count += 1
                    if qstar > 0:
                        qpos_states += 1
                        if qstar == feasible_max:
                            match_count += 1
                else:
                    # We still count how often q*>0, but this doesn't enter "match" metric for theory region
                    if qstar > 0:
                        qpos_states += 1

        rows.append([
            r, tau, margin,
            total_states, feasible_states, in_region_count, qpos_states, match_count
        ])

        overall_total += total_states
        overall_feasible += feasible_states
        overall_in_region += in_region_count
        overall_qpos += qpos_states
        overall_match += match_count

    # Append results to CSV (append-only; create header if not exists)
    file_exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["r", "tau", "margin=(pA-pB)*tau-cu",
                        "total_states", "feasible_states",
                        "in_region_feasible", "qpos_states", "match_fullclear"])
        w.writerows(rows)
        # summary row
        w.writerow([])
        w.writerow(["SUMMARY",
                    "", "",
                    overall_total, overall_feasible,
                    overall_in_region, overall_qpos, overall_match])

    # Print a short summary for console
    rate = (overall_match / overall_qpos) if overall_qpos > 0 else float("nan")
    print(f"[Full-clear check] In-theory-region q*>0 states: {overall_qpos}, "
          f"full-clear matches: {overall_match}, match rate = {rate:.4f}")

    return {
        "rows": rows,
        "overall": {
            "total_states": overall_total,
            "feasible_states": overall_feasible,
            "in_region_feasible": overall_in_region,
            "qpos_states": overall_qpos,
            "match_fullclear": overall_match,
            "match_rate": rate,
        }
    }


def build_critical_surface(inst, solution, include_r0: bool = False):
    """
    Build the critical surface IB*(IA, r):
      For each (r, IA), find the smallest IB such that q*>0.
      If no IB triggers dispatch, put NaN.

    Returns:
      surf : 2D array with shape (len(r_list), len(IA_vals))
      r_list (descending): [N..1] or [N..0]
      IA_vals: as in solution
    """
    IB_vals = solution["IB_vals"]
    IA_vals = solution["bA_vals"]
    PI_list = solution["PI"]
    N = inst.N

    r_list = list(range(N, -1, -1)) if include_r0 else list(range(N, 0, -1))
    surf = np.full((len(r_list), len(IA_vals)), np.nan, dtype=float)

    for ri, r in enumerate(r_list):
        M = PI_list[r]  # (nI x nA)
        for j, IA in enumerate(IA_vals):
            # find smallest IB (in ascending IB_vals) such that q*>0
            trigger_IB = np.nan
            for i, IB in enumerate(IB_vals):
                if _feasible_full_clear_q(IB, IA) <= 0:
                    continue
                if int(M[i, j]) > 0:
                    trigger_IB = IB
                    break
            surf[ri, j] = trigger_IB

    return surf, r_list, IA_vals


def save_critical_surface_csv(surf, r_list, IA_vals, out_csv: str = "critical_surface.csv"):
    """
    Save the critical surface as a CSV with header:
      row0: ["r\\IA", IA0, IA1, ..., IAN]
      rows: [r, IB*(IA0,r), IB*(IA1,r), ...]
    """
    file_exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["r\\IA"] + list(IA_vals))
        for ri, r in enumerate(r_list):
            w.writerow([r] + [("" if np.isnan(v) else int(v)) for v in surf[ri, :]])
        w.writerow([])  # blank separator


def save_critical_surface_figure(inst, surf, r_list, IA_vals,
                                 out_png: str = "critical_surface.png",
                                 elev: int = 25, azim: int = 45, dpi: int = 180):
    """
    Save a 3D surface: X=IA, Y=r, Z=IB*.
    NaNs are masked so the surface will be missing where no dispatch is triggered.
    Axes start from 0 (IA, r), and IB* shown as-is (can be negative if your grid allows).
    """
    X, Y = np.meshgrid(IA_vals, r_list)  # X: IA (cols), Y: r (rows)
    Z = np.array(surf, dtype=float)

    Z_masked = np.ma.masked_invalid(Z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Surface
    surf_plot = ax.plot_surface(X, Y, Z_masked, cmap="viridis", linewidth=0, antialiased=True)

    ax.set_xlabel("IA (A backlog)")
    ax.set_ylabel("r (periods remaining)")
    ax.set_zlabel("IB* (dispatch threshold)")

    ax.set_xlim(left=0)  # start IA from 0
    ax.set_ylim(bottom=0, top=inst.N)  # r from 0..N
    # z-limit: don't force; let data talk. If you want to start from 0: ax.set_zlim(bottom=0)

    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf_plot, shrink=0.75, pad=0.08, label="IB*")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"[Critical surface] Saved 3D surface figure: {out_png}")


def run_full_check(inst, solution,
                   fullclear_csv="fullclear_report.csv",
                   surf_csv="critical_surface.csv",
                   surf_png="critical_surface.png",
                   include_r0=False):
    stats = check_full_clear(inst, solution, include_r0=include_r0, out_csv=fullclear_csv)
    surf, r_list, IA_vals = build_critical_surface(inst, solution, include_r0=include_r0)
    save_critical_surface_csv(surf, r_list, IA_vals, out_csv=surf_csv)
    save_critical_surface_figure(inst, surf, r_list, IA_vals, out_png=surf_png)
    return stats, surf


if __name__ == "__main__":
    if Instance is None or solveDP_AMO_Bpriority_dynamic is None:
        raise RuntimeError("Could not import minimalSolver. Run this by importing from your main, or put minimalSolver.py in PYTHONPATH.")

    # Example instance (match your defaults)
    inst = Instance(
        N=20, T=2.0,               # dt = 0.1
        lambdaA=8, lambdaB=5,
        h=0.1, pA=50.0, pB=10.0,
        cf=20.0, cu=1.0,
        minIB=-20, maxIB=20, maxbA=10,
        IB0=20
    )
    solution = solveDP_AMO_Bpriority_dynamic(inst)

    # Run full check and produce outputs
    run_full_check(inst, solution,
                   fullclear_csv="fullclear_report.csv",
                   surf_csv="critical_surface.csv",
                   surf_png="critical_surface.png",
                   include_r0=False)
    print("Done. Wrote: fullclear_report.csv, critical_surface.csv, critical_surface.png")
