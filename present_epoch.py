# present_epoch.py
import csv
import numpy as np

def simulate_trajectory_with_events(inst, solution, IB0, bA0, seed=2025):
    """
    Simulate one trajectory following the optimal policy and return row-wise series
    for CSV reporting.

    Returns (lists of length N = inst.N):
      times      : start time of each epoch
      IB_start   : inventory at B at start-of-epoch (before dispatch & event)
      IA         : backlog at A at start-of-epoch (consolidated demand before dispatch)
      dispatch_q : planned q at start of epoch
      events     : "A", "B", or "-" (NONE)
      A_demand_q : 1 or "" (empty if no A in this epoch)
      B_demand_q : 1 or "" (empty if no B in this epoch)
      IB_end     : inventory at B at end-of-epoch (after dispatch & event)
    """
    rng = np.random.default_rng(seed)
    IB_vals, bA_vals = solution["IB_vals"], solution["bA_vals"]
    PI = solution["PI"]
    pi0, piA, piB = solution["pi"]
    dt = inst.dt()

    # Try to read CLAMP_TO_GRID from the solver module (default True)
    CLAMP_TO_GRID = True
    try:
        mod = __import__(inst.__module__)
        if hasattr(mod, "CLAMP_TO_GRID"):
            CLAMP_TO_GRID = getattr(mod, "CLAMP_TO_GRID")
    except Exception:
        pass

    times, IB_start, IA, dispatch_q, events, A_demand_q, B_demand_q, IB_end = \
        [], [], [], [], [], [], [], []

    IB, bA = IB0, bA0
    N = inst.N
    for k in range(N):
        r = N - k
        times.append(round(k * dt, 6))

        # Record start-of-epoch state BEFORE dispatch and event
        IB_start.append(IB)
        IA.append(bA)

        # Planned dispatch (policy lookup at start-of-epoch)
        IBc = max(inst.minIB, min(inst.maxIB, IB))
        bAc = max(0, min(inst.maxbA, bA))
        i = int(np.where(IB_vals == IBc)[0][0])
        j = int(np.where(bA_vals == bAc)[0][0])
        q = int(PI[r][i, j])
        dispatch_q.append(q)

        # One AMO event
        u = rng.random()
        if u < pi0:
            scen = "NONE"
        elif u < pi0 + piA:
            scen = "A"
        else:
            scen = "B"

        # Realize transitions with B priority
        if scen == "NONE":
            events.append("-")
            A_demand_q.append("")
            B_demand_q.append("")
            qeff = q
            IB = IB - qeff
            bA = max(0, bA - qeff)

        elif scen == "A":
            events.append("A")
            A_demand_q.append(1)
            B_demand_q.append("")
            qeff = q
            IB = IB - qeff
            bA = max(0, bA - qeff + 1)

        else:  # scen == "B"
            events.append("B")
            A_demand_q.append("")
            B_demand_q.append(1)
            base_IB = IB - (1 if IB > 0 else 0)   # B consumes first iff stock>0 (lost sale if IB<=0)
            qeff = min(q, max(base_IB, 0))
            IB = base_IB - qeff
            bA = max(0, bA - qeff)

        if CLAMP_TO_GRID:
            IB = max(inst.minIB, min(inst.maxIB, IB))
            bA = max(0, min(inst.maxbA, bA))

        # Closing inventory after this period
        IB_end.append(IB)

    return times, IB_start, IA, dispatch_q, events, A_demand_q, B_demand_q, IB_end


def write_epoch_csv_append(outfile, label,
                           times, IB_start, IA, dispatch_q, events, A_demand_q, B_demand_q, IB_end):
    """
    Append one run to CSV in ROW-WISE form (one series per row).
    Row order per run (then a blank separator row):
      label,<your label>
      time,<t0>,...,<tN-1>
      IB_start,<IB0>,...,<IBN-1>
      IA,<ia0>,...,<iaN-1>
      dispatch_q,<q0>,...,<qN-1>
      event,<e0>,...,<eN-1>         (in {"A","B","-"})
      A_demand,<a0>,...,<aN-1>      (1 or empty)
      B_demand,<b0>,...,<bN-1>      (1 or empty)
      IB_end,<IB1>,...,<IBN>        (closing inventory each period; note IB_end[k]==IB_start[k+1])
    """
    with open(outfile, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", str(label)])
        w.writerow(["time", *times])
        w.writerow(["IB_start", *IB_start])
        w.writerow(["ConsolidatedA", *IA])
        w.writerow(["dispatch_q", *dispatch_q])
        w.writerow(["event", *events])
        w.writerow(["A_demand", *A_demand_q])
        w.writerow(["B_demand", *B_demand_q])
        w.writerow(["IB_end", *IB_end])
        w.writerow([])  # blank line between runs


def present_epoch(inst, solution, IB0, bA0=0, seed=2025, outfile="epoch.csv", label=""):
    """
    High-level wrapper:
      simulate → append rows to CSV in the required order (with IB_start and IB_end)
    """
    times, IB_start, IA, dq, ev, Ad, Bd, IB_end = simulate_trajectory_with_events(
        inst, solution, IB0, bA0, seed
    )
    if label.strip() == "":
        label = f"[IB0={IB0}, bA0={bA0}, seed={seed}]"

    write_epoch_csv_append(outfile, label, times, IB_start, IA, dq, ev, Ad, Bd, IB_end)
    print(f"Appended CSV results to {outfile}: {label}")
