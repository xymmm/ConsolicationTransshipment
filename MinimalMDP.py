# Finite-horizon MDP with AMO (NONE/A_ONLY/B_ONLY) and "B priority".
# Actions at (IB, bA): q ∈ {0,1,...,min(IB,bA)} (including wait).
# Value function indexing :
#   V[r] = optimal expected total cost with r periods remaining (terminal V[0] = 0).
#   We compute FORWARD: r = 1..N using V[r-1]. (backwards)
# User-facing “t=0 (start)” maps to internal r=N (see get_optimal_expected_cost_user_t).

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

# === NEW: use the presentation helper (kept separate so solver stays clean)
from present_epoch import present_epoch

# -------------------------
# Problem instance
# -------------------------
@dataclass(frozen=True)
class Instance:
    N: int
    T: float
    lambdaA: float
    lambdaB: float
    h: float
    pA: float
    pB: float
    cf: float
    cu: float
    minIB: int
    maxIB: int
    maxbA: int
    IB0: int = 0            # initial inventory at B (user input)
    tail: float = 0.0       # unused in AMO

    def dt(self) -> float:
        return self.T / self.N

# -------------------------
# Global toggles
# -------------------------
PAY_FIXED_ON_REALIZED: bool = True  # fixed cost charged when q_realized>0 (else on planned q>0)
CLAMP_TO_GRID: bool = True          # clamp next states into grid bounds

# -------------------------
# Demand probabilities
# -------------------------
def computeDemandProbability_AMO(inst: Instance) -> Tuple[float, float, float]:
    # At most one event in a small dt: NONE, A_ONLY, B_ONLY (renormalized if needed)
    dt = inst.dt()
    piA = inst.lambdaA * dt
    piB = inst.lambdaB * dt
    pi0 = max(0.0, 1.0 - (piA + piB))
    s = pi0 + piA + piB
    if s <= 0.0:
        return 1.0, 0.0, 0.0
    return pi0 / s, piA / s, piB / s

# -------------------------
# Costs
# -------------------------
def computeImmediateCost_closing(IB_end: int, bA_end: int, inst: Instance) -> float:
    # End-of-period cost: dt * [ h*max(IB_end,0) + pB*max(-IB_end,0) + pA*bA_end ]
    dt = inst.dt()
    holding = inst.h * max(IB_end, 0)
    backlogB = inst.pB * max(-IB_end, 0)
    backlogA = inst.pA * bA_end
    return dt * (holding + backlogB + backlogA)

def computeTransshipmentCost(q_planned: int, q_realized: int, inst: Instance) -> float:
    # Dispatch cost = fixed + variable
    pay_fixed = (q_realized > 0) if PAY_FIXED_ON_REALIZED else (q_planned > 0)
    fixed = inst.cf if pay_fixed else 0.0
    variable = inst.cu * q_realized
    return fixed + variable

# -------------------------
# State grid
# -------------------------
def buildStateGrid(inst: Instance):
    IB_vals = np.arange(inst.minIB, inst.maxIB + 1, dtype=int)
    bA_vals = np.arange(0, inst.maxbA + 1, dtype=int)
    IB2i = {v: i for i, v in enumerate(IB_vals)}
    bA2j = {v: j for j, v in enumerate(bA_vals)}
    return IB_vals, bA_vals, IB2i, bA2j

# -------------------------
# DP solver (FORWARD in r = periods remaining)
# -------------------------
def solveDP_AMO_Bpriority_dynamic(inst: Instance):
    pi0, piA, piB = computeDemandProbability_AMO(inst)
    pis = np.array([pi0, piA, piB], dtype=float)  # [NONE, A_ONLY, B_ONLY]

    IB_vals, bA_vals, IB2i, bA2j = buildStateGrid(inst)
    nI, nA = len(IB_vals), len(bA_vals)

    # V[r] = cost with r periods remaining; V[0] = 0 terminal
    V  = [np.zeros((nI, nA), dtype=float) for _ in range(inst.N + 1)]
    # PI[r] = optimal action when r periods remain (no decision at r=0, but keep same length for simplicity)
    PI = [np.zeros((nI, nA), dtype=int)   for _ in range(inst.N + 1)]

    # Forward pass: r = 1..N uses V[r-1] as future
    for r in range(1, inst.N + 1):
        Vprev = V[r - 1]   # future value after this period
        for i, IB in enumerate(IB_vals):
            for j, bA in enumerate(bA_vals):
                max_feasible_q = max(0, min(IB, bA))
                best_cost = float("inf")
                best_q = 0

                for q in range(0, max_feasible_q + 1):
                    total_expected = 0.0

                    # Scenarios: 0=NONE, 1=A_ONLY, 2=B_ONLY
                    # We'll compute next-state and immediate costs, then add Vprev[next]
                    # NONE
                    q_eff = q
                    IB_end = IB - q_eff
                    bA_end = max(0, bA - q_eff)
                    if CLAMP_TO_GRID:
                        IBc = max(inst.minIB, min(inst.maxIB, IB_end))
                        bAc = max(0, min(inst.maxbA, bA_end))
                    else:
                        IBc, bAc = IB_end, bA_end
                    dc = computeTransshipmentCost(q_planned=q, q_realized=q_eff, inst=inst)
                    cc = computeImmediateCost_closing(IBc, bAc, inst)
                    total_expected += pis[0] * (dc + cc + Vprev[IB2i[IBc], bA2j[bAc]])

                    # A_ONLY
                    q_eff = q
                    IB_end = IB - q_eff
                    bA_end = max(0, bA - q_eff + 1)
                    if CLAMP_TO_GRID:
                        IBc = max(inst.minIB, min(inst.maxIB, IB_end))
                        bAc = max(0, min(inst.maxbA, bA_end))
                    else:
                        IBc, bAc = IB_end, bA_end
                    dc = computeTransshipmentCost(q_planned=q, q_realized=q_eff, inst=inst)
                    cc = computeImmediateCost_closing(IBc, bAc, inst)
                    total_expected += pis[1] * (dc + cc + Vprev[IB2i[IBc], bA2j[bAc]])

                    # B_ONLY with B priority
                    base_IB = IB - (1 if IB > 0 else 0)     # B consumes 1 if available
                    q_eff = min(q, max(base_IB, 0))         # then we ship to A from what's left
                    IB_end = base_IB - q_eff
                    bA_end = max(0, bA - q_eff)
                    if CLAMP_TO_GRID:
                        IBc = max(inst.minIB, min(inst.maxIB, IB_end))
                        bAc = max(0, min(inst.maxbA, bA_end))
                    else:
                        IBc, bAc = IB_end, bA_end
                    dc = computeTransshipmentCost(q_planned=q, q_realized=q_eff, inst=inst)
                    cc = computeImmediateCost_closing(IBc, bAc, inst)
                    total_expected += pis[2] * (dc + cc + Vprev[IB2i[IBc], bA2j[bAc]])

                    if total_expected < best_cost:
                        best_cost = total_expected
                        best_q = q

                PI[r][i, j] = best_q
                V[r][i, j]  = best_cost

    return {
        "V": V,                 # list of (nI x nA), r=0..N
        "PI": PI,               # list of (nI x nA), r=0..N (no decision at r=0)
        "IB_vals": IB_vals,
        "bA_vals": bA_vals,
        "pi": (pi0, piA, piB),
    }

# -------------------------
# Policy printer (optional)
# -------------------------
def print_optimal_policy(solution, inst: Instance, r: int,
                         IB_from: Optional[int] = None, IB_to: Optional[int] = None,
                         bA_from: Optional[int] = None, bA_to: Optional[int] = None):
    IB_vals = solution["IB_vals"]; bA_vals = solution["bA_vals"]; PI = solution["PI"]
    if r < 0 or r > inst.N:
        raise ValueError(f"r must be in [0..{inst.N}]")

    IB_lo = IB_vals.min() if IB_from is None else max(IB_vals.min(), min(IB_from, IB_vals.max()))
    IB_hi = IB_vals.max() if IB_to   is None else max(IB_vals.min(), min(IB_to,   IB_vals.max()))
    if IB_lo > IB_hi: IB_lo, IB_hi = IB_hi, IB_lo

    bA_lo = bA_vals.min() if bA_from is None else max(bA_vals.min(), min(bA_from, bA_vals.max()))
    bA_hi = bA_vals.max() if bA_to   is None else max(bA_vals.min(), min(bA_to,   bA_vals.max()))
    if bA_lo > bA_hi: bA_lo, bA_hi = bA_hi, bA_lo

    IB_index = {v: i for i, v in enumerate(IB_vals)}
    bA_index = {v: j for j, v in enumerate(bA_vals)}
    cols = list(range(IB_lo, IB_hi + 1))

    print(f"\n=== Optimal policy q* (r={r} periods remaining) ===")
    print(f"IB range: [{IB_lo}..{IB_hi}], bA range: [{bA_lo}..{bA_hi}]")
    header = ["bA\\IB"] + [f"{c:>4d}" for c in cols]
    print(" ".join(header))
    print("-" * (6 + 5 * len(cols)))

    for bA in range(bA_hi, bA_lo - 1, -1):
        row = [f"{bA:>5d}"]; j = bA_index[bA]
        for IB in cols:
            i = IB_index[IB]
            qstar = int(PI[r][i, j])
            feasible_max = max(0, min(IB, bA))
            cell = f"{qstar:>4d}" if 0 <= qstar <= feasible_max else "  - "
            row.append(cell)
        print(" ".join(row))

def print_policies_every5(solution, inst: Instance,
                          IB_from: Optional[int] = None, IB_to: Optional[int] = None,
                          bA_from: Optional[int] = None, bA_to: Optional[int] = None):
    for r in range(inst.N, -1, -1):
        if r % 5 == 0:
            print_optimal_policy(solution, inst, r=r,
                                 IB_from=IB_from, IB_to=IB_to,
                                 bA_from=bA_from, bA_to=bA_to)

# -------------------------
# Cost helpers (exact and simulation tally)
# -------------------------
def get_optimal_expected_cost_user_t(solution, inst: Instance, t_user: int, IB: int, bA: int) -> float:
    r = inst.N - t_user
    if not (0 <= r <= inst.N):
        raise ValueError(f"user t out of range: t_user={t_user} -> r={r}")
    IB_vals, bA_vals = solution["IB_vals"], solution["bA_vals"]
    IBc = max(inst.minIB, min(inst.maxIB, IB))
    bAc = max(0, min(inst.maxbA, bA))
    i0 = int(np.where(IB_vals == IBc)[0][0])
    j0 = int(np.where(bA_vals == bAc)[0][0])
    return float(solution["V"][r][i0, j0])

def simulate_realized_cost(inst: Instance, solution, IB0: int, bA0: int, seed: int = 2025) -> float:
    rng = np.random.default_rng(seed)
    IB_vals, bA_vals = solution["IB_vals"], solution["bA_vals"]
    PI = solution["PI"]
    pi0, piA, piB = solution["pi"]

    IB = IB0
    bA = bA0
    total_cost = 0.0

    for k in range(inst.N):
        r = inst.N - k  # periods remaining now
        IBc = max(inst.minIB, min(inst.maxIB, IB))
        bAc = max(0, min(inst.maxbA, bA))
        i = int(np.where(IB_vals == IBc)[0][0])
        j = int(np.where(bA_vals == bAc)[0][0])
        q = int(PI[r][i, j])

        u = rng.random()
        if u < pi0:
            scen = 0
        elif u < pi0 + piA:
            scen = 1
        else:
            scen = 2

        if scen == 0:
            q_eff = q
            IB_next = IB - q_eff
            bA_next = max(0, bA - q_eff)
        elif scen == 1:
            q_eff = q
            IB_next = IB - q_eff
            bA_next = max(0, bA - q_eff + 1)
        else:
            base_IB = IB - (1 if IB > 0 else 0)
            q_eff = min(q, max(base_IB, 0))
            IB_next = base_IB - q_eff
            bA_next = max(0, bA - q_eff)

        if CLAMP_TO_GRID:
            IB_next = max(inst.minIB, min(inst.maxIB, IB_next))
            bA_next = max(0, min(inst.maxbA, bA_next))

        dispatch_cost = computeTransshipmentCost(q_planned=q, q_realized=q_eff, inst=inst)
        closing_cost  = computeImmediateCost_closing(IB_next, bA_next, inst)
        total_cost += (dispatch_cost + closing_cost)

        IB, bA = IB_next, bA_next

    return float(total_cost)

def run_simulations(inst: Instance, solution, simN: int, base_seed: int, IB0: int, bA0: int):
    costs = []
    for r in range(simN):
        seed = base_seed + r
        c = simulate_realized_cost(inst, solution, IB0=IB0, bA0=bA0, seed=seed)
        costs.append(c)

    costs = np.array(costs, dtype=float)
    mean = float(costs.mean())
    std = float(costs.std(ddof=1)) if simN > 1 else 0.0
    z = 1.96  # 95% normal approx
    half_width = z * std / (simN ** 0.5) if simN > 1 else 0.0
    ci_low, ci_high = mean - half_width, mean + half_width
    return mean, std, ci_low, ci_high, costs

# -------------------------
# Demo / quick run
# -------------------------
if __name__ == "__main__":
    # === knobs for quick experiments ===
    SHOW_POLICY_GRIDS = True
    OUTFILE = "epoch.txt"
    RUN_LABEL = "Baseline"
    SEED = 2025

    # Example instance (edit as needed)
    inst = Instance(
        N=20, T=2.0,               # dt = 0.1
        lambdaA=8, lambdaB=5,
        h=0.1, pA=50.0, pB=10.0,
        cf=20.0, cu=1.0,
        minIB=-20, maxIB=20, maxbA=10,
        IB0=20                     # initial inventory at B (user input)
    )

    # Solve DP (exact)
    solution = solveDP_AMO_Bpriority_dynamic(inst)

    # (Optional) Show a few policy grids
    if SHOW_POLICY_GRIDS:
        print_policies_every5(solution, inst)

    # Report the optimal expected cost at user-facing t=0 (start), which maps to r=N
    opt_cost = get_optimal_expected_cost_user_t(solution, inst, t_user=0, IB=inst.IB0, bA=0)
    print(f"\n[Exact] Optimal expected cost at t=0 from (IB0={inst.IB0}, bA0=0): {opt_cost:.4f}")

    # Present one epoch trajectory to CSV
    present_epoch(inst, solution, IB0=inst.IB0, bA0=0, seed=2025,
                  outfile="epoch.csv", label="Run A")


    # (Optional) batch simulations for a CI (kept, since not part of old 4-row table)
    simN = 2000
    base_seed = 3260
    mean, std, lo, hi, costs = run_simulations(inst, solution, simN, base_seed, IB0=inst.IB0, bA0=0)
    print(f"\n[Tally over {simN} sims] mean={mean:.4f}, std={std:.4f}, 95% CI=({lo:.4f}, {hi:.4f})")
