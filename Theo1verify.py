# Theo1verify.py (runner only)
import os
from minimalSolver import (
    Instance,
    solveDP_AMO_Bpriority_dynamic,
    get_optimal_expected_cost_user_t,
    run_simulations,
    append_sim_results,
)

# presentation helpers live outside the solver:
from present_epoch import present_epoch
from present_policy import present_policy
from present_surface import present_surface

# import verify_theory functions
from verify_theory import run_full_check


def run_instance(label, params, sim_n=2000, seed=2025, base_seed=3260):
    print(f"\n{'=' * 60}")
    print(f"Running Instance: {label}")
    print(f"Params: {params}")
    print(f"{'=' * 60}")

    # 1. Create Instance
    inst = Instance(**params)

    # 2. Solve DP
    solution = solveDP_AMO_Bpriority_dynamic(inst)

    # 3. Optimal expected cost at start
    opt_cost = get_optimal_expected_cost_user_t(solution, inst, t_user=0, IB=inst.IB0, bA=0)
    print(f"[{label}] Exact Optimal Expected Cost (t=0): {opt_cost:.4f}")

    # 4. Presentation Outputs (using label prefixes to avoid overwriting)

    # A. Single trajectory epoch tracking
    epoch_file = f"{label}_epoch.csv"
    present_epoch(inst, solution, IB0=inst.IB0, bA0=0, seed=seed,
                  outfile=epoch_file, label=label)
    print(f"Saved epoch trace to {epoch_file}")

    # B. Policy Matrices
    policy_file = f"{label}_policy.csv"
    present_policy(inst, solution, outfile=policy_file, include_r0=False)
    print(f"Saved policy matrices to {policy_file}")

    # C. 3D Surface Plot
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    present_surface(inst, solution, outdir="figures", label=label, dpi=180)
    print(f"Saved surface plot to figures/{label}_policy_surface.png")

    # 5. Simulations
    print(f"Running {sim_n} simulations...")
    mean, std, lo, hi, costs = run_simulations(inst, solution, sim_n, base_seed, IB0=inst.IB0, bA0=0)
    print(f"[{label}] Tally: mean={mean:.4f}, std={std:.4f}, 95% CI=({lo:.4f}, {hi:.4f})")

    # Append to a common results file
    append_sim_results("sims.csv", label, base_seed, costs)

    # 6. Verify Theory (Theorem 1 Check)
    # This generates the critical surface CSVs and PNGs specific to the theorem verification
    print("Running theoretical verification (Theorem 1 checks)...")
    run_full_check(
        inst, solution,
        fullclear_csv=f"{label}_fullclear_report.csv",
        surf_csv=f"{label}_critical_surface.csv",
        surf_png=f"{label}_critical_surface.png",
        include_r0=False
    )
    print(f"Verification complete. See {label}_critical_surface.csv/.png")


if __name__ == "__main__":
    # Configuration for reasonable instances to verify Theorem 1
    # Theorem 1 requires pi1 == pi2 (pA == pB)

    instances_to_run = [
        {
            "label": "Thm1_Base",
            "params": {
                "N": 20, "T": 2.0,
                "lambdaA": 8, "lambdaB": 5,
                "h": 0.1,
                "pA": 10.0, "pB": 10.0,  # EQUAL penalties for Theorem 1
                "cf": 20.0, "cu": 1.0,
                "minIB": -20, "maxIB": 20, "maxbA": 10,
                "IB0": 20
            }
        },
        {
            "label": "Thm1_LowFixedCost",
            "params": {
                "N": 20, "T": 2.0,
                "lambdaA": 8, "lambdaB": 5,
                "h": 0.1,
                "pA": 10.0, "pB": 10.0,  # EQUAL penalties
                "cf": 5.0,  # Lower fixed cost -> should lower threshold
                "cu": 1.0,
                "minIB": -20, "maxIB": 20, "maxbA": 10,
                "IB0": 20
            }
        },
        {
            "label": "Thm1_HighPenalty",
            "params": {
                "N": 20, "T": 2.0,
                "lambdaA": 8, "lambdaB": 5,
                "h": 0.1,
                "pA": 20.0, "pB": 20.0,  # Higher EQUAL penalties -> urgent dispatch
                "cf": 20.0, "cu": 1.0,
                "minIB": -20, "maxIB": 20, "maxbA": 10,
                "IB0": 20
            }
        }
    ]

    # Main Loop
    for case in instances_to_run:
        run_instance(
            label=case["label"],
            params=case["params"],
            sim_n=2000,
            seed=2025,
            base_seed=3260
        )