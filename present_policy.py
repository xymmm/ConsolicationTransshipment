# present_policy.py  (IB = columns, IA = rows)
import os, csv, numpy as np

def present_policy(inst, solution, outfile="policy.csv", include_r0=False):
    """
    Append N (or N+1) policy matrices into one CSV.
    For each epoch r:
      - Header row: "IA\\IB", IB=min..max (ascending)
      - Rows: IA=0..maxbA (ascending)
      - Cell: q*(IB, IA; r)
    Blank line between epochs. Never overwrites.
    """
    PI = solution["PI"]
    IB_vals = solution["IB_vals"]   # ascending -> columns
    IA_vals = solution["bA_vals"]   # ascending -> rows

    r_values = range(inst.N, -1, -1) if include_r0 else range(inst.N, 0, -1)
    new_file = not os.path.exists(outfile)

    with open(outfile, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["# policy matrices; rows=IA, cols=IB; epoch annotated below"])
            w.writerow([])

        for r in r_values:
            w.writerow([f"epoch r={int(r)}"])
            header = ["IA\\IB"] + [int(v) for v in IB_vals.tolist()]
            w.writerow(header)

            PI_r = PI[r]  # shape (nI x nA), indexes: [i=IB, j=IA]
            for j, IA in enumerate(IA_vals):         # rows over IA
                row = [int(IA)]
                for i, IB in enumerate(IB_vals):     # columns over IB
                    row.append(int(PI_r[i, j]))
                w.writerow(row)

            w.writerow([])  # blank line between epochs

    print(f"✅ Appended {len(list(r_values))} matrices to {outfile} (rows=IA, cols=IB).")
