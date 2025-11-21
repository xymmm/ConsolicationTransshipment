import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set academic style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def load_thresholds(csv_file):
    """
    Extracts the switching curve from the CSV.
    Automatically selects the first row (largest 'r') to represent the
    start-of-horizon policy, ensuring comparable states across different N.
    """
    print(f"Loading {csv_file}...")

    if not os.path.exists(csv_file):
        print(f"  [Error] File not found: {csv_file}")
        return [], []

    try:
        # Read CSV. Assuming first column is 'r'.
        df = pd.read_csv(csv_file, index_col=0)

        # Select the row with the largest 'r' (Start of the horizon)
        # CSVs are typically sorted descending, so iloc[0] is usually max r.
        # We do this explicitly to handle N=20 vs N=200 correctly.
        start_r = df.index.max()

        # Robust check if start_r exists
        if start_r in df.index:
            data = df.loc[start_r]
        else:
            # Fallback to first row
            data = df.iloc[0]

        print(f"  -> Using policy at r={start_r} (Start of Horizon)")

        x = []
        y = []

        for col in df.columns:
            try:
                # The column header is Inventory (I2)
                inv_val = int(col)

                # The cell value is Critical Backlog (b1)
                val = data[col]

                # Check if valid number (not NaN, not empty string)
                if pd.notna(val) and str(val).strip() != "":
                    x.append(inv_val)
                    y.append(float(val))
            except ValueError:
                continue

        # Sort by Inventory to ensure the line connects correctly
        if x:
            xy_sorted = sorted(zip(x, y))
            x = [i[0] for i in xy_sorted]
            y = [i[1] for i in xy_sorted]

        print(f"  -> Found {len(x)} valid data points.")
        return x, y
    except Exception as e:
        print(f"  Error loading {csv_file}: {e}")
        return [], []


# --- Main Execution ---

# 1. Setup Data Sources
# We compare N=20 (Base & High Penalty) vs N=100 & N=200 (Base)
# Added folder prefix 'Thm1_Verify/' to paths
datasets = [
    {
        "file": "Thm1_Verify_p1larger/Thm1_Base_critical_surface.csv",
        "label": r"Base Case (N=20)",
        "color": "#1f77b4",  # Blue
        "style": "-o",
        "width": 1.5
    },
    {
        "file": "Thm1_Verify_p1larger/Thm1_HighPenalty_critical_surface.csv",
        "label": r"High Penalty (N=20)",
        "color": "#d62728",  # Red
        "style": "--^",
        "width": 1.5
    },
    {
        "file": "Thm1_Verify_p1larger/Thm1_Base_N100_critical_surface.csv",
        "label": r"Base Case (N=100)",
        "color": "#2ca02c",  # Green
        "style": "-s",
        "width": 1.5
    },
    {
        "file": "Thm1_Verify_p1larger/Thm1_Base_N200_critical_surface.csv",
        "label": r"Base Case (N=200)",
        "color": "#9467bd",  # Purple
        "style": "-d",
        "width": 1.5
    }
]

# 2. Create Plot
plt.figure(figsize=(10, 6))

plotted_count = 0
for ds in datasets:
    x, y = load_thresholds(ds["file"])
    if x and y:
        plt.plot(x, y, ds["style"],
                 linewidth=ds["width"],
                 markersize=6,
                 color=ds["color"],
                 label=ds["label"],
                 alpha=0.8)
        plotted_count += 1

if plotted_count == 0:
    print("\nERROR: No data was plotted. Please check if the 'Thm1_Verify' folder exists and contains CSV files.")
else:
    # 3. Annotation and Formatting
    plt.title("Validation of Theorem 1: switching curver with a larger pi_1", fontsize=16, pad=20)
    plt.xlabel(r"Retailer 2 Inventory ($I_2$)", fontsize=14)
    plt.ylabel(r"Critical Backlog Threshold ($\overline{b}_1$)", fontsize=14)

    # Add region labels
    plt.text(6, 10, "DISPATCH REGION\n(Above Curves)", color='green', weight='bold', fontsize=11, ha='center')
    plt.text(12, 2, "WAIT REGION\n(Below Curves)", color='gray', weight='bold', fontsize=11, ha='center')

    plt.legend(loc='upper right', frameon=True, fontsize=11, framealpha=0.95)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.tight_layout()

    # 4. Save
    outfile = "Thm1_Verify_p1larger/Theorem1_SwitchingCurve_p1larger.png"
    plt.savefig(outfile, dpi=300)
    print(f"\nFigure saved to {outfile}")
    # plt.show()