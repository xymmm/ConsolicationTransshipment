import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set academic plotting style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)


def load_thresholds(csv_file, target_r):
    """
    Extracts the switching curve for a specific time step r.
    Filters out empty values to ensure the plotted line is continuous.
    """
    print(f"Reading {csv_file} for epoch r={target_r}...")

    if not os.path.exists(csv_file):
        print(f"  [Error] File not found: {csv_file}")
        return [], []

    try:
        # Read CSV. index_col=0 handles the 'r' column
        df = pd.read_csv(csv_file, index_col=0)

        # Locate the target row (Start of Horizon)
        if target_r in df.index:
            data = df.loc[target_r]
        else:
            print(f"  [Warning] r={target_r} not found in {csv_file}. Using first row.")
            data = df.iloc[0]

        x = []  # Inventory (I2)
        y = []  # Critical Backlog (b1)

        for col in df.columns:
            try:
                # Column headers are Inventory Levels
                inv_val = int(col)
                val = data[col]

                # Robust check for valid numbers (filters out NaNs and empty strings)
                if pd.notna(val) and str(val).strip() != "":
                    x.append(inv_val)
                    y.append(float(val))
            except ValueError:
                continue

        # Sort by Inventory (x) to ensure the line connects correctly from left to right
        if x:
            xy_sorted = sorted(zip(x, y))
            x = [i[0] for i in xy_sorted]
            y = [i[1] for i in xy_sorted]
            print(f"  -> Extracted {len(x)} valid points.")
        else:
            print("  -> No valid points found.")

        return x, y

    except Exception as e:
        print(f"  [Error] Exception reading file: {e}")
        return [], []


# --- Configuration ---

datasets = [
    {
        "file": "Thm1_Base_critical_surface.csv",
        "r": 20,
        "label": "Low Res (N=20)",
        "color": "#999999",  # Grey
        "style": "--o",  # Dashed with circles
        "width": 1.5,
        "alpha": 0.6
    },
    {
        "file": "Thm1_Base_N100_critical_surface.csv",
        "r": 100,
        "label": "High Res (N=100)",
        "color": "#1f77b4",  # Blue
        "style": "-s",  # Solid with squares
        "width": 2,
        "alpha": 0.8
    },
    {
        "file": "Thm1_Base_N200_critical_surface.csv",
        "r": 200,
        "label": "Very High Res (N=200)",
        "color": "#d62728",  # Red
        "style": "-^",  # Solid with triangles
        "width": 2.5,
        "alpha": 1.0
    }
]

# --- Plotting ---
plt.figure(figsize=(10, 6))

plotted_count = 0
for ds in datasets:
    x, y = load_thresholds(ds["file"], ds["r"])
    if x and y:
        plt.plot(x, y, ds["style"],
                 linewidth=ds["width"],
                 markersize=6,
                 color=ds["color"],
                 label=ds["label"],
                 alpha=ds["alpha"])
        plotted_count += 1

if plotted_count == 0:
    print("\nERROR: No data was plotted. Please ensure CSV files are in the same folder.")
else:
    # Formatting
    plt.title("Convergence of Optimal Switching Curves\n(Base Case: $C_f=20, \pi=10$)", fontsize=16, pad=15)
    plt.xlabel("Retailer 2 Inventory ($I_2$)", fontsize=14)
    plt.ylabel("Critical Backlog Threshold ($\overline{b}_1$)", fontsize=14)

    # Add region labels
    plt.text(6, 13, "DISPATCH REGION", color='green', weight='bold', fontsize=11, ha='center')
    plt.text(12, 2, "WAIT REGION", color='gray', weight='bold', fontsize=11, ha='center')

    # Arrow indicating convergence direction (optional, adjust coordinates if needed)
    plt.annotate('Convergence', xy=(6, 10), xytext=(6, 12),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=11, ha='center')

    plt.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save
    outfile = "Theorem1_Convergence_Plot.png"
    plt.savefig(outfile, dpi=300)
    print(f"\nSuccess! Convergence Plot saved to {outfile}")
    # plt.show()