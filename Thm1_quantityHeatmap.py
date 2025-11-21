import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set academic plotting style (white background)
sns.set_theme(style="white", context="paper", font_scale=1.2)

def parse_policy_csv(filename, target_r):
    """
    Parses the custom CSV format to extract the policy matrix for a specific epoch.
    The file contains multiple matrices separated by headers like 'epoch r=20'.
    """
    print(f"Reading file: {filename} for r={target_r}...")
    matrix_data = []
    reading = False
    header_found = False
    cols = []

    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return None

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Check for the start of the target epoch section
        # Note: We look for exact match or ensure we don't match partials like r=200 when looking for r=20
        if f"epoch r={target_r}" in line:
            reading = True
            continue

        # Stop reading if we hit the next epoch header
        if reading and "epoch r=" in line:
            break

        if reading:
            # Skip empty lines
            if not line:
                continue

            # The first line after 'epoch' tag is usually the header (IA\IB, -20, -19...)
            if not header_found:
                # Split by comma
                parts = line.split(',')
                # The first part is "IA\IB", the rest are column headers (I2 values)
                # We convert the headers to integers
                try:
                    cols = [int(x) for x in parts[1:] if x]
                    header_found = True
                except ValueError:
                    continue  # Skip if header line is malformed
            else:
                # Data rows
                parts = line.split(',')
                if len(parts) < 2: continue

                try:
                    row_idx = int(parts[0])  # This is b1 (Backlog)
                    vals = [int(x) for x in parts[1:] if x]

                    # Store as a dictionary or list row
                    row_dict = {'b1': row_idx}
                    for c_idx, val in enumerate(vals):
                        if c_idx < len(cols):
                            row_dict[cols[c_idx]] = val
                    matrix_data.append(row_dict)
                except ValueError:
                    continue

    if not matrix_data:
        print(f"Error: Could not find data for r={target_r}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(matrix_data)
    df = df.set_index('b1')

    # Sort index and columns to ensure the heatmap is oriented correctly
    df = df.sort_index(ascending=False)  # High backlog at top
    df = df.sort_index(axis=1)  # Low inventory to High inventory left-to-right

    return df

def plot_heatmap(df_policy, title_suffix, output_filename):
    if df_policy is None:
        return

    plt.figure(figsize=(12, 8))

    # Filter columns to show a relevant range if the grid is huge
    # For visualization, -5 to 20 is usually the interesting area
    cols_to_show = [c for c in df_policy.columns if c >= -5]
    df_plot = df_policy[cols_to_show]

    # Create Heatmap
    # We use a custom colour map: 0 (Wait) is light grey, higher q is darker blue/green
    cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=False, as_cmap=True)

    # Mask zeros to make the "Wait" region distinct (e.g., white or grey)
    mask = df_plot == 0

    ax = sns.heatmap(df_plot,
                     annot=True,  # Show numbers in cells
                     fmt="d",  # Integer format
                     cmap="Blues",  # Colour scheme
                     mask=mask,  # Hide 0s from the colour map
                     cbar_kws={'label': 'Dispatch Quantity ($q^*$)'},
                     linewidths=.5,
                     linecolor='lightgrey')

    # Colour the masked values (Wait region) explicitly
    ax.set_facecolor('#f7f7f7')  # Light grey for "Wait" action

    # Formatting
    plt.title(f"Optimal Dispatch Policy ($q^*$) Heatmap\n{title_suffix}", fontsize=16)
    plt.xlabel("Retailer 2 Inventory Level ($I_2$)", fontsize=14)
    plt.ylabel("Retailer 1 Backlog Level ($b_1$)", fontsize=14)

    # Add an annotation explaining the logic
    plt.text(len(df_plot.columns) + 1, 1,
             "Numbers indicate\ndispatch quantity.\n\nEmpty/Grey cells\nindicate 'Wait'.",
             fontsize=12, ha='left')

    plt.tight_layout()

    # Save
    plt.savefig(output_filename, dpi=300)
    print(f"Heatmap saved to {output_filename}")
    plt.close() # Close to free memory for next loop

# --- Main Execution ---



# change csv paths as needed
configs = [
    {
        "file": "Thm1_Verify_p1larger/Thm1_Base_policy.csv",
        "r": 20,
        "title": "Base Case (N=20)",
        "out": "Thm1_Verify_p1larger/Policy_Heatmap_N20_p1larger.png"
    },
    {
        "file": "Thm1_Verify_p1larger/Thm1_Base_N100_policy.csv",
        "r": 100,
        "title": "High Res (N=100)",
        "out": "Thm1_Verify_p1larger/Policy_Heatmap_N100_p1larger.png"
    },
    {
        "file": "Thm1_Verify_p1larger/Thm1_Base_N200_policy.csv",
        "r": 200,
        "title": "Very High Res (N=200)",
        "out": "Thm1_Verify_p1larger/Policy_Heatmap_N200_p1larger.png"
    }
]

for conf in configs:
    df = parse_policy_csv(conf["file"], target_r=conf["r"])
    plot_heatmap(df, conf["title"], conf["out"])