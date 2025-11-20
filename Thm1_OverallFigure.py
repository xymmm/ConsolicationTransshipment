import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set academic style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def load_thresholds(csv_file):
    """
    Extracts the switching curve for r=20.
    Handles empty/NaN values robustly to ensure plotting works.
    """
    print(f"Loading {csv_file}...")
    try:
        # Read CSV. Assuming first column is 'r'.
        df = pd.read_csv(csv_file, index_col=0)

        # We target r=20 (start of horizon)
        if 20 in df.index:
            data = df.loc[20]
        else:
            print(f"  Warning: r=20 not found. Using first row available.")
            data = df.iloc[0]

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

        print(f"  Found {len(x)} valid data points.")
        return x, y
    except Exception as e:
        print(f"  Error loading {csv_file}: {e}")
        return [], []


# --- Main Execution ---

# 1. Load Data (Base vs High Penalty)
x_base, y_base = load_thresholds("Thm1_Base_critical_surface.csv")
x_high, y_high = load_thresholds("Thm1_HighPenalty_critical_surface.csv")

# 2. Create Plot
plt.figure(figsize=(8, 5))

# Plot Base Case (Blue Circles)
if x_base:
    plt.plot(x_base, y_base, marker='o', markersize=6, linestyle='-', linewidth=2,
             color='#1f77b4', label=r'Base Case ($\pi=10$)')

# Plot High Penalty Case (Red Triangles)
if x_high:
    plt.plot(x_high, y_high, marker='^', markersize=6, linestyle='--', linewidth=2,
             color='#d62728', label=r'High Penalty ($\pi=20$)')

# 3. Annotation and Formatting
plt.title("Validation of Theorem 1: Switching Curves", fontsize=14)
plt.xlabel(r"Retailer 2 Inventory ($I_2$)", fontsize=12)
plt.ylabel(r"Critical Backlog Threshold ($\overline{b}_1$)", fontsize=12)

# Add region labels (Positions adjusted to be generally safe)
plt.text(5, 12, "DISPATCH REGION\n(Above Curve)", color='green', weight='bold', fontsize=10)
plt.text(12, 2, "WAIT REGION\n(Below Curve)", color='gray', weight='bold', fontsize=10)

plt.legend(loc='best')
plt.grid(True, alpha=0.4)
plt.tight_layout()

# 4. Save
plt.savefig("Theorem1_Validation_Plot.png", dpi=300)
print("Figure saved to Theorem1_Validation_Plot.png")
plt.show()