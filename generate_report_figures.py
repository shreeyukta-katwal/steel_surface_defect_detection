import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Style and Environment Setup
# ---------------------------------------------------------
# Create figures directory
os.makedirs('figures', exist_ok=True)

# Set clean professional styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#e2e8f0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#e2e8f0'
})

# Define a cohesive professional color palette
COLOR_NEUTRAL = '#4A7BB0'  # Soft professional steel blue
COLOR_WARN = '#E05A47'     # Vibrant tomato red for underrepresented class
COLOR_BASE_VL = '#5B9BD5'  # Lighter blue for baseline val-loss
COLOR_BASE_PF = '#1F4E79'  # Dark solid blue for baseline peak-f1
COLOR_CBAM_VL = '#F4B183'  # Lighter orange for cbam val-loss
COLOR_CBAM_PF = '#C55A11'  # Solid rust orange/red for cbam peak-f1

# ---------------------------------------------------------
# FIGURE 1: Class Distribution Bar Chart
# ---------------------------------------------------------
print("Generating Figure 1: Class Distribution...")
classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
counts = [536, 135, 3341, 905]

fig1, ax1 = plt.subplots(figsize=(8, 5))

# Draw bars with distinct warning color for Class 2
colors_fig1 = [COLOR_NEUTRAL, COLOR_WARN, COLOR_NEUTRAL, COLOR_NEUTRAL]
bars = ax1.bar(classes, counts, color=colors_fig1, width=0.6, edgecolor='#2c3e50', linewidth=0.8)

# Add exact counts above the bars
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 4),  # 4 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=11, fontweight='bold',
                 color='#2d3748')

# Customizations
ax1.set_xlabel('Defect Class', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Number of Training Images (positive instances)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_title('Class Distribution in Severstal Training Set (n=8,797)', fontsize=13, fontweight='bold', pad=15)
ax1.set_ylim(0, 3700)
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
ax1.tick_params(axis='both', which='major', labelsize=11)

# Subtle decoration for Class 2 to draw attention
ax1.annotate('Severely Underrepresented\n(2.5% of positive instances)',
             xy=('Class 2', 135),
             xytext=(0.8, 800),
             arrowprops=dict(facecolor='#2d3748', edgecolor='#2d3748', arrowstyle="->", lw=1.2, connectionstyle="arc3,rad=-0.1"),
             fontsize=10, color='#9b2c2c', fontweight='bold', ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f5", ec="#feb2b2", lw=0.8))

plt.tight_layout()
fig1.savefig('figures/fig4_class_distribution.png')
plt.close(fig1)


# ---------------------------------------------------------
# FIGURE 2: Macro F1 Comparison Across Seeds
# ---------------------------------------------------------
print("Generating Figure 2: Macro F1 Comparison...")
seeds = ['Seed 42', 'Seed 0', 'Seed 7']
x = np.arange(len(seeds))
width = 0.2

# Series data
base_vl = [0.8341, 0.8367, 0.8628]
base_pf = [0.8598, 0.8629, 0.8671]
cbam_vl = [0.8777, 0.8318, 0.8484]
cbam_pf = [0.8777, 0.8695, 0.8740]

# Calculate means and stds for dashed lines and legend formatting
means = {
    'base_vl': np.mean(base_vl),
    'base_pf': np.mean(base_pf),
    'cbam_vl': np.mean(cbam_vl),
    'cbam_pf': np.mean(cbam_pf)
}

fig2, ax2 = plt.subplots(figsize=(10, 6.5))

# Draw bars with consistent colors and hatch patterns
rects1 = ax2.bar(x - 1.5 * width, base_vl, width, label='Baseline (Val-Loss): 0.8445 ± 0.0159',
                 color=COLOR_BASE_VL, edgecolor='#1f4e79', linewidth=1, hatch='///')
rects2 = ax2.bar(x - 0.5 * width, base_pf, width, label='Baseline (Peak-F1): 0.8633 ± 0.0037',
                 color=COLOR_BASE_PF, edgecolor='#0f283e', linewidth=1)
rects3 = ax2.bar(x + 0.5 * width, cbam_vl, width, label='CBAM (Val-Loss): 0.8526 ± 0.0232',
                 color=COLOR_CBAM_VL, edgecolor='#c55a11', linewidth=1, hatch='\\\\\\')
rects4 = ax2.bar(x + 1.5 * width, cbam_pf, width, label='CBAM (Peak-F1): 0.8737 ± 0.0041',
                 color=COLOR_CBAM_PF, edgecolor='#7a370a', linewidth=1)

# Draw subtle horizontal reference lines for series means
ax2.axhline(means['base_vl'], color=COLOR_BASE_VL, linestyle='--', alpha=0.6, linewidth=1, zorder=1)
ax2.axhline(means['base_pf'], color=COLOR_BASE_PF, linestyle='--', alpha=0.6, linewidth=1, zorder=1)
ax2.axhline(means['cbam_vl'], color=COLOR_CBAM_VL, linestyle='--', alpha=0.6, linewidth=1, zorder=1)
ax2.axhline(means['cbam_pf'], color=COLOR_CBAM_PF, linestyle='--', alpha=0.6, linewidth=1, zorder=1)

# Customizations
ax2.set_xlabel('Random Seed Context', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_title('Macro F1 by Seed: Val-Loss vs Peak-F1 Checkpoint Selection', fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(seeds, fontsize=11)
ax2.set_ylim(0.80, 0.90)
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

# Legend settings
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True, fontsize=10.5)

# Tight layout adjustments to prevent clipping the legend
plt.tight_layout()
fig2.savefig('figures/fig6_macro_f1_comparison.png', bbox_inches='tight')
plt.close(fig2)


# ---------------------------------------------------------
# FIGURE 3: Segmentation Dice Score vs Epoch
# ---------------------------------------------------------
print("Generating Figure 3: Segmentation Dice Score vs Epoch...")

# Load CSV files
df_base = pd.read_csv('logs/baseline_history.csv')
df_cbam = pd.read_csv('logs/cbam_history.csv')

fig3, ax3 = plt.subplots(figsize=(10, 6))

# Plot lines using actual data
line_base, = ax3.plot(df_base['epoch'], df_base['dice'], color='#2980B9', linewidth=2.2, label='Baseline U-Net')
line_cbam, = ax3.plot(df_cbam['epoch'], df_cbam['dice'], color='#E67E22', linewidth=2.2, label='CBAM U-Net')

# Highlight early volatility region (epochs 1 to 9)
# Use axvspan with a soft red overlay to represent volatility/instability
ax3.axvspan(1, 9, color='#E74C3C', alpha=0.07, zorder=0, label='Early instability (Epochs 1–9)')

# Text label inside volatility region
ax3.text(5, 0.63, "Early instability\n(oscillations: 0.58–0.79)", color='#C0392B', fontsize=10.5,
         fontweight='bold', ha='center', va='center',
         bbox=dict(facecolor='white', alpha=0.85, edgecolor='#E74C3C', boxstyle='round,pad=0.4', lw=0.5))

# Mark final epoch values (Epoch 40)
# Base: 0.8372, CBAM: 0.8368
val_base_40 = df_base.loc[df_base['epoch'] == 40, 'dice'].values[0]
val_cbam_40 = df_cbam.loc[df_cbam['epoch'] == 40, 'dice'].values[0]

# Scatter plot of these final points
ax3.scatter(40, val_base_40, color='#1F4E79', s=55, zorder=5, edgecolor='white', linewidth=1)
ax3.scatter(40, val_cbam_40, color='#C55A11', s=55, zorder=5, edgecolor='white', linewidth=1)

# Annotate final values with offset to prevent overlap
ax3.annotate(f'Baseline Final: {val_base_40:.4f}',
             xy=(40, val_base_40),
             xytext=(40.6, val_base_40 + 0.007),
             fontsize=10.5, color='#1F4E79', fontweight='bold',
             arrowprops=dict(arrowstyle="-", color='#1F4E79', alpha=0.5, lw=1))

ax3.annotate(f'CBAM Final: {val_cbam_40:.4f}',
             xy=(40, val_cbam_40),
             xytext=(40.6, val_cbam_40 - 0.011),
             fontsize=10.5, color='#C55A11', fontweight='bold',
             arrowprops=dict(arrowstyle="-", color='#C55A11', alpha=0.5, lw=1))

# Mark CBAM peak value (Epoch 38: Dice = 0.8393)
# Find actual peak from data
peak_epoch = df_cbam['dice'].idxmax() + 1
peak_val = df_cbam['dice'].max()
print(f"CBAM Peak detected at Epoch {peak_epoch} with Dice={peak_val:.4f}")

# Plot a golden star marker at peak
ax3.scatter(peak_epoch, peak_val, color='#F1C40F', marker='*', s=150, zorder=6, edgecolor='#D35400', linewidth=1.2, label='CBAM Peak Dice')

# Annotate peak value with curved arrow
ax3.annotate(f'CBAM Peak: {peak_val:.4f} (Epoch {peak_epoch})',
             xy=(peak_epoch, peak_val),
             xytext=(peak_epoch - 14, peak_val + 0.015),
             fontsize=10.5, color='#D35400', fontweight='bold',
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color='#D35400', lw=1.2),
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='#E67E22', boxstyle='round,pad=0.3', lw=0.5))

# Plot customizations
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold', labelpad=10)
ax3.set_ylabel('Dice Score', fontsize=12, fontweight='bold', labelpad=10)
ax3.set_title('Segmentation Dice Score vs Training Epoch: Baseline vs CBAM U-Net', fontsize=13, fontweight='bold', pad=15)
ax3.set_xlim(0, 44)  # Extend limits to make room for final epoch label
ax3.set_ylim(0.55, 0.87)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.tick_params(axis='both', which='major', labelsize=11)

# Position legend
ax3.legend(loc='lower right', frameon=True, fontsize=10.5)

plt.tight_layout()
fig3.savefig('figures/fig7_dice_vs_epoch.png')
plt.close(fig3)


# ---------------------------------------------------------
# FIGURE 4: Train vs Validation Loss — Classification
# ---------------------------------------------------------
print("Generating Figure 4: Train vs Validation Loss (Classification)...")

df_base_cls = pd.read_csv('logs/baseline_seed42.csv')
df_cbam_cls = pd.read_csv('logs/cbam_seed42.csv')

fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

TRAIN_COLOR_BASE = '#1F4E79'   # deep navy   – Baseline train
VAL_COLOR_BASE   = '#5B9BD5'   # sky blue    – Baseline val
TRAIN_COLOR_CBAM = '#C55A11'   # rust orange – CBAM train
VAL_COLOR_CBAM   = '#F4B183'   # peach       – CBAM val

for ax, df, model_label, tc, vc in [
    (axes4[0], df_base_cls, 'Baseline ResNet18', TRAIN_COLOR_BASE, VAL_COLOR_BASE),
    (axes4[1], df_cbam_cls, 'CBAM-ResNet18',    TRAIN_COLOR_CBAM, VAL_COLOR_CBAM),
]:
    epochs = df['epoch']
    train  = df['train_loss']
    val    = df['val_loss']

    ax.plot(epochs, train, color=tc, linewidth=2.2, label='Train Loss',
            marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.4)
    ax.plot(epochs, val,   color=vc, linewidth=2.2, label='Val Loss',
            marker='s', markersize=4, markerfacecolor='white', markeredgewidth=1.4,
            linestyle='--')

    # Shade the gap between train and val to highlight overfitting
    ax.fill_between(epochs, train, val,
                    where=(val >= train), interpolate=True,
                    color=vc, alpha=0.12, label='Val > Train (gap)')
    ax.fill_between(epochs, train, val,
                    where=(val < train), interpolate=True,
                    color=tc, alpha=0.10, label='Train > Val')

    # Mark best val epoch
    best_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
    best_val   = df['val_loss'].min()
    ax.scatter(best_epoch, best_val, color='#F1C40F', marker='*', s=160,
               zorder=6, edgecolor='#8B6914', linewidth=1.2, label=f'Best Val (Ep {int(best_epoch)})')

    ax.set_title(model_label, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold', labelpad=8)
    ax.legend(fontsize=9.5, loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=10)

fig4.suptitle('Train vs Validation Loss — Classification (Seed 42)',
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig4.savefig('figures/fig8_classification_train_val_loss.png', bbox_inches='tight')
plt.close(fig4)
print("  Saved: figures/fig8_classification_train_val_loss.png")


# ---------------------------------------------------------
# FIGURE 5: Train vs Validation Loss — Segmentation
# ---------------------------------------------------------
print("Generating Figure 5: Train vs Validation Loss (Segmentation)...")

fig5, axes5 = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

for ax, df, model_label, tc, vc in [
    (axes5[0], df_base, 'Baseline U-Net', TRAIN_COLOR_BASE, VAL_COLOR_BASE),
    (axes5[1], df_cbam, 'CBAM U-Net',    TRAIN_COLOR_CBAM, VAL_COLOR_CBAM),
]:
    epochs = df['epoch']
    train  = df['train_loss']
    val    = df['val_loss']

    ax.plot(epochs, train, color=tc, linewidth=2.2, label='Train Loss',
            marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.4)
    ax.plot(epochs, val,   color=vc, linewidth=2.2, label='Val Loss',
            marker='s', markersize=4, markerfacecolor='white', markeredgewidth=1.4,
            linestyle='--')

    ax.fill_between(epochs, train, val,
                    where=(val >= train), interpolate=True,
                    color=vc, alpha=0.12, label='Val > Train (gap)')
    ax.fill_between(epochs, train, val,
                    where=(val < train), interpolate=True,
                    color=tc, alpha=0.10, label='Train > Val')

    # Mark best val epoch
    best_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
    best_val   = df['val_loss'].min()
    ax.scatter(best_epoch, best_val, color='#F1C40F', marker='*', s=160,
               zorder=6, edgecolor='#8B6914', linewidth=1.2, label=f'Best Val (Ep {int(best_epoch)})')

    ax.set_title(model_label, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold', labelpad=8)
    ax.legend(fontsize=9.5, loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=10)

fig5.suptitle('Train vs Validation Loss — Segmentation',
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig5.savefig('figures/fig9_segmentation_train_val_loss.png', bbox_inches='tight')
plt.close(fig5)
print("  Saved: figures/fig9_segmentation_train_val_loss.png")


print("All figures generated successfully!")
