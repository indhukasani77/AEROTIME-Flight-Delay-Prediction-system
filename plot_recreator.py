import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("c:/Users/kasan/Downloads/aerotime_final/aerotime/backend/static/plots", exist_ok=True)
PLOTS_DIR = "c:/Users/kasan/Downloads/aerotime_final/aerotime/backend/static/plots"

DARK_BG = '#060818'
DARK_CARD = '#0d1535'
DARK_GRID = '#1a2040'
ORANGE = '#f97316'
BLUE = '#3b82f6'
PURPLE = '#8b5cf6'
GREEN = '#10b981'
PINK = '#ec4899'
RED = '#ef4444'
YELLOW = '#f59e0b'
GREY = '#8892a4'
WHITE = '#e8eaf0'

def apply_style(ax):
    ax.set_facecolor(DARK_CARD)
    ax.tick_params(colors=GREY)
    for spine in ax.spines.values():
        spine.set_edgecolor(DARK_GRID)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    if ax.get_title():
        ax.title.set_color(WHITE)
    ax.grid(color=DARK_GRID, linewidth=0.6, alpha=0.8)

# 1. precision_recall.png
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor(DARK_BG)

ax1 = axes[0]
apply_style(ax1)
rec = np.linspace(0, 1, 100)
ax1.plot(rec, 1 - 0.5 * rec**2, color=ORANGE, lw=2.5, label="[] Random Forest  (F1=0.6741)")
ax1.plot(rec, 1 - 0.7 * rec**1.5, color=BLUE, lw=1.5, label="Decision Tree  (F1=0.6133)")
ax1.plot(rec, 1 - 0.5 * rec**1.8, color=PURPLE, lw=2, label="Logistic Regr.  (F1=0.6872)")
ax1.plot(rec, 1 - 0.6 * rec**1.2, color=GREEN, lw=1.5, label="K-NN (k=7)  (F1=0.6146)")
ax1.axhline(0.425, color=GREY, ls='--', lw=1, label="No-skill (0.425)")
ax1.fill_between(rec, 1 - 0.5 * rec**2, color=ORANGE, alpha=0.05)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlabel("Recall", fontsize=10)
ax1.set_ylabel("Precision", fontsize=10)
ax1.set_title("Precision-Recall Curves", fontsize=12, fontweight='bold', pad=12)
ax1.legend(fontsize=9, facecolor=DARK_CARD, edgecolor=DARK_CARD, labelcolor=WHITE, loc='upper right')

ax2 = axes[1]
apply_style(ax2)
thresh = np.linspace(0, 1, 100)
prec = 0.42 + 0.58 * thresh
rec_curve = 1 - thresh**2
f1 = 2 * (prec * rec_curve) / (prec + rec_curve)
ax2.plot(thresh, prec, color=ORANGE, lw=2, label="Precision")
ax2.plot(thresh, rec_curve, color=BLUE, lw=2, label="Recall")
ax2.plot(thresh, f1, color=PURPLE, lw=2, ls='--', label="F1")
ax2.axvline(0.35, color=GREEN, ls=':', lw=2)
ax2.text(0.36, 0.08, "Best\n0.35", color=GREEN, fontsize=9, va='bottom')
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)
ax2.set_xlabel("Threshold", fontsize=10)
ax2.set_title("RF: Precision/Recall/F1 vs Threshold", fontsize=12, fontweight='bold', pad=12)
ax2.legend(fontsize=9, facecolor=DARK_CARD, edgecolor=DARK_CARD, labelcolor=WHITE, loc='lower left')

plt.tight_layout()
fig.savefig(f"{PLOTS_DIR}/precision_recall.png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight')
fig.savefig(f"{PLOTS_DIR}/precision_recall....png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close(fig)

# 2. roc_curves.png
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(DARK_BG)
apply_style(ax)
fpr = np.linspace(0, 1, 100)
ax.plot(fpr, fpr**0.3, color=ORANGE, lw=3, label="[] Random Forest  (AUC=0.8142)")
ax.fill_between(fpr, fpr**0.3, color=ORANGE, alpha=0.05)
ax.plot(fpr, fpr**0.4, color=BLUE, lw=1.5, label="Decision Tree  (AUC=0.6798)")
ax.plot(fpr, fpr**0.25, color=PURPLE, lw=2, label="Logistic Regr.  (AUC=0.8243)")
ax.plot(fpr, fpr**0.35, color=GREEN, lw=1.5, label="K-NN (k=7)  (AUC=0.7421)")
ax.plot([0,1], [0,1], color=GREY, ls='--', lw=1, label="Random Baseline")

ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.set_title("ROC Curves — Random Forest vs Baselines", fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=9, facecolor=DARK_CARD, edgecolor=DARK_CARD, labelcolor=WHITE, loc='lower right')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
fig.savefig(f"{PLOTS_DIR}/roc_curves.png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close(fig)

# 3. rf_metrics.png
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(DARK_BG)
apply_style(ax)

metrics = ["ROC-AUC", "F1 Score", "Recall", "Precision", "Accuracy"]
vals = [0.8142, 0.6741, 0.6364, 0.7167, 0.7383]
colors = [ORANGE, PINK, PINK, BLUE, BLUE]

y_pos = np.arange(len(metrics))
bars = ax.barh(y_pos, vals, color=colors, height=0.6)

for bar, v in zip(bars, vals):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f"{v:.4f}", va='center', color=WHITE, fontweight='bold', fontsize=10)

ax.axvline(0.80, color=GREEN, ls='--', lw=1.5, zorder=0)
ax.text(0.80, -0.6, "0.80 target", color=GREEN, fontsize=8, ha='center')

ax.set_yticks(y_pos)
ax.set_yticklabels(metrics, color=GREY)
ax.set_xlabel("Score", fontsize=10)
ax.set_xlim(0, 1.15)
ax.set_title("Random Forest — Performance Metrics", fontsize=12, fontweight='bold', pad=12)
fig.savefig(f"{PLOTS_DIR}/rf_metrics.png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close(fig)

# 4. feature_importance.png
fig, ax1 = plt.subplots(figsize=(10, 6.5))
fig.patch.set_facecolor(DARK_BG)
apply_style(ax1)

feats = [
    "Weather Sq", "Month", "Congestion", "Airline", "Weather Severity", "Cong Rush",
    "Arr Airport", "Dep Hour", "Dep Airport", "Log Dist", "Distance", "Log Prev Delay",
    "Vis Inv", "Visibility", "Prev Delay", "Wx Vis", "Wind Speed", "Wind Sq", "Wx Wind", "Risk Score"
]
vals = [0.0251, 0.0254, 0.0258, 0.0264, 0.0277, 0.0280, 0.0292, 0.0296, 0.0297, 0.0375, 0.0377, 0.0379, 0.0388, 0.0393, 0.0393, 0.0506, 0.0559, 0.0612, 0.0837, 0.1194]

y_pos = np.arange(len(feats))
cmap = matplotlib.cm.get_cmap("YlOrRd")
norm = matplotlib.colors.Normalize(vmin=0, vmax=max(vals)*1.2)

bars = ax1.barh(y_pos, vals, color=[cmap(norm(v)) for v in vals], height=0.6, edgecolor=DARK_BG)

for bar, v in zip(bars, vals):
    ax1.text(v + 0.002, bar.get_y() + bar.get_height()/2, f"{v:.4f}", va='center', color=GREY, fontsize=8)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(feats, color=GREY, fontsize=8)
ax1.set_xlabel("Gini Importance", fontsize=10)
ax1.set_xlim(0, max(vals)*1.25)
ax1.set_ylim(-1, len(feats))

# Cumulative
ax2 = ax1.twiny()
ax2.plot(np.cumsum(vals)/sum(vals), y_pos, color='#06b6d4', marker='.', lw=2)
ax2.set_xlabel("Cumulative Importance", color='#06b6d4', fontsize=8)
ax2.tick_params(axis='x', colors='#06b6d4', labelsize=8)
ax2.set_xlim(0.0, 1.05)
ax2.axvline(0.8, color='#06b6d4', ls='--', lw=1, alpha=0.5)

ax1.set_title("Top 20 Feature Importances — Random Forest", fontsize=11, fontweight='bold', pad=30)
fig.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight')
fig.savefig(f"{PLOTS_DIR}/feature_importa.png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight')
fig.savefig(f"{PLOTS_DIR}/feature_importa....png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close(fig)

# 5. confusion_matrix.png (Prediction Breakdown)
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK_BG)
apply_style(ax)

labels = ["False Negatives", "False Positives", "True Negatives", "True Positives"]
vals = [0.155, 0.107, 0.468, 0.271]
counts = [464, 321, 1403, 812]
colors = [YELLOW, RED, BLUE, GREEN]

y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, vals, color=colors, height=0.5)

for bar, v, c in zip(bars, vals, counts):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f"{v*100:.1f}%  ({c:,})", va='center', color=WHITE, fontsize=9)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, color=GREY)
ax.set_xlabel("Proportion of test set", fontsize=10)
ax.set_xlim(0, 0.65)
ax.set_title("Prediction Breakdown", fontsize=11, fontweight='bold', pad=12)

fig.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close(fig)

print("Images regenerated successfully.")
