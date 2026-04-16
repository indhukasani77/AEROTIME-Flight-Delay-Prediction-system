"""
AeroTime — train_model.py  (updated)
Run this ONCE before starting app.py:
    python train_model.py

Trains:
  1. Random Forest   (baseline)
  2. XGBoost         (primary / production)

Saves to  models/:
  random_forest.pkl
  xgboost_model.pkl
  label_encoder.pkl
  model_stats.pkl    ← loaded by /api/model-stats route

Generates plots to  static/plots/:
  confusion_matrix.png
  roc_curves.png
  feature_importance.png
  precision_recall.png
  rf_metrics.png
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing     import LabelEncoder, label_binarize
import xgboost as xgb

# ── PATHS ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
PLOTS_DIR  = os.path.join(BASE_DIR, 'static', 'plots')
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURES = ['weather_severity', 'congestion', 'wind_speed', 'visibility',
            'dep_hour', 'month', 'day_of_week', 'airline_enc', 'distance']
CLASS_NAMES = ['On Time', 'Minor Delay', 'Major Delay']

# ── shared dark style ──
DARK_BG   = '#060818'
DARK_CARD = '#0d1535'
DARK_GRID = '#1a2040'
ORANGE    = '#f97316'
GREEN     = '#10b981'
BLUE      = '#3b82f6'
PURPLE    = '#8b5cf6'
PINK      = '#ec4899'
TEAL      = '#06b6d4'
WHITE     = '#e8eaf0'
MUTED     = '#8892a4'

PALETTE   = [ORANGE, GREEN, BLUE, PURPLE, PINK, TEAL]

def apply_dark_style(fig, axes=None):
    fig.patch.set_facecolor(DARK_BG)
    if axes is None:
        return
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(DARK_CARD)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        if ax.get_title():
            ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK_GRID)
        ax.grid(color=DARK_GRID, linewidth=0.6, alpha=0.8)

# ════════════════════════════════════════════════
#  STEP 1 — BUILD / LOAD DATASET
# ════════════════════════════════════════════════
CSV_PATH = os.path.join(BASE_DIR, 'data', 'flights.csv')

if os.path.exists(CSV_PATH):
    print(f"📂 Loading real dataset from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    REQUIRED = ['weather_severity', 'congestion', 'wind_speed', 'visibility',
                'dep_hour', 'month', 'day_of_week', 'airline', 'distance', 'delay_minutes']
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"⚠️  Missing columns: {missing} — falling back to synthetic data.")
        df = None
else:
    df = None

if df is None:
    print("🔧 Generating synthetic training data (10,000 flights)…")
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({
        'weather_severity': np.random.randint(1, 11, n),
        'congestion':       np.random.randint(1, 11, n),
        'wind_speed':       np.random.randint(0,  60, n),
        'visibility':       np.random.uniform(1,  15, n),
        'dep_hour':         np.random.randint(0,  24, n),
        'month':            np.random.randint(1,  13, n),
        'day_of_week':      np.random.randint(0,   7, n),
        'airline':          np.random.choice(['AA','DL','UA','WN','B6','AS','NK'], n),
        'distance':         np.random.randint(200, 3000, n),
    })
    AIRLINE_BIAS = {'AA': 5, 'DL': -3, 'UA': 4, 'WN': 8, 'B6': 3, 'AS': -5, 'NK': 12}

    def calc_delay(row):
        d  = row['weather_severity'] * 3.5 + row['congestion'] * 2.8
        d += max(0, row['wind_speed'] - 20) * 0.8
        d += max(0, 5 - row['visibility']) * 6
        d += 18 if 15 <= row['dep_hour'] <= 19 else 0
        d += AIRLINE_BIAS.get(row['airline'], 0)
        return max(0, d * 0.28)

    df['delay_minutes'] = df.apply(calc_delay, axis=1)

def to_label(mins):
    if   mins < 5:  return 0
    elif mins < 15: return 1
    else:           return 2

df['target']      = df['delay_minutes'].apply(to_label)
le                = LabelEncoder()
df['airline_enc'] = le.fit_transform(df['airline'])

X = df[FEATURES]
y = df['target']

print(f"📊 Dataset: {len(df):,} rows | class distribution: {dict(y.value_counts().sort_index())}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# binarised labels needed for multi-class ROC / PR curves
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# ════════════════════════════════════════════════
#  STEP 2 — TRAIN RANDOM FOREST
# ════════════════════════════════════════════════
print("\n🌲 Training Random Forest…")
rf = RandomForestClassifier(n_estimators=150, max_depth=12,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)
rf_acc   = accuracy_score(y_test, rf_pred)
rf_f1    = f1_score(y_test, rf_pred, average='weighted')
try:
    rf_auc = roc_auc_score(y_test, rf_proba, multi_class='ovr', average='weighted')
except Exception:
    rf_auc = 0.0

print(f"   Accuracy : {rf_acc:.4f}")
print(f"   F1       : {rf_f1:.4f}")
print(f"   AUC-ROC  : {rf_auc:.4f}")

# ════════════════════════════════════════════════
#  STEP 3 — TRAIN XGBOOST
# ════════════════════════════════════════════════
print("\n⚡ Training XGBoost…")
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6,
    learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, random_state=42,
    eval_metric='mlogloss', use_label_encoder=False
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)

xgb_pred  = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)
xgb_acc   = accuracy_score(y_test, xgb_pred)
xgb_f1    = f1_score(y_test, xgb_pred, average='weighted')
try:
    xgb_auc = roc_auc_score(y_test, xgb_proba, multi_class='ovr', average='weighted')
except Exception:
    xgb_auc = 0.0

print(f"   Accuracy : {xgb_acc:.4f}")
print(f"   F1       : {xgb_f1:.4f}")
print(f"   AUC-ROC  : {xgb_auc:.4f}")

cr_dict   = classification_report(y_test, xgb_pred,
                                   target_names=CLASS_NAMES, output_dict=True)
cm_matrix = confusion_matrix(y_test, xgb_pred).tolist()
fi_dict   = dict(zip(FEATURES, xgb_model.feature_importances_.tolist()))
fi_sorted = dict(sorted(fi_dict.items(), key=lambda x: -x[1]))

# ════════════════════════════════════════════════
#  STEP 4 — GENERATE PLOTS
# ════════════════════════════════════════════════
print("\n🎨 Generating plots…")

# ── helpers ──
def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   ✓ {path}")

# ── 1. CONFUSION MATRIX ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
apply_dark_style(fig, ax)

cm_arr = np.array(cm_matrix)
im = ax.imshow(cm_arr, cmap='YlOrRd', aspect='auto')

# cell labels
thresh = cm_arr.max() / 2.0
for i in range(3):
    for j in range(3):
        val = cm_arr[i, j]
        pct = val / cm_arr[i].sum() * 100
        ax.text(j, i, f'{val}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white' if val < thresh else DARK_BG)

ax.set_xticks([0, 1, 2]); ax.set_xticklabels(CLASS_NAMES, fontsize=9, color=WHITE)
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(CLASS_NAMES, fontsize=9, color=WHITE)
ax.set_xlabel('Predicted Label', fontsize=10, color=WHITE, labelpad=8)
ax.set_ylabel('True Label',      fontsize=10, color=WHITE, labelpad=8)
ax.set_title('Confusion Matrix — XGBoost (Primary Model)',
             fontsize=11, color=WHITE, pad=12, fontweight='bold')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors=MUTED, labelsize=8)
cbar.outline.set_edgecolor(DARK_GRID)

# accuracy annotation
acc_txt = f'Accuracy: {xgb_acc*100:.2f}%   F1: {xgb_f1:.4f}   AUC: {xgb_auc:.4f}'
fig.text(0.5, 0.02, acc_txt, ha='center', fontsize=9,
         color=ORANGE, fontweight='bold')

plt.tight_layout(rect=[0, 0.05, 1, 1])
save(fig, 'confusion_matrix.png')

# ── 2. ROC CURVES ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
apply_dark_style(fig, ax)

colors_roc = [ORANGE, GREEN, BLUE, PURPLE]
models_roc = [
    ('XGBoost',       xgb_proba, xgb_auc,  colors_roc[0], '-'),
    ('Random Forest', rf_proba,  rf_auc,   colors_roc[1], '--'),
]

for name, proba, auc_val, col, ls in models_roc:
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
        lbl = f'{name} – {cls}' if len(models_roc) > 1 else cls
        ax.plot(fpr, tpr, color=col, lw=1.4, linestyle=ls, alpha=0.75)

# macro average — XGBoost
fpr_all, tpr_all = {}, {}
for i in range(3):
    fpr_all[i], tpr_all[i], _ = roc_curve(y_test_bin[:, i], xgb_proba[:, i])

all_fpr   = np.unique(np.concatenate(list(fpr_all.values())))
mean_tpr  = np.zeros_like(all_fpr)
for i in range(3):
    mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
mean_tpr /= 3
ax.plot(all_fpr, mean_tpr, color=ORANGE, lw=2.5, linestyle='-',
        label=f'XGBoost Macro AUC = {xgb_auc:.3f}')

# macro average — RF
fpr_rf, tpr_rf = {}, {}
for i in range(3):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_bin[:, i], rf_proba[:, i])
all_fpr_rf  = np.unique(np.concatenate(list(fpr_rf.values())))
mean_tpr_rf = np.zeros_like(all_fpr_rf)
for i in range(3):
    mean_tpr_rf += np.interp(all_fpr_rf, fpr_rf[i], tpr_rf[i])
mean_tpr_rf /= 3
ax.plot(all_fpr_rf, mean_tpr_rf, color=GREEN, lw=2.5, linestyle='--',
        label=f'Random Forest Macro AUC = {rf_auc:.3f}')

ax.plot([0, 1], [0, 1], color=MUTED, lw=1, linestyle=':', alpha=0.6,
        label='Random Classifier')

ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate',  fontsize=10)
ax.set_title('ROC Curves — XGBoost vs Random Forest',
             fontsize=11, color=WHITE, pad=12, fontweight='bold')
ax.legend(fontsize=8, facecolor=DARK_CARD, edgecolor=DARK_GRID,
          labelcolor=WHITE, loc='lower right')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
plt.tight_layout()
save(fig, 'roc_curves.png')

# ── 3. FEATURE IMPORTANCE ───────────────────────
FEAT_LABELS = {
    'weather_severity': 'Weather Severity',
    'congestion':       'Airport Congestion',
    'wind_speed':       'Wind Speed',
    'visibility':       'Visibility',
    'dep_hour':         'Departure Hour',
    'month':            'Month',
    'day_of_week':      'Day of Week',
    'airline_enc':      'Airline',
    'distance':         'Flight Distance',
}

fig, ax = plt.subplots(figsize=(8, 5))
apply_dark_style(fig, ax)

names  = [FEAT_LABELS.get(k, k) for k in fi_sorted.keys()]
values = list(fi_sorted.values())
y_pos  = np.arange(len(names))

bars = ax.barh(y_pos, values, color=[PALETTE[i % len(PALETTE)] for i in range(len(names))],
               edgecolor='none', height=0.65)

for bar, val in zip(bars, values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha='left', fontsize=8.5,
            color=WHITE, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=9, color=WHITE)
ax.set_xlabel('Importance Score', fontsize=10)
ax.set_title('Feature Importance — XGBoost',
             fontsize=11, color=WHITE, pad=12, fontweight='bold')
ax.invert_yaxis()
ax.set_xlim(0, max(values) * 1.18)
plt.tight_layout()
save(fig, 'feature_importance.png')

# ── 4. PRECISION / RECALL CURVES ────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
apply_dark_style(fig, ax)

pr_colors = [ORANGE, GREEN, BLUE]
for i, (cls, col) in enumerate(zip(CLASS_NAMES, pr_colors)):
    prec, rec, _ = precision_recall_curve(y_test_bin[:, i], xgb_proba[:, i])
    ap = average_precision_score(y_test_bin[:, i], xgb_proba[:, i])
    ax.plot(rec, prec, color=col, lw=2,
            label=f'{cls} (AP = {ap:.3f})')
    ax.fill_between(rec, prec, alpha=0.08, color=col)

ax.set_xlabel('Recall',    fontsize=10)
ax.set_ylabel('Precision', fontsize=10)
ax.set_title('Precision-Recall Curves — XGBoost',
             fontsize=11, color=WHITE, pad=12, fontweight='bold')
ax.legend(fontsize=9, facecolor=DARK_CARD, edgecolor=DARK_GRID,
          labelcolor=WHITE)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
plt.tight_layout()
save(fig, 'precision_recall.png')

# ── 5. RF METRICS OVERVIEW ──────────────────────
fig = plt.figure(figsize=(10, 6))
apply_dark_style(fig)
gs  = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4)

# (a) accuracy bar
ax_acc = fig.add_subplot(gs[0, :2])
apply_dark_style(fig, ax_acc)
model_names = ['Random\nForest', 'XGBoost']
accs        = [rf_acc * 100, xgb_acc * 100]
bar_colors  = [GREEN, ORANGE]
brs = ax_acc.bar(model_names, accs, color=bar_colors, edgecolor='none', width=0.45)
for br, v in zip(brs, accs):
    ax_acc.text(br.get_x() + br.get_width() / 2, v + 0.3,
                f'{v:.2f}%', ha='center', va='bottom', fontsize=10,
                color=WHITE, fontweight='bold')
ax_acc.set_ylim(0, 100)
ax_acc.set_ylabel('Accuracy (%)', fontsize=9)
ax_acc.set_title('Model Accuracy', fontsize=10, color=WHITE, fontweight='bold')
ax_acc.yaxis.grid(True, color=DARK_GRID, linewidth=0.7)
ax_acc.set_axisbelow(True)

# (b) AUC gauge (simple donut)
ax_donut = fig.add_subplot(gs[0, 2])
apply_dark_style(fig, ax_donut)
auc_val  = xgb_auc
wedge_c  = [ORANGE, DARK_GRID]
sizes    = [auc_val, 1 - auc_val]
wedges, _ = ax_donut.pie(sizes, colors=wedge_c, startangle=90,
                          wedgeprops={'width': 0.45, 'edgecolor': DARK_BG})
ax_donut.text(0, 0, f'{auc_val:.3f}', ha='center', va='center',
              fontsize=13, color=ORANGE, fontweight='bold')
ax_donut.set_title('XGBoost AUC', fontsize=10, color=WHITE, fontweight='bold', pad=4)

# (c) per-class F1 bar
ax_f1 = fig.add_subplot(gs[1, :2])
apply_dark_style(fig, ax_f1)
f1_vals = [cr_dict[c]['f1-score'] for c in CLASS_NAMES]
f1_cols  = [GREEN, '#f59e0b', '#ef4444']
brs2 = ax_f1.bar(CLASS_NAMES, f1_vals, color=f1_cols, edgecolor='none', width=0.5)
for br, v in zip(brs2, f1_vals):
    ax_f1.text(br.get_x() + br.get_width() / 2, v + 0.01,
               f'{v:.3f}', ha='center', va='bottom', fontsize=9,
               color=WHITE, fontweight='bold')
ax_f1.set_ylim(0, 1.1)
ax_f1.set_ylabel('F1 Score', fontsize=9)
ax_f1.set_title('Per-Class F1 Score — XGBoost', fontsize=10, color=WHITE, fontweight='bold')
ax_f1.yaxis.grid(True, color=DARK_GRID, linewidth=0.7)
ax_f1.set_axisbelow(True)

# (d) summary text box
ax_txt = fig.add_subplot(gs[1, 2])
apply_dark_style(fig, ax_txt)
ax_txt.axis('off')
summary = (
    f"XGBoost\n"
    f"Acc  {xgb_acc*100:.2f}%\n"
    f"F1   {xgb_f1:.4f}\n"
    f"AUC  {xgb_auc:.4f}\n\n"
    f"Random Forest\n"
    f"Acc  {rf_acc*100:.2f}%\n"
    f"F1   {rf_f1:.4f}\n"
    f"AUC  {rf_auc:.4f}"
)
ax_txt.text(0.1, 0.95, summary, transform=ax_txt.transAxes,
            fontsize=9, color=WHITE, va='top', linespacing=1.8,
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=DARK_CARD,
                      edgecolor=DARK_GRID, linewidth=1))

fig.suptitle('Random Forest & XGBoost — Full Metrics Overview',
             fontsize=12, color=WHITE, fontweight='bold', y=1.01)
save(fig, 'rf_metrics.png')

# ════════════════════════════════════════════════
#  STEP 5 — SAVE MODELS & STATS
# ════════════════════════════════════════════════
joblib.dump(rf,        os.path.join(MODEL_DIR, 'random_forest.pkl'))
joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
joblib.dump(le,        os.path.join(MODEL_DIR, 'label_encoder.pkl'))

stats = {
    "rf_accuracy":           rf_acc,
    "xgb_accuracy":          xgb_acc,
    "rf_f1":                 rf_f1,
    "xgb_f1":                xgb_f1,
    "rf_auc":                rf_auc,
    "xgb_auc":               xgb_auc,
    "confusion_matrix":      cm_matrix,
    "feature_importance":    fi_sorted,
    "classification_report": cr_dict,
    "model_version":         "XGBoost v3.2 + RF v1.0",
    "training_samples":      len(X_test),
}
joblib.dump(stats, os.path.join(MODEL_DIR, 'model_stats.pkl'))

print(f"\n✅ All models saved to:  {MODEL_DIR}/")
print(f"✅ All plots  saved to:  {PLOTS_DIR}/")
print(f"\n📊 Final Summary:")
print(f"   Random Forest → Accuracy: {rf_acc*100:.2f}%  F1: {rf_f1:.4f}  AUC: {rf_auc:.4f}")
print(f"   XGBoost       → Accuracy: {xgb_acc*100:.2f}%  F1: {xgb_f1:.4f}  AUC: {xgb_auc:.4f}")
print(f"\n🚀 Now run:  python app.py")