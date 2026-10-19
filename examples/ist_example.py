"""Example: International Stroke Trial (14-day death, prevalence 9.2%), development sample of 2000.

Compares the natural fit + Platt, balanced weights + Platt and prevalence-preserving augmentation on the rest of
the trial. Requires lightgbm and pandas, and the processed file ist_prep.csv, which prepare_ist.py (in the replication
material of the imbcal paper) builds from the public IST database (https://datashare.ed.ac.uk/handle/10283/124).
Usage: python ist_example.py PATH/ist_prep.csv
"""
import sys
import numpy as np, pandas as pd
import lightgbm as lgb
from imbcal import ImbalanceCalibratedClassifier, probability_report

d = pd.read_csv(sys.argv[1])
cat = ["conciencia", "subtipo", "heparina_asig"] + [f"RDEF{k}" for k in range(1, 9)]
X = pd.get_dummies(d.drop(columns=["y", "fecha", "pais", "hospital", "def_score"]), columns=cat,
                   drop_first=True, dtype=float).to_numpy(float)
X = (X - X.mean(0)) / X.std(0)
y = d.y.to_numpy()
rng = np.random.default_rng(0)
dev = rng.choice(len(y), 2000, replace=False); test = np.setdiff1d(np.arange(len(y)), dev)
base = lgb.LGBMClassifier(n_estimators=100, num_leaves=4, min_child_samples=10, learning_rate=0.05, verbose=-1)
for strategy in ["none", "weights", "augment"]:
    m = ImbalanceCalibratedClassifier(base, strategy=strategy, random_state=1).fit(X[dev], y[dev])
    r = probability_report(y[test], m.predict_proba(X[test])[:, 1])
    print(f"{strategy:8s} selected={m.selected_.label():40s} log-loss={r['log_loss']:.4f} brier={r['brier']:.4f} "
          f"slope={r['calibration_slope']:.2f} auc={r['auc']:.3f}")
