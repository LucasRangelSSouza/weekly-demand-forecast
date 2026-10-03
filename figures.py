"""Figures: error by model and level, error by horizon, and one family with its colours and forecasts.

  python figures.py figures
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecast as F

out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.grid": True, "grid.color": "#E5E7EB", "axes.axisbelow": True})
GREY, ORANGE, BLUE, GREEN, NAVY = "#8A8A8A", "#E8871E", "#006699", "#2E8B57", "#003366"
NAMES = {"seasonal_naive": ("Same week last year", GREY), "moving_average": ("Mean of last 8 weeks", ORANGE),
         "global_gbm": ("Boosting per colour", BLUE), "top_down": ("Boosting per family, split by colour share", GREEN)}
res, fams, cols, preds, actual = F.run()

# 1. WAPE by model at two levels, and by horizon
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
x = np.arange(2)
for j, (m, (label, c)) in enumerate(NAMES.items()):
    vals = [100 * res["models"][m]["wape_colour"], 100 * res["models"][m]["wape_family"]]
    xs = x + (j - 1.5) * 0.2
    a1.bar(xs, vals, 0.2, color=c, label=label)
    for xx, v in zip(xs, vals):
        a1.text(xx, v + 1, f"{v:.0f}", ha="center", fontsize=8)
a1.set_xticks(x, ["Family and colour", "Family"]); a1.set_ylabel("WAPE (%), 1 to 8 weeks ahead")
a1.set_title("Error by model and level"); a1.legend(fontsize=7.5, frameon=False, loc="upper right")
for m, (label, c) in NAMES.items():
    h = res["models"][m]["wape_colour_by_horizon"]
    a2.plot(list(map(int, h)), [100 * v for v in h.values()], marker="o", ms=3.5, color=c, label=label)
a2.set_xlabel("Weeks ahead"); a2.set_ylabel("WAPE (%), family and colour"); a2.set_title("Error by horizon")
a2.set_ylim(0, None)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_error.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)

# 2. One family: history, and the forecasts from the last origin
fam = max(fams, key=lambda f: fams[f].sum())
t = F.BACKTEST[-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.7), gridspec_kw={"width_ratios": [1.5, 1]})
weeks = np.arange(F.WEEKS)
a1.plot(weeks, fams[fam], color=NAVY, lw=1.2)
a1.axvline(t, color=GREY, ls=":", lw=1); a1.text(t - 2, a1.get_ylim()[1] * 0.92, "forecast origin", ha="right", fontsize=8, color=GREY)
a1.set_xlabel("Week"); a1.set_ylabel("Pieces sold per week"); a1.set_title(f"Family '{fam}': three years of weekly sales, with promotions")
mine = [k for k in cols if k[0] == fam]
share = np.array([cols[k] for k in mine]) / np.maximum(fams[fam], 1)
a2.stackplot(weeks, share, colors=plt.cm.tab20(np.arange(len(mine))), alpha=0.9)
a2.set_xlim(0, F.WEEKS - 1); a2.set_ylim(0, 1); a2.set_xlabel("Week"); a2.set_ylabel("Share of the family's sales")
a2.set_title("Colour shares drift, launch and stop")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_family.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)
print("ok")
