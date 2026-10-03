"""Weekly demand forecast by product family and colour for a textile manufacturer, on synthetic sales.

Synthetic data with the shape of the problem: three years of weekly sales (pieces) for 10 product families, each sold
in 4 to 10 colours. Families follow two collections a year (winter and summer peaks), a slow trend and promotions;
colours change share as fashion moves, some are launched midway and some are discontinued. Weekly noise is
over-dispersed, as real sales are.

Four forecasts, 1 to 8 weeks ahead, evaluated with a rolling origin over the last 26 weeks:
  seasonal_naive   the same week last year
  moving_average   the mean of the last 8 weeks
  global_gbm       one gradient-boosting model for every family-colour series, trained on lags, week of year and the
                   same week last year (a direct model: one row per origin and horizon)
  top_down         the same model at family level, split into colours by each colour's share of the last 8 weeks

Metric: WAPE (sum of absolute errors over sum of actuals) and bias, at family-colour level and at family level.

  python forecast.py      # prints the table, writes results.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

RNG = np.random.default_rng(29)
HERE = os.path.dirname(os.path.abspath(__file__))
WEEKS, LAGS, HORIZON = 156, 8, 8
BACKTEST = list(range(WEEKS - 26 - HORIZON, WEEKS - HORIZON + 1, 2))  # forecast origins (last week of known data)
FAMILIES = ["t-shirt", "polo", "dress", "leggings", "hoodie", "shorts", "pyjamas", "blouse", "trousers", "jacket"]


def generate():
    fam_series, colours = {}, {}
    week = np.arange(WEEKS)
    for f, fam in enumerate(FAMILIES):
        base = RNG.uniform(400, 2500)
        winter = RNG.uniform(-0.5, 0.9)  # some families sell in winter, some in summer
        season = 1 + winter * np.exp(-((week % 52 - 27) ** 2) / 40) + (0.6 - winter * 0.5) * np.exp(-((week % 52 - 46) ** 2) / 40)
        trend = 1 + RNG.normal(0.0015, 0.002) * week
        promo = np.ones(WEEKS)
        for w in RNG.choice(WEEKS, 6, replace=False):
            promo[w:w + 2] *= RNG.uniform(1.4, 2.0)
        level = base * np.clip(season, 0.2, None) * trend * promo
        n_col = int(RNG.integers(4, 11))
        logits = RNG.normal(0, 0.8, n_col)[:, None] + np.cumsum(RNG.normal(0, 0.06, (n_col, WEEKS)), axis=1)
        alive = np.ones((n_col, WEEKS), bool)
        for c in range(n_col):
            if RNG.random() < 0.2:
                alive[c, :int(RNG.integers(40, 120))] = False  # launched midway
            elif RNG.random() < 0.15:
                alive[c, int(RNG.integers(60, 140)):] = False  # discontinued
        share = np.where(alive, np.exp(logits), 0)
        share /= share.sum(axis=0, keepdims=True)
        mean = level[None, :] * share
        sales = RNG.negative_binomial(20, 20 / (20 + mean + 1e-9))  # over-dispersed counts around the mean
        for c in range(n_col):
            colours[(fam, f"colour {c + 1}")] = sales[c].astype(float)
        fam_series[fam] = sales.sum(axis=0).astype(float)
    return fam_series, colours


def features(y, t, h):
    """Known at origin t, for a forecast of week t + h."""
    lags = y[t - LAGS + 1:t + 1][::-1]
    target_week = t + h
    last_year = y[target_week - 52] if target_week - 52 >= 0 else np.nan
    return [*lags, lags.mean(), lags[:4].mean(), last_year, target_week % 52, h, float((y[:t + 1] > 0).sum())]


def design(series: dict, origins, family_index=None):
    X, Y, keys = [], [], []
    for key, y in series.items():
        fid = FAMILIES.index(key if isinstance(key, str) else key[0])
        for t in origins:
            for h in range(1, HORIZON + 1):
                if t + h < WEEKS:
                    X.append(features(y, t, h) + [fid])
                    Y.append(y[t + h])
                    keys.append((key, t, h))
    return np.array(X, float), np.array(Y, float), keys


def gbm():
    return HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0)


def wape(actual, pred):
    return float(np.abs(actual - pred).sum() / actual.sum())


def bias(actual, pred):
    return float((pred - actual).sum() / actual.sum())


def run():
    fams, cols = generate()
    train_origins = range(60, BACKTEST[0] - HORIZON + 1)  # leaves a full year for the same-week-last-year feature
    # bottom-up: one model for every family-colour series
    Xc, Yc, _ = design(cols, train_origins)
    col_model = gbm().fit(Xc, Yc)
    # family model, for top-down
    Xf, Yf, _ = design(fams, train_origins)
    fam_model = gbm().fit(Xf, Yf)

    preds = {m: {} for m in ("seasonal_naive", "moving_average", "global_gbm", "top_down")}
    actual = {}
    Xb, _, kb = design(cols, BACKTEST)
    gb = dict(zip(kb, np.clip(col_model.predict(Xb), 0, None)))
    Xfb, _, kfb = design(fams, BACKTEST)
    fb = dict(zip(kfb, np.clip(fam_model.predict(Xfb), 0, None)))
    for (fam, col), y in cols.items():
        for t in BACKTEST:
            recent_share = cols[(fam, col)][t - LAGS + 1:t + 1].sum() / max(fams[fam][t - LAGS + 1:t + 1].sum(), 1)
            for h in range(1, HORIZON + 1):
                if t + h >= WEEKS:
                    continue
                k = ((fam, col), t, h)
                actual[k] = y[t + h]
                preds["seasonal_naive"][k] = y[t + h - 52]
                preds["moving_average"][k] = y[t - LAGS + 1:t + 1].mean()
                preds["global_gbm"][k] = gb[k]
                preds["top_down"][k] = fb[(fam, t, h)] * recent_share
    keys = list(actual)
    a = np.array([actual[k] for k in keys])
    out = {"series": len(cols), "families": len(fams), "origins": len(BACKTEST), "forecasts": len(keys), "models": {}}
    for m, p in preds.items():
        pv = np.array([p[k] for k in keys])
        # family level: sum colours of the same family, origin and horizon
        agg_a, agg_p = {}, {}
        for k, av, pr in zip(keys, a, pv):
            fk = (k[0][0], k[1], k[2])
            agg_a[fk] = agg_a.get(fk, 0) + av
            agg_p[fk] = agg_p.get(fk, 0) + pr
        fa, fp = np.array(list(agg_a.values())), np.array([agg_p[k] for k in agg_a])
        by_h = {}
        for h in range(1, HORIZON + 1):
            idx = [i for i, k in enumerate(keys) if k[2] == h]
            by_h[h] = round(wape(a[idx], pv[idx]), 3)
        out["models"][m] = {"wape_colour": round(wape(a, pv), 3), "bias_colour": round(bias(a, pv), 3),
                            "wape_family": round(wape(fa, fp), 3), "bias_family": round(bias(fa, fp), 3),
                            "wape_colour_by_horizon": by_h}
    return out, fams, cols, preds, actual


if __name__ == "__main__":
    out, *_ = run()
    print(f"{out['series']} series, {out['origins']} origins, {out['forecasts']} forecasts")
    for m, r in out["models"].items():
        print(f"{m:16s} WAPE colour {r['wape_colour']:.3f}  family {r['wape_family']:.3f}  bias colour {r['bias_colour']:+.3f}")
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
