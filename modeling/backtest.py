"""
Backtesting, diagnostics and verified corrections.

What can and cannot be tested
  * Projection accuracy and probability calibration are tested against real outcomes, always.
  * Betting ROI needs historical lines. There are no free historical lines, so ROI is only simulated for
    games whose odds you have stored (every `python cli.py odds` run stores a snapshot).

How corrections are found without fooling ourselves
  * Every prediction used here is out-of-sample: walk-forward ratings and time-ordered cross-validation.
  * History is split by DATE: corrections are fitted on the earlier ~70% and must improve the later ~30%
    (the holdout) to be kept. Anything that doesn't is rejected, and the reason is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm
from sklearn.linear_model import LogisticRegression

from config import BETTING, MODEL
from modeling.game_model import GameModel
from modeling.player_model import PlayerModel, build_training_frame, rotation_from_games, roster_for_team
from modeling.ratings import RatingEngine
from modeling.tuning import Tuning, logit

HOLDOUT_FRAC = 0.30
GRID = {"half_life": [25.0, 35.0, 50.0, 70.0, 100.0], "ridge": [10.0, 25.0, 50.0, 100.0], "carry": [0.3, 0.55, 0.75, 0.9]}
MIN_IMPROVEMENT = 0.003         # a correction must cut holdout error / log loss by at least 0.3%
Progress = Optional[Callable[[str], None]]


def default_rating_kwargs() -> Dict[str, float]:
    return {"half_life": MODEL.RATING_HALF_LIFE_DAYS, "ridge": MODEL.RATING_RIDGE,
            "pace_ridge": MODEL.PACE_RIDGE, "carry": MODEL.PRIOR_SEASON_CARRY}


@dataclass
class Finding:
    name: str
    adopted: bool
    detail: str
    metric: str = ""
    before: Optional[float] = None
    after: Optional[float] = None


def _say(progress: Progress, msg: str):
    if progress:
        progress(msg)


# ── fitting helpers ──────────────────────────────────────────────────────────
def fit_models(g: pd.DataFrame, tg: pd.DataFrame, pb: pd.DataFrame, rating_kw: Optional[Dict] = None):
    """Fit game + player models on the given frames (no saving). Returns (GameModel, PlayerModel)."""
    eng = RatingEngine(**(rating_kw or {}))
    gm = GameModel().fit(g, engine=eng)
    frame = build_training_frame(pb, tg, gm.last_wf)
    pm = PlayerModel().fit(frame)
    return gm, pm


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_platt(p: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.array([logit(v) for v in p]).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(x, np.asarray(y, dtype=int))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.array([logit(v) for v in p])
    return 1.0 / (1.0 + np.exp(-(a * x + b)))


def reliability(p: Sequence[float], y: Sequence[float], edges=(0.0, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0001)) -> List[Dict]:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 5:
            rows.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "mean_p": float(p[m].mean()), "rate": float(y[m].mean())})
    return rows


def holdout_cut_date(oof: pd.DataFrame, frac: float = HOLDOUT_FRAC) -> Optional[pd.Timestamp]:
    """Date separating the earlier (tuning) games from the later (holdout) games, among scored games."""
    d = oof.loc[oof["scored"], "game_date"].sort_values()
    if len(d) < 100:
        return None
    return pd.Timestamp(d.iloc[int(len(d) * (1 - frac))])


# ── diagnostics: what does the backtest find? ────────────────────────────────
def game_diagnostics(gm: GameModel) -> Dict:
    o = gm.oof[gm.oof["scored"]].copy()
    if o.empty:
        return {"n": 0}
    o["err_m"], o["err_t"] = o["margin"] - o["pred_margin"], o["total"] - o["pred_total"]
    o["month"] = pd.to_datetime(o["game_date"]).dt.strftime("%Y-%m")
    by_month = [{"month": m, "n": len(d), "mae_m": d["err_m"].abs().mean(), "mae_t": d["err_t"].abs().mean(),
                 "bias_m": d["err_m"].mean(), "bias_t": d["err_t"].mean()} for m, d in o.groupby("month")]
    p = norm.cdf(o["pred_margin"] / gm.margin_sigma)
    y = (o["margin"] > 0).astype(float)
    return {
        "n": int(len(o)), "mae_m": float(o["err_m"].abs().mean()), "mae_t": float(o["err_t"].abs().mean()),
        "mae_m_naive": float((o["margin"] - o["naive_margin"]).abs().mean()), "mae_t_naive": float((o["total"] - o["naive_total"]).abs().mean()),
        "bias_m": float(o["err_m"].mean()), "bias_t": float(o["err_t"].mean()),
        "bias_m_se": float(o["err_m"].std() / math.sqrt(len(o))), "bias_t_se": float(o["err_t"].std() / math.sqrt(len(o))),
        "sigma_m": float(gm.margin_sigma), "sigma_t": float(gm.total_sigma),
        "by_month": by_month, "win_reliability": reliability(p, y), "win_logloss": logloss(p, y),
        "win_accuracy": float(((p > 0.5) == (y > 0.5)).mean()), "pick_accuracy_vs_naive": float(
            (np.sign(o["pred_margin"]) == np.sign(o["margin"])).mean()),
    }


def _nb_over(mu: np.ndarray, line: np.ndarray, r: float) -> np.ndarray:
    mu = np.maximum(mu, 0.05)
    return 1.0 - nbinom.cdf(np.floor(line), r, r / (r + mu))


def prop_calibration_sample(pm: PlayerModel, offsets: Sequence[float] = (-4, -2, 0, 2, 4)) -> pd.DataFrame:
    """
    Probability-calibration sample for the points distribution: lines placed at several offsets around each
    out-of-fold prediction give a spread of P(over) values, each checked against what really happened.
    (Real sportsbook lines aren't available historically; this tests the distribution, not the market.)
    """
    o = pm.oof
    rows = []
    for off in offsets:
        line = np.floor(o["pred"].values + off) + 0.5
        p = _nb_over(o["pred"].values, line, pm.nb_r)
        rows.append(pd.DataFrame({"game_date": o["game_date"].values, "p": p, "y": (o["pts"].values > line).astype(float)}))
    return pd.concat(rows, ignore_index=True).sort_values("game_date").reset_index(drop=True)


def player_diagnostics(pm: PlayerModel) -> Dict:
    o = pm.oof.copy()
    if o.empty:
        return {"n": 0}
    o["err"] = o["pts"] - o["pred"]
    o["month"] = pd.to_datetime(o["game_date"]).dt.strftime("%Y-%m")
    tiers = []
    for name, lo, hi in (("bench (<20 exp min)", 0, 20), ("rotation (20-30)", 20, 30), ("stars (30+)", 30, 99)):
        m = (o["ewm_minutes"] >= lo) & (o["ewm_minutes"] < hi)
        if m.sum() > 50:
            d = o[m]
            tiers.append({"tier": name, "n": int(m.sum()), "mae": float(d["err"].abs().mean()), "mae_form": float((d["pts"] - d["ewm_pts"]).abs().mean()), "bias": float(d["err"].mean())})
    miss = []
    for name, lo, hi in (("no teammates out", -1, 1e-9), ("some out (<20% of minutes)", 1e-9, 0.2), ("many out (20%+)", 0.2, 9)):
        m = (o["miss_share"] > lo) & (o["miss_share"] <= hi) if lo >= 0 else (o["miss_share"] <= hi)
        if m.sum() > 50:
            d = o[m]
            miss.append({"bucket": name, "n": int(m.sum()), "mae": float(d["err"].abs().mean()), "mae_form": float((d["pts"] - d["ewm_pts"]).abs().mean()), "bias": float(d["err"].mean())})
    by_month = [{"month": m, "n": len(d), "mae": d["err"].abs().mean(), "bias": d["err"].mean()} for m, d in o.groupby("month")]
    # skill test against a "recent form" line: does the model pick the right side more than half the time?
    line = np.floor(o["ewm_pts"].values) + 0.5
    over_actual = o["pts"].values > line
    model_over = o["pred"].values > line
    gap = np.abs(o["pred"].values - line)
    big = gap >= 1.5
    sample = prop_calibration_sample(pm)
    return {
        "n": int(len(o)), "mae": float(o["err"].abs().mean()), "mae_form": float((o["pts"] - o["ewm_pts"]).abs().mean()),
        "bias": float(o["err"].mean()), "minutes_mae": float((o["minutes"] - o["pred_min"]).abs().mean()),
        "tiers": tiers, "teammates_out": miss, "by_month": by_month,
        "form_line_accuracy": float((model_over == over_actual).mean()), "form_line_accuracy_big_gap": float((model_over == over_actual)[big].mean()) if big.sum() > 30 else None,
        "form_line_n_big": int(big.sum()),
        "calibration": reliability(sample["p"], sample["y"]), "calibration_logloss": logloss(sample["p"], sample["y"]),
        "nb_r": float(pm.nb_r),
    }


# ── tuning team-rating parameters ────────────────────────────────────────────
def _rating_errors(g: pd.DataFrame, kw: Dict, cut: pd.Timestamp, cache: Dict) -> Dict[str, float]:
    key = tuple(sorted(kw.items()))
    if key not in cache:
        wf, _, _ = RatingEngine(**kw).walk_forward(g)
        ok = (wf["gno_h"] >= MODEL.MIN_TEAM_GAMES_FOR_EVAL) & (wf["gno_a"] >= MODEL.MIN_TEAM_GAMES_FOR_EVAL) & wf["r_margin"].notna()
        w = wf[ok]
        em, et = (w["margin"] - w["r_margin"].astype(float)).abs(), (w["total"] - w["r_total"].astype(float)).abs()
        tune = (w["game_date"] < cut).values
        cache[key] = {"m_tune": em[tune].mean(), "t_tune": et[tune].mean(), "m_hold": em[~tune].mean(), "t_hold": et[~tune].mean(),
                      "n_tune": int(tune.sum()), "n_hold": int((~tune).sum())}
    return cache[key]


def tune_ratings(g: pd.DataFrame, cut: pd.Timestamp, base_kw: Optional[Dict] = None, grid: Optional[Dict] = None,
                 progress: Progress = None) -> Tuple[Dict, List[Finding], List[Dict]]:
    """
    Coordinate search over the rating parameters, scored on games BEFORE `cut`; the winner is adopted only if
    it also beats the starting parameters on games on/after `cut`.
    Returns (kwargs_to_use, findings, trials).
    """
    base = dict(base_kw or default_rating_kwargs())
    grid = grid or GRID
    cache: Dict = {}
    ref = _rating_errors(g, base, cut, cache)
    if ref["n_tune"] < 100 or ref["n_hold"] < 50:
        return base, [Finding("team rating parameters", False, f"not enough scored games to tune safely ({ref['n_tune']} tuning / {ref['n_hold']} holdout)")], []

    def obj(e, which):    # dimensionless: relative margin error + relative total error vs the starting parameters
        return e[f"m_{which}"] / ref[f"m_{which}"] + e[f"t_{which}"] / ref[f"t_{which}"]

    best = dict(base)
    trials: List[Dict] = []
    for param in ("half_life", "ridge", "carry"):
        scores = []
        for v in grid.get(param, []):
            kw = dict(best)
            kw[param] = v
            if param == "ridge":
                kw["pace_ridge"] = v
            _say(progress, f"  trying {param}={v:g} ...")
            e = _rating_errors(g, kw, cut, cache)
            scores.append((obj(e, "tune"), v, kw, e))
            trials.append({"param": param, "value": v, "margin_mae": e["m_tune"], "total_mae": e["t_tune"], "score": obj(e, "tune")})
        if scores:
            sc, v, kw, _ = min(scores, key=lambda t: t[0])
            if sc < obj(_rating_errors(g, best, cut, cache), "tune") - 1e-9:
                best = kw
    changed = {k: best[k] for k in ("half_life", "ridge", "carry") if abs(best[k] - base[k]) > 1e-9}
    if not changed:
        return base, [Finding("team rating parameters", False, "the current parameters were already the best of the combinations tried")], trials
    e_base, e_best = _rating_errors(g, base, cut, cache), _rating_errors(g, best, cut, cache)
    hb, ha = obj(e_base, "hold"), obj(e_best, "hold")
    desc = ", ".join(f"{k} {base[k]:g} -> {best[k]:g}" for k in changed)
    detail = (f"{desc}. On the later holdout games: margin MAE {e_base['m_hold']:.2f} -> {e_best['m_hold']:.2f}, "
              f"total MAE {e_base['t_hold']:.2f} -> {e_best['t_hold']:.2f}")
    if ha < hb * (1 - MIN_IMPROVEMENT):
        return best, [Finding("team rating parameters", True, detail, "holdout margin+total error (relative)", hb, ha)], trials
    return base, [Finding("team rating parameters", False, "better on the earlier games but not on the later holdout, so rejected: " + detail, "holdout margin+total error (relative)", hb, ha)], trials


# ── bias and probability corrections ─────────────────────────────────────────
def fit_corrections(game_oof: pd.DataFrame, sigma_m: float, prop_sample: Optional[pd.DataFrame], cut: pd.Timestamp) -> Tuple[Dict, List[Finding]]:
    """Fit bias / calibration on games before `cut`, keep only what improves games on/after `cut`."""
    out: Dict = {"margin_scale": 1.0, "margin_sigma": None, "margin_bias": 0.0, "total_bias": 0.0, "win_cal": None, "prop_cal": None}
    findings: List[Finding] = []
    o = game_oof[game_oof["scored"]].sort_values("game_date")
    tune, hold = o[o["game_date"] < cut], o[o["game_date"] >= cut]
    if len(tune) < 100 or len(hold) < 50:
        findings.append(Finding("bias and calibration", False, f"not enough scored games ({len(tune)} tuning / {len(hold)} holdout)"))
        return out, findings

    # Margin stretch: ridge shrinkage and time-decay can leave projected margins too close to zero (favourites
    # under-projected). Regress actual margin on projected margin; slope > 1 means "stretch".
    x, yv = tune["pred_margin"].values, tune["margin"].values
    slope, icpt = np.polyfit(x, yv, 1)
    resid = yv - (slope * x + icpt)
    se_slope = resid.std(ddof=2) / (x.std() * math.sqrt(len(x)))
    if 0.7 <= slope <= 1.6 and abs(slope - 1.0) > 2.5 * se_slope:
        hx, hy = hold["pred_margin"].values, hold["margin"].values
        before, after = np.abs(hy - hx).mean(), np.abs(hy - (slope * hx + icpt)).mean()
        if after < before * (1 - MIN_IMPROVEMENT):
            out["margin_scale"] = float(slope)
            out["margin_sigma"] = float(resid.std(ddof=2))
            findings.append(Finding("margin stretch", True, f"actual margins moved {slope:.2f}x as far as projected (1.00 = right); stretching projections cuts holdout margin MAE {before:.2f} -> {after:.2f}", "holdout margin MAE", before, after))
        else:
            findings.append(Finding("margin stretch", False, f"slope {slope:.2f} on earlier games, but it did not help on the holdout (margin MAE {before:.2f} -> {after:.2f})", "holdout margin MAE", before, after))
    else:
        findings.append(Finding("margin stretch", False, f"projected margins were the right size (slope {slope:.2f} +/- {1.96 * se_slope:.2f}; 1.00 is perfect)"))
    ms = out["margin_scale"]
    sigma_used = out["margin_sigma"] or sigma_m

    for label, actual, pred, key, floor in (("margin bias", "margin", "pred_margin", "margin_bias", 0.3), ("total bias", "total", "pred_total", "total_bias", 0.5)):
        scale = ms if key == "margin_bias" else 1.0
        err = (tune[actual] - scale * tune[pred]).values
        mean, se = err.mean(), err.std(ddof=1) / math.sqrt(len(err))
        if abs(mean) < floor or abs(mean / se) < 2.5:
            findings.append(Finding(label, False, f"measured {mean:+.2f} pts (+/-{1.96 * se:.2f}); not large and consistent enough to correct"))
            continue
        herr = (hold[actual] - scale * hold[pred]).values
        before, after = np.abs(herr).mean(), np.abs(herr - mean).mean()
        if after < before * (1 - 0.002):
            out[key] = float(mean)
            findings.append(Finding(label, True, f"the model was off by {mean:+.2f} pts on average; correcting it cuts holdout MAE {before:.2f} -> {after:.2f}", "holdout MAE", before, after))
        else:
            findings.append(Finding(label, False, f"measured {mean:+.2f} pts earlier, but correcting it did not help on the holdout (MAE {before:.2f} -> {after:.2f})", "holdout MAE", before, after))

    def win_p(d):
        return norm.cdf((d["pred_margin"].values * ms + out["margin_bias"]) / sigma_used)
    ptune, ytune = win_p(tune), (tune["margin"] > 0).astype(float).values
    phold, yhold = win_p(hold), (hold["margin"] > 0).astype(float).values
    a, b = fit_platt(ptune, ytune)
    before, after = logloss(phold, yhold), logloss(apply_platt(phold, a, b), yhold)
    if 0.4 <= a <= 2.0 and after < before * (1 - MIN_IMPROVEMENT):
        out["win_cal"] = (a, b)
        findings.append(Finding("win-probability calibration", True, f"slope {a:.2f} (1.00 = already calibrated); holdout log loss {before:.4f} -> {after:.4f}", "holdout log loss", before, after))
    else:
        findings.append(Finding("win-probability calibration", False, f"win probabilities were already well calibrated (fitted slope {a:.2f}); holdout log loss {before:.4f} vs {after:.4f}", "holdout log loss", before, after))

    if prop_sample is not None and len(prop_sample) > 500:
        pt, ph = prop_sample[prop_sample["game_date"] < cut], prop_sample[prop_sample["game_date"] >= cut]
        if len(pt) > 300 and len(ph) > 150:
            a, b = fit_platt(pt["p"].values, pt["y"].values)
            before, after = logloss(ph["p"].values, ph["y"].values), logloss(apply_platt(ph["p"].values, a, b), ph["y"].values)
            if 0.4 <= a <= 2.0 and after < before * (1 - 0.002):
                out["prop_cal"] = (a, b)
                findings.append(Finding("over/under probability calibration", True, f"slope {a:.2f}; holdout log loss {before:.4f} -> {after:.4f}", "holdout log loss", before, after))
            else:
                findings.append(Finding("over/under probability calibration", False, f"points distribution already well calibrated (fitted slope {a:.2f}); holdout log loss {before:.4f} vs {after:.4f}", "holdout log loss", before, after))
    return out, findings


@dataclass
class Derived:
    tuning: Tuning
    gm0: GameModel
    pm0: PlayerModel
    gm1: GameModel
    pm1: PlayerModel
    findings: List[Finding]
    trials: List[Dict]
    cut: Optional[pd.Timestamp]
    diagnostics: Dict
    before_after: List[Dict]


def derive_tuning(g: pd.DataFrame, tg: pd.DataFrame, pb: pd.DataFrame, tune_rating: bool = True,
                  grid: Optional[Dict] = None, progress: Progress = None) -> Derived:
    """The whole backtest pipeline on the given (already time-limited) frames. Nothing is saved here."""
    _say(progress, "fitting the model with current default settings (this is the baseline)")
    base_kw = default_rating_kwargs()
    gm0, pm0 = fit_models(g, tg, pb, base_kw)
    cut = holdout_cut_date(gm0.oof)
    diagnostics = {"game": game_diagnostics(gm0), "player": player_diagnostics(pm0)}
    findings: List[Finding] = []
    trials: List[Dict] = []
    gm1, pm1, rating_adopted = gm0, pm0, {}
    if cut is None:
        findings.append(Finding("all corrections", False, "not enough history to hold out a verification period (need roughly 300+ scored games)"))
        return Derived(Tuning(), gm0, pm0, gm0, pm0, findings, trials, None, diagnostics, [])
    if tune_rating:
        _say(progress, "tuning team-rating parameters (each trial is a full walk-forward replay)")
        best_kw, f_r, trials = tune_ratings(g, cut, base_kw, grid, progress)
        findings += f_r
        if f_r and f_r[0].adopted:
            rating_adopted = {k: float(best_kw[k]) for k in ("half_life", "ridge", "pace_ridge", "carry")}
            _say(progress, "refitting game and player models with the tuned ratings")
            gm1, pm1 = fit_models(g, tg, pb, best_kw)
    _say(progress, "measuring bias and calibration, verifying each correction on the held-out later games")
    corr, f_c = fit_corrections(gm1.oof, gm1.margin_sigma, prop_calibration_sample(pm1), cut)
    findings += f_c
    tuning = Tuning(rating=rating_adopted, margin_scale=corr["margin_scale"], margin_sigma=corr["margin_sigma"],
                    margin_bias=corr["margin_bias"], total_bias=corr["total_bias"], win_cal=corr["win_cal"],
                    prop_cal=corr["prop_cal"], created=date.today().isoformat(),
                    evidence={"holdout_from": str(cut.date()), "findings": [f.__dict__ for f in findings]})
    return Derived(tuning, gm0, pm0, gm1, pm1, findings, trials, cut, diagnostics, _before_after(gm0, gm1, pm0, pm1, tuning, cut))


def _before_after(gm0, gm1, pm0, pm1, tuning: Tuning, cut) -> List[Dict]:
    """Holdout-only comparison of the default model vs the corrected model."""
    rows = []
    a = gm0.oof[gm0.oof["scored"] & (gm0.oof["game_date"] >= cut)].set_index("game_key")
    b = gm1.oof[gm1.oof["scored"] & (gm1.oof["game_date"] >= cut)].set_index("game_key")
    common = a.index.intersection(b.index)
    if len(common) >= 30:
        a, b = a.loc[common], b.loc[common]
        bm, bt = b["pred_margin"] * tuning.margin_scale + tuning.margin_bias, b["pred_total"] + tuning.total_bias
        rows.append({"metric": "margin MAE (pts)", "before": float((a["margin"] - a["pred_margin"]).abs().mean()), "after": float((b["margin"] - bm).abs().mean()), "lower_better": True})
        rows.append({"metric": "total MAE (pts)", "before": float((a["total"] - a["pred_total"]).abs().mean()), "after": float((b["total"] - bt).abs().mean()), "lower_better": True})
        rows.append({"metric": "margin bias (pts)", "before": float((a["margin"] - a["pred_margin"]).mean()), "after": float((b["margin"] - bm).mean()), "lower_better": None})
        rows.append({"metric": "total bias (pts)", "before": float((a["total"] - a["pred_total"]).mean()), "after": float((b["total"] - bt).mean()), "lower_better": None})
        y = (a["margin"] > 0).astype(float).values
        p0 = norm.cdf(a["pred_margin"].values / gm0.margin_sigma)
        p1 = norm.cdf(bm.values / (tuning.margin_sigma or gm1.margin_sigma))
        if tuning.win_cal:
            p1 = apply_platt(p1, *tuning.win_cal)
        rows.append({"metric": "home-win log loss", "before": logloss(p0, y), "after": logloss(p1, y), "lower_better": True})
    s0, s1 = prop_calibration_sample(pm0), prop_calibration_sample(pm1)
    h0, h1 = s0[s0["game_date"] >= cut], s1[s1["game_date"] >= cut]
    if len(h0) > 150 and len(h1) > 150:
        p1 = apply_platt(h1["p"].values, *tuning.prop_cal) if tuning.prop_cal else h1["p"].values
        rows.append({"metric": "over/under log loss (points)", "before": logloss(h0["p"].values, h0["y"].values), "after": logloss(p1, h1["y"].values), "lower_better": True})
        o0, o1 = pm0.oof[pm0.oof["game_date"] >= cut], pm1.oof[pm1.oof["game_date"] >= cut]
        rows.append({"metric": "player points MAE", "before": float((o0["pts"] - o0["pred"]).abs().mean()), "after": float((o1["pts"] - o1["pred"]).abs().mean()), "lower_better": True})
    return rows
