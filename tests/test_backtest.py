import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

from modeling.backtest import (Finding, apply_platt, derive_tuning, fit_corrections, fit_platt, holdout_cut_date, logloss,
                               reliability, tune_ratings)
from modeling.frames import build_team_games, games_frame
from modeling.tuning import Tuning, platt
from nbadata.warehouse import Warehouse
from tests.synthetic import build_league

SIG = 12.0


def make_oof(n=1200, seed=0, total_bias=0.0, bias_only_early=False, margin_scale=1.0):
    """Fake out-of-sample game predictions with a known flaw."""
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(n) // 6, unit="D")
    pred_m = rng.normal(0, 6, n)
    # margin_scale = 1: forecasts are exactly calibrated (E[margin | pred] = pred). > 1: the real edge is smaller
    # than the forecasts claim, i.e. over-confident predictions.
    margin = pred_m / margin_scale + rng.normal(0, SIG, n)
    pred_t = rng.normal(224, 8, n)
    bias = np.where(np.arange(n) < n * 0.6, total_bias, 0.0) if bias_only_early else total_bias
    total = pred_t + bias + rng.normal(0, 17, n)
    return pd.DataFrame({"game_key": [f"g{i}" for i in range(n)], "game_date": dates, "margin": margin, "total": total,
                         "pred_margin": pred_m, "pred_total": pred_t, "scored": True})


class Utilities(unittest.TestCase):
    def test_platt_roundtrip_and_logloss(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0.1, 0.9, 4000)
        true_p = np.array([platt(v, 0.7, 0.2) for v in p])        # the "world" is less extreme than the forecasts
        y = (rng.random(4000) < true_p).astype(float)
        a, b = fit_platt(p, y)
        self.assertAlmostEqual(a, 0.7, delta=0.12)
        self.assertAlmostEqual(b, 0.2, delta=0.12)
        self.assertLess(logloss(apply_platt(p, a, b), y), logloss(p, y))
        self.assertAlmostEqual(platt(0.37, 1.0, 0.0), 0.37, places=6)

    def test_reliability_bins(self):
        p = np.r_[np.full(50, 0.3), np.full(50, 0.7)]
        y = np.r_[np.zeros(35), np.ones(15), np.zeros(15), np.ones(35)]
        rows = reliability(p, y)
        self.assertEqual([r["n"] for r in rows], [50, 50])
        self.assertAlmostEqual(rows[0]["rate"], 0.3)
        self.assertAlmostEqual(rows[1]["rate"], 0.7)


class CorrectionsAreVerified(unittest.TestCase):
    def run_fit(self, oof):
        cut = holdout_cut_date(oof)
        return fit_corrections(oof, SIG, None, cut)

    def test_real_total_bias_is_found_and_adopted(self):
        corr, findings = self.run_fit(make_oof(total_bias=3.0))
        self.assertAlmostEqual(corr["total_bias"], 3.0, delta=1.0)
        f = next(f for f in findings if f.name == "total bias")
        self.assertTrue(f.adopted)
        self.assertLess(f.after, f.before)

    def test_no_bias_means_no_correction(self):
        corr, findings = self.run_fit(make_oof(total_bias=0.0))
        self.assertEqual(corr["total_bias"], 0.0)
        self.assertEqual(corr["margin_bias"], 0.0)
        self.assertFalse(next(f for f in findings if f.name == "total bias").adopted)

    def test_bias_that_disappears_in_holdout_is_rejected(self):
        """The overfitting trap: a flaw present early on that is gone later must NOT be 'corrected'."""
        corr, findings = self.run_fit(make_oof(total_bias=4.0, bias_only_early=True))
        f = next(f for f in findings if f.name == "total bias")
        self.assertFalse(f.adopted)
        self.assertEqual(corr["total_bias"], 0.0)

    def test_overconfident_win_probabilities_are_recalibrated(self):
        corr, findings = self.run_fit(make_oof(margin_scale=1.8, seed=3))
        self.assertIsNotNone(corr["win_cal"])
        self.assertLess(corr["win_cal"][0], 0.9)                  # slope < 1: pull probabilities toward 50%
        self.assertTrue(next(f for f in findings if f.name == "win-probability calibration").adopted)

    def test_calibrated_win_probabilities_are_left_alone(self):
        corr, _ = self.run_fit(make_oof(margin_scale=1.0, seed=5))
        self.assertIsNone(corr["win_cal"])

    def test_compressed_margins_are_stretched_and_the_gain_is_verified(self):
        # actual margins move 1.4x as far as projected: favourites are under-projected
        corr, findings = self.run_fit(make_oof(margin_scale=1 / 1.4, seed=7, n=2400))
        self.assertAlmostEqual(corr["margin_scale"], 1.4, delta=0.2)
        self.assertIsNotNone(corr["margin_sigma"])
        f = next(f for f in findings if f.name == "margin stretch")
        self.assertTrue(f.adopted)
        self.assertLess(f.after, f.before)

    def test_right_sized_margins_are_not_stretched(self):
        corr, findings = self.run_fit(make_oof(margin_scale=1.0, seed=9, n=2400))
        self.assertEqual(corr["margin_scale"], 1.0)
        self.assertIsNone(corr["margin_sigma"])
        self.assertFalse(next(f for f in findings if f.name == "margin stretch").adopted)

    def test_stretch_that_only_existed_early_is_rejected(self):
        """Verification: compression in the earlier games that is gone in the holdout must not be 'fixed'."""
        early, late = make_oof(n=1500, margin_scale=1 / 1.5, seed=11), make_oof(n=900, margin_scale=1.0, seed=12)
        late["game_date"] = late["game_date"] + (early["game_date"].max() - late["game_date"].min() + pd.Timedelta(days=1))
        corr, findings = self.run_fit(pd.concat([early, late], ignore_index=True))
        f = next(f for f in findings if f.name == "margin stretch")
        if f.adopted:                       # if adopted it must genuinely have helped on the holdout
            self.assertLess(f.after, f.before)
        else:
            self.assertEqual(corr["margin_scale"], 1.0)

    def test_too_little_data_is_reported_not_guessed(self):
        corr, findings = fit_corrections(make_oof(n=120), SIG, None, pd.Timestamp("2025-01-15"))
        self.assertEqual(corr["total_bias"], 0.0)
        self.assertIn("not enough", findings[0].detail)


class TuningStore(unittest.TestCase):
    def test_roundtrip_apply_and_reset(self):
        import tempfile
        t = Tuning(rating={"half_life": 70.0, "ridge": 25.0, "pace_ridge": 25.0, "carry": 0.55}, margin_scale=1.25, margin_sigma=10.5,
                   margin_bias=0.8, total_bias=-2.0, win_cal=(0.8, 0.05), prop_cal=(0.9, 0.0), created="2026-11-01")
        with tempfile.TemporaryDirectory() as d:
            t.save(d)
            t2 = Tuning.load(d)
            self.assertEqual(t2.rating["half_life"], 70.0)
            self.assertEqual((t2.margin_bias, t2.total_bias, t2.win_cal, t2.prop_cal), (0.8, -2.0, (0.8, 0.05), (0.9, 0.0)))
            self.assertEqual((t2.margin_scale, t2.margin_sigma), (1.25, 10.5))
            self.assertTrue(Tuning.reset(d))
            self.assertTrue(Tuning.load(d).is_default())
            self.assertFalse(Tuning.reset(d))
        self.assertTrue(Tuning.load("/no/such/dir").is_default())
        self.assertEqual(len(t.describe()), 6)

    def test_apply_game_and_prop(self):
        pred = {"home_pts": 112.0, "away_pts": 108.0, "margin": 4.0, "total": 220.0, "margin_sigma": 12.0, "total_sigma": 18.0, "home_win_prob": 0.63}
        t = Tuning(margin_bias=1.0, total_bias=-3.0)
        out = t.apply_game(pred)
        self.assertEqual((out["margin"], out["total"]), (5.0, 217.0))
        self.assertAlmostEqual(out["home_pts"] - out["away_pts"], 5.0)
        self.assertAlmostEqual(out["home_win_prob"], float(norm.cdf(5.0 / 12.0)))
        self.assertIs(Tuning().apply_game(pred), pred)               # default tuning changes nothing
        st = Tuning(margin_scale=1.5, margin_sigma=10.0).apply_game(pred)      # stretch: 4.0 -> 6.0, and the new sigma is used
        self.assertAlmostEqual(st["margin"], 6.0)
        self.assertEqual(st["margin_sigma"], 10.0)
        self.assertAlmostEqual(st["home_win_prob"], float(norm.cdf(6.0 / 10.0)))
        probs = {"over": 0.60, "under": 0.38, "push": 0.02}
        cal = Tuning(prop_cal=(0.5, 0.0)).apply_prop(probs)
        self.assertAlmostEqual(cal["over"] + cal["under"] + cal["push"], 1.0)
        self.assertEqual(cal["push"], 0.02)
        self.assertLess(cal["over"], 0.60)                           # slope 0.5 pulls toward 50/50
        self.assertGreater(cal["over"], cal["under"])


class RatingTuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wh = Warehouse(":memory:")
        build_league(cls.wh, seasons=(2023, 2024), games_per_team=50, seed=13)
        cls.tg = build_team_games(cls.wh.team_box_df(), cls.wh.games_df())
        cls.g = games_frame(cls.tg)
        cls.grid = {"half_life": [20.0, 60.0, 150.0], "ridge": [10.0, 50.0], "carry": [0.3, 0.8]}

    def test_deliberately_bad_parameters_are_fixed_and_verified_on_holdout(self):
        bad = {"half_life": 4.0, "ridge": 600.0, "pace_ridge": 600.0, "carry": 0.0}
        from modeling.backtest import _rating_errors
        gm_dates = self.g[(self.g["gno_h"] >= 15) & (self.g["gno_a"] >= 15)]["game_date"].sort_values()
        cut = pd.Timestamp(gm_dates.iloc[int(len(gm_dates) * 0.7)])
        best, findings, trials = tune_ratings(self.g, cut, bad, self.grid)
        self.assertTrue(findings[0].adopted, findings[0].detail)
        self.assertLess(findings[0].after, findings[0].before)
        self.assertGreater(best["half_life"], 4.0)
        self.assertLess(best["ridge"], 600.0)
        self.assertGreaterEqual(len(trials), 7)

    def test_nothing_changes_when_data_is_too_thin(self):
        small = self.g.head(120)
        cut = small["game_date"].iloc[80]
        best, findings, _ = tune_ratings(small, cut, None, self.grid)
        self.assertFalse(findings[0].adopted)
        self.assertIn("not enough", findings[0].detail)


class FullPipeline(unittest.TestCase):
    def test_derive_tuning_end_to_end_and_honest_gating(self):
        wh = Warehouse(":memory:")
        build_league(wh, seasons=(2023, 2024), games_per_team=45, seed=17)
        tg = build_team_games(wh.team_box_df(), wh.games_df())
        g = games_frame(tg)
        pb = wh.player_box_df()
        pb["game_date"] = pd.to_datetime(pb["game_date"])
        msgs = []
        d = derive_tuning(g, tg, pb, tune_rating=True, grid={"half_life": [30.0, 100.0], "ridge": [25.0], "carry": [0.55]}, progress=msgs.append)
        self.assertIsNotNone(d.cut)
        self.assertTrue(any("baseline" in m for m in msgs))
        names = {f.name for f in d.findings}
        self.assertTrue({"team rating parameters", "margin bias", "total bias", "win-probability calibration"} <= names)
        self.assertEqual(d.diagnostics["game"]["n"], d.gm0.metrics["n_games_scored"])
        self.assertTrue(d.diagnostics["player"]["calibration"])
        metrics = {r["metric"] for r in d.before_after}
        self.assertTrue({"margin MAE (pts)", "total MAE (pts)", "home-win log loss"} <= metrics)
        # synthetic data has no bias and well-calibrated noise: the gate should not invent corrections
        self.assertEqual((d.tuning.margin_bias, d.tuning.total_bias), (0.0, 0.0))
        # whatever was adopted must not be worse on the holdout
        for r in d.before_after:
            if r["lower_better"] and r["metric"] in ("margin MAE (pts)", "total MAE (pts)"):
                self.assertLessEqual(r["after"], r["before"] * 1.01)
        self.assertEqual(d.tuning.evidence["holdout_from"], str(d.cut.date()))


if __name__ == "__main__":
    unittest.main()
