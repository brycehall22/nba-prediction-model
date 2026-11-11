import unittest

import numpy as np
import pandas as pd

from modeling.frames import build_team_games, games_frame
from modeling.game_model import GameModel
from modeling.player_model import (FEATURES, PlayerModel, build_inference_frame, build_training_frame,
                                   compute_player_features, fit_dispersion, historical_missing,
                                   missing_features, over_under_probs, rotation_from_games, training_mask, _nb_dist)
from modeling.ratings import RatingEngine
from nbadata.teams import resolve_team
from nbadata.warehouse import Warehouse
from tests.synthetic import build_league


class FeatureLeakage(unittest.TestCase):
    def test_features_use_only_earlier_games(self):
        base = pd.DataFrame({"player_key": "p", "game_date": pd.date_range("2025-01-01", periods=12),
                             "game_key": [f"g{i}" for i in range(12)], "season": "s", "starter": 1.0,
                             "pts": np.arange(10.0, 22.0), "minutes": 30.0, "fga": 12.0, "fta": 4.0, "fg3a": 5.0, "tov": 2.0})
        a = compute_player_features(base)
        mod = base.copy()
        mod.loc[mod.index >= 8, "pts"] = 999.0          # change game 8+ results
        b = compute_player_features(mod)
        cols = ["ewm_pts", "last3_pts", "last5_pts", "last10_pts", "std10_pts", "season_avg_pts"]
        self.assertTrue(np.allclose(a.loc[:8, cols].fillna(-1), b.loc[:8, cols].fillna(-1)))   # rows 0..8 unaffected (row 8's features see games 0..7)
        self.assertTrue(np.isnan(a.loc[0, "ewm_pts"]))   # nothing before the first game
        self.assertEqual(a.loc[3, "last3_pts"], np.mean([10, 11, 12]))


class RotationLogic(unittest.TestCase):
    def test_missing_share(self):
        games = [{"a": 34.0, "b": 30.0, "c": 20.0, "d": 4.0}] * 5
        rot = rotation_from_games(games)
        self.assertEqual(set(rot), {"a", "b", "c"})          # d is below the rotation cutoff
        f = missing_features(rot, {"a"})
        self.assertAlmostEqual(f["miss_share"], 34 / 84)
        self.assertEqual(f["top_missing"], 1.0)
        self.assertEqual(missing_features(rot, set())["miss_share"], 0.0)
        self.assertEqual(missing_features({}, {"a"})["miss_share"], 0.0)


class NegBinomial(unittest.TestCase):
    def test_dispersion_recovered(self):
        rng = np.random.default_rng(0)
        mu = rng.uniform(6, 30, 20000)
        r_true = 15.0
        y = _nb_dist(mu, r_true).rvs(random_state=1)
        self.assertAlmostEqual(fit_dispersion(y, mu), r_true, delta=2.5)

    def test_probabilities_and_push(self):
        p = over_under_probs(20.0, 20.5, 18.0)
        self.assertAlmostEqual(p["over"] + p["under"], 1.0)
        self.assertEqual(p["push"], 0.0)
        q = over_under_probs(20.0, 20.0, 18.0)               # integer line: push mass exists
        self.assertGreater(q["push"], 0.03)
        self.assertAlmostEqual(q["over"] + q["under"] + q["push"], 1.0)
        # monotone in the line and in the mean
        self.assertGreater(over_under_probs(20, 17.5, 18)["over"], over_under_probs(20, 22.5, 18)["over"])
        self.assertGreater(over_under_probs(24, 20.5, 18)["over"], over_under_probs(18, 20.5, 18)["over"])


class PlayerModelEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wh = Warehouse(":memory:")
        build_league(cls.wh, seasons=(2023, 2024), games_per_team=70, seed=21)
        cls.pb = cls.wh.player_box_df()
        cls.tg = build_team_games(cls.wh.team_box_df(), cls.wh.games_df())
        cls.g = games_frame(cls.tg)
        cls.wf, cls.state, _ = RatingEngine().walk_forward(cls.g)
        cls.frame = build_training_frame(cls.pb, cls.tg, cls.wf)
        cls.model = PlayerModel().fit(cls.frame)

    def test_training_frame_is_clean(self):
        self.assertGreater(len(self.frame), 5000)
        m = training_mask(self.frame)
        self.assertGreater(m.sum(), 3000)
        feats = self.frame.loc[m]
        # std10 needs 3 games and season-to-date stats don't exist on a player's first game of a season; both
        # are legitimate gaps that the boosted model handles natively. Nothing else may be missing.
        allowed = {"std10_pts", "season_avg_pts", "season_avg_min"}
        nan_cols = {c for c in FEATURES if feats[c].isna().any()}
        self.assertTrue(nan_cols <= allowed, nan_cols - allowed)
        first_of_season = feats.groupby(["player_key", "season"]).cumcount() == 0
        self.assertTrue(first_of_season[feats["season_avg_pts"].isna()].all())

    def test_model_not_worse_than_baselines_and_calibrated(self):
        m = self.model.metrics
        self.assertLessEqual(m["pts_mae"], m["pts_mae_ewma_baseline"] * 1.01)
        self.assertTrue(0.72 < m["coverage_80"] < 0.88, m["coverage_80"])
        self.assertTrue(abs(m["pts_bias"]) < 0.4, m["pts_bias"])
        self.assertTrue(1 < m["nb_r"] < 500)

    def test_missing_teammates_feature_is_informative(self):
        # in the synthetic league team points are fixed, so when stars are out the rest score more
        df = self.frame[training_mask(self.frame)]
        hi = df[df["miss_share"] > 0.2]; lo = df[df["miss_share"] == 0.0]
        self.assertGreater(len(hi), 100)
        uplift_hi = (hi["pts"] - hi["ewm_pts"]).mean(); uplift_lo = (lo["pts"] - lo["ewm_pts"]).mean()
        self.assertGreater(uplift_hi, uplift_lo + 0.3)

    def test_inference_matches_training_feature_path(self):
        """A past game rebuilt through the inference path gets the same features as in the training frame."""
        key = self.g["game_key"].iloc[len(self.g) // 2]
        row = self.g[self.g["game_key"] == key].iloc[0]
        gd = row["game_date"]
        pb_before = self.pb[self.pb["game_date"] < gd.strftime("%Y-%m-%d")].copy()
        tgr = self.wf[self.wf["game_key"] == key].iloc[0]
        inf = build_inference_frame(pb_before, self.tg, int(row["home_id"]), int(row["away_id"]), True, gd,
                                    tgr["r_home_pts"], tgr["r_away_pts"], tgr["r_poss"])
        train = self.frame[(self.frame["game_key"] == key) & (self.frame["team_id"] == row["home_id"])].set_index("player_key")
        both = inf.set_index("player_key").join(train[["ewm_pts", "last5_pts", "season_avg_pts", "n_prior"]], rsuffix="_t", how="inner")
        self.assertGreater(len(both), 3)
        for c in ("ewm_pts", "last5_pts", "season_avg_pts", "n_prior"):
            self.assertTrue(np.allclose(both[c], both[c + "_t"], equal_nan=True), c)

    def test_out_players_flagged_and_predictions_sane(self):
        gd = self.g["game_date"].max() + pd.Timedelta(days=1)
        h, a = resolve_team("BOS").id, resolve_team("LAL").id
        star = "bos player1"
        inj = pd.DataFrame([{"team_id": h, "player_key": star, "status": "Out"}])
        inf = build_inference_frame(self.pb, self.tg, h, a, True, gd, 112.0, 108.0, 99.0, injuries=inj)
        self.assertTrue(inf.loc[inf["player_key"] == star, "out"].iloc[0])
        self.assertGreater(inf["miss_share"].iloc[0], 0.1)
        pred = self.model.predict_frame(inf)
        self.assertTrue(pred["proj_pts"].between(0, 60).all())
        self.assertTrue(pred["proj_min"].between(0, 48).all())
        healthy = build_inference_frame(self.pb, self.tg, h, a, True, gd, 112.0, 108.0, 99.0)
        p_out = self.model.predict_frame(inf).set_index("player_key")["proj_pts"]
        p_in = self.model.predict_frame(healthy).set_index("player_key")["proj_pts"]
        others = [p for p in p_in.index if p != star and p in p_out.index]
        self.assertGreater(p_out[others].sum(), p_in[others].sum())        # teammates absorb the missing production


if __name__ == "__main__":
    unittest.main()
