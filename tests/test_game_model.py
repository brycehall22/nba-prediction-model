import tempfile
import unittest

import numpy as np

from modeling.frames import build_team_games, games_frame
from modeling.game_model import GameModel
from nbadata.teams import resolve_team
from nbadata.warehouse import Warehouse
from tests.synthetic import build_league


class GameModelOnSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wh = Warehouse(":memory:")
        cls.truth = build_league(wh, seasons=(2023, 2024), games_per_team=70, seed=11)
        cls.tg = build_team_games(wh.team_box_df(), wh.games_df())
        cls.g = games_frame(cls.tg)
        cls.model = GameModel().fit(cls.g)

    def test_metrics_sane_and_beat_naive(self):
        m = self.model.metrics
        self.assertGreater(m["n_games_scored"], 200)
        self.assertLess(m["margin_mae"], m["margin_mae_naive"])
        self.assertLess(m["total_mae"], m["total_mae_naive"])
        self.assertLess(m["home_win_logloss"], m["home_win_logloss_baseline"])
        # data was generated with ~7.5 eff-pt noise per team: margin sigma must be near that scale, not ~2
        self.assertTrue(8 < m["margin_sigma"] < 16, m["margin_sigma"])
        self.assertTrue(10 < m["total_sigma"] < 22, m["total_sigma"])

    def test_predict_internal_consistency(self):
        h, a = resolve_team("BOS").id, resolve_team("LAL").id
        p = self.model.predict(h, a, self.tg, self.g["game_date"].max() + np.timedelta64(1, "D"))
        self.assertAlmostEqual(p["home_pts"] - p["away_pts"], p["margin"], places=6)
        self.assertAlmostEqual(p["home_pts"] + p["away_pts"], p["total"], places=6)
        self.assertTrue(0 < p["home_win_prob"] < 1)
        # swapping venues must favour whoever is at home
        q = self.model.predict(a, h, self.tg, self.g["game_date"].max() + np.timedelta64(1, "D"))
        self.assertGreater(p["margin"] + q["margin"], 0)     # both home edges sum positive

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            self.model.save(d)
            m2 = GameModel.load(d)
            h, a = resolve_team("DEN").id, resolve_team("MIA").id
            self.assertAlmostEqual(self.model.predict(h, a)["margin"], m2.predict(h, a)["margin"])
        self.assertIsNone(GameModel.load("/nonexistent_dir_xyz"))


class ResidualModelGate(unittest.TestCase):
    def test_no_rest_effect_means_no_residual_model(self):
        wh = Warehouse(":memory:")
        build_league(wh, seasons=(2023, 2024), games_per_team=70, seed=11)
        g = games_frame(build_team_games(wh.team_box_df(), wh.games_df()))
        m = GameModel().fit(g).metrics
        self.assertFalse(m["residual_model_margin"] or m["residual_model_total"])

    def test_real_back_to_back_penalty_is_found_with_right_sign(self):
        wh = Warehouse(":memory:")
        build_league(wh, seasons=(2022, 2023, 2024), games_per_team=70, seed=5, b2b_penalty=6.0, day_step=1.0)
        tg = build_team_games(wh.team_box_df(), wh.games_df())
        g = games_frame(tg)
        model = GameModel().fit(g)
        self.assertTrue(model.metrics["residual_model_margin"])
        h, a = resolve_team("BOS").id, resolve_team("LAL").id
        last = g["game_date"].max()
        day = last + np.timedelta64(1, "D")
        # make BOS (home) look like it played yesterday and LAL fully rested -> margin should drop vs the reverse
        import pandas as pd
        fake = pd.DataFrame([{"game_key": "x", "team_id": h, "game_date": last, "season": "s", "season_type": 2, "is_home": 1,
                              "opp_id": 0, "pts": 0, "opp_pts": 0, "poss": 99, "rest": 2, "b2b": 0, "g_last4": 0, "team_game_no": 1}])
        tired_home = model.predict(h, a, pd.concat([tg, fake]), day)["margin"]
        rested = model.predict(h, a, tg[tg["team_id"] != h], day)["margin"]       # no recent BOS game -> rested
        self.assertLess(tired_home, rested)


if __name__ == "__main__":
    unittest.main()
