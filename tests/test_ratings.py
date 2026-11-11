import unittest

import numpy as np
import pandas as pd

from modeling.frames import build_team_games, games_frame
from modeling.ratings import RatingEngine
from nbadata.warehouse import Warehouse
from tests.synthetic import build_league


def load(wh):
    tg = build_team_games(wh.team_box_df(), wh.games_df())
    return tg, games_frame(tg)


class RatingsRecoverTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wh = Warehouse(":memory:")
        cls.truth = build_league(cls.wh, seasons=(2023, 2024), games_per_team=70, seed=3)
        cls.tg, cls.g = load(cls.wh)

    def test_frames_shapes(self):
        self.assertEqual(len(self.tg), 2 * len(self.g))
        self.assertTrue(self.g["poss"].between(80, 120).all())
        first = self.tg.sort_values("game_date").groupby("team_id").head(1)
        self.assertTrue((first["rest"] == 7).all())          # first game of data = fully rested placeholder
        self.assertTrue(self.tg["b2b"].isin([0.0, 1.0]).all())

    def test_final_ratings_correlate_with_truth(self):
        state = RatingEngine().fit_asof(self.g)
        ids = list(self.truth["off"])
        for key, attr in (("off", "off"), ("dfn", "dfn")):
            est = np.array([getattr(state, attr)[t] for t in ids])
            tru = np.array([self.truth[key][t] for t in ids])
            self.assertGreater(np.corrcoef(est, tru)[0, 1], 0.8, key)
        self.assertAlmostEqual(state.hca, self.truth["hca"], delta=1.5)
        self.assertAlmostEqual(state.mu, 115.0, delta=1.5)

    def test_walk_forward_has_no_lookahead(self):
        """Changing results on/after day D must not change any prediction for games on days <= D."""
        eng = RatingEngine()
        base, _, _ = eng.walk_forward(self.g)
        cut = self.g["game_date"].sort_values().iloc[len(self.g) // 2]
        g2 = self.g.copy()
        late = g2["game_date"] >= cut
        g2.loc[late, "home_pts"] = g2.loc[late, "home_pts"] + 40      # absurd, would shift ratings if leaked
        g2["margin"] = g2["home_pts"] - g2["away_pts"]
        g2["total"] = g2["home_pts"] + g2["away_pts"]
        alt, _, _ = eng.walk_forward(g2)
        before = base["game_date"] <= cut
        a, b = base.loc[before, "r_margin"].astype(float), alt.loc[before, "r_margin"].astype(float)
        # games ON the cut day are predicted from strictly earlier games, so they match too
        self.assertTrue(np.allclose(a, b, atol=1e-9), f"max diff {np.abs(a - b).max():.3g}")
        self.assertFalse(np.allclose(base.loc[~before, "r_margin"].astype(float), alt.loc[~before, "r_margin"].astype(float)))

    def test_ratings_beat_naive_baselines_on_known_truth(self):
        wf, _, _ = RatingEngine().walk_forward(self.g)
        ok = (wf["gno_h"] >= 15) & (wf["gno_a"] >= 15)
        mae_r = (wf.loc[ok, "margin"] - wf.loc[ok, "r_margin"].astype(float)).abs().mean()
        mae_c = (wf.loc[ok, "margin"] - wf.loc[ok, "margin"].mean()).abs().mean()
        self.assertLess(mae_r, mae_c)


if __name__ == "__main__":
    unittest.main()
