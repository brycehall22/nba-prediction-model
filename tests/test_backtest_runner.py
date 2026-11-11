import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import modeling.backtest as bt
from backtest_runner import replay_week, run_backtest
from modeling.backtest import Finding
from nbadata.warehouse import Warehouse
from oddsfeed.service import OddsService
from services import Engine
from tests.synthetic import build_league
from tests.test_pipeline import NoClient, game_odds, prop


def make_engine(seed=23, games_per_team=50):
    wh = Warehouse(":memory:")
    build_league(wh, seasons=(2023, 2024), games_per_team=games_per_team, seed=seed)
    tmp = tempfile.TemporaryDirectory()
    return Engine(wh, OddsService(wh, client=NoClient()), models_dir=tmp.name), tmp


class WeekReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng, cls.tmp = make_engine()
        cls.last = pd.Timestamp(cls.eng.wh.counts()["last_date"])
        week = cls.eng.wh.games_df(start=(cls.last - pd.Timedelta(days=6)).date().isoformat(), status="final")
        # store lines for a few games of that week so betting can be simulated
        for _, g in week.head(4).iterrows():
            cls.eng.odds.import_game_lines([game_odds(g["game_date"], int(g["home_team_id"]), int(g["away_team_id"]), spread=-2.5, total=224.5)])
        cls.week = week
        cls.rep = replay_week(cls.eng, tune_rating=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_structure_and_window(self):
        r = self.rep
        self.assertNotIn("error", r)
        self.assertEqual(r["end"], str(self.last.date()))
        self.assertEqual(len(r["games"]), len(self.week))
        self.assertTrue(r["games"]["date"].between(r["start"], r["end"]).all())
        self.assertGreater(len(r["players"]), 100)
        for k in ("naive", "base", "corr"):
            self.assertIn("margin_mae", r["game_summary"][k])
        self.assertGreater(r["game_summary"]["base"]["margin_mae"], 0)

    def test_models_were_fitted_only_on_data_before_the_week(self):
        d = self.rep["derived"]
        self.assertLess(pd.Timestamp(d.gm0.trained_through), pd.Timestamp(self.rep["start"]))
        self.assertLess(pd.Timestamp(d.pm0.trained_through), pd.Timestamp(self.rep["start"]))

    def test_betting_is_simulated_only_where_lines_were_stored(self):
        b = self.rep["betting"]
        self.assertEqual(b["games_with_lines"], 4)
        self.assertGreater(b["base_all"]["bets"], 0)
        self.assertEqual(b["base_all"]["bets"], b["base_all"]["wins"] + b["base_all"]["losses"] + sum(1 for x in self.rep["bets_base"] if x["result"] == "push"))
        self.assertTrue(all(x["game_key"] in set(self.week["game_key"]) for x in self.rep["bets_base"]))

    def test_misses_are_tagged(self):
        m = self.rep["misses"]
        self.assertTrue(m["top"])
        self.assertTrue(all(row["tags"] for row in m["top"]))
        errs = [abs(r["actual_pts"] - r["proj_corr"]) for r in m["top"]]
        self.assertEqual(errs, sorted(errs, reverse=True))

    def test_no_lookahead_changing_last_day_results_changes_no_prediction(self):
        eng2, tmp2 = make_engine()
        try:
            for _, g in self.week.head(4).iterrows():          # identical stored lines: player projections blend them in
                eng2.odds.import_game_lines([game_odds(g["game_date"], int(g["home_team_id"]), int(g["away_team_id"]), spread=-2.5, total=224.5)])
            last_day = self.last.date().isoformat()
            with eng2.wh.connect() as c:
                keys = [r["game_key"] for r in c.execute("SELECT game_key FROM games WHERE game_date=?", (last_day,))]
                for k in keys:                                   # wreck the final day's results everywhere they are stored
                    c.execute("UPDATE games SET home_pts=home_pts+40 WHERE game_key=?", (k,))
                    c.execute("UPDATE team_box SET pts=pts+40 WHERE game_key=? AND is_home=1", (k,))
                    c.execute("UPDATE player_box SET pts=pts+15 WHERE game_key=?", (k,))
            alt = replay_week(eng2, tune_rating=False)
            cols = ["base_margin", "base_total", "corr_margin", "corr_total"]
            a = self.rep["games"].set_index("game_key")[cols]
            b = alt["games"].set_index("game_key")[cols]
            self.assertTrue(np.allclose(a.sort_index().values, b.sort_index().values, atol=1e-9), "predictions moved when only later results changed")
            self.assertFalse(np.allclose(self.rep["games"]["actual_margin"].values, alt["games"]["actual_margin"].values))   # results did change
            # day-by-day replay: players' projections on the first days cannot depend on the last day's box scores
            first_day = self.rep["players"]["date"].min()
            ka = self.rep["players"][self.rep["players"]["date"] == first_day].set_index(["game", "player_key"])["proj_base"].sort_index()
            kb = alt["players"][alt["players"]["date"] == first_day].set_index(["game", "player_key"])["proj_base"].sort_index()
            self.assertTrue(np.allclose(ka.values, kb.values, atol=1e-9))
        finally:
            tmp2.cleanup()

    def test_errors_are_messages_not_crashes(self):
        r = replay_week(self.eng, end="2031-01-01", tune_rating=False)
        self.assertIn("error", r)
        self.assertIn("No completed games", r["error"])
        small, tmp = make_engine(games_per_team=8)
        try:
            self.assertIn("about 300+", replay_week(small, tune_rating=False)["error"])
        finally:
            tmp.cleanup()


class RunBacktest(unittest.TestCase):
    def test_clean_data_changes_nothing_and_writes_nothing(self):
        eng, tmp = make_engine(seed=29)
        try:
            res = run_backtest(eng, apply=True, tune_rating=False)
            self.assertFalse(res["applied"])
            self.assertEqual(res["adopted"], [])
            self.assertTrue(eng.tuning.is_default())
            import os
            self.assertFalse(os.path.exists(os.path.join(eng.models_dir, "tuning.json")))
            self.assertTrue(res["derived"].diagnostics["game"]["by_month"])
        finally:
            tmp.cleanup()

    def test_adopted_corrections_are_saved_applied_and_change_predictions(self):
        eng, tmp = make_engine(seed=31)
        try:
            eng.train()
            from nbadata.teams import resolve_team
            h, a = resolve_team("BOS").id, resolve_team("LAL").id
            day = (pd.Timestamp(eng.wh.counts()["last_date"]) + pd.Timedelta(days=2)).date().isoformat()
            before = eng.predict_game(h, a, day, with_props=False)["pred"]
            forced = ({"margin_scale": 1.0, "margin_sigma": None, "margin_bias": 1.5, "total_bias": -4.0, "win_cal": (0.8, 0.0), "prop_cal": (0.9, 0.0)},
                      [Finding("total bias", True, "forced for the test", "holdout MAE", 10.0, 9.5)])
            with mock.patch.object(bt, "fit_corrections", return_value=forced):
                res = run_backtest(eng, apply=True, tune_rating=False)
            self.assertTrue(res["applied"])
            import os
            self.assertTrue(os.path.exists(os.path.join(eng.models_dir, "tuning.json")))
            after = eng.predict_game(h, a, day, with_props=False)["pred"]
            self.assertAlmostEqual(after["total"] - before["total"], -4.0, delta=0.3)       # same models, bias applied
            self.assertAlmostEqual(after["margin"] - before["margin"], 1.5, delta=0.3)
            eng2 = Engine(eng.wh, eng.odds, models_dir=eng.models_dir)                       # survives a restart
            self.assertEqual(eng2.tuning.total_bias, -4.0)
            self.assertIn("total bias correction -4.00 pts", eng2.status()["corrections"][0] if len(eng2.status()["corrections"]) == 1 else " ".join(eng2.status()["corrections"]))
            # dry run: nothing is written
            eng3, tmp3 = make_engine(seed=31)
            try:
                with mock.patch.object(bt, "fit_corrections", return_value=forced):
                    res3 = run_backtest(eng3, apply=False, tune_rating=False)
                self.assertFalse(res3["applied"])
                self.assertTrue(eng3.tuning.is_default())
            finally:
                tmp3.cleanup()
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
