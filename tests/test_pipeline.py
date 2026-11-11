import tempfile
import unittest

import numpy as np
import pandas as pd

from nbadata.teams import resolve_team
from nbadata.timeutil import iso_utc, now_utc
from nbadata.warehouse import Warehouse
from oddsfeed.schema import GameOdds, PropOdds
from oddsfeed.service import OddsService
from services import Engine, NotReady
from tests.synthetic import build_league

BOS, LAL = resolve_team("BOS").id, resolve_team("LAL").id


class NoClient:
    configured = False


def game_odds(date, h, a, book="draftkings", spread=-3.5, total=224.5):
    return GameOdds(game_date=date, home_team_id=h, away_team_id=a, commence_utc=None, book=book,
                    fetched_at=iso_utc(now_utc()), source="test", spread_home=spread, spread_home_price=-110,
                    spread_away_price=-110, total=total, over_price=-110, under_price=-110, ml_home=-160, ml_away=140)


def prop(date, h, a, name, line, book="draftkings", over=-115, under=-105):
    from nbadata.players import normalize_name
    return PropOdds(game_date=date, home_team_id=h, away_team_id=a, commence_utc=None, book=book,
                    fetched_at=iso_utc(now_utc()), source="test", market="player_points", player_name=name,
                    player_key=normalize_name(name), line=line, over_price=over, under_price=under)


def res_players_with_props(res):
    return len({b["player_key"] for b in res["prop_bets"]})


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wh = Warehouse(":memory:")
        build_league(cls.wh, seasons=(2023, 2024), games_per_team=70, seed=31)
        cls.odds = OddsService(cls.wh, client=NoClient())
        cls.eng = Engine(cls.wh, cls.odds, models_dir=cls.tmp.name)
        cls.report = cls.eng.train()
        cls.last = pd.Timestamp(cls.wh.counts()["last_date"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_train_persists_and_reloads(self):
        self.assertIn("margin_mae", self.report["game"])
        self.assertIn("pts_mae", self.report["player"])
        eng2 = Engine(self.wh, self.odds, models_dir=self.tmp.name)
        self.assertIsNotNone(eng2.game_model)
        self.assertIsNotNone(eng2.player_model)
        self.assertEqual(eng2.status()["models"]["game"], self.report["trained_through"])

    def test_untrained_engine_gives_actionable_error(self):
        with tempfile.TemporaryDirectory() as d:
            eng = Engine(self.wh, self.odds, models_dir=d)
            with self.assertRaises(NotReady) as cm:
                eng.predict_game(BOS, LAL)
            self.assertIn("cli.py", str(cm.exception))

    def test_future_game_with_market_props_and_injury(self):
        day = (self.last + pd.Timedelta(days=3)).date().isoformat()
        self.odds.import_game_lines([game_odds(day, BOS, LAL)])
        self.odds.import_props([prop(day, BOS, LAL, "BOS Player1", 22.5), prop(day, BOS, LAL, "BOS Player2", 17.5),
                                prop(day, BOS, LAL, "LAL Player1", 22.5), prop(day, BOS, LAL, "Totally Unknown Guy", 9.5)])
        self.wh.replace_injuries([{"team_id": BOS, "player_key": "bos player2", "player_name": "BOS Player2", "status": "Out", "detail": ""}])
        res = self.eng.predict_game(BOS, LAL, day)
        self.assertEqual(len(res["game_bets"]), 6)
        self.assertTrue(res["market"]["available"])
        names = {b["player_key"] for b in res["prop_bets"]}
        self.assertIn("bos player1", names)
        self.assertIn("lal player1", names)
        self.assertNotIn("bos player2", names)                 # ruled out -> no bet offered
        self.assertNotIn("totally unknown guy", names)         # unmatched names are skipped, not guessed
        bos2 = next(p for p in res["home_players"] if p["player_key"] == "bos player2")
        self.assertTrue(bos2["out"])
        self.assertEqual(bos2["proj_pts"], 0.0)
        live_total = sum(p["proj_pts"] for p in res["home_players"] if not p["out"])
        self.assertTrue(80 < live_total < 150, live_total)     # reconciled to a sensible team total
        for b in res["prop_bets"]:
            self.assertAlmostEqual(b["model_prob"] + (1 - b["model_prob"] - b["push_prob"]) + b["push_prob"], 1.0)

    def test_log_grade_report_roundtrip_matches_independent_recompute(self):
        g = self.wh.games_df(status="final")
        row = g.iloc[len(g) // 2]
        h, a, day = int(row["home_team_id"]), int(row["away_team_id"]), row["game_date"]
        pbs = self.wh.player_box_df()
        top = pbs[(pbs["game_key"] == row["game_key"]) & (pbs["team_id"] == h) & (pbs["played"] == 1)].sort_values("minutes", ascending=False).head(3)
        self.odds.import_game_lines([game_odds(day, h, a, spread=-2.5, total=227.5)])
        self.odds.import_props([prop(day, h, a, r["player_name"], 18.5) for _, r in top.iterrows()])
        res = self.eng.predict_game(h, a, day)
        self.assertTrue(res["prop_bets"], "expected prop bets for the played game")
        n = self.eng.log.log(res["game_bets"] + res["prop_bets"], res["game_key"], day, "draftkings")
        self.assertGreater(n, 6)
        self.assertEqual(self.eng.log.log(res["game_bets"] + res["prop_bets"], res["game_key"], day, "draftkings"), 0)   # dedupe
        picks = self.wh.query_df("SELECT kind, IFNULL(player_key,'') pk, COUNT(*) n FROM predictions WHERE game_key=? AND is_pick=1 GROUP BY kind, pk",
                                 [row["game_key"]])
        self.assertTrue((picks["n"] == 1).all())                               # exactly one pick per spread/total/moneyline/player line
        self.assertTrue({"spread", "total", "moneyline", "prop"} <= set(picks["kind"]))
        self.assertEqual(int((picks["kind"] == "prop").sum()), int(res_players_with_props(res)))
        graded = self.eng.log.grade()
        self.assertEqual(graded, n)
        preds = self.wh.query_df("SELECT * FROM predictions WHERE game_key=?", [row["game_key"]])
        hp, ap = row["home_pts"], row["away_pts"]
        sp_home = preds[(preds["kind"] == "spread") & (preds["side"] == "HOME")].iloc[0]
        expect = "push" if (hp - ap) - 2.5 == 0 else ("win" if (hp - ap) - 2.5 > 0 else "loss")
        self.assertEqual(sp_home["result"], expect)
        tot_over = preds[(preds["kind"] == "total") & (preds["side"] == "OVER")].iloc[0]
        self.assertEqual(tot_over["result"], "win" if hp + ap > 227.5 else "loss")
        p_over = preds[(preds["kind"] == "prop") & (preds["side"] == "OVER")].iloc[0]
        actual = float(top[top["player_name"] == p_over["player_name"]]["pts"].iloc[0])
        self.assertEqual(p_over["actual"], actual)
        self.assertEqual(p_over["result"], "win" if actual > 18.5 else "loss")
        self.assertIsNotNone(p_over["closing_price"])
        rep = self.eng.log.report()
        self.assertEqual(rep["n"], n)
        self.assertIn("prop", rep["by_kind"])
        pr = self.eng.log.detailed_report()                                    # default view = picks only
        self.assertEqual(pr["overall"]["bets"], len(picks))
        self.assertEqual(self.eng.log.detailed_report(picks_only=False)["overall"]["bets"], n)

    def test_learned_market_weights_reach_the_bet_evaluation(self):
        from modeling.tuning import Tuning
        day = (self.last + pd.Timedelta(days=7)).date().isoformat()
        self.odds.import_game_lines([game_odds(day, BOS, LAL, spread=-1.5, total=235.5)])
        self.odds.import_props([prop(day, BOS, LAL, "BOS Player1", 14.5)])
        saved = self.eng.tuning
        try:
            base = self.eng.predict_game(BOS, LAL, day)
            self.eng.tuning = Tuning(prop_market_weight=0.9, game_market_weight=0.9)
            tuned = self.eng.predict_game(BOS, LAL, day)
        finally:
            self.eng.tuning = saved
        def gap(res, kind):
            return np.mean([abs(b["blended_prob"] - b["market_fair_prob"]) for b in res["game_bets"] + res["prop_bets"]
                            if b["kind"] == kind and b["market_fair_prob"] is not None])
        for kind in ("prop", "spread", "total"):
            self.assertLess(gap(tuned, kind), gap(base, kind), kind)           # more market weight -> closer to the market's price

    def test_preseason_games_are_excluded_from_modelling(self):
        wh = Warehouse(":memory:")
        build_league(wh, seasons=(2023,), games_per_team=30, seed=2)
        with wh.connect() as c:
            c.execute("UPDATE games SET season_type=1 WHERE game_date < '2023-11-05'")
        n_pre = int(wh.query_df("SELECT COUNT(*) n FROM games WHERE season_type=1")["n"].iloc[0])
        self.assertGreater(n_pre, 0)
        eng = Engine(wh, OddsService(wh, client=NoClient()), models_dir=self.tmp.name)
        tg, g, pb = eng._frames()
        self.assertEqual(len(g), len(wh.games_df()) - n_pre)
        self.assertTrue((g["season_type"] != 1).all())
        self.assertFalse(pb["game_key"].isin(wh.games_df().query("season_type == 1")["game_key"]).any())

    def test_slate_runs_and_survives_a_bad_game(self):
        day = (self.last + pd.Timedelta(days=5)).date().isoformat()
        DEN, MIA = resolve_team("DEN").id, resolve_team("MIA").id
        self.odds.import_game_lines([game_odds(day, DEN, MIA), game_odds(day, BOS, LAL)])
        out = self.eng.slate(day, with_props=False, log=False)
        self.assertEqual(len(out), 2)
        self.assertTrue(all("error" not in r for r in out))


if __name__ == "__main__":
    unittest.main()
