"""One pick per line: marking, logging, migration of old two-sided logs, the lean check and learned market weights."""

import contextlib
import io
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import numpy as np

import cli
from ev_engine import GameEV, evaluate_prop, mark_picks
from modeling.player_model import over_under_probs
from modeling.tuning import Tuning
from nbadata.warehouse import SCHEMA, Warehouse
from predictions import PredictionLog


def bet(kind, side, ev, prob=0.5, line=20.5, player_key=None, **kw):
    b = {"kind": kind, "side": side, "line": line, "price": -110, "model_prob": prob, "push_prob": 0.0, "market_fair_prob": 0.5,
         "blended_prob": prob, "ev_pct": ev, "kelly_pct": 0.0, "label": "none", "warning": None, "player_key": player_key,
         "player": player_key, "projection": kw.pop("projection", None)}
    b.update(kw)
    return b


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


class Marking(unittest.TestCase):
    def test_higher_ev_side_is_the_pick_per_group(self):
        bets = mark_picks([bet("prop", "OVER", -5, player_key="a"), bet("prop", "UNDER", 4, player_key="a"),
                           bet("prop", "OVER", 2, player_key="b"), bet("prop", "UNDER", -9, player_key="b")])
        picks = {(b["player_key"], b["side"]) for b in bets if b["is_pick"]}
        self.assertEqual(picks, {("a", "UNDER"), ("b", "OVER")})

    def test_ties_go_to_the_more_probable_side(self):
        bets = mark_picks([bet("total", "OVER", -4.5, prob=0.49), bet("total", "UNDER", -4.5, prob=0.51)])
        self.assertEqual([b["side"] for b in bets if b["is_pick"]], ["UNDER"])

    def test_game_evaluate_gives_exactly_one_pick_per_market(self):
        pred = {"margin": 8.0, "total": 230.0, "margin_sigma": 12.0, "total_sigma": 18.0}
        market = {"available": True, "home_spread": -4.5, "away_spread": 4.5, "home_spread_odds": -110, "away_spread_odds": -110,
                  "total": 224.5, "total_over_odds": -110, "total_under_odds": -110, "home_moneyline": -190, "away_moneyline": 160,
                  "fair": {"home_cover": 0.5, "over": 0.5, "home_win": 0.64}}
        bets = GameEV().evaluate(pred, market, weight=0.0)
        self.assertEqual(len(bets), 6)
        picks = [b for b in bets if b["is_pick"]]
        self.assertEqual(sorted(b["kind"] for b in picks), ["moneyline", "spread", "total"])
        self.assertEqual({(b["kind"], b["side"]) for b in picks}, {("moneyline", "HOME"), ("spread", "HOME"), ("total", "OVER")})

    def test_prop_evaluate_marks_one_side(self):
        bets = evaluate_prop(over_under_probs(28.0, 22.5, 18.0), 22.5, -110, -110, fair_over=0.5, extra={"player_key": "x"})
        self.assertEqual([b["side"] for b in bets if b["is_pick"]], ["OVER"])


class Logging(unittest.TestCase):
    def test_one_pick_per_group_and_later_line_moves_do_not_add_picks(self):
        wh = Warehouse(":memory:")
        log = PredictionLog(wh)
        k = "20261101_MIA_LAL"
        first = mark_picks([bet("prop", "OVER", -3, line=24.5, player_key="luka"), bet("prop", "UNDER", 6, line=24.5, player_key="luka")])
        self.assertEqual(log.log(first, k, "2026-11-01", "draftkings"), 2)
        self.assertEqual(log.log(first, k, "2026-11-01", "draftkings"), 0)                       # identical re-log: nothing new
        moved = mark_picks([bet("prop", "OVER", 7, line=25.5, player_key="luka"), bet("prop", "UNDER", -2, line=25.5, player_key="luka")])
        self.assertEqual(log.log(moved, k, "2026-11-01", "draftkings"), 2)                       # new line is stored (for CLV)...
        df = wh.query_df("SELECT side, line, is_pick FROM predictions ORDER BY id")
        self.assertEqual(int(df["is_pick"].sum()), 1)                                            # ...but there is still only one pick
        pick = df[df["is_pick"] == 1].iloc[0]
        self.assertEqual((pick["side"], pick["line"]), ("UNDER", 24.5))                         # the first batch's choice stands
        other = mark_picks([bet("prop", "OVER", 1, line=10.5, player_key="other"), bet("prop", "UNDER", -1, line=10.5, player_key="other")])
        log.log(other, k, "2026-11-01", "draftkings")
        self.assertEqual(int(wh.query_df("SELECT SUM(is_pick) n FROM predictions")["n"].iloc[0]), 2)   # a new player gets their own pick


class Migration(unittest.TestCase):
    def test_old_two_sided_logs_get_picks_backfilled(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.db")
            old_ddl = SCHEMA[SCHEMA.index("CREATE TABLE IF NOT EXISTS predictions"):SCHEMA.index("CREATE INDEX IF NOT EXISTS idx_pred_date")]
            old_ddl = old_ddl.replace("    is_pick INTEGER,                    -- 1 = the side the model actually chose for this line (one per game/market/player/book)\n", "")
            self.assertNotIn("is_pick", old_ddl)
            c = sqlite3.connect(path)
            c.executescript(old_ddl)
            ins = ("INSERT INTO predictions(created_at, game_key, game_date, kind, side, player_key, line, price, book, model_prob, blended_prob, ev_pct) "
                   "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)")
            rows = [  # group 1: both sides logged together; UNDER has the higher EV
                ("2026-11-01T10:00:00Z", "g1", "2026-11-01", "prop", "OVER", "luka", 24.5, -110, "draftkings", .4, .4, -9.0),
                ("2026-11-01T10:00:00Z", "g1", "2026-11-01", "prop", "UNDER", "luka", 24.5, -110, "draftkings", .6, .6, 5.0),
                # group 2: logged twice; the first batch favoured OVER, a later one UNDER -> the FIRST batch decides
                ("2026-11-01T10:00:00Z", "g1", "2026-11-01", "prop", "OVER", "kessler", 12.5, -110, "draftkings", .55, .55, 3.0),
                ("2026-11-01T10:00:00Z", "g1", "2026-11-01", "prop", "UNDER", "kessler", 12.5, -110, "draftkings", .45, .45, -8.0),
                ("2026-11-02T09:00:00Z", "g1", "2026-11-01", "prop", "OVER", "kessler", 13.5, -110, "draftkings", .4, .4, -9.0),
                ("2026-11-02T09:00:00Z", "g1", "2026-11-01", "prop", "UNDER", "kessler", 13.5, -110, "draftkings", .6, .6, 8.0),
                # group 3: a spread, sides have opposite line signs
                ("2026-11-01T10:00:00Z", "g1", "2026-11-01", "spread", "HOME", None, -3.5, -110, "draftkings", .52, .52, -1.0),
                ("2026-11-01T10:00:00Z", "g1", "2026-11-01", "spread", "AWAY", None, 3.5, -110, "draftkings", .48, .48, -4.0)]
            c.executemany(ins, rows)
            c.commit(); c.close()
            wh = Warehouse(path)                                              # opening an old database migrates it
            df = wh.query_df("SELECT player_key, kind, side, line, is_pick FROM predictions ORDER BY id")
            self.assertEqual(int(df["is_pick"].sum()), 3)
            chosen = {(r["player_key"] if isinstance(r["player_key"], str) else r["kind"], r["side"], r["line"]) for _, r in df[df["is_pick"] == 1].iterrows()}
            self.assertEqual(chosen, {("luka", "UNDER", 24.5), ("kessler", "OVER", 12.5), ("spread", "HOME", -3.5)})
            Warehouse(path)                                                   # opening again is harmless
            self.assertEqual(int(wh.query_df("SELECT SUM(is_pick) n FROM predictions")["n"].iloc[0]), 3)


def seed_graded_props(wh, n_lines=40, pick_side="OVER", proj_minus_line=2.0, actual_minus_line=0.2, win_rate=0.5, seed=0):
    rng = np.random.default_rng(seed)
    with wh.connect() as c:
        for i in range(n_lines):
            win = rng.random() < win_rate
            for side in ("OVER", "UNDER"):
                is_pick = 1 if side == pick_side else 0
                res = ("win" if win else "loss") if is_pick else ("loss" if win else "win")
                c.execute("INSERT INTO predictions(created_at, game_key, game_date, kind, side, player_key, player_name, line, price, book, model_prob, "
                          "market_prob, blended_prob, ev_pct, projection, result, actual, closing_line, closing_price, graded_at, is_pick) "
                          "VALUES('2026-11-01T10:00:00Z',?,?,'prop',?,?,?,20.5,-110,'draftkings',?,0.5,0.55,?, ?, ?, ?, 20.5, -110,'2026-11-02T00:00:00Z',?)",
                          (f"g{i}", "2026-11-01", side, f"p{i}", f"Player {i}", 0.58 if is_pick else 0.42, 6.0 if is_pick else -12.0,
                           20.5 + proj_minus_line, res, 20.5 + actual_minus_line, is_pick))


class ReportViews(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.db = os.path.join(self.tmp.name, "wh.db")

    def tearDown(self):
        os.chdir(self._cwd)
        self.tmp.cleanup()

    def test_picks_only_is_meaningful_and_both_sides_cancel(self):
        wh = Warehouse(self.db)
        seed_graded_props(wh, n_lines=40, win_rate=0.35)
        log = PredictionLog(wh)
        picks = log.detailed_report(picks_only=True)
        both = log.detailed_report(picks_only=False)
        self.assertEqual(picks["overall"]["bets"], 40)
        self.assertAlmostEqual(picks["overall"]["win_rate"], 14 / 40, delta=0.2)
        self.assertEqual(both["overall"]["bets"], 80)
        self.assertEqual(both["overall"]["wins"], both["overall"]["losses"])                    # complements cancel exactly
        code, out = run(["--db", self.db, "report", "--both-sides"])
        self.assertIn("every side of every line", out)
        self.assertIn("cancel", out)

    def test_one_sided_lean_is_called_out_with_the_diagnosis(self):
        wh = Warehouse(self.db)
        seed_graded_props(wh, n_lines=40, pick_side="UNDER", proj_minus_line=-2.0, actual_minus_line=-0.1)
        code, out = run(["--db", self.db, "report"])
        self.assertIn("Lean check", out)
        self.assertIn("<-- 100% on UNDER", out)
        self.assertIn("average projection minus line: -2.00", out)
        self.assertIn("systematically below the sportsbook lines", out)
        self.assertIn("the MODEL being off", out)

    def test_real_edge_pattern_is_not_blamed_on_the_model(self):
        wh = Warehouse(self.db)
        seed_graded_props(wh, n_lines=40, pick_side="OVER", proj_minus_line=2.0, actual_minus_line=1.9)
        code, out = run(["--db", self.db, "report"])
        self.assertIn("what a real edge would look like", out)

    def test_clv_with_almost_no_movement_is_not_called_behind_the_close(self):
        wh = Warehouse(self.db)
        seed_graded_props(wh, n_lines=40)
        with wh.connect() as c:                      # 1 of 80 closing prices differs slightly: CLV prints as -0.00
            c.execute("UPDATE predictions SET closing_price=-111 WHERE id=1")
        code, out = run(["--db", self.db, "report"])
        self.assertIn("Not informative yet", out)
        self.assertNotIn("behind the close", out)


class LearnedBlendWeights(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.db = os.path.join(self.tmp.name, "wh.db")

    def tearDown(self):
        os.chdir(self._cwd)
        self.tmp.cleanup()

    def seed_overconfident_model(self, n):
        """The market's probability is right; the model's is noisy and over-confident."""
        rng = np.random.default_rng(3)
        wh = Warehouse(self.db)
        with wh.connect() as c:
            for i in range(n):
                true_p = rng.uniform(0.35, 0.65)
                model_p = float(np.clip(true_p + rng.normal(0, 0.30), 0.03, 0.97))
                win = rng.random() < true_p
                c.execute("INSERT INTO predictions(created_at, game_key, game_date, kind, side, player_key, line, price, book, model_prob, market_prob, "
                          "blended_prob, ev_pct, result, actual, graded_at, is_pick) VALUES('2026-11-01T10:00:00Z',?,?,'prop','OVER',?,20.5,-110,'draftkings',?,?,?,5.0,?,20,'x',1)",
                          (f"g{i}", "2026-11-01", f"p{i}", model_p, true_p, 0.5 * model_p + 0.5 * true_p, "win" if win else "loss"))

    def test_strong_evidence_is_saved_and_reloaded(self):
        self.seed_overconfident_model(400)
        rep = PredictionLog(Warehouse(self.db)).detailed_report()
        v = rep["blend_weights"]["prop"]
        self.assertTrue(v["apply_ok"])
        self.assertGreater(v["best_w"], 0.6)
        self.assertLess(v["ll_best"], v["ll_current"])
        code, out = run(["--db", self.db, "report"])
        self.assertIn("python cli.py report --apply-weights", out)
        self.assertFalse(os.path.exists("models/tuning.json"))                  # not saved until asked
        code, out = run(["--db", self.db, "report", "--apply-weights"])
        self.assertIn("Saved: props market blend 50% ->", out)
        t = Tuning.load("models")
        self.assertGreater(t.prop_market_weight, 0.6)
        self.assertIsNone(t.game_market_weight)
        self.assertTrue(any("player-prop market blend" in d for d in t.describe()))

    def test_thin_evidence_is_refused(self):
        self.seed_overconfident_model(60)
        code, out = run(["--db", self.db, "report", "--apply-weights"])
        self.assertIn("Nothing saved", out)
        self.assertIn("need 150+ picks", out)
        self.assertFalse(os.path.exists("models/tuning.json"))


if __name__ == "__main__":
    unittest.main()
