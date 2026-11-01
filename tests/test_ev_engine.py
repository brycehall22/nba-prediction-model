import unittest

from ev_engine import GameEV, blend_win_prob, evaluate_prop, ev_per_unit, kelly_fraction
from modeling.player_model import over_under_probs
from oddsfeed.schema import american_to_decimal


class SpreadSignRegression(unittest.TestCase):
    """The old engine returned 0.72 for a fair favourite and 0.50 for a clear underdog."""
    ev = GameEV()

    def test_fair_favourite_is_about_50_percent(self):
        p = self.ev.spread_probs(margin=5.5, sigma=12.0, home_spread=-5.5)
        self.assertAlmostEqual(p["home"], 0.5, delta=0.02)

    def test_model_says_home_wins_by_5_but_line_is_minus_10(self):
        p = self.ev.spread_probs(margin=5.0, sigma=12.0, home_spread=-10.5)
        self.assertLess(p["home"], 0.40)
        self.assertGreater(p["away"], 0.60)

    def test_home_underdog_direction(self):
        # home is +7.5 underdog but model thinks they lose by only 3 -> home covers more than half
        p = self.ev.spread_probs(margin=-3.0, sigma=12.0, home_spread=7.5)
        self.assertGreater(p["home"], 0.6)

    def test_integer_spread_has_push_mass_and_sums_to_one(self):
        p = self.ev.spread_probs(margin=5.0, sigma=12.0, home_spread=-5.0)
        self.assertTrue(0.02 < p["push"] < 0.05)
        self.assertAlmostEqual(p["home"] + p["away"] + p["push"], 1.0)

    def test_totals(self):
        p = self.ev.total_probs(total=228.0, sigma=18.0, line=224.5)
        self.assertGreater(p["over"], 0.5)
        self.assertEqual(p["push"], 0.0)
        self.assertAlmostEqual(self.ev.total_probs(224.5, 18, 224.5)["over"], 0.5)


class EVAndKelly(unittest.TestCase):
    def test_ev_matches_hand_calc(self):
        self.assertAlmostEqual(ev_per_unit(0.55, 0.45, -110), 0.55 * (100 / 110) - 0.45)
        self.assertAlmostEqual(ev_per_unit(0.5, 0.5, -110), -0.0454545, places=5)   # the vig
        self.assertAlmostEqual(ev_per_unit(0.45, 0.45, -110), 0.45 * (100 / 110) - 0.45)   # push refunds stake

    def test_kelly_zero_without_edge_and_capped(self):
        self.assertEqual(kelly_fraction(0.50, 0.50, -110), 0.0)
        self.assertLessEqual(kelly_fraction(0.95, 0.05, 200, cap_pct=5), 0.05)
        f = kelly_fraction(0.60, 0.40, -110, fraction=1.0, cap_pct=100)
        b = american_to_decimal(-110) - 1
        self.assertAlmostEqual(f, (0.6 * b - 0.4) / b)

    def test_blend_toward_market(self):
        self.assertAlmostEqual(blend_win_prob(0.60, 0.0, 0.50, 0.5), 0.55)
        self.assertAlmostEqual(blend_win_prob(0.60, 0.0, None, 0.5), 0.60)
        self.assertAlmostEqual(blend_win_prob(0.60, 0.0, 0.50, 0.0), 0.60)
        # with a push, the blend acts on the conditional probability and keeps the push mass
        b = blend_win_prob(0.45, 0.05, 0.5, 1.0)
        self.assertAlmostEqual(b, 0.5 * 0.95)


class GameBets(unittest.TestCase):
    pred = {"margin": 8.0, "total": 230.0, "margin_sigma": 12.0, "total_sigma": 18.0}
    market = {"available": True, "home_spread": -4.5, "away_spread": 4.5, "home_spread_odds": -110, "away_spread_odds": -110,
              "total": 224.5, "total_over_odds": -110, "total_under_odds": -110, "home_moneyline": -190, "away_moneyline": 160,
              "fair": {"home_cover": 0.5, "over": 0.5, "home_win": 0.64}}

    def test_all_six_bets_and_directions(self):
        bets = GameEV().evaluate(self.pred, self.market, weight=0.0)
        self.assertEqual(len(bets), 6)
        by = {(b["kind"], b["side"]): b for b in bets}
        self.assertGreater(by[("spread", "HOME")]["ev_pct"], by[("spread", "AWAY")]["ev_pct"])
        self.assertGreater(by[("total", "OVER")]["ev_pct"], by[("total", "UNDER")]["ev_pct"])
        self.assertGreater(by[("moneyline", "HOME")]["model_prob"], 0.7)
        self.assertAlmostEqual(by[("spread", "HOME")]["edge_pts"], 3.5)
        self.assertAlmostEqual(by[("spread", "AWAY")]["edge_pts"], -3.5)

    def test_market_weight_shrinks_edges(self):
        raw = {(b["kind"], b["side"]): b["ev_pct"] for b in GameEV().evaluate(self.pred, self.market, weight=0.0)}
        shr = {(b["kind"], b["side"]): b["ev_pct"] for b in GameEV().evaluate(self.pred, self.market, weight=0.5)}
        self.assertLess(shr[("spread", "HOME")], raw[("spread", "HOME")])

    def test_no_market_no_bets(self):
        self.assertEqual(GameEV().evaluate(self.pred, {"available": False}), [])


class PropBets(unittest.TestCase):
    def test_over_under_and_warning(self):
        probs = over_under_probs(mu=28.0, line=22.5, r=18.0)
        bets = evaluate_prop(probs, 22.5, -110, -110, fair_over=0.5, weight=0.5)
        over, under = bets
        self.assertGreater(over["ev_pct"], 0)
        self.assertLess(under["ev_pct"], 0)
        self.assertIsNotNone(over["warning"])             # model far above market -> flagged
        self.assertLess(over["blended_prob"], over["model_prob"])
        self.assertEqual(under["kelly_pct"], 0.0)

    def test_push_on_integer_line_reduces_both_sides(self):
        a = evaluate_prop(over_under_probs(20, 20.0, 18), 20.0, -110, -110)
        self.assertGreater(a[0]["push_prob"], 0.03)
        self.assertAlmostEqual(a[0]["model_prob"] + a[1]["model_prob"] + a[0]["push_prob"], 1.0)


if __name__ == "__main__":
    unittest.main()
