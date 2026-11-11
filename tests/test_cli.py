"""CLI behaviour with no network: provider outcomes are faked, and output is captured."""

import contextlib
import io
import os
import tempfile
import unittest
from datetime import timedelta
from unittest import mock

import requests

import cli
from nbadata.http import HttpClient, ProviderError
from nbadata.providers.base import GameStub
from nbadata.providers.nba_cdn import NBACDNProvider
from nbadata.teams import resolve_team
from nbadata.timeutil import today_eastern
from nbadata.warehouse import Warehouse, make_game_key

BOS, LAL = resolve_team("BOS"), resolve_team("LAL")


class FakeProv:
    def __init__(self, name, ok=True, stubs=(), fail_list=False):
        self.name, self.ok, self.stubs, self.fail_list = name, ok, list(stubs), fail_list

    def health_check(self):
        return {"provider": self.name, "ok": self.ok, "steps": [{"step": "request", "ok": self.ok, "detail": "HTTP 200" if self.ok else "HTTP 403 blocked"}]}

    def list_games(self, s, e):
        if self.fail_list:
            raise ProviderError("down")
        return [g for g in self.stubs if s.isoformat() <= g.game_date <= e.isoformat()]

    def supports(self, stub):
        return True

    def fetch_box(self, stub):
        return None


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


class CliBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self.tmp.name)                       # isolates ./models and ./.env
        self.db = os.path.join(self.tmp.name, "wh.db")
        env = {k: v for k, v in os.environ.items() if k != "ODDS_API_KEY"}
        self._env = mock.patch.dict(os.environ, env, clear=True)
        self._env.start()

    def tearDown(self):
        self._env.stop()
        os.chdir(self._cwd)
        self.tmp.cleanup()


class Doctor(CliBase):
    def _doctor(self, provs):
        with mock.patch("cli.default_providers", return_value=provs), \
                mock.patch("nbadata.providers.espn.ESPNProvider.fetch_injuries", return_value=[]):
            return run(["--db", self.db, "doctor"])

    def test_one_source_down_is_not_fatal_and_next_step_is_backfill(self):
        code, out = self._doctor([FakeProv("espn", True), FakeProv("nba_cdn", False)])
        self.assertEqual(code, 0)
        self.assertIn("usable via espn", out)
        self.assertIn("nba_cdn is down; not fatal", out)
        self.assertIn("ODDS_API_KEY", out)
        self.assertIn("Next step", out)
        self.assertIn("python cli.py backfill", out)

    def test_all_sources_down_explains_and_fails(self):
        code, out = self._doctor([FakeProv("espn", False), FakeProv("nba_cdn", False)])
        self.assertEqual(code, 1)
        self.assertIn("No game-data source is reachable", out)
        self.assertIn("VPN", out)


class Setup(CliBase):
    def test_stops_at_step_one_when_nothing_reachable(self):
        with mock.patch("cli.default_providers", return_value=[FakeProv("espn", False)]):
            code, out = run(["--db", self.db, "setup"])
        self.assertEqual(code, 1)
        self.assertIn("Step 1/5", out)
        self.assertNotIn("Step 2/5", out)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "models")))


class EmptyStates(CliBase):
    def test_backfill_with_zero_games_diagnoses_instead_of_silence(self):
        with mock.patch("nbadata.ingest.default_providers", return_value=[FakeProv("espn"), FakeProv("nba_cdn", fail_list=True)]):
            code, out = run(["--db", self.db, "backfill", "--seasons", "1"])
        self.assertEqual(code, 1)
        self.assertIn("No games were loaded at all", out)
        self.assertIn("espn", out)             # answered but listed zero games
        self.assertIn("nba_cdn", out)          # failed outright
        self.assertIn("doctor", out)

    def test_slate_on_empty_day_points_to_next_scheduled_games(self):
        wh = Warehouse(self.db)
        day = (today_eastern() + timedelta(days=6)).isoformat()
        wh.upsert_games([{"game_key": make_game_key(day, LAL.id, BOS.id), "game_date": day, "season": "2026-27", "season_type": 1,
                          "home_team_id": BOS.id, "away_team_id": LAL.id, "tipoff_utc": None, "status": "scheduled",
                          "home_pts": None, "away_pts": None, "espn_id": "1", "nba_id": None, "source": "espn"}])
        code, out = run(["--db", self.db, "slate"])
        self.assertEqual(code, 0)
        self.assertIn("No games found", out)
        self.assertIn(f"Next scheduled games: {day}", out)
        self.assertIn("preseason", out)
        self.assertIn("LAL @ BOS", out)

    def test_slate_with_empty_warehouse_says_what_to_run(self):
        code, out = run(["--db", self.db, "slate"])
        self.assertIn("No games found", out)
        self.assertIn("python cli.py update", out)

    def test_train_without_data_is_a_message_not_a_traceback(self):
        code, out = run(["--db", self.db, "train"])
        self.assertEqual(code, 1)
        self.assertIn("backfill", out)

    def test_status_ends_with_next_step(self):
        code, out = run(["--db", self.db, "status"])
        self.assertIn("Next step:", out)


def _insert_pred(wh, **kw):
    base = dict(created_at="2026-10-30T12:00:00Z", game_key="20261030_CHA_WAS", game_date="2026-10-30", kind="spread", side="HOME",
                player_key=None, player_name=None, line=-3.5, price=-110, book="draftkings", model_prob=0.55, market_prob=0.5,
                blended_prob=0.53, ev_pct=1.0, kelly=0.0, projection=None, result=None, actual=None, closing_line=None,
                closing_price=None, graded_at="2026-10-31T01:00:00Z", is_pick=1)
    base.update(kw)
    cols = ",".join(base)
    with wh.connect() as c:
        c.execute(f"INSERT INTO predictions({cols}) VALUES({','.join('?' * len(base))})", list(base.values()))


class Reporting(CliBase):
    def test_small_game_only_report_does_not_crash_and_explains_itself(self):
        """Mirrors the real failure case: a handful of graded game bets, no props (player_name is NULL)."""
        wh = Warehouse(self.db)
        for kind, side, res, act in (("moneyline", "HOME", "win", 5.0), ("moneyline", "AWAY", "loss", -5.0), ("spread", "HOME", "win", 5.0),
                                     ("spread", "AWAY", "loss", -5.0), ("total", "OVER", "loss", 210.0), ("total", "UNDER", "win", 210.0)):
            _insert_pred(wh, kind=kind, side=side, result=res, actual=act, ev_pct=4.0 if side in ("HOME", "UNDER") else -4.0)
        _insert_pred(wh, kind="total", side="OVER", line=221.5, result=None, graded_at=None)           # one still pending
        code, out = run(["--db", self.db, "report"])
        self.assertEqual(code, 0)
        for needle in ("BET PERFORMANCE REPORT", "model picks only", "Logged picks: 7", "Graded      : 6", "Pending: 1", "By market", "By side",
                       "claimed-EV bucket", "Lean check", "Closing-line value", "How much can you trust this?", "far too few", "Graded picks",
                       "ONE side"):
            self.assertIn(needle, out, needle)
        self.assertIn("Not available yet", out)                      # no closing prices stored

    def test_report_with_props_filters_and_all_flag(self):
        wh = Warehouse(self.db)
        for i in range(40):
            win = i % 2 == 0
            _insert_pred(wh, kind="prop", side="OVER", player_key="a b", player_name="A B", line=20.5, result="win" if win else "loss",
                         actual=24.0 if win else 15.0, ev_pct=5.0, closing_line=20.5, closing_price=-110)
        code, out = run(["--db", self.db, "report", "--kind", "prop", "--all"])
        self.assertEqual(code, 0)
        self.assertIn("market: prop", out)
        self.assertIn("A B OVER 20.5", out)
        self.assertIn("Not informative yet", out)                    # closing == shown: don't claim 'behind the close'
        self.assertNotIn("behind the close", out)
        self.assertIn("Calibration", out)
        code, out = run(["--db", self.db, "report", "--since", "2030-01-01"])
        self.assertIn("No bets are logged yet", out)

    def test_empty_report_and_grade_and_status_are_descriptive(self):
        code, out = run(["--db", self.db, "report"])
        self.assertIn("No bets are logged yet", out)
        code, out = run(["--db", self.db, "grade"])
        self.assertIn("Graded this run : 0", out)
        self.assertIn("Still pending", out)
        code, out = run(["--db", self.db, "status"])
        for needle in ("Game data", "Models", "Odds", "Bet log", "Next step:"):
            self.assertIn(needle, out)

    def test_grade_explains_pending_reasons(self):
        wh = Warehouse(self.db)
        _insert_pred(wh, result=None, graded_at=None, game_date="2026-12-01", game_key="20261201_LAL_BOS")
        code, out = run(["--db", self.db, "grade"])
        self.assertIn("Still pending   : 1", out)
        self.assertIn("2026-12-01", out)
        self.assertIn("not final", out)


class UpdateAndOdds(CliBase):
    def test_update_reports_what_changed_and_next_games(self):
        day = (today_eastern() + timedelta(days=3)).isoformat()
        stub = GameStub(make_game_key(day, LAL.id, BOS.id), day, "2026-27", 2, BOS.id, LAL.id, None, "scheduled", espn_id="9", source="espn")
        with mock.patch("nbadata.ingest.default_providers", return_value=[FakeProv("espn", stubs=[stub])]):
            code, out = run(["--db", self.db, "update"])
        self.assertEqual(code, 0)
        for needle in ("UPDATE:", "Games synced", "New box scores saved", "Injury entries loaded", "Data is current through", "Next scheduled games"):
            self.assertIn(needle, out, needle)
        self.assertIn(day, out)
        self.assertEqual(Warehouse(self.db).get_meta("last_update"), today_eastern().isoformat())

    def test_odds_prints_a_board_of_what_is_loaded(self):
        from nbadata.timeutil import iso_utc, now_utc
        from oddsfeed.the_odds_api import TheOddsAPI
        from tests import fixtures as fx
        commence = iso_utc(now_utc() + timedelta(hours=5))

        class StubClient:
            configured = True
            credits_remaining, credits_used = 487, 13
            def pull(self, include_props=True):
                ev = fx.odds_api_event("e1", "Boston Celtics", "Los Angeles Lakers", commence, {
                    "draftkings": {"spread": (-5.5, -110, -110), "total": (224.5, -108, -112), "ml": (-230, 190)},
                    "fanduel": {"spread": (-5.5, -112, -108), "total": (224.5, -110, -110), "ml": (-225, 185)}})
                pv = fx.odds_api_props("e1", "Boston Celtics", "Los Angeles Lakers", commence, {
                    "draftkings": [("Jayson Tatum", 27.5, -115, -105), ("Jaylen Brown", 24.5, -110, -110)], "fanduel": [("Jayson Tatum", 27.5, -110, -110)]})
                return TheOddsAPI.parse_game_odds([ev]), TheOddsAPI.parse_props(pv), ["2 games"]

        with mock.patch("oddsfeed.service.TheOddsAPI", StubClient):
            code, out = run(["--db", self.db, "odds"])
        self.assertEqual(code, 0)
        for needle in ("ODDS:", "Book lines", "Prop lines", "API credits", "487 remaining", "Lines on file", "LAL @ BOS", "-5.5", "224.5", "-230/190", "2/2"):
            self.assertIn(needle, out, needle)
        code, out = run(["--db", self.db, "odds"])             # immediately again: throttled, and says so
        self.assertIn("skipped: pulled recently", out)


class NetworkLayer(unittest.TestCase):
    def test_403_html_page_becomes_a_readable_error(self):
        class Resp:
            status_code = 403
            text = "<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD><BODY><H1>Access Denied</H1> You don't have permission</BODY></HTML>"
            headers = {}
        class Sess(requests.Session):
            def get(self, *a, **k):
                return Resp()
        client = HttpClient(session=Sess(), min_interval=0)
        with self.assertRaises(ProviderError) as cm:
            client.get_json("https://cdn.nba.com/x")
        msg = str(cm.exception)
        self.assertIn("HTTP 403", msg)
        self.assertIn("Access Denied", msg)
        self.assertNotIn("<", msg)
        self.assertIn("blocked by the site", msg)
        self.assertEqual(cm.exception.status, 403)

    def test_cdn_sends_browser_headers_by_default(self):
        h = NBACDNProvider().http.session.headers
        self.assertEqual(h["Referer"], "https://www.nba.com/")
        self.assertEqual(h["Origin"], "https://www.nba.com")

    def test_loky_cpu_count_preset_to_silence_wmic_warning(self):
        import config  # noqa: F401
        self.assertTrue(os.environ.get("LOKY_MAX_CPU_COUNT", "").isdigit())


if __name__ == "__main__":
    unittest.main()
