import os
import tempfile
import unittest
from datetime import date

from nbadata.providers.base import GameStub
from nbadata.providers.espn import ESPNProvider
from nbadata.providers.nba_cdn import NBACDNProvider, parse_iso_minutes
from nbadata.ingest import Ingestor
from nbadata.teams import resolve_team
from nbadata.warehouse import Warehouse
from oddsfeed.csv_import import import_game_lines, import_props
from oddsfeed.schema import american_to_prob, devig_two_way
from oddsfeed.service import OddsService
from oddsfeed.the_odds_api import TheOddsAPI, OddsAPIError
from tests import fixtures as fx

BOS, LAL, DEN = resolve_team("BOS"), resolve_team("LAL"), resolve_team("DEN")


class FakeProvider:
    """Stands in for a provider so Ingestor logic can be tested without a network."""
    def __init__(self, name, stubs, boxes, fail=False):
        self.name, self._stubs, self._boxes, self.fail = name, stubs, boxes, fail
    def list_games(self, s, e):
        from nbadata.http import ProviderError
        if self.fail:
            raise ProviderError("down")
        return [g for g in self._stubs if s.isoformat() <= g.game_date <= e.isoformat()]
    def supports(self, stub):
        return True
    def fetch_box(self, stub):
        from nbadata.http import ProviderError
        if self.fail:
            raise ProviderError("down")
        return self._boxes.get(stub.game_key)


class ESPNParsing(unittest.TestCase):
    def test_scoreboard_times_and_teams(self):
        data = fx.espn_scoreboard([
            {"id": 11, "date": "2025-01-16T00:30Z", "home": "BOS", "away": "LAL", "home_pts": 110, "away_pts": 100},
            {"id": 12, "date": "2025-01-16T03:00Z", "home": "GS", "away": "NO", "home_pts": 99, "away_pts": 101},
            {"id": 13, "date": "2025-02-16T23:00Z", "home": "TEAM", "away": "STRIPES", "home_name": "Team LeBron"},  # All-Star -> skipped
        ])
        stubs = ESPNProvider.parse_scoreboard(data)
        self.assertEqual(len(stubs), 2)
        bos = stubs[0]
        self.assertEqual(bos.game_date, "2025-01-15")          # 00:30Z is still the 15th in Eastern time
        self.assertEqual(bos.game_key, "20250115_LAL_BOS")
        self.assertEqual((bos.home_pts, bos.away_pts, bos.status), (110, 100, "final"))
        self.assertEqual(stubs[1].game_key, "20250115_NOP_GSW")  # ESPN's GS / NO abbreviations resolve

    def test_scheduled_game_has_no_score(self):
        data = fx.espn_scoreboard([{"id": 1, "date": "2030-01-01T00:00Z", "home": "BOS", "away": "LAL", "final": False}])
        stub = ESPNProvider.parse_scoreboard(data)[0]
        self.assertEqual(stub.status, "scheduled")
        self.assertIsNone(stub.home_pts)

    def test_summary_parses_by_label_and_validates(self):
        stub = ESPNProvider.parse_scoreboard(fx.espn_scoreboard([
            {"id": 11, "date": "2025-01-16T00:30Z", "home": "BOS", "away": "LAL", "home_pts": 110, "away_pts": 100}]))[0]
        data = fx.espn_summary("BOS", "LAL", fx.roster("Boston", 110), fx.roster("LA", 100), 110, 100)
        box = ESPNProvider.parse_summary(data, stub)
        self.assertIsNotNone(box)
        self.assertEqual(box.validate(), [])
        starters = [p for p in box.players if p.team_id == BOS.id and p.starter]
        self.assertEqual(len(starters), 5)
        p = box.players[0]
        self.assertAlmostEqual(p.minutes, 34.5)
        self.assertEqual((p.fgm, p.fga, p.fg3m, p.fg3a, p.ftm, p.fta), (5, 10, 1, 3, 2, 2))
        dnp = [x for x in box.players if x.player_name.endswith("Benchwarmer")]
        self.assertTrue(all(x.played == 0 and x.minutes == 0 for x in dnp))
        rows = {r["team_id"]: r for r in box.team_rows()}
        self.assertEqual(rows[BOS.id]["pts"], 110)
        self.assertEqual(rows[BOS.id]["fga"], 90)              # 9 players x 10 FGA

    def test_summary_with_missing_players_is_none_not_crash(self):
        stub = GameStub("k", "2025-01-15", "2024-25", 2, BOS.id, LAL.id, None, "final", 1, 1)
        self.assertIsNone(ESPNProvider.parse_summary({"boxscore": {}}, stub))
        self.assertIsNone(ESPNProvider.parse_summary({}, stub))

    def test_validate_catches_bad_totals(self):
        stub = ESPNProvider.parse_scoreboard(fx.espn_scoreboard([
            {"id": 11, "date": "2025-01-16T00:30Z", "home": "BOS", "away": "LAL", "home_pts": 130, "away_pts": 100}]))[0]
        box = ESPNProvider.parse_summary(fx.espn_summary("BOS", "LAL", fx.roster("B", 110), fx.roster("L", 100), 130, 100), stub)
        self.assertTrue(any("!=" in p for p in box.validate()))

    def test_injuries_and_embedded_odds(self):
        inj = ESPNProvider.parse_injuries({"injuries": [{"displayName": "Boston Celtics", "injuries": [
            {"athlete": {"displayName": "Jayson Tatum"}, "status": "Out", "shortComment": "Achilles"}]}]})
        self.assertEqual((inj[0]["team_id"], inj[0]["player_key"], inj[0]["status"]), (BOS.id, "jayson tatum", "Out"))
        data = fx.espn_scoreboard([{"id": 1, "date": "2030-01-02T00:30Z", "home": "BOS", "away": "LAL", "final": False,
                                    "odds": {"provider": {"name": "DraftKings"}, "details": "BOS -5.5", "overUnder": 224.5,
                                             "homeTeamOdds": {"moneyLine": -230}, "awayTeamOdds": {"moneyLine": 190}}}])
        o = ESPNProvider.parse_scoreboard_odds(data)[0]
        self.assertEqual((o["spread_home"], o["total"], o["ml_home"], o["book"]), (-5.5, 224.5, -230, "draftkings"))
        data["events"][0]["competitions"][0]["odds"][0]["details"] = "LAL -2"   # away favourite -> home spread positive
        self.assertEqual(ESPNProvider.parse_scoreboard_odds(data)[0]["spread_home"], 2.0)


class CDNParsing(unittest.TestCase):
    def test_minutes_parser(self):
        self.assertAlmostEqual(parse_iso_minutes("PT34M12.00S"), 34.2)
        self.assertEqual(parse_iso_minutes(""), 0.0)
        self.assertAlmostEqual(parse_iso_minutes("PT05M"), 5.0)

    def test_schedule_and_box(self):
        sched = NBACDNProvider.parse_schedule(fx.cdn_schedule([
            {"id": "0022400500", "date": "2025-01-15", "utc": "2025-01-16T00:30:00Z", "home": "BOS", "away": "LAL", "home_pts": 110, "away_pts": 100},
            {"id": "0022400501", "date": "2025-01-16", "utc": "2025-01-17T01:00:00Z", "home": "DEN", "away": "BOS", "status": 1}]))
        self.assertEqual([g.game_key for g in sched], ["20250115_LAL_BOS", "20250116_BOS_DEN"])
        self.assertEqual(sched[0].nba_id, "0022400500")
        self.assertEqual(sched[0].season_type, 2)
        self.assertEqual(sched[1].status, "scheduled")
        box = NBACDNProvider.parse_boxscore(fx.cdn_boxscore(fx.roster("B", 110), fx.roster("L", 100), 110, 100), sched[0])
        self.assertEqual(box.validate(), [])
        self.assertTrue(any(p.dnp_reason for p in box.players))


class IngestFallback(unittest.TestCase):
    def setUp(self):
        self.wh = Warehouse(":memory:")
        espn_stub = ESPNProvider.parse_scoreboard(fx.espn_scoreboard([
            {"id": 11, "date": "2025-01-16T00:30Z", "home": "BOS", "away": "LAL", "home_pts": 110, "away_pts": 100}]))[0]
        cdn_stub = NBACDNProvider.parse_schedule(fx.cdn_schedule([
            {"id": "0022400500", "date": "2025-01-15", "utc": "2025-01-16T00:30:00Z", "home": "BOS", "away": "LAL", "home_pts": 110, "away_pts": 100}]))[0]
        self.stub = espn_stub
        self.box = NBACDNProvider.parse_boxscore(fx.cdn_boxscore(fx.roster("B", 110), fx.roster("L", 100), 110, 100), cdn_stub)
        self.assertIsNotNone(self.box)

    def test_falls_back_when_primary_down_and_is_idempotent(self):
        primary = FakeProvider("espn", [self.stub], {}, fail=True)
        secondary = FakeProvider("nba_cdn", [self.stub], {self.stub.game_key: self.box})
        ing = Ingestor(self.wh, [primary, secondary])
        rep = ing.backfill(seasons=["2024-25"])
        self.assertEqual(rep.boxes_saved, 1)
        self.assertEqual(rep.provider_errors.get("espn", 0) > 0, True)
        counts = self.wh.counts()
        self.assertEqual((counts["games_with_box"], counts["player_rows"]), (1, 20))
        ing.backfill(seasons=["2024-25"])                       # re-run: nothing duplicated
        self.assertEqual(self.wh.counts()["player_rows"], 20)

    def test_bad_box_is_rejected_not_stored(self):
        bad = fx.cdn_boxscore(fx.roster("B", 110), fx.roster("L", 100), 140, 100)   # header score disagrees with players
        stub = self.stub
        from nbadata.providers.nba_cdn import NBACDNProvider as C
        sched = C.parse_schedule(fx.cdn_schedule([{"id": "x", "date": "2025-01-15", "utc": "2025-01-16T00:30:00Z", "home": "BOS", "away": "LAL", "home_pts": 140, "away_pts": 100}]))[0]
        box = C.parse_boxscore(bad, sched)
        ing = Ingestor(self.wh, [FakeProvider("espn", [stub], {stub.game_key: box})])
        rep = ing.backfill(seasons=["2024-25"])
        self.assertEqual((rep.boxes_saved, rep.boxes_failed), (0, 1))
        self.assertEqual(self.wh.counts()["games_with_box"], 0)


class IngestEmptyFallthrough(unittest.TestCase):
    def test_empty_primary_does_not_block_secondary(self):
        wh = Warehouse(":memory:")
        stub = ESPNProvider.parse_scoreboard(fx.espn_scoreboard([
            {"id": 11, "date": "2025-01-16T00:30Z", "home": "BOS", "away": "LAL", "home_pts": 110, "away_pts": 100}]))[0]
        ing = Ingestor(wh, [FakeProvider("espn", [], {}), FakeProvider("nba_cdn", [stub], {})])
        self.assertEqual(ing.sync_games(date(2025, 1, 1), date(2025, 1, 31)), 1)
        self.assertEqual(wh.counts()["games"], 1)


class OddsParsing(unittest.TestCase):
    def setUp(self):
        self.ev = fx.odds_api_event("e1", "Boston Celtics", "Los Angeles Lakers", "2030-01-16T00:30:00Z", {
            "draftkings": {"spread": (-5.5, -110, -110), "total": (224.5, -108, -112), "ml": (-230, 190)},
            "fanduel": {"spread": (-5.5, -112, -108), "total": (224.5, -110, -110), "ml": (-225, 185)},
            "betmgm": {"spread": (-6.0, -110, -110), "total": (225.0, -110, -110), "ml": (-240, 195)},
        })

    def test_game_odds_normalisation(self):
        rows = TheOddsAPI.parse_game_odds([self.ev], "2030-01-15T12:00:00Z")
        dk = next(r for r in rows if r.book == "draftkings")
        self.assertEqual((dk.home_team_id, dk.away_team_id, dk.game_date), (BOS.id, LAL.id, "2030-01-15"))
        self.assertEqual((dk.spread_home, dk.spread_home_price, dk.spread_away_price), (-5.5, -110, -110))
        self.assertEqual((dk.total, dk.over_price, dk.under_price), (224.5, -108, -112))
        self.assertEqual((dk.ml_home, dk.ml_away), (-230, 190))

    def test_unknown_teams_skipped(self):
        bad = dict(self.ev, home_team="Harlem Globetrotters")
        self.assertEqual(TheOddsAPI.parse_game_odds([bad]), [])

    def test_props_normalisation_and_name_folding(self):
        ev = fx.odds_api_props("e1", "Boston Celtics", "Los Angeles Lakers", "2030-01-16T00:30:00Z", {
            "draftkings": [("Luka Dončić", 28.5, -115, -105), ("Jayson Tatum", 27.5, -110, -110)],
            "fanduel": [("Luka Doncic", 28.5, -110, -110)]})
        rows = TheOddsAPI.parse_props(ev, "2030-01-15T12:00:00Z")
        luka = [r for r in rows if r.player_key == "luka doncic"]
        self.assertEqual(len(luka), 2)                           # accent-folded key matches across books
        self.assertEqual({r.book for r in luka}, {"draftkings", "fanduel"})
        dk = next(r for r in luka if r.book == "draftkings")
        self.assertEqual((dk.line, dk.over_price, dk.under_price), (28.5, -115, -105))

    def test_missing_key_gives_actionable_error(self):
        client = TheOddsAPI(api_key=None)
        client.api_key = None
        with self.assertRaises(OddsAPIError) as cm:
            client.fetch_game_odds()
        self.assertIn("ODDS_API_KEY", str(cm.exception))


class OddsServiceFlow(unittest.TestCase):
    def setUp(self):
        self.wh = Warehouse(":memory:")
        class StubClient:
            configured = True
            credits_remaining, credits_used = 480, 20
            calls = 0
            def pull(inner, include_props=True):
                inner.calls += 1
                ev = fx.odds_api_event("e1", "Boston Celtics", "Los Angeles Lakers", "2030-01-16T00:30:00Z", {
                    "draftkings": {"spread": (-5.5, -110, -110), "total": (224.5, -108, -112), "ml": (-230, 190)},
                    "fanduel": {"spread": (-5.5, -112, -108), "total": (224.5, -110, -110), "ml": (-225, 185)},
                    "betmgm": {"spread": (-6.0, -110, -110), "total": (225.0, -110, -110), "ml": (-240, 195)}})
                pev = fx.odds_api_props("e1", "Boston Celtics", "Los Angeles Lakers", "2030-01-16T00:30:00Z", {
                    "draftkings": [("Jayson Tatum", 27.5, -115, -105)], "fanduel": [("Jayson Tatum", 27.5, -110, -110)],
                    "betmgm": [("Jayson Tatum", 28.5, -110, -110)]})
                return TheOddsAPI.parse_game_odds([ev], "2030-01-15T12:00:00Z"), TheOddsAPI.parse_props(pev, "2030-01-15T12:00:00Z"), ["ok"]
        self.client = StubClient()
        self.svc = OddsService(self.wh, client=self.client)

    def test_refresh_store_and_lookup(self):
        rep = self.svc.refresh(force=True)
        self.assertTrue(rep.ok)
        self.assertEqual((rep.games, rep.props, rep.credits_remaining), (3, 3, 480))
        m = self.svc.get_game_market(BOS.id, LAL.id, "2030-01-15")
        self.assertTrue(m["available"])
        self.assertEqual((m["book"], m["home_spread"], m["away_spread"], m["total"]), ("draftkings", -5.5, 5.5, 224.5))
        self.assertEqual(m["n_books"], 3)
        self.assertEqual(m["consensus_spread"], -5.5)            # median of -5.5, -5.5, -6.0
        # fair cover prob uses only books quoting the same -5.5 line (dk + fd), de-vigged
        self.assertEqual(m["fair"]["n_spread"], 2)
        self.assertTrue(0.49 < m["fair"]["home_cover"] < 0.52)
        self.assertGreater(m["fair"]["home_win"], 0.65)
        self.assertEqual(self.wh.get_meta("odds_credits_remaining"), "480")

    def test_refresh_throttled_to_protect_credits(self):
        self.svc.refresh(force=True)
        rep = self.svc.refresh()
        self.assertTrue(rep.skipped)
        self.assertEqual(self.client.calls, 1)

    def test_props_lookup_keeps_dk_line_and_flags_other_books(self):
        self.svc.refresh(force=True)
        props = self.svc.get_props(BOS.id, LAL.id, "2030-01-15")
        self.assertEqual(len(props), 1)
        p = props[0]
        self.assertEqual((p["player_key"], p["line"], p["over_odds"], p["under_odds"], p["book"]), ("jayson tatum", 27.5, -115, -105, "draftkings"))
        self.assertEqual(p["n_books"], 3)
        self.assertEqual(p["consensus_line"], 27.5)
        self.assertEqual(len(p["others"]), 3)
        self.assertTrue(0.47 < p["fair_over"] < 0.53)

    def test_no_key_falls_back_to_espn_lines_with_clear_message(self):
        class NoKey:
            configured = False
        class FakeESPN:
            def fetch_scoreboard_odds(inner, d):
                from nbadata.timeutil import today_eastern
                if d != today_eastern():
                    return []
                return [{"game_date": d.isoformat(), "home_team_id": BOS.id, "away_team_id": LAL.id, "commence_utc": None,
                         "book": "draftkings", "spread_home": -4.5, "spread_home_price": None, "total": 220.5, "ml_home": -200, "ml_away": 170}]
        svc = OddsService(Warehouse(":memory:"), client=NoKey(), espn=FakeESPN())
        rep = svc.refresh(force=True)
        self.assertTrue(rep.ok)
        self.assertEqual(rep.source, "espn")
        self.assertTrue(any("ODDS_API_KEY" in m for m in rep.messages))
        m = svc.get_game_market(BOS.id, LAL.id)
        self.assertEqual((m["home_spread"], m["home_spread_odds"]), (-4.5, -110))   # missing juice defaults to -110
        self.assertTrue(m["prices_assumed"])


class CsvImport(unittest.TestCase):
    def test_legacy_props_csv_and_canonical_games_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "props.csv")
            with open(p, "w") as fh:
                fh.write("Game,Player,Points Line,Over Odds,Under Odds\nCHA Hornets at SA Spurs,Ryan Kalkbrenner,6,-107,-110\nbad row,x,1,1,1\n")
            props = import_props(p)
            self.assertEqual(len(props), 1)
            self.assertEqual((props[0].player_key, props[0].line, props[0].over_price, props[0].home_team_id),
                             ("ryan kalkbrenner", 6.0, -107, resolve_team("SAS").id))
            g = os.path.join(d, "g.csv")
            with open(g, "w", encoding="utf-8") as fh:
                fh.write("game_date,away_team,home_team,spread_home,total,ml_home,ml_away\n2030-01-15,Los Angeles Lakers,Boston Celtics,-5.5,224.5,\u2212230,+190\n")
            rows = import_game_lines(g)
            self.assertEqual((rows[0].spread_home, rows[0].ml_home, rows[0].ml_away), (-5.5, -230, 190))


class Devig(unittest.TestCase):
    def test_devig_sums_to_one(self):
        a, b = devig_two_way(-110, -110)
        self.assertAlmostEqual(a, 0.5)
        a, b = devig_two_way(-230, 190)
        self.assertAlmostEqual(a + b, 1.0)
        self.assertGreater(american_to_prob(-230), american_to_prob(-110))
        self.assertIsNone(devig_two_way(None, -110))


if __name__ == "__main__":
    unittest.main()
