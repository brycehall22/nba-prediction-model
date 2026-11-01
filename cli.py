#!/usr/bin/env python
"""
NBA prediction model command line.

  python cli.py setup                  first-time setup in one go: check sources, load history, train, pull odds
  python cli.py doctor                 check every data source + your odds key
  python cli.py backfill               one-time historical load (resumable; safe to Ctrl+C and re-run)
  python cli.py update                 daily: new results, schedule, injuries
  python cli.py odds [--force]         pull DraftKings (+ other books) lines and player props
  python cli.py train                  fit team + player models and print validation tables
  python cli.py slate [--date D]       today's games: projections, market, best bets
  python cli.py predict HOME AWAY      one matchup (team names or abbreviations)
  python cli.py grade                  grade logged bets against final box scores
  python cli.py report [--kind K] [--since D] [--all]   ROI, calibration, closing-line value, bet list
  python cli.py status                 what's in the warehouse
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from config import BETTING, DATA
from nbadata.ingest import Ingestor, default_providers
from nbadata.teams import resolve_team, team_abbr
from nbadata.timeutil import today_eastern
from nbadata.warehouse import Warehouse


W = 84


def _f(x, nd=1, default="-"):
    return default if x is None or x != x else f"{x:.{nd}f}"


def _sign(x, nd=1, default="-"):
    return default if x is None or x != x else f"{x:+.{nd}f}"


def _pct(x, nd=1, default="-"):
    return default if x is None or x != x else f"{100 * x:.{nd}f}"


def section(title: str):
    print(f"\n-- {title} " + "-" * max(4, W - 4 - len(title)))


def table(headers, rows, aligns=None, indent="  "):
    """Plain ASCII table. aligns: string of 'l'/'r' per column."""
    if not rows:
        return
    aligns = aligns or "l" * len(headers)
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
    fmt = lambda row: indent + "  ".join((c.ljust(w) if a == "l" else c.rjust(w)) for c, w, a in zip(row, widths, aligns))
    print(fmt(headers))
    print(indent + "  ".join("-" * w for w in widths))
    for r in cells:
        print(fmt(r))


def _ago(minutes):
    if minutes is None:
        return "never"
    return f"{minutes:.0f} min ago" if minutes < 120 else (f"{minutes / 60:.1f} h ago" if minutes < 2880 else f"{minutes / 1440:.1f} days ago")


def _check_providers(verbose: bool = True):
    """Run each provider's health check. Returns (names_ok, names_failed)."""
    results = []
    for prov in default_providers():
        results.append((prov.name, prov.health_check()))
    ok = [n for n, r in results if r["ok"]]
    bad = [n for n, r in results if not r["ok"]]
    if verbose:
        for name, res in results:
            if res["ok"]:
                label = "OK  "
            else:
                label = "WARN" if ok else "FAIL"      # a down fallback is only a warning when another source works
            print(f"[{label}] {name}" + (f"  ({res['note']})" if res.get("note") else ""))
            for st in res["steps"]:
                print(f"        {'ok' if st['ok'] else 'XX'}  {st['step']}: {st['detail']}")
    return ok, bad


NO_SOURCE_HELP = """No game-data source is reachable from this machine, so nothing can be loaded.
  Common causes: a VPN/proxy or corporate network blocking the sites, a firewall, or no internet.
  Try: disconnect any VPN, run from a different network, then re-run `python cli.py doctor`.
  If only the NBA CDN shows 403 but ESPN works, that is fine: ESPN is the primary source."""


def next_step(wh: Warehouse) -> str:
    from modeling.game_model import GameModel
    c = wh.counts()
    if c["games_with_box"] == 0:
        return "python cli.py backfill        (loads historical box scores; nothing else works until this has data)"
    gm = GameModel.load()
    if gm is None:
        return "python cli.py train           (fits the models on the loaded history)"
    if c["last_date"] and gm.trained_through and (date.fromisoformat(c["last_date"]) - date.fromisoformat(gm.trained_through)).days > 14:
        return f"python cli.py train           (models are older than the data: trained through {gm.trained_through}, data through {c['last_date']})"
    if not wh.get_meta("odds_last_refresh"):
        return "python cli.py odds            (pulls lines; needs ODDS_API_KEY for DraftKings and props)"
    return "python cli.py slate           (today's projections and flagged bets)"


def _season_table(wh: Warehouse):
    sc = wh.season_counts()
    rows = [[r["season"], int(r["games"]), int(r["final"]), int(r["with_box"]), int(r["final"] - r["with_box"]), r["first_date"], r["last_box"] or "-"]
            for _, r in sc.iterrows()]
    table(["season", "games", "final", "box scores", "missing", "first game", "last box score"], rows, "lrrrrll")


def cmd_doctor(args) -> int:
    from oddsfeed.the_odds_api import OddsAPIError, TheOddsAPI
    print("DOCTOR: checks every outside service the program depends on, using one real request each.")
    print("  OK = working   WARN = failing but not needed because another source works   FAIL = needed and not working")
    section("Game data sources (schedule, box scores, player history)")
    ok, bad = _check_providers(verbose=True)
    if ok:
        print(f"\n-> Game data: usable via {', '.join(ok)}" + (f" ({', '.join(bad)} is down; not fatal, it is only a fallback)" if bad else ""))
    else:
        print("\n-> " + NO_SOURCE_HELP)
    section("Injury report (ESPN)")
    try:
        from nbadata.providers.espn import ESPNProvider
        rows = ESPNProvider().fetch_injuries()
        outs = sum(1 for r in rows if str(r["status"]).lower().startswith(("out", "doubtful")))
        print(f"[OK  ] {len(rows)} injury entries ({outs} Out/Doubtful). Players ruled out get no prop bet and teammates' projections rise.")
    except Exception as e:
        print(f"[WARN] unavailable: {e}\n       Props still work; the teammates-out adjustment is skipped.")
    section("Odds (The Odds API)")
    client = TheOddsAPI()
    if not client.configured:
        print("[WARN] No ODDS_API_KEY set. Without it only ESPN's embedded game lines are available (no DraftKings props).")
        print("       Get a free key at https://the-odds-api.com, then put ODDS_API_KEY=your_key in a file named .env in this folder")
        print("       (or in this terminal:  set ODDS_API_KEY=your_key)")
    else:
        try:
            info = client.check_key()
            print(f"[OK  ] The Odds API key works: {info['events']} upcoming events listed, credits remaining: {info['credits_remaining']}")
            print("       (this check costs no credits; a full odds pull costs about 3 + 1 per game with props)")
        except OddsAPIError as e:
            print(f"[FAIL] {e}")
    wh = Warehouse(args.db)
    c = wh.counts()
    section(f"Local warehouse ({wh.path})")
    print(f"  {c['games_with_box']} games with box scores ({c['first_date']} .. {c['last_date']}), {c['player_rows']} player rows")
    if c["games"]:
        _season_table(wh)
    section("Next step")
    print(f"  {next_step(wh)}")
    return 0 if ok else 1


def _progress(i, total):
    if i == 1 or i % 25 == 0 or i == total:
        print(f"  box scores {i}/{total}", flush=True)


def cmd_backfill(args) -> int:
    wh = Warehouse(args.db)
    ing = Ingestor(wh)
    from config import recent_seasons
    seasons = recent_seasons(args.seasons) if args.seasons else None
    print(f"BACKFILL: loading schedules and box scores for {seasons or recent_seasons(DATA.BACKFILL_SEASONS)} into {wh.path}")
    print("  Only games not already stored are downloaded, so this is safe to interrupt (Ctrl+C) and re-run.")
    rep = ing.backfill(seasons, progress=_progress)
    section("Result")
    print(f"  New box scores saved : {rep.boxes_saved}")
    print(f"  Failed this run      : {rep.boxes_failed}")
    print(f"  Still missing        : {rep.boxes_remaining}   (finished games whose box score has not been downloaded yet)")
    for m in rep.messages[-8:]:
        print("  -", m)
    if rep.provider_errors:
        print("  Source errors:", rep.provider_errors)
    c = wh.counts()
    if c["games"] == 0:
        print("\nNo games were loaded at all.")
        if rep.empty_providers:
            print(f"  These sources answered but listed zero games: {', '.join(rep.empty_providers)}  (the response format may have changed)")
        if rep.provider_errors:
            print(f"  These sources failed outright: {', '.join(rep.provider_errors)}")
        print("  Run `python cli.py doctor` for the exact failing step. " + NO_SOURCE_HELP.split("\n")[1].strip())
        return 1
    section("Warehouse contents by season (preseason excluded)")
    _season_table(wh)
    if c["games_with_box"] == 0:
        print("\nThe schedule loaded but no box scores could be fetched. Run `python cli.py doctor`; re-running backfill resumes.")
        return 1
    print(f"\nWarehouse now holds {c['games_with_box']} games with box scores ({c['first_date']} .. {c['last_date']}), {c['player_rows']} player rows.")
    print(f"Next step:  {next_step(wh)}")
    return 0 if rep.boxes_remaining == 0 or rep.boxes_saved > 0 else 1


def cmd_update(args) -> int:
    wh = Warehouse(args.db)
    before = wh.counts()
    print("UPDATE: pulling results from the last few days, the upcoming schedule, and the injury report.")
    rep = Ingestor(wh).update(progress=_progress)
    after = wh.counts()
    section("Result")
    print(f"  Games synced (schedule + scores) : {rep.games_seen}")
    print(f"  New box scores saved             : {rep.boxes_saved}   ({before['games_with_box']} -> {after['games_with_box']} games with box scores)")
    print(f"  Box scores still missing         : {rep.boxes_remaining}")
    print(f"  Injury entries loaded            : {rep.injuries}")
    print(f"  Data is current through          : {after['last_date']}")
    upcoming = wh.games_df(start=today_eastern().isoformat(), status="scheduled")
    if not upcoming.empty:
        nxt = upcoming["game_date"].min()
        print(f"  Next scheduled games             : {nxt} ({int((upcoming['game_date'] == nxt).sum())} games; {len(upcoming)} on the schedule through {upcoming['game_date'].max()})")
    for m in rep.messages:
        print("  -", m)
    if rep.provider_errors:
        print("  Source errors:", rep.provider_errors, "(run `python cli.py doctor`)")
    return 0


def cmd_odds(args) -> int:
    from oddsfeed.csv_import import import_game_lines, import_props
    from oddsfeed.service import OddsService
    wh = Warehouse(args.db)
    svc = OddsService(wh)
    if args.import_games:
        rows = import_game_lines(args.import_games)
        svc.import_game_lines(rows)
        print(f"Imported {len(rows)} game-line rows from {args.import_games} (book column decides which book each row is stored under).")
    if args.import_props:
        rows = import_props(args.import_props)
        svc.import_props(rows)
        print(f"Imported {len(rows)} player-prop rows from {args.import_props}.")
    if args.import_games or args.import_props:
        return 0
    print("ODDS: pulling game lines (moneyline, spread, total) from all US books" + ("" if args.no_props else ", plus player-points props for games starting soon") + ".")
    rep = svc.refresh(force=args.force, include_props=not args.no_props)
    section("Result")
    print(f"  Status          : {'ok' if rep.ok else 'FAILED'}   source: {rep.source or '-'}" + ("   (skipped: pulled recently)" if rep.skipped else ""))
    print(f"  Book lines      : {rep.games}   (one row per game per sportsbook)")
    print(f"  Prop lines      : {rep.props}   (one row per player per sportsbook)")
    if rep.credits_remaining is not None:
        print(f"  API credits     : {rep.credits_remaining} remaining" + (f", {rep.credits_used} used this month" if rep.credits_used is not None else ""))
    for m in rep.messages:
        print("  -", m)
    board = svc.board()
    if board:
        section(f"Lines on file ({TARGET_BOOK_LABEL} preferred; spread is the HOME team's line, negative = home favored)")
        rows = [[b["game_date"], b["matchup"], b["book"], _sign(b["spread_home"]), _f(b["total"]), f"{b['ml_home']}/{b['ml_away']}" if b["ml_home"] is not None else "-",
                 b["n_books"], f"{b['props_target_book']}/{b['props_any_book']}"] for b in board]
        table(["date", "game", "book", "spread", "total", "moneyline h/a", "books", "props dk/any"], rows, "llllrrrr")
        print("  'props dk/any' = players with a points line at the target book / at any book.")
    elif rep.ok and not rep.skipped:
        print("  No upcoming NBA games have lines posted right now (normal in the off-season or before the next slate is released).")
    return 0 if rep.ok else 1


def _verdict(model, base, lower_is_better=True):
    if model is None or base is None or model != model or base != base or base == 0:
        return ""
    gain = (base - model) / base * 100 if lower_is_better else (model - base) / base * 100
    return f"{abs(gain):.1f}% {'lower' if gain > 0 else 'higher'} error than the baseline (" + ("better)" if gain > 0 else "worse)")


def cmd_train(args) -> int:
    from services import Engine, NotReady
    eng = Engine(Warehouse(args.db))
    print("TRAIN: fitting team ratings, the game model and the player-points model on everything in the warehouse.")
    try:
        rep = eng.train(progress=lambda m: print(f"  {m} ..."))
    except (NotReady, ValueError) as e:
        print(e); return 1
    g, p = rep["game"], rep["player"]
    print(f"\nTrained through {rep['trained_through']}. Every number below is OUT-OF-SAMPLE: each game was predicted using only earlier games.")
    section("GAME MODEL: predicts final margin (home minus away) and total points")
    print(f"  Scored on {g['n_games_scored']} games (early-season games, where ratings are unreliable, are skipped).")
    table(["", "this model", "ratings only", "old-style baseline*", "this model vs baseline"],
          [["margin error (MAE, pts)", _f(g["margin_mae"], 2), _f(g["margin_mae_ratings_only"], 2), _f(g["margin_mae_naive"], 2), _verdict(g["margin_mae"], g["margin_mae_naive"])],
           ["total error (MAE, pts)", _f(g["total_mae"], 2), _f(g["total_mae_ratings_only"], 2), _f(g["total_mae_naive"], 2), _verdict(g["total_mae"], g["total_mae_naive"])]], "lrrrl")
    print("  * baseline = each team's season scoring average + 3 points of home court (what the previous version of the app did).")
    print(f"  Typical miss (sigma): margin {_f(g['margin_sigma'], 1)} pts, total {_f(g['total_sigma'], 1)} pts. About 68% of games finish within one sigma of the projection;")
    print("  these sigmas are what turns a projection into a cover / over probability, so they matter as much as the projection.")
    print(f"  Home-court edge fitted from the data: {_f(g['hca_fitted'], 2)} efficiency points (roughly 2-3 points of margin).")
    print(f"  Home-win prediction: accuracy {_f(100 * g['home_win_accuracy'], 1)}%, log loss {_f(g['home_win_logloss'], 3)} vs {_f(g['home_win_logloss_baseline'], 3)} for always guessing the base rate (lower is better).")
    print(f"  Bias: margin {_sign(g['margin_bias'], 2)}, total {_sign(g['total_bias'], 2)} (should be near 0).")
    print(f"  Rest / back-to-back adjustment kept: margin={g['residual_model_margin']}, total={g['residual_model_total']} (only kept when it clearly lowers out-of-fold error).")
    section("PLAYER MODEL: predicts points for rotation players")
    print(f"  Scored on {p['n_scored']} out-of-fold player-games (trained on {p['n_rows']}).")
    table(["", "MAE (pts)", "this model vs it"],
          [["this model", _f(p["pts_mae"], 2), ""],
           ["recent-form baseline (EWMA)", _f(p["pts_mae_ewma_baseline"], 2), _verdict(p["pts_mae"], p["pts_mae_ewma_baseline"])],
           ["season-average baseline", _f(p["pts_mae_season_avg_baseline"], 2), _verdict(p["pts_mae"], p["pts_mae_season_avg_baseline"])]], "lrl")
    print(f"  Minutes error {_f(p['minutes_mae'], 2)} min.  Bias {_sign(p['pts_bias'], 2)} pts (near 0 is good).")
    print(f"  Spread of outcomes: negative-binomial dispersion r={_f(p['nb_r'], 1)}; the model's 80% interval contained the real score {_f(100 * p['coverage_80'], 1)}% of the time (target ~80%).")
    if p["by_tier"]:
        print("  By player type:")
        table(["tier", "player-games", "model MAE", "recent-form MAE"],
              [[t, v["n"], f"{v['mae']:.2f}", f"{v['mae_ewma']:.2f}"] for t, v in p["by_tier"].items()], "lrrr", indent="    ")
    section("How to read this")
    print("  MAE = average absolute miss in points; lower is better. A model that can't beat its baseline has no information.")
    print("  These numbers measure projection accuracy only. There are no free historical prop lines, so profit can't be measured yet:")
    print("  run `python cli.py slate` on game days (it logs each bet), then `update`, `grade`, `report` to measure real ROI and closing-line value.")
    print(f"\nNext step:  {next_step(Warehouse(args.db))}")
    return 0


def _bet_line(b):
    if b["kind"] == "prop":
        what = f"{b['player'][:22]:22} {b['side']:5} {b['line']:g}"
    else:
        ln = "" if b["line"] is None else (f"{b['line']:+g}" if b["kind"] == "spread" else f"{b['line']:g}")
        what = f"{b['kind']:9} {b['side']:5} {ln:>8}"
    return what


def _print_game(res, top: int, show_all: bool):
    if "error" in res:
        print(f"\n{res['away']} @ {res['home']}: ERROR {res['error']}"); return
    p, m = res["pred"], res["market"]
    h, a = team_abbr(res["home_id"]), team_abbr(res["away_id"])
    print(f"\n{'=' * W}\n{res['away']} @ {res['home']}  ({res['game_date']})")
    print(f"  Model projection : {h} {_f(p['home_pts'])} - {a} {_f(p['away_pts'])}   (margin {_sign(p['margin'])} for {h}, total {_f(p['total'])}, {h} win {100 * p['home_win_prob']:.0f}%)")
    if p.get("rest_home") is not None:
        print(f"  Rest days        : {h} {p['rest_home']:.0f}, {a} {p['rest_away']:.0f}   (typical miss: margin +/-{_f(p['margin_sigma'], 1)}, total +/-{_f(p['total_sigma'], 1)})")
    if m.get("available"):
        edge = (p["margin"] + m["home_spread"]) if m.get("home_spread") is not None else None
        print(f"  Market ({m['book']}, {m['n_books']} books): spread {h} {_sign(m['home_spread'])}, total {_f(m['total'])}, "
              f"moneyline {m['home_moneyline']}/{m['away_moneyline']}" + ("   [juice not provided; -110 assumed]" if m.get("prices_assumed") else ""))
        if edge is not None:
            print(f"  Model vs market : margin edge {_sign(edge)} pts toward {h if edge > 0 else a}; total edge {_sign(p['total'] - m['total'])} pts ({'over' if p['total'] > m['total'] else 'under'})")
    else:
        print("  Market           : no lines available (run `python cli.py odds`)")
    shown = False
    bets = sorted(res["game_bets"], key=lambda b: -b["ev_pct"])
    pick = bets if show_all else [b for b in bets if b["label"] != "none"][:top]
    props = sorted(res["prop_bets"], key=lambda b: -b["ev_pct"])
    pickp = props if show_all else [b for b in props if b["label"] != "none"][:top]
    if pick or pickp:
        print(f"  {'bet':41} {'odds':>6} {'EV%':>7} {'p(win)':>7} {'model':>6} {'kelly%':>7}")
    for b in pick:
        print(f"  * {_bet_line(b):39} {b['price']:>+6d} {b['ev_pct']:>+7.1f} {100 * b['blended_prob']:>6.1f}% {100 * b['model_prob']:>5.1f}% {b['kelly_pct']:>7.1f}" + ("  !" if b["warning"] else ""))
        shown = True
    for b in pickp:
        print(f"  * {_bet_line(b):39} {b['price']:>+6d} {b['ev_pct']:>+7.1f} {100 * b['blended_prob']:>6.1f}% {100 * b['model_prob']:>5.1f}% {b['kelly_pct']:>7.1f}  proj {b['projection']:.1f} in {b['proj_min']:.0f}m"
              + ("  !" if b["warning"] else ""))
        shown = True
    if not shown and m.get("available"):
        print(f"  No bets at or above {BETTING.EV_THRESHOLD_PCT:g}% EV for this game (use --all to see every evaluated bet).")


def cmd_slate(args) -> int:
    from services import Engine, NotReady
    eng = Engine(Warehouse(args.db))
    d = date.fromisoformat(args.date) if args.date else today_eastern()
    try:
        results = eng.slate(d, with_props=not args.no_props, log=not args.no_log)
    except NotReady as e:
        print(e); return 1
    if not results:
        print(f"No games found for {d}.")
        upcoming = eng.wh.games_df(start=d.isoformat(), status="scheduled")
        if upcoming.empty:
            print("The warehouse has no upcoming schedule. Run `python cli.py update` (and `python cli.py odds` for lines).")
        else:
            nxt = upcoming["game_date"].min()
            day = upcoming[upcoming["game_date"] == nxt]
            kinds = {1: "preseason", 3: "playoffs", 5: "play-in"}
            tag = kinds.get(day["season_type"].dropna().astype(int).min() if day["season_type"].notna().any() else 2, "")
            first = day.iloc[0]
            print(f"Next scheduled games: {nxt} ({len(day)} game{'s' if len(day) != 1 else ''}{', ' + tag if tag else ''}), "
                  f"e.g. {team_abbr(int(first['away_team_id']))} @ {team_abbr(int(first['home_team_id']))}.  Try: python cli.py slate --date {nxt}")
        return 0
    ok = [r for r in results if "error" not in r]
    with_mkt = [r for r in ok if r["market"].get("available")]
    n_game_bets = sum(len(r["game_bets"]) for r in ok)
    n_props = sum(len(r["prop_bets"]) for r in ok) // 2
    flagged = sum(1 for r in ok for b in r["game_bets"] + r["prop_bets"] if b["label"] != "none")
    print(f"SLATE for {d}: {len(results)} games, {len(with_mkt)} with market lines. Evaluated {n_game_bets} game bets and {n_props} player-prop lines; "
          f"{flagged} bet sides clear the {BETTING.EV_THRESHOLD_PCT:g}% EV bar.")
    if not args.no_log:
        print("(Every evaluated bet is logged at the price shown, so `grade` and `report` can measure how the model really does.)")
    for r in results:
        _print_game(r, args.top, args.all)
    section("Legend")
    print("  EV%     expected profit per $1 staked, after blending the model with the de-vigged market price")
    print(f"          (weights: games {BETTING.GAME_MARKET_WEIGHT:.0%} market, props {BETTING.PROP_MARKET_WEIGHT:.0%} market; see config.py)")
    print("  p(win)  blended win probability used for EV;  model = the model's own probability before blending")
    print(f"  kelly%  suggested stake as % of bankroll ({BETTING.KELLY_FRACTION:g}x Kelly, capped at {BETTING.MAX_KELLY_BET_PCT:g}%): an upper bound, not advice")
    print("  !       model and market disagree strongly. This usually means the book knows something (injury, lineup); check before trusting it")
    return 0


def cmd_predict(args) -> int:
    from services import Engine, NotReady
    h, a = resolve_team(args.home), resolve_team(args.away)
    if not h or not a:
        print(f"Could not resolve team: {args.home if not h else args.away}"); return 1
    eng = Engine(Warehouse(args.db))
    try:
        res = eng.predict_game(h.id, a.id, date.fromisoformat(args.date) if args.date else today_eastern())
    except NotReady as e:
        print(e); return 1
    _print_game(res, args.top, args.all)
    section("Projected points by player (sorted by projected minutes)")
    for side in ("home", "away"):
        print(f"  {res[side]}")
        rows = []
        for p in res[f"{side}_players"][:10]:
            tag = "OUT" if p["out"] else (p["status"] or "")
            rows.append([p["player"], _f(p["proj_pts"]), _f(p["proj_min"]), _f(p["last5_pts"]), _f(p["ewm_pts"]), tag])
        table(["player", "proj pts", "proj min", "last 5 avg", "recent form", "status"], rows, "lrrrrl", indent="    ")
    print("  'recent form' = exponentially weighted points average (recent games count more). 'OUT' players are excluded from props.")
    return 0


def cmd_grade(args) -> int:
    from nbadata.timeutil import iso_utc, now_utc
    from predictions import PredictionLog
    log = PredictionLog(Warehouse(args.db))
    started = iso_utc(now_utc())
    print("GRADE: comparing logged bets with final scores and box scores in the warehouse.")
    n = log.grade()
    d = log.grade_detail(started)
    section("Result")
    print(f"  Graded this run : {n}   (wins {d['done'].get('win', 0)}, losses {d['done'].get('loss', 0)}, pushes {d['done'].get('push', 0)}, voided {d['done'].get('void', 0)})")
    print("  A 'void' is a prop where the player did not play; it is refunded and excluded from ROI.")
    pend = d["pending"]
    waiting = sum(x["n"] for x in pend)
    print(f"  Still pending   : {waiting}")
    if pend:
        table(["game date", "bets", "why still pending"], [[x["date"], x["n"], "game final but not graded" if x["final"] else "game not final / box score not loaded yet"] for x in pend[:12]], "lrl", indent="    ")
        print("  Pending bets are normal for games not yet played. For finished games run `python cli.py update` first, then grade again.")
    print("\nNext step:  python cli.py report")
    return 0


def cmd_report(args) -> int:
    from predictions import PredictionLog
    log = PredictionLog(Warehouse(args.db))
    rep = log.detailed_report(min_ev=args.min_ev, since=args.since, kind=args.kind)
    print("=" * W)
    print("BET PERFORMANCE REPORT" + (f"   (market: {args.kind})" if args.kind else "") + (f"   (games since {args.since})" if args.since else ""))
    print("=" * W)
    if rep["logged"] == 0:
        print("No bets are logged yet. Run `python cli.py slate` on a game day (it logs every bet it evaluates), then `update`, `grade`, `report`.")
        return 0
    print(f"  Logged bets : {rep['logged']}   covering games {rep['first_game']} .. {rep['last_game']}")
    print(f"  Graded      : {rep['graded']}      Pending: {rep['pending']} (game not played yet or not loaded: run `update`, then `grade`)      Voided: {rep['void']}")
    print("  How to read: each bet is staked at 1 unit at the price shown when first logged. ROI = units won / bets placed.")
    print("  'claimed EV' is what the model promised when it logged the bet; ROI is what actually happened. 'expected win%' is the model's average probability.")
    if rep["graded"] == 0:
        print("\nNothing graded yet. Run `python cli.py update` then `python cli.py grade` after the games finish.")
        return 0
    cols = ["bets", "W-L-P", "win%", "expected win%", "units", "ROI%", "+/-95%", "claimed EV%"]
    al = "lrrrrrrrr"

    def row(name, v):
        return [name, v["bets"], f"{v['wins']}-{v['losses']}-{v['pushes']}", _pct(v["win_rate"]), _pct(v["expected_win_rate"]),
                _sign(v["units"], 2), _sign(v["roi_pct"]), (f"{1.96 * v['roi_se_pct']:.0f}" if v["roi_se_pct"] is not None and v["bets"] >= 10 else "n/a"), _sign(v["avg_ev_pct"])]
    o = rep["overall"]
    section("Overall")
    table(["", *cols], [row("all graded bets", o)], al)
    print("  Note: this row includes BOTH sides of every line the program evaluated, so even a model with no edge shows a slightly negative ROI")
    print("  (roughly the sportsbook's built-in margin, about -4% to -5% at -110). The flagged section below is the one that measures the strategy.")
    if rep.get("flagged"):
        section(f"Flagged value bets only (claimed EV >= {rep['min_ev']:g}%): the bets the program would actually recommend")
        table(["", *cols], [row("flagged", rep["flagged"])], al)
    else:
        print(f"\n  No graded bet had claimed EV >= {rep['min_ev']:g}% yet.")
    section("By market")
    table(["market", *cols], [row(k, v) for k, v in rep["by_kind"].items()], al)
    section("By side (is one direction doing better than the other?)")
    table(["bet type", *cols], [row(k, v) for k, v in rep["by_side"].items()], al)
    section("By claimed-EV bucket (a working model should do better as claimed EV rises)")
    table(["claimed EV", *cols], [row(k, v) for k, v in rep["by_ev_bucket"].items()], al)

    section("Closing-line value (did the market move toward our side after we bet?)")
    only_one_snapshot = "price_clv_n" in o and abs(o["price_clv_pts"]) < 1e-9 and o.get("line_unchanged_pct", 100) >= 99.9
    if only_one_snapshot:
        print("  Not informative yet: the latest odds on file are the same ones the bets were logged from, so nothing can have moved.")
        print("  To measure it, pull odds again close to tip-off (`python cli.py odds --force`) BEFORE running `grade`.")
    elif "price_clv_n" in o:
        verdict = "beating the close" if o["price_clv_pts"] > 0 else "behind the close"
        print(f"  Price CLV: {_sign(o['price_clv_pts'], 2)} implied-probability points over {o['price_clv_n']} bets  ({verdict}).")
        print("  Positive means the price we got was better than the last price we saw; beating the close consistently is the best early sign of a real edge.")
    else:
        print("  Not available yet: closing prices come from later odds pulls. Run `python cli.py odds` shortly before games start so a late snapshot exists.")
    if "line_n" in o and not only_one_snapshot:
        print(f"  Line movement: the number moved in our favor on {o['line_beat_pct']:.0f}% of {o['line_n']} spread/total/prop bets ({o['line_unchanged_pct']:.0f}% did not move).")

    section("How much can you trust this?")
    n, se = o["bets"], o["roi_se_pct"]
    if n < 30:
        print(f"  Only {n} graded bets: far too few to conclude anything. A handful of results is mostly luck; the +/-95% column shows n/a below 10 bets.")
    if se is not None:
        lo, hi = o["roi_pct"] - 1.96 * se, o["roi_pct"] + 1.96 * se
        needed = int((19.6 * max(o["roi_se_pct"] * sqrt_n(n) / 100, 0.9)) ** 2)
        print(f"  With {n} graded bets the 95% range for true ROI is roughly {lo:+.0f}% to {hi:+.0f}% (observed {o['roi_pct']:+.1f}%).")
        print(f"  Narrowing that to about +/-10% takes roughly {needed} bets at these odds. Until then, treat ROI as noise and watch calibration and CLV instead.")
    if rep.get("flagged"):
        fl = rep["flagged"]
        if fl["bets"] < 30:
            print(f"  Flagged bets: only {fl['bets']}. A flagged ROI of {fl['roi_pct']:+.0f}% on that few bets says almost nothing either way.")
    c = rep.get("calibration")
    section("Calibration (is the model better than the market's own price?)")
    if c:
        print(f"  Log loss (lower is better): model {c['logloss_model']:.4f}, market {c['logloss_market']:.4f}; best blend puts {c['best_market_weight']:.0%} weight on the market ({c['n']} bets).")
        print("  " + ("Enough bets to consider changing GAME_MARKET_WEIGHT / PROP_MARKET_WEIGHT in config.py." if c["enough_data"]
                      else "Fewer than 300 bets: too noisy to change the market-blend settings yet."))
    else:
        print("  Needs at least 30 graded bets that have a market probability (not shown with --kind/--since filters).")

    recent = rep["recent"] if args.all else rep["recent"][:15]
    section(f"Graded bets ({'all' if args.all else 'most recent 15; add --all for every bet'}, {len(rep['recent'])} total)")
    rows = []
    for r in recent:
        away_home = r["game_key"].split("_")[1:3]
        pn = r.get("player_name")
        who = (pn + " " if isinstance(pn, str) and pn else "") + r["side"]
        line = "" if r["line"] is None or r["line"] != r["line"] else (f"{r['line']:+g}" if r["kind"] == "spread" else f"{r['line']:g}")
        act = "" if r["actual"] is None or r["actual"] != r["actual"] else f"{r['actual']:g}"
        rows.append([r["game_date"], f"{away_home[0]}@{away_home[1]}", r["kind"], f"{who} {line}".strip(), f"{r['price']:+d}", _pct(r["blended_prob"], 0), _sign(r["ev_pct"]), r["result"].upper(), act])
    table(["game date", "game", "market", "bet", "price", "p(win)%", "EV%", "result", "actual"], rows, "lllrrrrll")
    print("  'actual' = the margin for that side (spread/moneyline), the game total, or the player's points.")
    return 0


def sqrt_n(n):
    return n ** 0.5


def cmd_status(args) -> int:
    from services import Engine
    wh = Warehouse(args.db)
    eng = Engine(wh)
    s = eng.status()
    print("STATUS")
    section("Game data")
    print(f"  {s['games_with_box']} games with box scores ({s['first_date']} .. {s['last_date']}), {s['player_rows']} player rows, {s['games']} games on file incl. schedule")
    lu = wh.get_meta("last_update")
    print(f"  Last `update` run: {lu or 'never'}")
    if s["games"]:
        _season_table(wh)
    section("Models")
    gm, pm = eng.game_model, eng.player_model
    if gm is None:
        print("  Not trained yet (python cli.py train).")
    else:
        print(f"  Game model trained through {gm.trained_through}; player model through {pm.trained_through if pm else 'n/a'}.")
        if s["last_date"] and gm.trained_through:
            lag = (date.fromisoformat(s["last_date"]) - date.fromisoformat(gm.trained_through)).days
            print(f"  Data is {lag} days newer than the models." + ("  Consider retraining (ratings still refresh daily; trained parts are fitted error sizes and the player model)." if lag > 14 else ""))
    section("Odds")
    o = s["odds"]
    print(f"  Source: {o['source'] or 'none'}   last pulled: {_ago(o['last_refresh_minutes'])}   API key: {'set' if o['api_key_configured'] else 'NOT set'}"
          + (f"   credits remaining: {o['credits_remaining']}" if o["credits_remaining"] else ""))
    print(f"  Odds snapshots stored: {s['odds_snapshots']}  (each `odds` run adds one; they are what closing-line value is computed from)")
    section("Bet log")
    n = wh.query_df("SELECT COUNT(*) n, SUM(result IS NULL) pending, SUM(result IN ('win','loss','push')) graded FROM predictions").iloc[0]
    print(f"  Logged {int(n['n'])} bets: {int(n['graded'] or 0)} graded, {int(n['pending'] or 0)} pending. (`grade` then `report`)")
    print(f"\nNext step:  {next_step(wh)}")
    return 0


def cmd_setup(args) -> int:
    """Everything needed for a first run, stopping with a clear message at the first thing that is not working."""
    import argparse as _ap
    print("Step 1/5  Checking data sources")
    ok, bad = _check_providers(verbose=False)
    if not ok:
        print("  " + NO_SOURCE_HELP.replace("\n", "\n  "))
        print("  (details: python cli.py doctor)")
        return 1
    print(f"  usable via {', '.join(ok)}" + (f"; {', '.join(bad)} unavailable (fallback only)" if bad else ""))
    print("\nStep 2/5  Loading historical box scores (this is the slow part; Ctrl+C is safe, re-run resumes)")
    if cmd_backfill(args) != 0 and Warehouse(args.db).counts()["games_with_box"] == 0:
        return 1
    print("\nStep 3/5  Pulling recent results, schedule and injuries")
    cmd_update(args)
    print("\nStep 4/5  Training models")
    if cmd_train(args) != 0:
        return 1
    print("\nStep 5/5  Pulling odds")
    from oddsfeed.the_odds_api import TheOddsAPI
    if TheOddsAPI().configured:
        cmd_odds(_ap.Namespace(db=args.db, force=True, no_props=False, import_games=None, import_props=None))
    else:
        print("  Skipped: no ODDS_API_KEY. Add it to a .env file (see .env.example) and run `python cli.py odds`.")
    print(f"\nSetup finished. Next step:  {next_step(Warehouse(args.db))}")
    return 0


TARGET_BOOK_LABEL = "DraftKings"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NBA prediction model", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--db", default=None, help=f"warehouse path (default {DATA.DB_PATH})")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    p = sub.add_parser("setup"); p.add_argument("--seasons", type=int, help="how many seasons back (default 3)"); p.set_defaults(fn=cmd_setup)
    p = sub.add_parser("backfill"); p.add_argument("--seasons", type=int, help="how many seasons back"); p.set_defaults(fn=cmd_backfill)
    sub.add_parser("update").set_defaults(fn=cmd_update)
    p = sub.add_parser("odds"); p.add_argument("--force", action="store_true"); p.add_argument("--no-props", action="store_true")
    p.add_argument("--import-games", metavar="CSV"); p.add_argument("--import-props", metavar="CSV"); p.set_defaults(fn=cmd_odds)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    for name, fn in (("slate", cmd_slate), ("predict", cmd_predict)):
        p = sub.add_parser(name)
        if name == "predict":
            p.add_argument("home"); p.add_argument("away")
        p.add_argument("--date"); p.add_argument("--top", type=int, default=6); p.add_argument("--all", action="store_true")
        p.add_argument("--no-props", action="store_true"); p.add_argument("--no-log", action="store_true"); p.set_defaults(fn=fn)
    sub.add_parser("grade").set_defaults(fn=cmd_grade)
    p = sub.add_parser("report", help="ROI, calibration and closing-line value from graded bets")
    p.add_argument("--since", metavar="YYYY-MM-DD", help="only games on/after this date")
    p.add_argument("--kind", choices=["spread", "total", "moneyline", "prop"], help="only one market")
    p.add_argument("--min-ev", type=float, default=None, help="claimed-EV%% threshold for 'flagged' bets (default from config)")
    p.add_argument("--all", action="store_true", help="list every graded bet, not just the latest 15")
    p.set_defaults(fn=cmd_report)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if not args.verbose:
        logging.getLogger("urllib3").setLevel(logging.ERROR)     # retry chatter; our own errors are reported cleanly
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; re-run to resume.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
