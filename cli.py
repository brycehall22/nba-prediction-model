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
  python cli.py backtest               replay history as if live, find corrections, keep only the ones verified on later games
  python cli.py backtest --week        replay one week day by day: baseline model vs corrected model (--ending DATE)
  python cli.py grade                  grade logged bets against final box scores
  python cli.py report [--kind K] [--since D] [--all] [--apply-weights]   picks' ROI, lean, calibration, closing-line value
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
        print(f"  * {_bet_line(b):39} {b['price']:>+6d} {b['ev_pct']:>+7.1f} {100 * b['blended_prob']:>6.1f}% {100 * b['model_prob']:>5.1f}% {b['kelly_pct']:>7.1f}" + ("  PICK" if b.get("is_pick") and show_all else "") + ("  !" if b["warning"] else ""))
        shown = True
    for b in pickp:
        print(f"  * {_bet_line(b):39} {b['price']:>+6d} {b['ev_pct']:>+7.1f} {100 * b['blended_prob']:>6.1f}% {100 * b['model_prob']:>5.1f}% {b['kelly_pct']:>7.1f}  proj {b['projection']:.1f} in {b['proj_min']:.0f}m"
              + ("  PICK" if b.get("is_pick") and show_all else "") + ("  !" if b["warning"] else ""))
        shown = True
    if not shown and m.get("available"):
        print(f"  No picks at or above {BETTING.EV_THRESHOLD_PCT:g}% EV for this game (use --all to see both sides of every line; PICK marks the model's choice).")


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
    flagged = sum(1 for r in ok for b in r["game_bets"] + r["prop_bets"] if b["label"] != "none" and b.get("is_pick"))
    print(f"SLATE for {d}: {len(results)} games, {len(with_mkt)} with market lines. Evaluated {n_game_bets // 2} game lines and {n_props} player-prop lines; "
          f"{flagged} picks clear the {BETTING.EV_THRESHOLD_PCT:g}% EV bar.")
    if not args.no_log:
        print("(For every line the model takes one side, its PICK, and logs it at the price shown, so `grade` and `report` can measure how it really does.)")
    for r in results:
        _print_game(r, args.top, args.all)
    section("Legend")
    print("  EV%     expected profit per $1 staked, after blending the model with the de-vigged market price")
    print(f"          (weights: games {BETTING.GAME_MARKET_WEIGHT:.0%} market, props {BETTING.PROP_MARKET_WEIGHT:.0%} market; see config.py)")
    print("  p(win)  blended win probability used for EV;  model = the model's own probability before blending")
    print(f"  kelly%  suggested stake as % of bankroll ({BETTING.KELLY_FRACTION:g}x Kelly, capped at {BETTING.MAX_KELLY_BET_PCT:g}%): an upper bound, not advice")
    print("  !       model and market disagree strongly. This usually means the book knows something (injury, lineup); check before trusting it")
    print(f"  Note    moneyline/spread sides the model itself gives <{BETTING.MIN_MODEL_PROB_TO_FLAG:.0%} are never flagged,")
    print("          even if EV is positive (stops longshot dog bets the model still thinks will lose). Use --all to see them.")
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
    from modeling.tuning import Tuning
    from config import MODEL
    log = PredictionLog(Warehouse(args.db))
    picks_only = not args.both_sides
    rep = log.detailed_report(min_ev=args.min_ev, since=args.since, kind=args.kind, picks_only=picks_only)
    print("=" * W)
    print("BET PERFORMANCE REPORT" + (" (every side of every line)" if not picks_only else " (model picks only)")
          + (f"   market: {args.kind}" if args.kind else "") + (f"   games since {args.since}" if args.since else ""))
    print("=" * W)
    if rep["rows_logged"] == 0:
        print("No bets are logged yet. Run `python cli.py slate` on a game day (it logs every bet it evaluates), then `update`, `grade`, `report`.")
        return 0
    if picks_only:
        print(f"  Logged picks: {rep['logged']} lines   covering games {rep['first_game']} .. {rep['last_game']}   "
              f"({rep['rows_logged']} rows stored including the opposite sides, which are kept for closing-line value)")
    else:
        print(f"  Logged rows : {rep['logged']}   covering games {rep['first_game']} .. {rep['last_game']}")
    print(f"  Graded      : {rep['graded']}      Pending: {rep['pending']} (game not played yet or not loaded: run `update`, then `grade`)      Voided: {rep['void']}")
    if picks_only:
        print("  How to read: for every line the model takes ONE side (the one with the higher expected value) and that pick is staked at 1 unit at the")
        print("  price shown when first logged. A coin-flip model would win about 50% and lose roughly the sportsbook's margin (about -4% to -5% ROI at -110).")
    else:
        print("  How to read: this view counts BOTH sides of every line, so wins and losses cancel to ~50% by construction. Use it only for comparison;")
        print("  the default view (picks only) is the one that measures the model. ROI = units won / bets placed.")
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
    section("Overall" if not picks_only else "Overall: every pick")
    table(["", *cols], [row("all graded picks" if picks_only else "all graded rows", o)], al)
    if not picks_only:
        print("  Note: this row includes BOTH sides of every line, so even a model with no edge shows a slightly negative ROI (the sportsbook's margin).")
    if rep.get("flagged"):
        section(f"Flagged picks only (claimed EV >= {rep['min_ev']:g}%): the bets the program would actually recommend")
        table(["", *cols], [row("flagged", rep["flagged"])], al)
        fl = rep["flagged"]
        if fl["expected_win_rate"] is not None and fl["win_rate"] is not None and fl["bets"] >= 30 and fl["expected_win_rate"] - fl["win_rate"] > 0.05:
            print(f"  The model expected to win {100 * fl['expected_win_rate']:.1f}% of these and won {100 * fl['win_rate']:.1f}%: its claimed edges were larger than what happened (over-confident).")
    else:
        print(f"\n  No graded pick had claimed EV >= {rep['min_ev']:g}% yet.")
    section("By market")
    table(["market", *cols], [row(k, v) for k, v in rep["by_kind"].items()], al)
    section("By side (which side the model picks, and how each side performed)")
    table(["bet type", *cols], [row(k, v) for k, v in rep["by_side"].items()], al)
    section("By claimed-EV bucket (a working model should do better as claimed EV rises)")
    table(["claimed EV", *cols], [row(k, v) for k, v in rep["by_ev_bucket"].items()], al)

    section("Lean check (does the model pile onto one side?)")
    lean = rep["lean"]
    if not lean:
        print("  No picks logged yet.")
    for k, v in lean.items():
        split = ", ".join(f"{sd} {n}" for sd, n in v["counts"].items())
        flag = ""
        if v["n"] >= 30 and v["top_share"] >= 0.70:
            flag = f"   <-- {100 * v['top_share']:.0f}% on {v['top_side']}"
        print(f"  {k:10} {v['n']:>4} picks: {split}{flag}")
        if k == "prop" and v.get("avg_proj_minus_line") is not None:
            ap = v.get("avg_actual_minus_line")
            print(f"             average projection minus line: {v['avg_proj_minus_line']:+.2f} pts"
                  + (f"   average actual minus line: {ap:+.2f} pts ({v['n_graded']} graded)" if ap is not None else ""))
    pl = lean.get("prop")
    if pl and pl["n"] >= 30 and pl["top_share"] >= 0.70:
        print(f"  A lean this one-sided is almost never {pl['n']} independent edges. It means the model's projections sit systematically {'below' if pl['top_side'] == 'UNDER' else 'above'} the sportsbook lines.")
        if pl.get("avg_actual_minus_line") is not None and abs(pl["avg_actual_minus_line"]) < 0.5 * abs(pl["avg_proj_minus_line"]):
            print("  Players actually landed close to the lines on average, so the gap is the MODEL being off, not the market. Two fixes: raise the market blend")
            print("  (next section), and run `python cli.py backtest` to check the player projections for bias.")
        else:
            print("  Players did land away from the lines in the model's direction, which is what a real edge would look like; keep watching it.")

    section("Closing-line value (did the market move toward our side after we bet?)")
    only_one_snapshot = "price_clv_n" in o and abs(o["price_clv_pts"]) < 0.01 and o.get("line_unchanged_pct", 100) >= 95
    if only_one_snapshot:
        print(f"  Not informative yet: the latest odds on file are essentially the ones the picks were logged from ({o.get('line_unchanged_pct', 100):.0f}% of lines never moved).")
        print("  To measure it, pull odds again close to tip-off (`python cli.py odds --force`) BEFORE running `grade`.")
    elif "price_clv_n" in o:
        verdict = "beating the close" if o["price_clv_pts"] > 0 else "behind the close"
        print(f"  Price CLV: {_sign(o['price_clv_pts'], 2)} implied-probability points over {o['price_clv_n']} picks  ({verdict}).")
        print("  Positive means the price we got was better than the last price we saw; beating the close consistently is the best early sign of a real edge.")
    else:
        print("  Not available yet: closing prices come from later odds pulls. Run `python cli.py odds` shortly before games start so a late snapshot exists.")
    if "line_n" in o and not only_one_snapshot:
        print(f"  Line movement: the number moved in our favor on {o['line_beat_pct']:.0f}% of {o['line_n']} spread/total/prop picks ({o['line_unchanged_pct']:.0f}% did not move).")

    section("How much can you trust this?")
    n, se = o["bets"], o["roi_se_pct"]
    if n < 30:
        print(f"  Only {n} graded picks: far too few to conclude anything. A handful of results is mostly luck; the +/-95% column shows n/a below 10 bets.")
    if se is not None:
        lo, hi = o["roi_pct"] - 1.96 * se, o["roi_pct"] + 1.96 * se
        needed = int((19.6 * max(o["roi_se_pct"] * sqrt_n(n) / 100, 0.9)) ** 2)
        print(f"  With {n} graded picks the 95% range for true ROI is roughly {lo:+.0f}% to {hi:+.0f}% (observed {o['roi_pct']:+.1f}%).")
        print(f"  Narrowing that to about +/-10% takes roughly {needed} picks at these odds. Until then, treat ROI as noise and watch calibration and CLV instead.")
    if rep.get("flagged") and rep["flagged"]["bets"] < 30:
        print(f"  Flagged picks: only {rep['flagged']['bets']}. A flagged ROI of {rep['flagged']['roi_pct']:+.0f}% on that few bets says almost nothing either way.")

    c = rep.get("calibration")
    section("Calibration (is the model better than the market's own price?)")
    if c:
        print(f"  Log loss (lower is better): model {c['logloss_model']:.4f}, market {c['logloss_market']:.4f}; best blend puts {c['best_market_weight']:.0%} weight on the market ({c['n']} picks).")
        print("  " + ("Enough picks to act on this; see the next section." if c["enough_data"] else "Fewer than 300 picks overall: treat this as a hint, see the next section for the per-market evidence."))
    else:
        print("  Needs at least 30 graded picks that have a market probability.")

    section("Market blend: how much weight should the market's price get?")
    bw = rep.get("blend_weights") or {}
    if not bw:
        print("  Needs at least 30 graded picks with a market price in a group (props / game bets).")
    else:
        print("  Final probability = (1 - w) x model + w x market. The weight that minimises log loss on your graded picks is shown against what is configured.")
        rows = []
        for grp, v in bw.items():
            verdict = (f"change to {v['best_w']:.0%}" if v["apply_ok"] else
                       ("need 150+ picks" if v["n"] < 150 else ("current setting is close to best" if abs(v["best_w"] - v["current_w"]) < 0.10 else "gain too small to act on")))
            rows.append(["props" if grp == "prop" else "game bets", v["n"], _f(v["ll_model"], 4), _f(v["ll_market"], 4), f"{v['best_w']:.0%}", f"{v['current_w']:.0%}", verdict])
        table(["group", "graded picks", "model only", "market only", "best weight", "configured", "verdict"], rows, "lrrrrrl")
        ok = {g: v for g, v in bw.items() if v["apply_ok"]}
        if args.apply_weights:
            if not ok:
                print("  Nothing saved: no group has enough evidence yet (needs 150+ picks, a log-loss gain of 0.01+, and a change of 10+ points).")
            else:
                t = Tuning.load(MODEL.MODELS_DIR)
                if "prop" in ok:
                    t.prop_market_weight = float(ok["prop"]["best_w"])
                if "game" in ok:
                    t.game_market_weight = float(ok["game"]["best_w"])
                t.save(MODEL.MODELS_DIR)
                print("  Saved: " + "; ".join(f"{'props' if g == 'prop' else 'game bets'} market blend {v['current_w']:.0%} -> {v['best_w']:.0%}" for g, v in ok.items()))
                print("  Used automatically by `slate` and `predict`. Undo with `python cli.py backtest --reset` (removes all saved corrections).")
        elif ok:
            print("  The evidence is strong enough to act on. Save it with:  python cli.py report --apply-weights")

    recent = rep["recent"] if args.all else rep["recent"][:15]
    section(f"Graded picks ({'all' if args.all else 'most recent 15; add --all for every pick'}, {len(rep['recent'])} total)")
    rows = []
    for r in recent:
        parts = str(r["game_key"]).split("_")
        away_home = parts[1:3] if len(parts) >= 3 else [str(r["game_key"]), ""]
        pn = r.get("player_name")
        who = (pn + " " if isinstance(pn, str) and pn else "") + r["side"]
        line = "" if r["line"] is None or r["line"] != r["line"] else (f"{r['line']:+g}" if r["kind"] == "spread" else f"{r['line']:g}")
        act = "" if r["actual"] is None or r["actual"] != r["actual"] else f"{r['actual']:g}"
        rows.append([r["game_date"], (f"{away_home[0]}@{away_home[1]}" if away_home[1] else away_home[0]), r["kind"], f"{who} {line}".strip(), f"{r['price']:+d}", _pct(r["blended_prob"], 0), _sign(r["ev_pct"]), r["result"].upper(), act])
    table(["game date", "game", "market", "pick", "price", "p(win)%", "EV%", "result", "actual"], rows, "lllrrrrll")
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
    print("  Corrections in effect (from `backtest`): " + ("; ".join(s["corrections"]) if s["corrections"] else "none"))
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


def _fmt_dir(before, after, lower_better):
    if lower_better is None or before is None or after is None:
        return ""
    if abs(after - before) <= 0.002 * max(abs(before), 1e-9):
        return "same"
    better = after < before if lower_better else after > before
    return "better" if better else "worse"


def _print_findings(findings):
    for f in findings:
        print(f"  [{'ADOPTED ' if f.adopted else 'rejected'}] {f.name}: {f.detail}")


def _rel_table(rows, indent="  "):
    table(["predicted range", "games", "model said", "actually happened"],
          [[f"{100 * r['lo']:.0f}%-{100 * r['hi']:.0f}%", r["n"], f"{100 * r['mean_p']:.1f}%", f"{100 * r['rate']:.1f}%"] for r in rows], "lrrr", indent=indent)


def cmd_backtest(args) -> int:
    from services import Engine, NotReady
    from backtest_runner import replay_week, run_backtest
    from modeling.tuning import Tuning
    from config import MODEL
    wh = Warehouse(args.db)
    eng = Engine(wh)
    if args.reset:
        removed = Tuning.reset(eng.models_dir)
        print("Corrections removed: the model now behaves exactly as trained." if removed else "No saved corrections to remove.")
        if removed:
            print("(If these corrections included tuned rating parameters, run `python cli.py train` to refit the models without them.)")
        return 0
    try:
        eng._frames()
    except NotReady as e:
        print(e); return 1
    say = lambda m: print(f"  {m}")
    if args.week:
        return _backtest_week(eng, args, say)

    print("BACKTEST: replaying history as if it were live, using only information available at the time.")
    print("  Nothing is scored in-sample: ratings are walk-forward and the models use time-ordered cross-validation.")
    print("  Corrections are fitted on the EARLIER ~70% of games and kept only if they improve the LATER ~30% (the holdout).")
    print("  No free source has historical betting lines, so this measures accuracy and probability calibration, not profit.")
    print("  On a full history this takes several minutes (it refits the models and replays many rating settings); progress is shown below.")
    try:
        res = run_backtest(eng, apply=not args.dry_run, tune_rating=not args.no_tune, seasons=args.seasons, progress=say)
    except ValueError as e:
        print(e); return 1
    d = res["derived"]
    gd, pd_ = d.diagnostics["game"], d.diagnostics["player"]
    if gd.get("n", 0) == 0:
        print("\nNot enough history to backtest. Run `python cli.py backfill` first.")
        return 1
    print(f"\nScored {gd['n']} games across {', '.join(res['seasons'])}" + (f"; holdout (verification) period starts {d.cut.date()}." if d.cut is not None else "."))

    section("1. Game accuracy with the current settings")
    table(["", "this model", "old-style baseline*", "improvement"],
          [["margin error (MAE, pts)", _f(gd["mae_m"], 2), _f(gd["mae_m_naive"], 2), _verdict(gd["mae_m"], gd["mae_m_naive"]).split(" error")[0].strip()],
           ["total error (MAE, pts)", _f(gd["mae_t"], 2), _f(gd["mae_t_naive"], 2), _verdict(gd["mae_t"], gd["mae_t_naive"]).split(" error")[0].strip()]], "lrrl")
    print("  * baseline = season scoring averages + 3 home-court points (the previous version's method).")
    print(f"  Average bias: margin {_sign(gd['bias_m'], 2)} (+/-{1.96 * gd['bias_m_se']:.2f}), total {_sign(gd['bias_t'], 2)} (+/-{1.96 * gd['bias_t_se']:.2f}). "
          "A bias larger than its +/- range is a systematic error that can be corrected.")
    print(f"  Typical miss: margin {_f(gd['sigma_m'], 1)} pts, total {_f(gd['sigma_t'], 1)} pts. Picked the winner {_f(100 * gd['pick_accuracy_vs_naive'], 1)}% of the time.")
    print("  Error over time (does the model get worse at certain points of the season?):")
    table(["month", "games", "margin MAE", "total MAE", "margin bias", "total bias"],
          [[m["month"], m["n"], _f(m["mae_m"], 2), _f(m["mae_t"], 2), _sign(m["bias_m"], 2), _sign(m["bias_t"], 2)] for m in gd["by_month"]], "lrrrrr", indent="    ")

    section("2. Are the win probabilities honest?")
    print("  Games grouped by the home-win probability the model gave; the last column is how often the home team really won.")
    print("  A calibrated model has the two columns close. A model that says 80% but wins 65% is over-confident.")
    _rel_table(gd["win_reliability"])
    print(f"  Log loss {_f(gd['win_logloss'], 4)} (lower is better; always guessing 50% scores 0.6931).")

    section("3. Player points")
    print(f"  {pd_['n']} out-of-fold player-games. Error {_f(pd_['mae'], 2)} pts vs {_f(pd_['mae_form'], 2)} for just using recent form ({_verdict(pd_['mae'], pd_['mae_form']).split(' error')[0]}).  Minutes error {_f(pd_['minutes_mae'], 2)}.")
    table(["player type", "player-games", "model MAE", "recent-form MAE", "bias"],
          [[t["tier"], t["n"], _f(t["mae"], 2), _f(t["mae_form"], 2), _sign(t["bias"], 2)] for t in pd_["tiers"]], "lrrrr", indent="    ")
    if pd_["teammates_out"]:
        print("  Does it handle absences? (teammates missing from the usual rotation)")
        table(["situation", "player-games", "model MAE", "recent-form MAE", "bias"],
              [[t["bucket"], t["n"], _f(t["mae"], 2), _f(t["mae_form"], 2), _sign(t["bias"], 2)] for t in pd_["teammates_out"]], "lrrrr", indent="    ")
    acc = pd_["form_line_accuracy"]
    print(f"  Skill check: against a 'recent form' line the model picked the right side of over/under {_f(100 * acc, 1)}% of the time"
          + (f" ({_f(100 * pd_['form_line_accuracy_big_gap'], 1)}% when it disagreed with that line by 1.5+ pts, {pd_['form_line_n_big']} cases)" if pd_["form_line_accuracy_big_gap"] is not None else "") + ".")
    print("  (Sportsbook lines are sharper than a recent-form line, so treat this as 'has information beyond recent form', not as a betting win rate.)")
    print("  Calibration of the points distribution (lines placed around each projection; actual over-rate should match):")
    _rel_table(pd_["calibration"])

    if d.trials:
        section("4. Team-rating parameter search (score on EARLIER games; 2.000 = current settings, lower is better)")
        best = sorted(d.trials, key=lambda t: t["score"])[:8]
        table(["parameter", "value", "margin MAE", "total MAE", "score"], [[t["param"], f"{t['value']:g}", _f(t["margin_mae"], 3), _f(t["total_mae"], 3), _f(t["score"], 4)] for t in best], "lrrrr", indent="    ")
        print("  (best 8 of the combinations tried)")
    section(f"{5 if d.trials else 4}. What the backtest found")
    _print_findings(d.findings)

    if d.before_after:
        section(f"{6 if d.trials else 5}. Before vs after, scored ONLY on the held-out games (from {d.cut.date()})")
        table(["metric", "before", "after", "verdict"],
              [[r["metric"], _f(r["before"], 4 if "log" in r["metric"] else 2), _f(r["after"], 4 if "log" in r["metric"] else 2), _fmt_dir(r["before"], r["after"], r["lower_better"])] for r in d.before_after], "lrrl")
        print("  'before' = the default model; 'after' = with the adopted corrections. If nothing was adopted the two match.")

    section("Outcome")
    n_ad = len(res["adopted"])
    if args.dry_run:
        print(f"  Dry run: nothing was saved. {n_ad} correction{'s' if n_ad != 1 else ''} passed verification." + ("  Re-run without --dry-run to apply them." if n_ad else ""))
    elif res["applied"]:
        print(f"  Applied and saved {n_ad} verified correction{'s' if n_ad != 1 else ''} to {res['saved_to']}/tuning.json:")
        for line in eng.tuning.describe():
            print(f"    - {line}")
        print("  They are used automatically by `slate` and `predict`. Undo any time with `python cli.py backtest --reset`.")
    else:
        print("  No correction passed verification on the held-out games, so the model was left unchanged. That is a normal, healthy result:")
        print("  it means nothing the backtest found was big and consistent enough to be worth acting on.")
        if res["previous"]:
            print("  (Corrections saved by an earlier run are still in effect: " + "; ".join(res["previous"]) + ". `backtest --reset` removes them.)")
    print("\nTo test how this would have played out on a recent week: python cli.py backtest --week")
    return 0


def _backtest_week(eng, args, say) -> int:
    from backtest_runner import replay_week
    print("BACKTEST WEEK: replaying a week day by day, exactly as it would have run live.")
    print("  The models are re-fitted on games from BEFORE the week; each day's predictions use only earlier days.")
    print("  Two versions are compared on the same games: the default model, and the model with the corrections")
    print("  the backtest would have found using only pre-week history.")
    res = replay_week(eng, end=args.ending, tune_rating=not args.no_tune, progress=say)
    if "error" in res:
        print("\n" + res["error"]); return 1
    G, P, gs, ps, bet = res["games"], res["players"], res["game_summary"], res["player_summary"], res["betting"]
    d = res["derived"]
    print(f"\nWeek {res['start']} .. {res['end']}: {len(G)} games replayed; models fitted on {res['n_pre_games']} earlier games.")
    print("  Assumption: the injury report matched who actually sat out (rotation players who didn't play are treated as 'Out').")
    print("  This is optimistic, since late scratches weren't known in advance.")

    section("Game by game (margin = home minus away)")
    rows = []
    for _, r in G.iterrows():
        rows.append([r["date"], r["matchup"], _sign(r["actual_margin"], 0), _sign(r["base_margin"]), _sign(r["corr_margin"]), _sign(r["actual_margin"] - r["corr_margin"]),
                     f"{r['actual_total']:.0f}", _f(r["base_total"]), _f(r["corr_total"]), _sign(r["actual_total"] - r["corr_total"])])
    table(["date", "game", "actual", "model", "corrected", "miss", "actual tot", "model tot", "corr tot", "miss"], rows, "llrrrrrrrr")

    section("Game summary (lower error is better)")
    names = [("naive", "old-style baseline"), ("base", "default model"), ("corr", "model + corrections")]
    table(["predictor", "margin MAE", "total MAE", "margin bias", "total bias", "winner right"],
          [[lab, _f(gs[k]["margin_mae"], 2), _f(gs[k]["total_mae"], 2), _sign(gs[k]["margin_bias"], 2), _sign(gs[k]["total_bias"], 2), f"{100 * gs[k]['winner_correct']:.0f}%"] for k, lab in names], "lrrrrr")
    print("  bias = actual minus projected, averaged (positive = the model was too low).")
    print(f"  Home-win log loss: default {_f(gs['base']['win_logloss'], 3)}, corrected {_f(gs['corr']['win_logloss'], 3)}.")

    if ps:
        section("Player points")
        table(["predictor", "MAE (pts)", "bias", "right side of form-line", "over/under log loss"],
              [["recent form only", _f(ps["mae_form"], 2), "", "", ""],
               ["default model", _f(ps["base"]["mae"], 2), _sign(ps["base"]["bias"], 2), f"{100 * ps['base']['form_line_accuracy']:.0f}%", _f(ps["base"]["calibration_logloss"], 4)],
               ["model + corrections", _f(ps["corr"]["mae"], 2), _sign(ps["corr"]["bias"], 2), f"{100 * ps['corr']['form_line_accuracy']:.0f}%", _f(ps["corr"]["calibration_logloss"], 4)]], "lrrrr")
        print(f"  {ps['n']} player-games (projected 12+ minutes, played).")

    section("Betting simulation")
    if bet["games_with_lines"] == 0:
        print("  No stored betting lines for this week, so bets can't be simulated. Run `python cli.py odds` before games to build")
        print("  up history; every pull is stored and later weeks can then be replayed with real prices.")
    else:
        print(f"  Stored lines exist for {bet['games_with_lines']} of {len(G)} games. The model takes one side of each line (its pick); flat 1-unit stakes at the stored price (flagged = EV >= {BETTING.EV_THRESHOLD_PCT:g}%).")
        def row(lab, b):
            return [lab, b["bets"], f"{b['wins']}-{b['losses']}", _sign(b["units"], 2), _sign(b["roi_pct"]) if b["roi_pct"] is not None else "n/a"]
        table(["", "bets", "W-L", "units", "ROI%"],
              [row("default model, flagged picks", bet["base_flagged"]), row("corrected model, flagged picks", bet["corr_flagged"]),
               row("default model, every pick", bet["base_all"]), row("corrected model, every pick", bet["corr_all"])], "lrrrr")
        print("  A week of bets is a tiny sample: ROI swings of tens of percent are normal noise.")

    m = res["misses"]
    if m["top"]:
        section("Biggest player misses (corrected projection vs actual)")
        table(["date", "game", "player", "projected", "actual", "miss", "proj min", "actual min", "likely cause"],
              [[r["date"], r["game"], r["player"][:22], _f(r["proj_corr"]), f"{r['actual_pts']:.0f}", _sign(r["actual_pts"] - r["proj_corr"]), _f(r["proj_min"], 0), f"{r['actual_min']:.0f}", ", ".join(r["tags"])] for r in m["top"]], "lllrrrrrl")
        if m["tag_counts"]:
            print(f"  Of {m['n_big']} misses of 8+ points (out of {m['n']} player-games): " + ", ".join(f"{k} {v}" for k, v in sorted(m['tag_counts'].items(), key=lambda kv: -kv[1])) + " (a miss can have several causes).")
            print("  'minutes surprise' = played 8+ minutes more/less than projected (foul trouble, injury, blowout, role change); these are the hardest to predict.")

    section("What would have improved")
    _print_findings(d.findings)
    t = d.tuning
    if not t.describe():
        print("  No correction passed verification on pre-week data, so the corrected model equals the default model and the two rows above match.")
    else:
        print("  Effect of the adopted corrections on this week (negative = lower error = better):")
        if t.rating or t.margin_scale != 1.0 or t.margin_bias or t.total_bias:
            print(f"    margin MAE {_sign(gs['corr']['margin_mae'] - gs['base']['margin_mae'], 2)} pts ({gs['base']['margin_mae']:.2f} -> {gs['corr']['margin_mae']:.2f}),"
                  f" total MAE {_sign(gs['corr']['total_mae'] - gs['base']['total_mae'], 2)} pts ({gs['base']['total_mae']:.2f} -> {gs['corr']['total_mae']:.2f})")
        if t.win_cal or t.margin_scale != 1.0:
            print(f"    home-win log loss {_sign(gs['corr']['win_logloss'] - gs['base']['win_logloss'], 4)} ({gs['base']['win_logloss']:.4f} -> {gs['corr']['win_logloss']:.4f})")
        if t.prop_cal and ps:
            print(f"    over/under log loss {_sign(ps['corr']['calibration_logloss'] - ps['base']['calibration_logloss'], 4)} ({ps['base']['calibration_logloss']:.4f} -> {ps['corr']['calibration_logloss']:.4f})")
    gb = gs["base"]["total_bias"]
    if abs(gb) >= 2:
        print(f"  Note: the default model's totals ran {abs(gb):.1f} pts too {'low' if gb > 0 else 'high'} this week. One week can't prove a bias (the full backtest tests that over the whole history).")
    print("  One week is a small sample: differences of a fraction of a point in MAE are noise. Use `python cli.py backtest` for the full-history verdict.")
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
    p = sub.add_parser("backtest", help="replay history, find corrections, verify them on later games")
    p.add_argument("--week", action="store_true", help="replay one week day by day instead of the full history")
    p.add_argument("--ending", metavar="YYYY-MM-DD", help="with --week: last day of the week (default: latest data)")
    p.add_argument("--dry-run", action="store_true", help="report only; do not save any corrections")
    p.add_argument("--reset", action="store_true", help="remove saved corrections")
    p.add_argument("--no-tune", action="store_true", help="skip the slower team-rating parameter search")
    p.add_argument("--seasons", type=int, help="limit to the most recent N seasons")
    p.set_defaults(fn=cmd_backtest)
    p = sub.add_parser("report", help="ROI, calibration and closing-line value from graded bets")
    p.add_argument("--since", metavar="YYYY-MM-DD", help="only games on/after this date")
    p.add_argument("--kind", choices=["spread", "total", "moneyline", "prop"], help="only one market")
    p.add_argument("--min-ev", type=float, default=None, help="claimed-EV%% threshold for 'flagged' bets (default from config)")
    p.add_argument("--all", action="store_true", help="list every graded pick, not just the latest 15")
    p.add_argument("--both-sides", action="store_true", help="count both sides of every line (old behaviour; wins and losses cancel)")
    p.add_argument("--apply-weights", action="store_true", help="save the market-blend weights the graded picks support (needs 150+ picks)")
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
