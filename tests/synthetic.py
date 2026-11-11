"""
Synthetic NBA league written into a Warehouse. FOR TESTS ONLY: it lets the whole pipeline
(frames -> ratings -> game model -> player model -> EV) be exercised offline with known
ground truth. Results on this data say nothing about real-world accuracy.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from nbadata.teams import TEAMS, team_abbr
from nbadata.warehouse import Warehouse, make_game_key

BASE_MIN = [35, 33, 31, 28, 26, 22, 18, 14, 10, 6]
USAGE = [1.35, 1.15, 1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]


def season_label(y: int) -> str:
    return f"{y}-{(y + 1) % 100:02d}"


def build_league(wh: Warehouse, seasons: Tuple[int, ...] = (2023, 2024), games_per_team: int = 60,
                 seed: int = 7, hca: float = 2.5, truth: Optional[Dict] = None,
                 b2b_penalty: float = 0.0, day_step: float = 2.0) -> Dict:
    rng = np.random.default_rng(seed)
    ids = [t.id for t in TEAMS]
    if truth is None:
        truth = {
            "off": {t: rng.normal(0, 3.0) for t in ids},
            "dfn": {t: rng.normal(0, 3.0) for t in ids},
            "pace": {t: rng.normal(0, 2.0) for t in ids},
            "hca": hca,
        }
    # player skill: some players much better scorers than others (usage multipliers)
    skill = {t: rng.lognormal(0, 0.18, size=10) for t in ids}
    games, team_rows, player_rows = [], [], []
    last_played: Dict[int, date] = {}
    n_rounds = games_per_team
    for sy in seasons:
        start = date(sy, 10, 22)
        for rnd in range(n_rounds):
            d = start + timedelta(days=int(rnd * day_step) + rnd // 7)
            perm = rng.permutation(ids)
            # play 12 of 15 possible games per round so rest/b2b varies
            for k in range(0, 24, 2):
                home, away = int(perm[k]), int(perm[k + 1])
                poss = 99 + truth["pace"][home] + truth["pace"][away] + rng.normal(0, 2.5)
                pen_h = b2b_penalty if last_played.get(home) and (d - last_played[home]).days == 1 else 0.0
                pen_a = b2b_penalty if last_played.get(away) and (d - last_played[away]).days == 1 else 0.0
                last_played[home] = last_played[away] = d
                eff_h = 115 + truth["hca"] + truth["off"][home] + truth["dfn"][away] - pen_h + rng.normal(0, 7.5)
                eff_a = 115 - truth["hca"] + truth["off"][away] + truth["dfn"][home] - pen_a + rng.normal(0, 7.5)
                hp, ap = int(round(eff_h * poss / 100)), int(round(eff_a * poss / 100))
                if hp == ap:
                    hp += 1
                key = make_game_key(d, away, home)
                games.append({"game_key": key, "game_date": d.isoformat(), "season": season_label(sy), "season_type": 2,
                              "home_team_id": home, "away_team_id": away, "tipoff_utc": None, "status": "final",
                              "home_pts": hp, "away_pts": ap, "espn_id": None, "nba_id": None, "source": "synthetic"})
                for tid, pts, is_home in ((home, hp, 1), (away, ap, 0)):
                    fta, oreb, tov = 22, 10, 13
                    fga = int(round(poss - 0.44 * fta + oreb - tov))
                    team_rows.append({"game_key": key, "team_id": tid, "is_home": is_home, "pts": pts, "fgm": int(fga * 0.46),
                                      "fga": fga, "fg3m": 12, "fg3a": 34, "ftm": 17, "fta": fta, "oreb": oreb, "dreb": 33,
                                      "reb": 43, "ast": 25, "tov": tov, "stl": 7, "blk": 5, "pf": 20})
                    # players: random absences redistribute minutes and shots
                    avail = rng.random(10) > 0.08
                    avail[:2] = avail[:2] | (rng.random(2) > 0.5)         # stars miss less
                    mins = np.array(BASE_MIN, dtype=float) * avail
                    mins = mins * (240.0 / max(mins.sum(), 1.0)) * np.clip(rng.normal(1, 0.04, 10), 0.9, 1.1)
                    mins = np.where(avail, np.clip(mins + rng.normal(0, 2.0, 10), 3, 42), 0)
                    w = mins * np.array(USAGE) * skill[tid] * (avail)
                    shares = rng.multinomial(pts, w / w.sum())
                    for i in range(10):
                        played = int(avail[i])
                        player_rows.append({
                            "game_key": key, "team_id": tid, "player_key": f"{team_abbr(tid).lower()} player{i + 1}",
                            "player_name": f"{team_abbr(tid)} Player{i + 1}", "player_id": None,
                            "starter": 1 if i < 5 else 0, "played": played, "minutes": float(mins[i]),
                            "pts": int(shares[i]), "reb": 4, "ast": 3, "fgm": int(shares[i] // 2.4), "fga": int(shares[i] / 1.1) + 1 if played else 0,
                            "fg3m": 1, "fg3a": 3, "ftm": 2, "fta": 3 if played else 0, "oreb": 1, "dreb": 3, "tov": 1 if played else 0,
                            "stl": 1, "blk": 0, "pf": 2, "plus_minus": 0, "dnp_reason": None if played else "DNP",
                        })
    wh.upsert_games(games)
    by_game: Dict[str, Tuple[List[dict], List[dict]]] = {g["game_key"]: ([], []) for g in games}
    for r in team_rows:
        by_game[r["game_key"]][0].append(r)
    for r in player_rows:
        by_game[r["game_key"]][1].append(r)
    gmap = {g["game_key"]: g for g in games}
    for k, (tr, pr) in by_game.items():
        wh.save_boxscore(gmap[k], tr, pr)
    return truth
