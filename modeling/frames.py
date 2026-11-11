"""Turn warehouse tables into model-ready frames. Pure pandas; no network."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def possessions(df: pd.DataFrame) -> pd.Series:
    return df["fga"] + 0.44 * df["fta"] - df["oreb"] + df["tov"]


def build_team_games(team_box: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    One row per team-game with efficiency, pace and schedule context.
    Rows are chronological. `poss` is the game-level possession estimate (mean of both teams).
    """
    if team_box.empty:
        return pd.DataFrame()
    tb = team_box.copy()
    tb["poss_team"] = possessions(tb)
    g = games.set_index("game_key")
    gposs = tb.groupby("game_key")["poss_team"].mean().rename("poss")
    pts_by = tb.pivot(index="game_key", columns="team_id", values="pts")

    rows = []
    for side, opp_side in (("home_team_id", "away_team_id"), ("away_team_id", "home_team_id")):
        d = g[[side, opp_side, "game_date", "season", "season_type"]].copy()
        d.columns = ["team_id", "opp_id", "game_date", "season", "season_type"]
        d["is_home"] = 1 if side == "home_team_id" else 0
        d = d.reset_index()
        rows.append(d)
    tg = pd.concat(rows, ignore_index=True)
    tg = tg.merge(tb[["game_key", "team_id", "pts"]], on=["game_key", "team_id"], how="inner")
    opp_pts = tb[["game_key", "team_id", "pts"]].rename(columns={"team_id": "opp_id", "pts": "opp_pts"})
    tg = tg.merge(opp_pts, on=["game_key", "opp_id"], how="inner").merge(gposs, on="game_key", how="left")
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    tg = tg.sort_values(["game_date", "game_key", "is_home"]).reset_index(drop=True)
    tg["off_eff"] = 100.0 * tg["pts"] / tg["poss"]
    tg["def_eff"] = 100.0 * tg["opp_pts"] / tg["poss"]
    return add_schedule_context(tg)


def add_schedule_context(tg: pd.DataFrame) -> pd.DataFrame:
    """rest days, back-to-back, games in previous 4 days, game number in season (all strictly pre-game)."""
    tg = tg.sort_values(["team_id", "game_date", "game_key"]).copy()
    prev = tg.groupby("team_id")["game_date"].shift(1)
    tg["rest"] = (tg["game_date"] - prev).dt.days.fillna(7).clip(upper=7).astype(float)
    tg["b2b"] = (tg["rest"] <= 1).astype(float)

    def g4(s: pd.Series) -> pd.Series:
        v = s.values.astype("datetime64[D]").astype(np.int64)
        out = np.zeros(len(v))
        j = 0
        for i in range(len(v)):
            while v[i] - v[j] > 3:
                j += 1
            out[i] = i - j      # games in the 3 days before, excluding this one
        return pd.Series(out, index=s.index)

    tg["g_last4"] = tg.groupby("team_id", group_keys=False)["game_date"].apply(g4)
    tg["team_game_no"] = tg.groupby(["team_id", "season"]).cumcount() + 1
    return tg.sort_values(["game_date", "game_key", "is_home"]).reset_index(drop=True)


def games_frame(tg: pd.DataFrame) -> pd.DataFrame:
    """One row per game (home perspective) with both teams' context."""
    h = tg[tg["is_home"] == 1].set_index("game_key")
    a = tg[tg["is_home"] == 0].set_index("game_key")
    g = pd.DataFrame({
        "game_date": h["game_date"], "season": h["season"], "season_type": h["season_type"],
        "home_id": h["team_id"], "away_id": h["opp_id"],
        "home_pts": h["pts"], "away_pts": h["opp_pts"], "poss": h["poss"],
        "rest_h": h["rest"], "rest_a": a["rest"].reindex(h.index),
        "b2b_h": h["b2b"], "b2b_a": a["b2b"].reindex(h.index),
        "g4_h": h["g_last4"], "g4_a": a["g_last4"].reindex(h.index),
        "gno_h": h["team_game_no"], "gno_a": a["team_game_no"].reindex(h.index),
    }).dropna(subset=["home_id", "away_id"])
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["total"] = g["home_pts"] + g["away_pts"]
    return g.reset_index().sort_values(["game_date", "game_key"]).reset_index(drop=True)


def context_for(tg: pd.DataFrame, team_id: int, game_date) -> dict:
    """Rest context for an UPCOMING game, from the team's completed games before `game_date`."""
    gd = pd.Timestamp(game_date)
    prior = tg[(tg["team_id"] == team_id) & (tg["game_date"] < gd)]
    if prior.empty:
        return {"rest": 7.0, "b2b": 0.0, "g_last4": 0.0}
    last = prior["game_date"].max()
    rest = float(min((gd - last).days, 7))
    recent = prior[prior["game_date"] >= gd - pd.Timedelta(days=3)]
    return {"rest": rest, "b2b": float(rest <= 1), "g_last4": float(len(recent))}
