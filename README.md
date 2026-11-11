# NBA Prediction Model

Projects NBA game margins/totals and player points, compares them with DraftKings (and other US books) lines,
and reports expected value. Command-line only (Windows cmd friendly). Built around a local data warehouse and an odds API instead of scrapers.

## Quick start (Windows cmd)

```bat
pip install -r requirements.txt
copy .env.example .env            :: then edit .env and paste your free key from https://the-odds-api.com
python cli.py setup               :: checks sources, loads history, trains, pulls odds; stops with a reason if anything is wrong
python cli.py slate               :: today's projections and flagged bets
```

`setup` is the slow one (history load: roughly 30-40 minutes). It is resumable: Ctrl+C is safe and re-running continues.
You can also run the steps yourself: `doctor` -> `backfill` -> `train` -> `odds` -> `slate`.
`python cli.py status` always ends with the next step to run.

Daily routine: `update` -> `odds` -> `slate`. After games finish: `update` -> `grade` -> `report`. Now and then: `backtest`.
Single matchup: `python cli.py predict BOS LAL --date 2026-11-02 --all`.

## Reading the output

Every command explains its own output, and ends with the next step to run. The one that needs the most care is `report`:

```bat
python cli.py report                 :: all graded bets
python cli.py report --kind prop     :: one market (spread | total | moneyline | prop)
python cli.py report --since 2026-11-01 --all
```

| Section | What it tells you |
|---|---|
| Logged / Graded / Pending / Voided | How many picks exist, how many have results, how many are waiting for games (run `update`, then `grade`). |
| Overall | **One pick per line**: the side the model liked better when it logged the line. A coin-flip model wins about 50% and loses roughly the sportsbook's margin (about -4% to -5% ROI at -110). `--both-sides` shows the old view where both sides are counted; there wins and losses cancel by construction. |
| Flagged picks | The picks the program would actually recommend (claimed EV above the threshold). This row measures the strategy. |
| By market / side / claimed-EV bucket | Whether one market or direction is carrying results, and whether higher claimed EV really does better. |
| Lean check | Which side the model picks. A one-sided lean (say 90% UNDER) means projections sit systematically away from the lines, which is a bias, not hundreds of separate edges. For props it shows average projection minus line next to average actual minus line to tell a model bias from a real edge. |
| expected win% vs win% | The model's own average probability against what happened. A big gap means over-confidence. |
| Closing-line value | Whether the price/line moved toward your side after you bet. Needs a later `odds` pull before tip-off. |
| How much can you trust this? | The 95% range for true ROI at your sample size and roughly how many picks are needed. Under ~300 picks, ROI is mostly luck. |
| Market blend | Log loss of the model alone vs the market alone, and the weight on the market that fits your graded picks best. `report --apply-weights` saves it once there are 150+ picks in a group and the gain is real. |

### How picks are recorded

For every line (spread, total, moneyline, or one player's points line) `slate` evaluates both sides and flags the one with the higher expected value as the **pick**.
Only the first batch logged for a game/market/player/book counts as the bet; later re-evaluations at a moved line are stored (for closing-line value) but are not counted again.
Logs from earlier versions are migrated automatically the first time the database is opened: the first batch's higher-EV side becomes the pick.

## Backtesting and corrections

```bat
python cli.py backtest                :: replay history, find corrections, keep only those verified on later games
python cli.py backtest --dry-run      :: same report, saves nothing
python cli.py backtest --no-tune      :: skip the (slower) team-rating parameter search
python cli.py backtest --week         :: replay the latest week day by day: default model vs corrected model
python cli.py backtest --week --ending 2026-11-08
python cli.py backtest --reset        :: remove all saved corrections
```

**What it measures.** Projection accuracy and probability calibration against real outcomes (never in-sample: walk-forward ratings, time-ordered cross-validation).
There is no free source of historical betting lines, so profit is only simulated for games whose odds you stored (every `odds` pull is stored); run `odds` regularly and old weeks become replayable with real prices.

**What it corrects.** Team-rating parameters (half-life, shrinkage, carry-over); a margin stretch when projected margins are systematically too close to zero; margin and total bias; win-probability and over/under probability calibration.
Each correction is fitted on the earlier ~70% of games and **kept only if it improves the later ~30%**; otherwise it is rejected and the report says why. Many runs will adopt nothing, which is a healthy result.
Adopted corrections are saved to `models/tuning.json` and used automatically by `slate` and `predict`. Learned market-blend weights from `report --apply-weights` live in the same file.

**Week replay.** Re-fits the models on games before the week, then predicts each day using only earlier data. It compares the default model with the corrections the backtest would have found from pre-week history alone.
It assumes the injury report matched who actually sat out (optimistic), and one week is a small sample, so treat differences of a fraction of a point as noise.

## Troubleshooting

| You see | Meaning / fix |
|---|---|
| `nba_cdn ... HTTP 403 Access Denied` | NBA.com's CDN blocked the request (bot protection, VPN, or network filter). Not fatal: ESPN is the primary source. If `doctor` says "usable via espn", carry on. |
| `No game-data source is reachable` | Both sources failed. Turn off VPN/proxy, try another network, re-run `doctor`. |
| `No games found for <date>` on `slate` | Off-season or no games that day. It prints the next scheduled date; use `slate --date`. |
| `No games were loaded at all` after `backfill` | It lists which source answered with zero games (format change) and which failed outright. |
| `Could not find the number of physical cores` (joblib/loky) | Harmless Windows warning; silenced automatically by `config.py`. |
| `No ODDS_API_KEY` | Add it to `.env` (`ODDS_API_KEY=...`) or `set ODDS_API_KEY=...`. Without it you only get ESPN game lines, no props. |

## Where data comes from

| Need | Primary | Fallback | Notes |
|---|---|---|---|
| Schedule, box scores, player history | ESPN public JSON | NBA.com static CDN | No keys. Stored once in `data/nba_warehouse.db`, then only new games are fetched (the old code made ~40 `stats.nba.com` calls per prediction). Each box score is validated (player points must sum to the team score) before it's saved. |
| Game lines + player props | The Odds API (`bookmakers` include DraftKings) | ESPN's embedded game lines (no props), then CSV import | Needs `ODDS_API_KEY`. |
| Injuries | ESPN injury report | none (adjustment skipped) | Players ruled out get no prop bet and teammates' projections rise. |

**Odds API credits.** One game-lines pull costs 3 credits (moneyline, spread, total) and props cost 1 credit per game, so a
~10-game day is about 13 credits. The free plan's 500/month covers roughly one refresh a day. The app refuses to re-pull within
`OddsConfig.MIN_REFRESH_MINUTES` and shows remaining credits. Check their pricing page that player props are included in your plan.

**Offline / no key.** `python cli.py odds --import-games lines.csv --import-props props.csv` loads lines from any source. Formats are in `oddsfeed/csv_import.py`.

## How the model works

* **Team ratings** (`modeling/ratings.py`): walk-forward, time-decayed ridge regression of points per 100 possessions on
  offence/defence/home-court, plus a pace model. Opponent-adjusted; each prediction uses only earlier games. Last season's
  ratings are carried over (shrunk) as the prior.
* **Game model** (`modeling/game_model.py`): ratings give margin/total; a rest/back-to-back adjustment is kept only if it improves
  out-of-fold error. **Margin and total sigmas are fitted from out-of-sample residuals** (the old code used ensemble spread, ~1-2 points).
* **Player model** (`modeling/player_model.py`): Poisson-loss gradient boosting on pre-game rolling features, expected team points/pace,
  blowout risk, and a "teammates missing" signal. Distribution is a negative binomial with fitted dispersion, so integer lines (pushes) work.
* **EV** (`ev_engine.py`): cover/over probabilities, push-aware EV and Kelly, then **blended toward the de-vigged market price**
  (`BETTING.GAME_MARKET_WEIGHT`, `PROP_MARKET_WEIGHT`). Strong model-vs-market disagreement is flagged, because it usually means
  the book knows something (lineup, injury) the model doesn't.

## Bugs fixed from the previous version

* Spread cover probability had the wrong sign (a fair favourite showed ~72%), inflating EV on favourites. Regression tests in `tests/test_ev_engine.py`.
* The app never used the trained models (it averaged season points and added 3.5) and ignored opponent defence.
* `cross_val_score(fit_params=...)` crashes on current scikit-learn, so training could not run.
* The scraper wrote misaligned columns (totals ended up in the spread column).
* The season was hard-coded to 2024-25. It is now derived from the date.

## What is and isn't verified

* `python -m unittest discover -s tests` runs 100+ tests (parsers, ingestion fallback, ratings recovery of known truth,
  no-lookahead checks, EV math, end-to-end predict -> log -> grade, CLI empty-state behaviour). They use a synthetic league and hand-built
  API payloads that mirror the documented response shapes.
* **The live endpoints (ESPN, NBA CDN, The Odds API) were not reachable from the environment this was built in**, so the parsers have
  not seen real responses. `python cli.py doctor` is the check: it fetches a real finished game, parses it, and verifies player
  points sum to the team score. If a source changed shape it tells you which step failed, and ingestion falls back to the next source.
* **Accuracy and profit on real data are unknown until you run `train` and then forward-test.** There are no free historical prop lines, so
  `train` reports projection accuracy (vs. recent-form and season-average baselines), not ROI. Beating closing lines is hard; the app logs
  every bet it shows, and `grade` + `report` measure ROI, calibration and closing-line value as results accumulate. After a few hundred
  graded bets `report` suggests a market-blend weight.

## Layout

```
cli.py  services.py  ev_engine.py  predictions.py  config.py
nbadata/     teams, players, warehouse (SQLite), http, ingest, providers/{espn,nba_cdn}
oddsfeed/    the_odds_api, service (refresh/consensus), csv_import, schema (de-vig)
modeling/    frames, ratings, game_model, player_model
tests/       unit + end-to-end tests (tests/synthetic.py is test-only data)
legacy/      previous implementation, retired (see legacy/README.md)
```
