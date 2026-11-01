# Legacy code (kept for reference, no longer used)

These files belong to the previous implementation and are not imported by the current code:

| File | Why it was retired |
|---|---|
| `scrape_odds.py`, `scrape_props.py` | Selenium/HTML scrapers for DraftKings. Page markup changes broke them, and they were already writing misaligned columns. Replaced by `oddsfeed/` (JSON API). |
| `data_collector.py` | `nba_api` / stats.nba.com calls: ~40 slow requests per prediction, frequent read timeouts. Replaced by `nbadata/` (local warehouse + ESPN / NBA CDN). |
| `model_trainer.py`, `train.py` | Used `cross_val_score(fit_params=...)`, which current scikit-learn rejects; the app never loaded the trained models anyway. Replaced by `modeling/` + `cli.py train`. |
| `predictor.py`, `betting_integration.py` | An older generation that imported modules deleted in the last commit. |
| `betting_data.py` | CSV reader for the scraper output. Replaced by `oddsfeed/`. |
| `enhanced_models/` | Models trained by the old pipeline (different features, xgboost objects). Not loadable by the new code. |
| `draftkings_*.csv` | Sample scraper output (misaligned columns). |
