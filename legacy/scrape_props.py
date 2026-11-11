#!/usr/bin/env python3
"""
DraftKings NBA Player Points Props Scraper — v3 for 2025/2026 layout.

Exact DOM per player prop row:

  data-testid="market-template"  (cb-market__template--2-columns-big-cells)
    │
    ├── data-testid="market-label"              ← PLAYER NAME
    │     (cb-market__label cb-market__label-row cb-market__label--text-left)
    │
    ├── data-testid="cb-market-buttons-slider"  ← OVER / UNDER odds buttons
    │     (cb-market-buttons-slider)
    │     Contains buttons with over odds and under odds
    │
    └── data-testid="cb-selection-picker"       ← LINE PICKER (carousel of alt lines)
          ├── button left-arrow
          ├── div cb-selection-picker__selections-wrapper
          │     └── div cb-selection-picker__selections-animation
          │           ├── button selection-0 ... (inactive lines)
          │           ├── button selection--prev
          │           ├── button selection--focused     ← CURRENT ACTIVE LINE
          │           │     ├── span.cb-selection-picker__selection-label  → "20+"
          │           │     ├── span[data-testid="cb-odds-update-arrow"]
          │           │     └── span.cb-selection-picker__selection-odds   → "-130"
          │           ├── button selection--next
          │           └── ...more inactive lines
          └── button right-arrow
"""

import argparse
import logging
import random
import re
import sys
import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from config import UI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scrape_props")

URL = "https://sportsbook.draftkings.com/leagues/basketball/nba?category=player-points&subcategory=points-o%2Fu"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.4 Safari/605.1.15",
]


def _build_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    ua = random.choice(USER_AGENTS)
    opts.add_argument(f"user-agent={ua}")
    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd("Network.setUserAgentOverride", {
        "userAgent": ua, "platform": "Windows NT 10.0; Win64; x64",
    })
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def _norm(text: str) -> str:
    return text.replace("\u2212", "-").replace("\u2013", "-").replace("−", "-").strip()


def _extract_game_title(section) -> str:
    """Get 'AWAY at HOME' from a game section header."""

    # event-nav-link <a> tags
    try:
        links = section.find_elements(By.CSS_SELECTOR, "a[data-testid='lp-nav-link'], a.event-nav-link")
        texts = [_norm(l.text) for l in links if l.text.strip()]
        if len(texts) >= 2:
            return f"{texts[0]} at {texts[1]}"
        for link in links:
            href = link.get_attribute("href") or ""
            from urllib.parse import unquote
            decoded = unquote(unquote(href))
            m = re.search(r"/event/(.+?)[-\s]*@[-\s]*(.+?)/(\d+)", decoded)
            if m:
                def s2n(s):
                    p = s.strip().split("-")
                    return p[0].upper() + " " + " ".join(x.capitalize() for x in p[1:]) if len(p) >= 2 else s.upper()
                return f"{s2n(m.group(1))} at {s2n(m.group(2))}"
    except Exception:
        pass

    # static wrapper / label-parlay text
    for sel in [".cb-market__static-wrapper", "[data-testid='label-parlay']"]:
        try:
            els = section.find_elements(By.CSS_SELECTOR, sel)
            texts = [_norm(e.text) for e in els if e.text.strip()]
            if len(texts) >= 2:
                return f"{texts[0]} at {texts[1]}"
            if texts and len(texts[0]) > 5:
                return texts[0]
        except Exception:
            pass

    # grep section text
    try:
        raw = _norm(section.text[:300])
        m = re.search(r"([A-Z]{2,3}\s+\w+)\s+(?:at|@|vs\.?)\s+([A-Z]{2,3}\s+\w+)", raw, re.IGNORECASE)
        if m:
            return f"{m.group(1)} at {m.group(2)}"
    except Exception:
        pass

    return ""


def _extract_player_name(row) -> str:
    """Extract player name from a market-template row."""

    # Primary: data-testid="market-label"
    label_els = row.find_elements(By.CSS_SELECTOR, "[data-testid='market-label']")
    if label_els:
        # Check links/spans inside label
        for el in label_els[0].find_elements(By.CSS_SELECTOR, "a, span"):
            t = _norm(el.text)
            if t and len(t) > 2 and " " in t and not re.match(r'^[OU\d+\-]', t):
                return t
        # Fallback: first good line from label text
        for line in _norm(label_els[0].text).split("\n"):
            line = line.strip()
            if line and len(line) > 2 and " " in line and not re.match(r'^[OU\d+\-]', line):
                return line

    return ""


def _extract_line_from_picker(row) -> str:
    """
    Get the points line from the focused selection in the picker carousel.

    The focused button has class 'cb-selection-picker__selection--focused'
    and contains:
      <span class="cb-selection-picker__selection-label">20+</span>
      <span class="cb-selection-picker__selection-odds">-130</span>
    """
    # Find the focused selection button
    focused = row.find_elements(
        By.CSS_SELECTOR,
        "button.cb-selection-picker__selection--focused, "
        "button[class*='selection--focused']"
    )
    if focused:
        # Get the label span (the line value like "20+" or "27.5")
        label_span = focused[0].find_elements(
            By.CSS_SELECTOR,
            ".cb-selection-picker__selection-label, "
            "span[class*='selection-label']"
        )
        if label_span:
            raw = _norm(label_span[0].text)
            # Strip trailing '+' (DK shows "20+" meaning over 20)
            raw = raw.rstrip("+")
            # Extract the number
            m = re.search(r'(\d+\.?\d*)', raw)
            if m:
                return m.group(1)

    # Fallback: any selection-label visible in the picker
    all_labels = row.find_elements(
        By.CSS_SELECTOR,
        ".cb-selection-picker__selection-label"
    )
    for lbl in all_labels:
        if lbl.is_displayed():
            raw = _norm(lbl.text).rstrip("+")
            m = re.search(r'(\d+\.?\d*)', raw)
            if m:
                return m.group(1)

    return ""


def _extract_odds_from_buttons(row) -> tuple:
    """
    Get over and under odds from the cb-market-buttons-slider.

    The slider contains buttons for over and under.
    Each button text typically has the odds like "-105" or "+110".
    The first button is usually Over, second is Under.
    """
    over_odds = ""
    under_odds = ""

    # Find buttons in the slider
    slider_btns = row.find_elements(
        By.CSS_SELECTOR,
        "[data-testid='cb-market-buttons-slider'] button, "
        ".cb-market-buttons-slider button"
    )

    if len(slider_btns) >= 2:
        over_raw = _norm(slider_btns[0].text)
        under_raw = _norm(slider_btns[1].text)

        # Extract odds number from each button
        m = re.search(r'([+-]\d+)', over_raw)
        if m:
            over_odds = m.group(1)
        m = re.search(r'([+-]\d+)', under_raw)
        if m:
            under_odds = m.group(1)

    # If slider didn't work, try component-builder-market-button
    if not over_odds:
        market_btns = row.find_elements(
            By.CSS_SELECTOR,
            "button[data-testid='component-builder-market-button']"
        )
        odds_found = []
        for btn in market_btns:
            t = _norm(btn.text)
            m = re.search(r'([+-]\d+)', t)
            if m:
                odds_found.append(m.group(1))
        if len(odds_found) >= 2:
            over_odds = odds_found[0]
            under_odds = odds_found[1]
        elif len(odds_found) == 1:
            over_odds = odds_found[0]

    return over_odds, under_odds


def scrape_props(headless: bool = True, max_retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, max_retries + 1):
        driver = None
        try:
            logger.info(f"Attempt {attempt}/{max_retries}: loading props page")
            driver = _build_driver(headless)
            driver.get(URL)

            # Wait for page structure
            for sel in ["[data-testid='marketboard']", "[data-testid='non-collapsible-wrapper']"]:
                try:
                    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
                    logger.info(f"Loaded (matched: {sel})")
                    break
                except Exception:
                    continue

            time.sleep(6)
            logger.info(f"Page: {len(driver.page_source)} chars")

            # ── Game sections ─────────────────────────────────────────
            sections = driver.find_elements(By.CSS_SELECTOR, "[data-testid='non-collapsible-wrapper']")
            if not sections:
                sections = driver.find_elements(By.CSS_SELECTOR, ".cb-non-collapsible")
            logger.info(f"{len(sections)} game sections")

            props = []
            seen = set()

            for section in sections:
                game_title = _extract_game_title(section)
                if not game_title:
                    continue
                logger.info(f"  Game: {game_title}")

                # Player rows
                player_rows = section.find_elements(By.CSS_SELECTOR, "[data-testid='market-template']")
                if not player_rows:
                    player_rows = section.find_elements(By.CSS_SELECTOR, "[class*='cb-market__template']")
                logger.info(f"    {len(player_rows)} player rows")

                for row in player_rows:
                    try:
                        # ── Player name ───────────────────────────
                        player_name = _extract_player_name(row)
                        if not player_name:
                            continue

                        key = f"{game_title}|{player_name}"
                        if key in seen:
                            continue

                        # ── Points line from selection picker ─────
                        points_line = _extract_line_from_picker(row)

                        # ── Over/Under odds from buttons slider ───
                        over_odds, under_odds = _extract_odds_from_buttons(row)

                        # ── Fallback: brute-force from row text ───
                        if not points_line:
                            rtxt = _norm(row.text)
                            m = re.search(r'(\d+\.5)', rtxt)
                            if not m:
                                m = re.search(r'(\d+)\+', rtxt)  # "20+" format
                            if m:
                                points_line = m.group(1)

                        if not over_odds or not under_odds:
                            rtxt = _norm(row.text)
                            odds_all = re.findall(r'([+-]\d{3})', rtxt)
                            if len(odds_all) >= 2:
                                over_odds = over_odds or odds_all[0]
                                under_odds = under_odds or odds_all[1]

                        # ── Store result ──────────────────────────
                        if points_line and (over_odds or under_odds):
                            props.append({
                                "Game": game_title,
                                "Player": player_name,
                                "Points Line": points_line,
                                "Over Odds": over_odds or "-110",
                                "Under Odds": under_odds or "-110",
                            })
                            seen.add(key)
                            logger.info(f"      ✓ {player_name}: {points_line}  O:{over_odds}  U:{under_odds}")
                        else:
                            logger.debug(f"      ✗ {player_name}: line={points_line} over={over_odds} under={under_odds}")

                    except Exception as e:
                        logger.debug(f"      Row error: {e}")

            if props:
                logger.info(f"Total: {len(props)} player props")
                return pd.DataFrame(props)

            # ── Debug dump ────────────────────────────────────────────
            logger.warning("No props found. Dumping debug info...")
            try:
                # Show data-testid values present
                tids = set()
                for el in driver.find_elements(By.XPATH, "//*[@data-testid]")[:200]:
                    tids.add(el.get_attribute("data-testid") or "")
                logger.info(f"  data-testid values: {sorted(tids)[:30]}")

                # Show market-template count and first one's HTML
                templates = driver.find_elements(By.CSS_SELECTOR, "[data-testid='market-template']")
                logger.info(f"  market-template count: {len(templates)}")
                if templates:
                    html = templates[0].get_attribute("outerHTML")[:2000]
                    logger.info(f"  First template HTML:\n{html}")

                # Show what the focused selection looks like
                focused = driver.find_elements(By.CSS_SELECTOR, "[class*='selection--focused']")
                logger.info(f"  Focused selections: {len(focused)}")
                if focused:
                    logger.info(f"  First focused HTML: {focused[0].get_attribute('outerHTML')[:500]}")

                # Show buttons-slider content
                sliders = driver.find_elements(By.CSS_SELECTOR, "[data-testid='cb-market-buttons-slider']")
                logger.info(f"  Button sliders: {len(sliders)}")
                if sliders:
                    logger.info(f"  First slider HTML: {sliders[0].get_attribute('outerHTML')[:500]}")
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait = 5 * attempt + random.uniform(0, 3)
                logger.info(f"Retrying in {wait:.0f}s...")
                time.sleep(wait)
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    logger.error("All attempts failed")
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Scrape DraftKings player points props")
    parser.add_argument("--output", "-o", default=UI.PROPS_FILE)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    df = scrape_props(headless=not args.visible, max_retries=args.retries)
    if df.empty:
        logger.error("No props scraped.")
        sys.exit(1)

    df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(df)} props to {args.output}")

    games = df["Game"].unique()
    print(f"\n{'='*60}")
    print(f"Player Points Props — {len(df)} props, {len(games)} games")
    print(f"{'='*60}")
    for game in games:
        gdf = df[df["Game"] == game]
        print(f"\n  {game} ({len(gdf)} players):")
        for _, r in gdf.head(5).iterrows():
            print(f"    {r['Player']:25s}  {r['Points Line']:>5s}  O:{r['Over Odds']:>5s}  U:{r['Under Odds']:>5s}")
        if len(gdf) > 5:
            print(f"    ... +{len(gdf)-5} more")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
