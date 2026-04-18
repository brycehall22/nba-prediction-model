#!/usr/bin/env python3
"""
DraftKings NBA Game Odds Scraper — updated for 2025/2026 site redesign.

The site now uses:
  - cb-market__template--2-columns   for each game row
  - event-nav-link <a> tags          for team names (href contains matchup)
  - cb-market__button                for odds cells (6 per row: spread, total, ML × 2 teams)

Usage:
    python scrape_odds.py
    python scrape_odds.py --visible          # show browser
    python scrape_odds.py --output my.csv
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
logger = logging.getLogger("scrape_odds")

URL = "https://sportsbook.draftkings.com/leagues/basketball/nba"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


def _build_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--lang=en-US")
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


def _normalize(text: str) -> str:
    """Normalize unicode minus signs and whitespace."""
    return text.replace("\u2212", "-").replace("−", "-").strip()


def _parse_team_names_from_href(href: str) -> tuple:
    """
    Parse team names from an event-nav-link href.
    Example href: /event/bkn-nets-%2540-phi-76ers/33800320
    %2540 = double-encoded '@' → '@'
    Returns (away_team_slug, home_team_slug) or None.
    """
    from urllib.parse import unquote
    decoded = unquote(unquote(href))  # double decode for %2540 → %40 → @
    # Pattern: /event/{away}-@-{home}/{id}  or  {away} @ {home}
    m = re.search(r"/event/(.+?)[-\s]*@[-\s]*(.+?)/(\d+)", decoded)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def _slug_to_display(slug: str) -> str:
    """Convert 'bkn-nets' to 'BKN Nets'."""
    parts = slug.split("-")
    if len(parts) >= 2:
        abbrev = parts[0].upper()
        mascot = " ".join(p.capitalize() for p in parts[1:])
        return f"{abbrev} {mascot}"
    return slug.upper()


def scrape_odds(headless: bool = True, max_retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, max_retries + 1):
        driver = None
        try:
            logger.info(f"Attempt {attempt}/{max_retries}: loading {URL}")
            driver = _build_driver(headless)
            driver.get(URL)

            # Wait for the new marketboard structure
            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='marketboard']"))
            )
            # Extra time for odds buttons to render
            time.sleep(6)

            # Debug: log what we found
            page_source_len = len(driver.page_source)
            logger.info(f"Page loaded ({page_source_len} chars)")

            # ── Find all game rows ────────────────────────────────────
            # Each game is a div with class containing 'cb-market__template--2-columns'
            game_rows = driver.find_elements(
                By.CSS_SELECTOR, "div[data-testid='market-template']"
            )
            if not game_rows:
                # Fallback: try the class name directly
                game_rows = driver.find_elements(
                    By.CSS_SELECTOR, ".cb-market__template--2-columns"
                )
            if not game_rows:
                # Broader fallback
                game_rows = driver.find_elements(
                    By.CSS_SELECTOR, "[class*='cb-market__template']"
                )

            logger.info(f"Found {len(game_rows)} game rows")

            games = []

            for row in game_rows:
                try:
                    # ── Extract team names ────────────────────────────
                    # Method 1: from event-nav-link hrefs
                    links = row.find_elements(By.CSS_SELECTOR, "a.event-nav-link, a[data-testid='lp-nav-link']")
                    away_name = ""
                    home_name = ""

                    if links:
                        # First link typically has the matchup in the href
                        href = links[0].get_attribute("href") or ""
                        parsed = _parse_team_names_from_href(href)
                        if parsed:
                            away_name = _slug_to_display(parsed[0])
                            home_name = _slug_to_display(parsed[1])

                    # Method 2: fallback to link text content
                    if not away_name and links:
                        link_texts = [_normalize(l.text) for l in links if l.text.strip()]
                        if len(link_texts) >= 2:
                            away_name = link_texts[0]
                            home_name = link_texts[1]
                        elif len(link_texts) == 1:
                            # Sometimes both teams are in one link separated by ' @ ' or newline
                            parts = re.split(r'\s*[@\n]\s*', link_texts[0])
                            if len(parts) >= 2:
                                away_name = parts[0]
                                home_name = parts[1]

                    # Method 3: try label elements
                    if not away_name:
                        labels = row.find_elements(
                            By.CSS_SELECTOR,
                            "[data-testid='label-parlay'], .cb-market__label, [class*='label']"
                        )
                        label_texts = [_normalize(l.text) for l in labels if l.text.strip()]
                        if len(label_texts) >= 2:
                            away_name = label_texts[0]
                            home_name = label_texts[1]

                    if not away_name or not home_name:
                        logger.debug(f"Could not extract team names from row, skipping")
                        continue

                    # ── Extract odds buttons ──────────────────────────
                    # There should be 6 buttons per game row:
                    # [away_spread, home_spread, away_total(O), home_total(U), away_ML, home_ML]
                    buttons = row.find_elements(
                        By.CSS_SELECTOR,
                        "button[data-testid='component-builder-market-button'], button.cb-market__button"
                    )
                    if not buttons:
                        buttons = row.find_elements(By.CSS_SELECTOR, "button[class*='cb-market__button']")

                    logger.info(f"  {away_name} @ {home_name} — {len(buttons)} buttons")

                    # Parse button text — each button contains the line and odds
                    # e.g., "+7.5\n-110" or "O 206.5\n-110" or "+250"
                    button_texts = []
                    for btn in buttons:
                        raw = _normalize(btn.text)
                        button_texts.append(raw)

                    # We expect 6 buttons: spread×2, total×2, ML×2
                    # But DK sometimes has different layouts, so be flexible
                    away_spread = ""
                    home_spread = ""
                    total_str = ""
                    away_ml = ""
                    home_ml = ""

                    if len(button_texts) >= 6:
                        away_spread = button_texts[0]
                        home_spread = button_texts[1]
                        # Total buttons contain O/U indicator
                        over_text = button_texts[2]
                        under_text = button_texts[3]
                        away_ml = button_texts[4]
                        home_ml = button_texts[5]

                        # Parse total from O/U buttons
                        # Format: "O 206.5\n-110" or "206.5\n-110"
                        total_val = ""
                        over_odds = ""
                        under_odds = ""

                        over_match = re.search(r"(\d+\.?\d*)", over_text)
                        if over_match:
                            total_val = over_match.group(1)
                        over_odds_match = re.search(r"([+-]\d+)(?:\s*$|\n)", over_text)
                        if over_odds_match:
                            over_odds = over_odds_match.group(1)
                        else:
                            # Odds might be on the second line
                            over_lines = over_text.split("\n")
                            if len(over_lines) >= 2:
                                over_odds = over_lines[-1].strip()

                        under_lines = under_text.split("\n")
                        if len(under_lines) >= 2:
                            under_odds = under_lines[-1].strip()
                        elif re.search(r"([+-]\d+)", under_text):
                            under_odds = re.search(r"([+-]\d+)", under_text).group(1)

                        if total_val:
                            total_str = f"{total_val} (O: {over_odds}, U: {under_odds})"

                        # Clean ML values (extract just odds number)
                        away_ml_match = re.search(r"([+-]\d+)", away_ml)
                        home_ml_match = re.search(r"([+-]\d+)", home_ml)
                        away_ml = away_ml_match.group(1) if away_ml_match else away_ml
                        home_ml = home_ml_match.group(1) if home_ml_match else home_ml

                    elif len(button_texts) >= 2:
                        # Minimal data — just MLs or partial
                        away_ml = button_texts[0]
                        home_ml = button_texts[1]

                    # Format spread strings
                    def _format_spread(raw):
                        lines = raw.split("\n")
                        if len(lines) >= 2:
                            return f"{lines[0].strip()} ({lines[1].strip()})"
                        return raw

                    games.append({
                        "Away Team": away_name,
                        "Home Team": home_name,
                        "Away Spread": _format_spread(away_spread) if away_spread else "",
                        "Home Spread": _format_spread(home_spread) if home_spread else "",
                        "Total": total_str,
                        "Away Moneyline": away_ml,
                        "Home Moneyline": home_ml,
                    })

                except Exception as e:
                    logger.warning(f"Error parsing game row: {e}")
                    continue

            if games:
                logger.info(f"Successfully scraped {len(games)} games")
                return pd.DataFrame(games)
            else:
                logger.warning("No games extracted from page")

                # Debug: dump some page structure
                try:
                    body = driver.find_element(By.TAG_NAME, "body")
                    classes = set()
                    for el in driver.find_elements(By.XPATH, "//*[@class]")[:200]:
                        for c in (el.get_attribute("class") or "").split():
                            if "market" in c.lower() or "event" in c.lower() or "odds" in c.lower():
                                classes.add(c)
                    logger.info(f"Relevant CSS classes found: {sorted(classes)[:30]}")
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

    logger.error("All scrape attempts failed")
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Scrape DraftKings NBA odds")
    parser.add_argument("--output", "-o", default=UI.ODDS_FILE, help="Output CSV path")
    parser.add_argument("--visible", action="store_true", help="Show browser window")
    parser.add_argument("--retries", type=int, default=3, help="Max retry attempts")
    args = parser.parse_args()

    df = scrape_odds(headless=not args.visible, max_retries=args.retries)

    if df.empty:
        logger.error("No data scraped. Check if games are scheduled and DraftKings is accessible.")
        sys.exit(1)

    df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(df)} games to {args.output}")

    print(f"\n{'='*60}")
    print(f"DraftKings NBA Odds — {len(df)} games")
    print(f"{'='*60}")
    for _, row in df.iterrows():
        print(f"  {row['Away Team']:25s} @ {row['Home Team']:25s}")
        print(f"    Spread: {row['Home Spread']:20s}  Total: {row['Total']}")
        print(f"    ML: {row['Away Moneyline']} / {row['Home Moneyline']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
