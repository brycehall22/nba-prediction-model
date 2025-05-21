from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import re
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('NBA Odds Scraper')

def scrape_draftkings_nba_odds():
    url = "https://sportsbook.draftkings.com/leagues/basketball/nba"
    
    # Set up Chrome with advanced stealth options for headless mode
    options = Options()
    
    # Use headless mode with anti-detection measures
    options.add_argument("--headless=new")  # New headless implementation
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    # Set a realistic window size
    options.add_argument("--window-size=1920,1080")
    
    # Add realistic user agent
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'
    ]
    options.add_argument(f"user-agent={random.choice(user_agents)}")
    
    # Add additional anti-detection options
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    
    # Set language to avoid international redirects
    options.add_argument("--lang=en-US")
    
    driver = webdriver.Chrome(options=options)
    
    try:
        # Add extra headers and modify navigator properties to avoid detection
        driver.execute_cdp_cmd("Network.setUserAgentOverride", {
            "userAgent": random.choice(user_agents),
            "platform": "Windows NT 10.0; Win64; x64"
        })
        
        # Modify WebDriver properties to avoid detection
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        
        # Load the page
        logger.info("Loading DraftKings NBA odds page...")
        driver.get(url)
        
        # Wait for content to load - specifically for the sportsbook-table
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "sportsbook-table"))
        )
        
        # Extra time for all odds to load
        time.sleep(5)
        
        logger.info("Page loaded, extracting game data...")
        
        # Find all table rows (tr) within the sportsbook-table
        # Each game has a pair of rows (one for each team)
        all_rows = driver.find_elements(By.XPATH, "//table[contains(@class, 'sportsbook-table')]//tr")
        
        logger.info(f"Found {len(all_rows)} table rows")
        
        games_data = []
        current_game = {}
        team_index = 0
        
        # Process each row
        for row in all_rows:
            try:
                # Check if this is a team row by looking for event-cell__name
                team_name_element = row.find_elements(By.XPATH, ".//div[contains(@class, 'event-cell__name-text')]")
                
                if team_name_element:
                    team_name = team_name_element[0].text.strip()
                    logger.info(f"Found team: {team_name}")
                    
                    # Get the spread, spread odds, total, and moneyline for this team
                    # The structure is in separate td elements within the same row
                    
                    # Get spread value and odds
                    spread_elements = row.find_elements(By.XPATH, ".//td[contains(@class, 'sportsbook-table__column-row')][1]//span[contains(@class, 'sportsbook-odds')]")
                    spread_line_elements = row.find_elements(By.XPATH, ".//td[contains(@class, 'sportsbook-table__column-row')][1]//div[contains(@aria-label, '+') or contains(@aria-label, '-')]")
                    
                    spread_value = ""
                    spread_odds = ""
                    
                    if spread_line_elements:
                        aria_label = spread_line_elements[0].get_attribute("aria-label")
                        if aria_label:
                            match = re.search(r'([+-]\d+\.?\d*)', aria_label)
                            if match:
                                spread_value = match.group(1)
                    
                    if spread_elements:
                        spread_odds = spread_elements[0].text.strip()
                    
                    # Get total value
                    total_elements = row.find_elements(By.XPATH, ".//td[contains(@class, 'sportsbook-table__column-row')][2]//div[contains(@aria-label, 'O') or contains(@aria-label, 'U')]")
                    total_odds_elements = row.find_elements(By.XPATH, ".//td[contains(@class, 'sportsbook-table__column-row')][2]//span[contains(@class, 'sportsbook-odds')]")
                    
                    total_indicator = ""
                    total_value = ""
                    total_odds = ""
                    
                    if total_elements:
                        aria_label = total_elements[0].get_attribute("aria-label")
                        if aria_label:
                            if "OVER" in aria_label.upper():
                                total_indicator = "O"
                            elif "UNDER" in aria_label.upper():
                                total_indicator = "U"
                            
                            match = re.search(r'(\d+\.?\d*)', aria_label)
                            if match:
                                total_value = match.group(1)
                    
                    if total_odds_elements:
                        total_odds = total_odds_elements[0].text.strip()
                    
                    # Get moneyline
                    moneyline_elements = row.find_elements(By.XPATH, ".//td[contains(@class, 'sportsbook-table__column-row')][3]//span[contains(@class, 'sportsbook-odds')]")
                    moneyline = ""
                    
                    if moneyline_elements:
                        moneyline = moneyline_elements[0].text.strip()
                    
                    # Based on team index, determine if this is away or home team
                    if team_index % 2 == 0:
                        # Away team (first in pair)
                        current_game = {
                            'Away Team': team_name,
                            'Away Spread': spread_value,
                            'Away Spread Odds': spread_odds,
                            'Away Total': f"{total_indicator} {total_value}" if total_value else "",
                            'Away Total Odds': total_odds,
                            'Away Moneyline': moneyline
                        }
                    else:
                        # Home team (second in pair)
                        current_game['Home Team'] = team_name
                        current_game['Home Spread'] = spread_value
                        current_game['Home Spread Odds'] = spread_odds
                        current_game['Home Total'] = f"{total_indicator} {total_value}" if total_value else ""
                        current_game['Home Total Odds'] = total_odds
                        current_game['Home Moneyline'] = moneyline

                        for key, value in current_game.items():
                            if isinstance(value, str):
                                current_game[key] = value.replace('\u2212', '-')
                            
                        # Format data for output
                        formatted_game = {
                            'Away Team': current_game['Away Team'],
                            'Home Team': current_game['Home Team'],
                            'Away Spread': f"{current_game['Away Spread']} ({current_game['Away Spread Odds']})" if current_game['Away Spread'] and current_game['Away Spread Odds'] else "",
                            'Home Spread': f"{current_game['Home Spread']} ({current_game['Home Spread Odds']})" if current_game['Home Spread'] and current_game['Home Spread Odds'] else "",
                            'Total': f"{current_game['Away Total'].split()[-1] if ' ' in current_game['Away Total'] else ''} (O: {current_game['Away Total Odds']}, U: {current_game['Home Total Odds']})" if current_game['Away Total'] and current_game['Home Total'] else "",
                            'Away Moneyline': current_game['Away Moneyline'],
                            'Home Moneyline': current_game['Home Moneyline']
                        }
                            
                        games_data.append(formatted_game)
                        logger.info(f"Added game: {formatted_game['Away Team']} @ {formatted_game['Home Team']}")
                    
                    team_index += 1
            
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(games_data)
        return df
    
    except Exception as e:
        logger.error(f"Error scraping data: {e}")
        return None
    
    finally:
        driver.quit()

def main():
    logger.info("Starting NBA odds scraper...")
    
    # Scrape the data
    odds_data = scrape_draftkings_nba_odds()
    
    if odds_data is not None and not odds_data.empty:
        # Print the results
        logger.info("\nDraftKings NBA Odds Data:")
        logger.info(f"Found {len(odds_data)} games")
        
        # Save to CSV
        odds_data.to_csv('draftkings_nba_odds.csv', index=False)
        logger.info("\nData saved to 'draftkings_nba_odds.csv'")
    else:
        logger.error("No data retrieved. Possible reasons: incorrect selectors, no games scheduled, or access restrictions.")

if __name__ == "__main__":
    main()