from selenium import webdriver
from selenium.webdriver.chrome.options import Options
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
logger = logging.getLogger('NBA Player Props Scraper')

def scrape_draftkings_player_points_props():
    url = "https://sportsbook.draftkings.com/leagues/basketball/nba?category=player-points&subcategory=points-o%2Fu"
    
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
    ]
    options.add_argument(f"user-agent={random.choice(user_agents)}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.execute_cdp_cmd("Network.setUserAgentOverride", {
            "userAgent": random.choice(user_agents),
            "platform": "Windows NT 10.0; Win64; x64"
        })
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        logger.info("Loading DraftKings NBA player points props page...")
        driver.get(url)
        
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "tbody"))
        )
        time.sleep(5)
        
        logger.info("Page loaded, extracting player props data...")
        
        game_sections = driver.find_elements(By.XPATH, "//div[contains(@class, 'sportsbook-event-accordion')]")
        logger.info(f"Found {len(game_sections)} game sections")
        
        props_data = []
        seen_players = set()  # To track unique player-game combinations
        
        for section_index, section in enumerate(game_sections):
            try:
                # Extract team names from the accordion title
                try:
                    title_wrapper = section.find_element(By.XPATH, ".//div[contains(@class, 'sportsbook-event-accordion__title')]")
                    team_elements = title_wrapper.find_elements(By.XPATH, ".//span[contains(@class, 'event-cell__name')]")
                    if len(team_elements) >= 2:
                        away_team = team_elements[0].text.strip()
                        home_team = team_elements[1].text.strip()
                    else:
                        raise Exception("Not enough team elements found in event-cell__name")
                except Exception as e:
                    logger.warning(f"Failed to extract team names directly: {str(e)}")
                    game_header = title_wrapper.text.strip()
                    game_header = re.sub(r'\s+', ' ', game_header)
                    teams_match = re.search(r'(.+?)(?: at | @ | v | vs\.? | VS\.? )(.+)', game_header, re.IGNORECASE)
                    if teams_match:
                        away_team = teams_match.group(1).strip()
                        home_team = teams_match.group(2).strip()
                    else:
                        away_team = "Unknown"
                        home_team = "Unknown"
                        logger.warning(f"Could not parse teams from: {game_header}")
                
                logger.info(f"Processing game {section_index + 1}: {away_team} at {home_team}")
                
                # Find the first relevant table body (avoid duplicates)
                table_bodies = section.find_elements(By.XPATH, ".//tbody")
                if not table_bodies:
                    logger.warning(f"No table body found for game {section_index + 1}")
                    continue
                
                # Process only the first tbody to avoid duplicates
                table_body = table_bodies[0]
                rows = table_body.find_elements(By.XPATH, ".//tr")
                logger.info(f"Found {len(rows)} player rows in this game")
                
                for row in rows:
                    try:
                        player_name = row.find_element(By.XPATH, ".//th//span").text.strip()
                        # Create a unique key for this player in this game
                        player_game_key = f"{away_team} at {home_team} - {player_name}"
                        player_game_key_2 = f"{away_team} at {home_team}"
                        if player_game_key in seen_players:
                            logger.info(f"Skipping duplicate entry for {player_name} in {away_team} at {home_team}")
                            continue

                        if player_game_key_2 == "Unknown at Unknown":
                            logger.info(f"Skipping duplicate entry for {player_name} in {away_team} at {home_team}")
                            continue
                        
                        cells = row.find_elements(By.XPATH, ".//td")
                        if len(cells) >= 2:
                            over_line = cells[0].find_element(By.XPATH, ".//span[contains(@class, 'line')]").text.replace("O ", "").strip()
                            over_odds = cells[0].find_element(By.XPATH, ".//span[contains(@class, 'odds')]").text.strip()
                            under_line = cells[1].find_element(By.XPATH, ".//span[contains(@class, 'line')]").text.replace("U ", "").strip()
                            under_odds = cells[1].find_element(By.XPATH, ".//span[contains(@class, 'odds')]").text.strip()
                            
                            prop_data = {
                                'Game': f"{away_team} at {home_team}",
                                'Player': player_name,
                                'Points Line': over_line,
                                'Over Odds': over_odds,
                                'Under Odds': under_odds
                            }
                            props_data.append(prop_data)
                            seen_players.add(player_game_key)
                            logger.info(f"Added prop: {player_name} O/U {over_line}")
                    except Exception as e:
                        logger.error(f"Error processing row: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Error processing game section {section_index + 1}: {str(e)}")
                continue
        
        df = pd.DataFrame(props_data)
        return df
    
    except Exception as e:
        logger.error(f"Error scraping data: {e}")
        return None
    
    finally:
        driver.quit()

def main():
    logger.info("Starting NBA player points props scraper...")
    props_data = scrape_draftkings_player_points_props()
    
    if props_data is not None and not props_data.empty:
        logger.info(f"Found {len(props_data)} player props")
        logger.info("\nSample data (first 5 rows):")
        logger.info(props_data.head(5))
        props_data.to_csv('draftkings_player_points_props.csv', index=False)
        logger.info("\nData saved to 'draftkings_player_points_props.csv'")
    else:
        logger.error("No data retrieved. Check selectors or network access.")

if __name__ == "__main__":
    main()