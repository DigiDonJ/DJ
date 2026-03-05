"""
RacingPost results scraper — yesterday's race finishing positions.

Ports the user's R results-scraping logic to Python, fixing:
- Empty DataFrame initialisation
- Non-numeric position values (PU, F, UR, DSQ, etc.)
- append=TRUE duplication issues
"""

import time
import logging
from datetime import date, timedelta
from typing import Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URI = "https://www.racingpost.com"
THROTTLE_SECONDS = 5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}

# Mapping of non-numeric finishing codes to position values
# (used for completeness; these horses are excluded from "won=1" labelling)
NON_FINISHER_CODES = {"PU", "F", "UR", "RO", "BD", "SU", "REF", "DSQ", "WO", "NR", "CO", "LFT"}


def _get(url: str, client: Optional[httpx.Client] = None) -> BeautifulSoup:
    if client:
        resp = client.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
    else:
        resp = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def _parse_position(pos_str: str) -> Optional[int]:
    """
    Parse a finishing position string.
    Returns int position for finishers, None for non-finishers (PU, F, UR, etc.).
    """
    raw = str(pos_str).strip().upper()
    # Take only first 2 chars (handles "1st", "2nd", "3rd" etc.)
    raw = raw[:2].rstrip("STNDRDTH").strip()
    if raw in NON_FINISHER_CODES or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def scrape_results_for_date(target_date: date) -> pd.DataFrame:
    """
    Scrape race results for a specific date.

    Returns DataFrame with columns:
        race_id, result_pos, horse_num, horse_name, race_date, race_time, racecourse
    """
    date_str = target_date.strftime("%Y-%m-%d")
    url = f"{BASE_URI}/results/{date_str}/time-order/"

    try:
        soup = _get(url)
    except httpx.HTTPStatusError as e:
        logger.warning(f"Could not fetch results for {date_str}: {e}")
        return pd.DataFrame()

    # Collect result page links
    links = []
    for el in soup.select('[class*="rp-timeView__timePanel"] [href]'):
        href = el.get("href", "")
        if href:
            links.append(f"{BASE_URI}{href}" if not href.startswith("http") else href)

    # Fallback: try direct xpath equivalent
    if not links:
        for el in soup.find_all(attrs={"href": True}):
            href = el.get("href", "")
            if "/results/" in href and date_str in href:
                full = f"{BASE_URI}{href}" if not href.startswith("http") else href
                if full not in links:
                    links.append(full)

    logger.info(f"Found {len(links)} result pages for {date_str}")

    all_records = []
    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        for i, result_url in enumerate(links):
            if i > 0:
                time.sleep(THROTTLE_SECONDS)
            try:
                records = _scrape_result_page(result_url, client=client)
                all_records.extend(records)
            except Exception as exc:
                logger.warning(f"Failed to scrape result page {result_url}: {exc}")

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    return df


def _scrape_result_page(url: str, client: Optional[httpx.Client] = None) -> list:
    """Scrape a single race result page, returning list of horse dicts."""
    soup = _get(url, client=client)

    main = soup.select_one("main.rp-resultsWrapper__content, main[class*='rp-results']")
    if main is None:
        main = soup

    race_id = main.get("data-analytics-race-id", "") if hasattr(main, "get") else ""
    race_date = main.get("data-analytics-race-date", "") if hasattr(main, "get") else ""
    race_time = main.get("data-analytics-race-time", "") if hasattr(main, "get") else ""
    racecourse = main.get("data-analytics-coursename", "") if hasattr(main, "get") else ""

    # Try selecting from soup directly if main is soup
    if not race_id:
        meta_el = soup.select_one("main[data-analytics-race-id]")
        if meta_el:
            race_id = meta_el.get("data-analytics-race-id", "")
            race_date = meta_el.get("data-analytics-race-date", "")
            race_time = meta_el.get("data-analytics-race-time", "")
            racecourse = meta_el.get("data-analytics-coursename", "")

    # Positions
    position_els = soup.select(
        '[data-test-selector="text-horsePosition"], '
        '.rp-horseTable__pos__numWrapper span[data-test-selector]'
    )
    positions = [_parse_position(el.get_text(strip=True)) for el in position_els]

    # Horse numbers (saddle cloth)
    horse_num_els = soup.select(".rp-horseTable__saddleClothNo")
    horse_nums = []
    for el in horse_num_els:
        try:
            horse_nums.append(int(el.get_text(strip=True)))
        except ValueError:
            horse_nums.append(None)

    # Horse names
    horse_name_els = soup.select(
        '[data-test-selector="link-horseName"], '
        '.rp-horseTable__horse a[data-test-selector]'
    )
    horse_names = [el.get_text(strip=True) for el in horse_name_els]

    n = len(horse_names)
    if n == 0:
        return []

    def _pad(lst, length, default=None):
        return list(lst) + [default] * max(0, length - len(lst))

    positions = _pad(positions, n)
    horse_nums = _pad(horse_nums, n)

    records = []
    for i in range(n):
        records.append({
            "race_id": race_id,
            "result_pos": positions[i],
            "horse_num": horse_nums[i],
            "horse_name": horse_names[i],
            "race_date": race_date,
            "race_time": race_time,
            "racecourse": racecourse,
        })

    return records


def scrape_yesterday_results() -> pd.DataFrame:
    """
    Convenience function: scrape yesterday's results.

    Returns DataFrame with columns:
        race_id, result_pos, horse_num, horse_name, race_date, race_time, racecourse
    """
    yesterday = date.today() - timedelta(days=1)
    return scrape_results_for_date(yesterday)


def save_results(df: pd.DataFrame, path: str = "data/output/results.csv") -> None:
    """Save results to CSV (overwrite — no append to avoid duplicates)."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} result rows to {path}")
