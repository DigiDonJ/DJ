"""
RacingPost racecard scraper.

Ports the user's R rvest scraping logic to Python (httpx + BeautifulSoup),
preserving the same XPath/CSS selectors and 5-second throttle between requests.
"""

import time
import logging
from datetime import date
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


def _get(url: str, client: Optional[httpx.Client] = None) -> BeautifulSoup:
    """Fetch a URL and return parsed BeautifulSoup."""
    if client:
        resp = client.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
    else:
        resp = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
    if resp.status_code == 403:
        raise PermissionError(
            "RacingPost blocked the request (403). "
            "Try again later or check your IP is not rate-limited."
        )
    if resp.status_code == 401:
        raise PermissionError("RacingPost requires a login to access this page.")
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def _safe_numeric(value: str) -> Optional[float]:
    """Convert string to float, returning None if not parseable."""
    try:
        return float(str(value).strip().replace(",", "").replace("£", "").replace("€", ""))
    except (ValueError, TypeError):
        return None


def scrape_time_order() -> pd.DataFrame:
    """
    Scrape today's race list from the RacingPost time-order page.

    Returns a DataFrame with columns:
        race_id, racecourse, race_date, race_time, num_runners, meeting_url
    """
    url = f"{BASE_URI}/racecards/time-order"
    soup = _get(url)

    # Try multiple selector patterns — RacingPost has changed class names over time
    items = (
        soup.select(".RC-meetingItem__link") or
        soup.select("a[data-race-id]") or
        soup.select(".rp-timeView__timePanel a[href*='/racecards/']") or
        soup.select("a[href*='/racecards/'][href*='-']")
    )

    records = []
    for item in items:
        race_id = (
            item.get("data-race-id") or
            item.get("data-analytics-race-id") or
            ""
        )
        racecourse = (
            item.get("data-racecourse") or
            item.get("data-analytics-coursename") or
            item.get_text(strip=True)[:30] or
            ""
        )
        race_date = item.get("data-race-date") or item.get("data-analytics-race-date") or ""
        race_time = item.get("data-race-time") or item.get("data-analytics-race-time") or ""
        href = item.get("href", "")
        meeting_url = f"{BASE_URI}{href}" if href and not href.startswith("http") else href

        # Extract race_id from URL if not in attributes (e.g. /racecards/2024-05-22/ascot/123456)
        if not race_id and "/racecards/" in href:
            parts = href.rstrip("/").split("/")
            if parts:
                race_id = parts[-1]

        num_el = item.select_one(".RC-meetingItem__numberOfRunners, .rp-timeView__runners")
        num_runners = num_el.get_text(strip=True) if num_el else ""

        if meeting_url:
            records.append({
                "race_id": race_id,
                "racecourse": racecourse,
                "race_date": race_date,
                "race_time": race_time,
                "num_runners": num_runners,
                "meeting_url": meeting_url,
            })

    if not records:
        logger.warning("No races found on time-order page. Page structure may have changed.")
        return pd.DataFrame(columns=["race_id", "racecourse", "race_date", "race_time", "num_runners", "meeting_url"])

    df = pd.DataFrame(records)
    # Drop rows without a URL
    df = df[df["meeting_url"].str.strip().astype(bool)]
    df = df.reset_index(drop=True)
    logger.info(f"Found {len(df)} races in time order.")
    return df


def scrape_race_card(url: str, client: Optional[httpx.Client] = None) -> pd.DataFrame:
    """
    Scrape one race card page and return a DataFrame of horses.

    Columns:
        racecourse, race_id, race_date, race_time, horse_no, horse_name,
        price_money, num_runners, race_terms, going, OR, TS, RPR,
        Flag1, Flag2, Flag3, Flag4, Flag5,
        draw, past_performance, last_run, horse_age, weight,
        trainer_RFT, jockey_allowance
    """
    soup = _get(url, client=client)
    main = soup.select_one("main.js-RC-mainContent, main.RC-content-wrapper, main[class*='RC-']")
    if main is None:
        main = soup

    # --- Race-level metadata ---
    sections = main.select("section[data-card-race-id]") if main != soup else soup.select("section[data-card-race-id]")

    if not sections:
        logger.warning(f"No race sections found at {url}")
        return pd.DataFrame()

    all_horses = []

    for section in sections:
        race_id = section.get("data-card-race-id", "")
        racecourse = section.get("data-card-coursename", "")
        race_date = section.get("data-card-race-date", "")
        race_time = section.get("data-card-race-time", "")

        # Prize money
        price_el = section.select_one('[data-test-selector="RC-headerBox__winner"] .RC-headerBox__infoRow__content')
        price_money = _safe_numeric(price_el.get_text(strip=True)) if price_el else None

        # Number of runners
        runners_el = section.select_one('[data-test-selector="RC-headerBox__runners"] .RC-headerBox__infoRow__content')
        num_runners = runners_el.get_text(strip=True) if runners_el else ""

        # Going
        going_el = section.select_one('[data-test-selector="RC-headerBox__going"] .RC-headerBox__infoRow__content')
        going = going_el.get_text(strip=True) if going_el else ""

        # Race terms
        terms_el = section.select_one('[data-test-selector="RC-headerBox__terms"]')
        race_terms = terms_el.get_text(strip=True) if terms_el else ""

        # --- Horse-level data ---
        horse_nos = [
            span.get("data-order-no", "")
            for span in section.select(".RC-runnerNumber span[data-order-no]")
        ]
        draws = [
            span.get("data-order-draw", "")
            for span in section.select(".RC-runnerNumber span[data-order-draw]")
        ]
        horse_names = [
            a.get_text(strip=True)
            for a in section.select('a.RC-runnerName, a[class*="RC-runnerName"]')
        ]
        past_performances = [
            el.get_text(strip=True)
            for el in section.select(".RC-runnerInfo__form")
        ]
        last_runs = [
            _safe_numeric(el.get_text(strip=True))
            for el in section.select(".RC-runnerStats__lastRun")
        ]
        horse_ages = [
            _safe_numeric(el.get_text(strip=True))
            for el in section.select(".RC-runnerAge")
        ]
        weights = [
            _safe_numeric(span.get("data-order-wgt", ""))
            for span in section.select(".RC-runnerWgt__carried[data-order-wgt]")
        ]
        trainer_rfts = [
            el.get_text(strip=True)
            for el in section.select('[data-test-selector="RC-cardPage-runnerTrainer-rtf"]')
        ]
        jockey_allowances = [
            el.get_text(strip=True)
            for el in section.select('[data-test-selector="RC-cardPage-runnerJockey-allowance"]')
        ]
        ors = [_safe_numeric(el.get_text(strip=True)) for el in section.select(".RC-runnerOr")]
        tss = [_safe_numeric(el.get_text(strip=True)) for el in section.select(".RC-runnerTs")]
        rprs = [_safe_numeric(el.get_text(strip=True)) for el in section.select(".RC-runnerRpr")]

        n = len(horse_names)
        if n == 0:
            continue

        def _pad(lst, length, default=None):
            return list(lst) + [default] * max(0, length - len(lst))

        horse_nos = _pad(horse_nos, n, "")
        draws = _pad(draws, n, "0")
        past_performances = _pad(past_performances, n, "")
        last_runs = _pad(last_runs, n)
        horse_ages = _pad(horse_ages, n)
        weights = _pad(weights, n)
        trainer_rfts = _pad(trainer_rfts, n)
        jockey_allowances = _pad(jockey_allowances, n)
        ors = _pad(ors, n)
        tss = _pad(tss, n)
        rprs = _pad(rprs, n)

        for i in range(n):
            or_val = ors[i]
            ts_val = tss[i]
            rpr_val = rprs[i]
            w_val = weights[i]

            flag1 = flag2 = flag3 = flag4 = flag5 = None
            if None not in (ts_val, rpr_val, or_val):
                flag1 = ts_val + rpr_val - or_val
                if w_val is not None:
                    flag2 = ts_val - w_val
                    flag3 = rpr_val - w_val
                    flag4 = (flag1 * 2) + (flag2 * 1) + (flag3 * 1.75)
                    flag5 = (flag1 + flag2 + flag3) / 3

            all_horses.append({
                "racecourse": racecourse,
                "race_id": race_id,
                "race_date": race_date,
                "race_time": race_time,
                "horse_no": _safe_numeric(horse_nos[i]),
                "horse_name": horse_names[i],
                "price_money": price_money,
                "num_runners": num_runners,
                "race_terms": race_terms,
                "going": going,
                "OR": or_val,
                "TS": ts_val,
                "RPR": rpr_val,
                "Flag1": flag1,
                "Flag2": flag2,
                "Flag3": flag3,
                "Flag4": flag4,
                "Flag5": flag5,
                "draw": _safe_numeric(draws[i]),
                "past_performance": past_performances[i],
                "last_run": last_runs[i],
                "horse_age": horse_ages[i],
                "weight": w_val,
                "trainer_RFT": _safe_numeric(trainer_rfts[i]) if trainer_rfts[i] else None,
                "jockey_allowance": jockey_allowances[i] if jockey_allowances[i] else None,
            })

    return pd.DataFrame(all_horses)


def scrape_all_races_today(
    progress_callback=None,
    url_list: list = None,
) -> pd.DataFrame:
    """
    Scrape all of today's racecards with a 5-second throttle between requests.

    Args:
        progress_callback: optional callable(current, total, racecourse) for progress updates
        url_list: override URL list (for testing); if None, fetches from time-order page

    Returns:
        Combined DataFrame of all horses across all races today.
    """
    if url_list is None:
        time_order = scrape_time_order()
        url_list = time_order["meeting_url"].tolist()

    all_dfs = []
    total = len(url_list)

    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        for i, url in enumerate(url_list):
            if i > 0:
                time.sleep(THROTTLE_SECONDS)
            try:
                df = scrape_race_card(url, client=client)
                if not df.empty:
                    all_dfs.append(df)
                    rc = df["racecourse"].iloc[0] if not df.empty else url
                    logger.info(f"[{i+1}/{total}] Scraped {rc} ({len(df)} horses)")
                    if progress_callback:
                        progress_callback(i + 1, total, rc)
            except Exception as exc:
                logger.warning(f"[{i+1}/{total}] Failed to scrape {url}: {exc}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    # Skip races where all last_run values are missing (non-runners / data unavailable)
    valid = combined.groupby("race_id")["last_run"].transform(lambda x: x.notna().any())
    combined = combined[valid].reset_index(drop=True)
    return combined
