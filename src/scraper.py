"""
RacingPost racecard scraper.

Extracts data from the __NEXT_DATA__ JSON embedded in each page (Next.js),
with a CSS/HTML fallback. Filters time-order URLs to valid race card patterns only.
"""

import json
import re
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

# Valid race card URL: /racecards/{course_id}/{course_name}/{date}/{race_id}/
RACE_URL_RE = re.compile(r"/racecards/\d+/[^/]+/\d{4}-\d{2}-\d{2}/\d+")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _get_response(url: str, client: Optional[httpx.Client] = None) -> httpx.Response:
    """Fetch a URL and return the raw response."""
    if client:
        resp = client.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
    else:
        resp = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
    if resp.status_code == 403:
        raise PermissionError(
            "RacingPost blocked the request (403). "
            "Try again later or check your IP is not rate-limited."
        )
    if resp.status_code in (401, 406):
        raise PermissionError(f"RacingPost rejected the request ({resp.status_code}) for {url}")
    resp.raise_for_status()
    return resp


def _get_soup(url: str, client: Optional[httpx.Client] = None) -> tuple:
    """Return (BeautifulSoup, raw_text) for a URL."""
    resp = _get_response(url, client)
    return BeautifulSoup(resp.text, "lxml"), resp.text


def _safe_numeric(value) -> Optional[float]:
    try:
        return float(str(value).strip().replace(",", "").replace("£", "").replace("€", "").replace("st", "").replace("lb", ""))
    except (ValueError, TypeError):
        return None


def _extract_next_data(raw_html: str) -> Optional[dict]:
    """Extract the __NEXT_DATA__ JSON embedded by Next.js, if present."""
    match = re.search(r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', raw_html, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _parse_race_from_next_data(data: dict, url: str) -> pd.DataFrame:
    """
    Extract horse data from the __NEXT_DATA__ JSON structure.
    RacingPost embeds race data under pageProps in various nested paths.
    """
    # Navigate to pageProps
    props = data.get("props", {}).get("pageProps", {})

    # Try multiple known paths where race card data lives
    race_card = (
        props.get("raceCard") or
        props.get("pageData", {}).get("raceCard") or
        props.get("race") or
        props.get("pageData", {}).get("race") or
        {}
    )

    runners = (
        race_card.get("runners") or
        race_card.get("horses") or
        props.get("runners") or
        []
    )

    if not runners:
        logger.debug(f"No runners found in __NEXT_DATA__ for {url}. Keys: {list(props.keys())}")
        return pd.DataFrame()

    # Race-level metadata
    race_id = str(
        race_card.get("raceId") or
        race_card.get("race_id") or
        props.get("raceId") or
        url.rstrip("/").split("/")[-1]
    )
    racecourse = (
        race_card.get("courseName") or
        race_card.get("course") or
        props.get("courseName") or
        ""
    )
    race_date = race_card.get("raceDate") or race_card.get("date") or ""
    race_time = race_card.get("raceTime") or race_card.get("time") or ""
    going = race_card.get("going") or race_card.get("goingDescription") or ""
    price_money = _safe_numeric(race_card.get("prizeMoney") or race_card.get("prize"))
    num_runners = len(runners)

    rows = []
    for runner in runners:
        horse_name = (
            runner.get("horseName") or
            runner.get("name") or
            runner.get("horse", {}).get("horseName") or
            ""
        )
        horse_no = _safe_numeric(runner.get("saddleClothNo") or runner.get("clothNumber") or runner.get("number"))
        draw = _safe_numeric(runner.get("draw") or runner.get("stall"))
        weight = _safe_numeric(runner.get("weightValue") or runner.get("weight"))
        horse_age = _safe_numeric(runner.get("age") or runner.get("horseAge"))
        last_run = _safe_numeric(runner.get("daysSinceLastRun") or runner.get("lastRun"))

        # Ratings — try nested and flat
        or_val = _safe_numeric(
            runner.get("officialRating") or
            runner.get("or") or
            runner.get("OR") or
            (runner.get("ratings") or {}).get("or")
        )
        ts_val = _safe_numeric(
            runner.get("topSpeed") or
            runner.get("ts") or
            runner.get("TS") or
            (runner.get("ratings") or {}).get("ts")
        )
        rpr_val = _safe_numeric(
            runner.get("rpr") or
            runner.get("RPR") or
            (runner.get("ratings") or {}).get("rpr")
        )

        trainer_rft = _safe_numeric(
            runner.get("trainerRFT") or
            runner.get("trainerForm") or
            (runner.get("trainer") or {}).get("runToForm")
        )
        jockey_allowance = runner.get("jockeyAllowance") or runner.get("allowance")

        # Compute flags
        flag1 = flag2 = flag3 = flag4 = flag5 = None
        if None not in (ts_val, rpr_val, or_val):
            flag1 = ts_val + rpr_val - or_val
            if weight is not None:
                flag2 = ts_val - weight
                flag3 = rpr_val - weight
                flag4 = (flag1 * 2) + (flag2 * 1) + (flag3 * 1.75)
                flag5 = (flag1 + flag2 + flag3) / 3

        rows.append({
            "racecourse": racecourse,
            "race_id": race_id,
            "race_date": race_date,
            "race_time": race_time,
            "horse_no": horse_no,
            "horse_name": horse_name,
            "price_money": price_money,
            "num_runners": num_runners,
            "race_terms": race_card.get("raceTitle") or "",
            "going": going,
            "OR": or_val,
            "TS": ts_val,
            "RPR": rpr_val,
            "Flag1": flag1,
            "Flag2": flag2,
            "Flag3": flag3,
            "Flag4": flag4,
            "Flag5": flag5,
            "draw": draw,
            "past_performance": runner.get("form") or runner.get("formString") or "",
            "last_run": last_run,
            "horse_age": horse_age,
            "weight": weight,
            "trainer_RFT": trainer_rft,
            "jockey_allowance": jockey_allowance,
        })

    return pd.DataFrame(rows)


def _parse_race_from_html(soup: BeautifulSoup, url: str) -> pd.DataFrame:
    """
    Fallback: extract horse data from HTML using multiple selector patterns.
    Tries RC- prefixed classes and rp- prefixed classes.
    """
    # Metadata — try multiple attribute patterns
    meta_el = (
        soup.select_one("section[data-card-race-id]") or
        soup.select_one("[data-race-id]") or
        soup.select_one("main[data-analytics-race-id]") or
        soup.select_one("[data-analytics-race-id]")
    )

    race_id = ""
    racecourse = ""
    race_date = ""
    race_time = ""
    going = ""

    if meta_el:
        race_id = (
            meta_el.get("data-card-race-id") or
            meta_el.get("data-race-id") or
            meta_el.get("data-analytics-race-id") or
            ""
        )
        racecourse = (
            meta_el.get("data-card-coursename") or
            meta_el.get("data-analytics-coursename") or
            ""
        )
        race_date = meta_el.get("data-card-race-date") or meta_el.get("data-analytics-race-date") or ""
        race_time = meta_el.get("data-card-race-time") or meta_el.get("data-analytics-race-time") or ""

    # Fall back to URL for race_id
    if not race_id:
        parts = url.rstrip("/").split("/")
        race_id = parts[-1] if parts else ""

    # Horse names — try multiple class patterns
    horse_name_els = (
        soup.select("a.RC-runnerName") or
        soup.select('[data-test-selector="link-horseName"]') or
        soup.select(".rp-horseTable__horse a") or
        soup.select('[class*="horseName"] a') or
        soup.select('[class*="runnerName"]')
    )
    horse_names = [el.get_text(strip=True) for el in horse_name_els]

    if not horse_names:
        logger.warning(f"No horses found via HTML selectors at {url}")
        return pd.DataFrame()

    n = len(horse_names)

    def _select_numeric(selectors: list, attr: str = None) -> list:
        for sel in selectors:
            els = soup.select(sel)
            if els:
                if attr:
                    return [_safe_numeric(e.get(attr)) for e in els]
                return [_safe_numeric(e.get_text(strip=True)) for e in els]
        return [None] * n

    def _select_text(selectors: list) -> list:
        for sel in selectors:
            els = soup.select(sel)
            if els:
                return [e.get_text(strip=True) for e in els]
        return [""] * n

    def _pad(lst, length, default=None):
        return list(lst) + [default] * max(0, length - len(lst))

    horse_nos = _pad(_select_numeric(
        ['.RC-runnerNumber span[data-order-no]', '.rp-horseTable__saddleClothNo'],
        attr="data-order-no"
    ) or _select_numeric(['.rp-horseTable__saddleClothNo']), n)

    draws = _pad(_select_numeric(
        ['.RC-runnerNumber span[data-order-draw]'],
        attr="data-order-draw"
    ), n)

    ors = _pad(_select_numeric([".RC-runnerOr", ".rp-horseTable__ratingValue--or", '[class*="runnerOr"]']), n)
    tss = _pad(_select_numeric([".RC-runnerTs", ".rp-horseTable__ratingValue--ts", '[class*="runnerTs"]']), n)
    rprs = _pad(_select_numeric([".RC-runnerRpr", ".rp-horseTable__ratingValue--rpr", '[class*="runnerRpr"]']), n)
    weights = _pad(_select_numeric(['.RC-runnerWgt__carried[data-order-wgt]'], attr="data-order-wgt"), n)
    horse_ages = _pad(_select_numeric([".RC-runnerAge", '[class*="runnerAge"]']), n)
    last_runs = _pad(_select_numeric([".RC-runnerStats__lastRun", '[class*="lastRun"]']), n)
    trainer_rfts = _pad(_select_text(['[data-test-selector="RC-cardPage-runnerTrainer-rtf"]']), n)
    past_perfs = _pad(_select_text([".RC-runnerInfo__form", ".rp-horseTable__horse__form"]), n)

    going_el = soup.select_one('[data-test-selector="RC-headerBox__going"] .RC-headerBox__infoRow__content')
    going = going_el.get_text(strip=True) if going_el else ""

    price_el = soup.select_one('[data-test-selector="RC-headerBox__winner"] .RC-headerBox__infoRow__content')
    price_money = _safe_numeric(price_el.get_text(strip=True)) if price_el else None

    rows = []
    for i in range(n):
        or_val, ts_val, rpr_val, w_val = ors[i], tss[i], rprs[i], weights[i]

        flag1 = flag2 = flag3 = flag4 = flag5 = None
        if None not in (ts_val, rpr_val, or_val):
            flag1 = ts_val + rpr_val - or_val
            if w_val is not None:
                flag2 = ts_val - w_val
                flag3 = rpr_val - w_val
                flag4 = (flag1 * 2) + (flag2 * 1) + (flag3 * 1.75)
                flag5 = (flag1 + flag2 + flag3) / 3

        rows.append({
            "racecourse": racecourse,
            "race_id": race_id,
            "race_date": race_date,
            "race_time": race_time,
            "horse_no": horse_nos[i],
            "horse_name": horse_names[i],
            "price_money": price_money,
            "num_runners": n,
            "race_terms": "",
            "going": going,
            "OR": or_val, "TS": ts_val, "RPR": rpr_val,
            "Flag1": flag1, "Flag2": flag2, "Flag3": flag3, "Flag4": flag4, "Flag5": flag5,
            "draw": draws[i],
            "past_performance": past_perfs[i],
            "last_run": last_runs[i],
            "horse_age": horse_ages[i],
            "weight": w_val,
            "trainer_RFT": _safe_numeric(trainer_rfts[i]) if trainer_rfts[i] else None,
            "jockey_allowance": None,
        })

    return pd.DataFrame(rows)


def scrape_time_order() -> pd.DataFrame:
    """
    Scrape today's race list from the RacingPost time-order page.
    Filters to only valid individual race card URLs.
    """
    url = f"{BASE_URI}/racecards/time-order"
    soup, raw = _get_soup(url)

    # First try __NEXT_DATA__ for race list
    next_data = _extract_next_data(raw)
    records = []

    if next_data:
        props = next_data.get("props", {}).get("pageProps", {})
        meetings = (
            props.get("meetings") or
            props.get("pageData", {}).get("meetings") or
            props.get("races") or
            props.get("pageData", {}).get("races") or
            []
        )
        for meeting in meetings:
            races = meeting.get("races") or [meeting] if meeting.get("raceId") else []
            for race in races:
                race_id = str(race.get("raceId") or race.get("id") or "")
                course = race.get("courseName") or race.get("course") or meeting.get("courseName") or ""
                race_date = race.get("raceDate") or race.get("date") or ""
                race_time = race.get("raceTime") or race.get("time") or ""
                course_id = str(race.get("courseId") or meeting.get("courseId") or "")
                url_slug = (race.get("courseSlug") or meeting.get("courseSlug") or course.lower().replace(" ", "-"))
                meeting_url = (
                    race.get("url") or
                    race.get("raceUrl") or
                    f"{BASE_URI}/racecards/{course_id}/{url_slug}/{race_date}/{race_id}/"
                )
                if not meeting_url.startswith("http"):
                    meeting_url = f"{BASE_URI}{meeting_url}"
                if race_id and RACE_URL_RE.search(meeting_url):
                    records.append({
                        "race_id": race_id,
                        "racecourse": course,
                        "race_date": race_date,
                        "race_time": race_time,
                        "num_runners": race.get("numberOfRunners") or "",
                        "meeting_url": meeting_url,
                    })

    # Fall back to HTML link scraping
    if not records:
        all_links = soup.select("a[href]")
        seen = set()
        for a in all_links:
            href = a.get("href", "")
            if RACE_URL_RE.search(href) and href not in seen:
                seen.add(href)
                meeting_url = f"{BASE_URI}{href}" if not href.startswith("http") else href
                parts = href.rstrip("/").split("/")
                race_id = parts[-1] if parts else ""
                race_date_m = re.search(r"\d{4}-\d{2}-\d{2}", href)
                race_date = race_date_m.group() if race_date_m else ""
                racecourse = parts[-3] if len(parts) >= 3 else ""

                race_id_attr = a.get("data-race-id") or a.get("data-analytics-race-id") or race_id
                racecourse_attr = a.get("data-racecourse") or a.get("data-analytics-coursename") or racecourse.replace("-", " ").title()
                race_time_attr = a.get("data-race-time") or a.get("data-analytics-race-time") or ""

                records.append({
                    "race_id": race_id_attr,
                    "racecourse": racecourse_attr,
                    "race_date": race_date,
                    "race_time": race_time_attr,
                    "num_runners": "",
                    "meeting_url": meeting_url,
                })

    if not records:
        logger.warning("No races found on time-order page.")
        return pd.DataFrame(columns=["race_id", "racecourse", "race_date", "race_time", "num_runners", "meeting_url"])

    df = pd.DataFrame(records).drop_duplicates("meeting_url").reset_index(drop=True)
    logger.info(f"Found {len(df)} races in time order.")
    return df


def scrape_race_card(url: str, client: Optional[httpx.Client] = None) -> pd.DataFrame:
    """
    Scrape one race card page. Tries __NEXT_DATA__ JSON first, then HTML selectors.
    """
    # Validate URL matches race card pattern before fetching
    if not RACE_URL_RE.search(url):
        logger.debug(f"Skipping non-race URL: {url}")
        return pd.DataFrame()

    soup, raw = _get_soup(url, client) if client is None else (_soup_with_client(url, client))

    # Try JSON extraction first (most reliable)
    next_data = _extract_next_data(raw)
    if next_data:
        df = _parse_race_from_next_data(next_data, url)
        if not df.empty:
            return df

    # Fall back to HTML
    df = _parse_race_from_html(soup, url)
    if df.empty:
        logger.warning(f"No data extracted from {url}")
    return df


def _soup_with_client(url: str, client: httpx.Client) -> tuple:
    """Fetch with an existing client, returning (soup, raw_text)."""
    resp = _get_response(url, client)
    return BeautifulSoup(resp.text, "lxml"), resp.text


def scrape_all_races_today(
    progress_callback=None,
    url_list: list = None,
) -> pd.DataFrame:
    """
    Scrape all of today's racecards with a 5-second throttle between requests.
    """
    if url_list is None:
        time_order = scrape_time_order()
        url_list = time_order["meeting_url"].tolist()

    # Filter to only valid race card URLs
    url_list = [u for u in url_list if RACE_URL_RE.search(u)]
    logger.info(f"Scraping {len(url_list)} valid race URLs")

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
                    rc = df["racecourse"].iloc[0]
                    logger.info(f"[{i+1}/{total}] Scraped {rc} — {url.split('/')[-2]} ({len(df)} horses)")
                    if progress_callback:
                        progress_callback(i + 1, total, rc)
                else:
                    logger.warning(f"[{i+1}/{total}] Empty result for {url}")
            except Exception as exc:
                logger.warning(f"[{i+1}/{total}] Failed to scrape {url}: {exc}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined.reset_index(drop=True)
