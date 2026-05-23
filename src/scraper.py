"""
RacingPost racecard scraper.

Extracts data from the __NEXT_DATA__ JSON embedded in each page (Next.js),
with a CSS/HTML fallback. Filters time-order URLs to valid race card patterns only.
"""

import csv
import json
import os
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

TODAY = date.today().strftime("%Y-%m-%d")
TODAY_RACE_URL_RE = re.compile(rf"/racecards/\d+/[^/]+/{re.escape(TODAY)}/\d+")

# UK countries we want to include (Ireland and France are excluded)
_UK_COUNTRIES = {"ENGLAND", "SCOTLAND", "WALES"}

# Derived from data/Coursecountry.csv at import time
_UK_COURSE_NAMES: set  # upper-cased course names, e.g. "GOODWOOD"
_UK_COURSE_SLUGS: set  # URL slugs, e.g. "chelmsford-city"


def _load_uk_courses() -> tuple:
    """Load UK course names and URL slugs from data/Coursecountry.csv."""
    names: set = set()
    slugs: set = set()
    cc_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "Coursecountry.csv",
    )
    if not os.path.exists(cc_path):
        return names, slugs
    try:
        with open(cc_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                country = row.get("Country", "").strip().upper()
                course = row.get("racecourse", "").strip()
                if country in _UK_COUNTRIES and course:
                    names.add(course.upper())
                    slugs.add(course.lower().replace(" ", "-"))
    except Exception as exc:
        logging.getLogger(__name__).warning(f"Could not load Coursecountry.csv: {exc}")
    return names, slugs


_UK_COURSE_NAMES, _UK_COURSE_SLUGS = _load_uk_courses()


def _is_uk_course(racecourse: str, meeting_url: str = "") -> bool:
    """Return True if the course is in England, Scotland, or Wales."""
    if not _UK_COURSE_NAMES:
        return True  # CSV not available — don't filter
    if racecourse and racecourse.strip().upper() in _UK_COURSE_NAMES:
        return True
    # Derive slug from URL: /racecards/{id}/{slug}/{date}/{race_id}/
    if meeting_url:
        parts = meeting_url.rstrip("/").split("/")
        if len(parts) >= 4:
            url_slug = parts[-3]
            if url_slug in _UK_COURSE_SLUGS:
                return True
    return False


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


def _is_runner_dict(obj: dict) -> bool:
    """
    Heuristic: True if this dict looks like a horse/runner entry.
    Runner dicts typically contain horse name + at least one of: jockey, draw, weight, age.
    """
    if not isinstance(obj, dict):
        return False
    name_keys = {"horseName", "name", "horse"}
    sig_keys = {
        "jockeyName", "jockey", "trainerName", "trainer",
        "draw", "saddleClothNo", "clothNumber", "number",
        "weight", "weightValue", "weightCarried",
        "age", "horseAge",
        "or", "OR", "officialRating",
        "rpr", "RPR", "topSpeed", "ts", "TS",
        "form", "lastRun", "daysSinceLastRun",
    }
    has_name = any(k in obj for k in name_keys)
    has_sig = any(k in obj for k in sig_keys)
    return has_name and has_sig


def _find_runners(node, path=""):
    """
    Recursively walk JSON looking for lists of runner-like dicts.
    Returns the first plausible list found (largest preferred).
    """
    candidates = []

    def walk(n, p):
        if isinstance(n, list) and n:
            if all(isinstance(x, dict) for x in n) and len(n) >= 2:
                # Count how many entries look runner-like
                runner_count = sum(_is_runner_dict(x) for x in n)
                if runner_count >= max(2, len(n) // 2):
                    candidates.append((len(n), p, n))
            for i, item in enumerate(n):
                walk(item, f"{p}[{i}]")
        elif isinstance(n, dict):
            for k, v in n.items():
                walk(v, f"{p}.{k}" if p else k)

    walk(node, path)
    if not candidates:
        return None, ""
    # Prefer the largest candidate (most likely to be the runners list)
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][2], candidates[0][1]


def _find_value(node, keys, max_depth=10):
    """Recursively search for the first non-empty value with one of the given keys."""
    if max_depth <= 0:
        return None
    if isinstance(node, dict):
        for k in keys:
            v = node.get(k)
            if v not in (None, "", []):
                return v
        for v in node.values():
            result = _find_value(v, keys, max_depth - 1)
            if result not in (None, "", []):
                return result
    elif isinstance(node, list):
        for item in node:
            result = _find_value(item, keys, max_depth - 1)
            if result not in (None, "", []):
                return result
    return None


def _runner_field(runner: dict, keys: list, nested_keys: dict = None):
    """
    Get a field from a runner dict, trying multiple key names + nested paths.
    nested_keys = {'horse': ['name', 'horseName'], 'ratings': ['or']}
    """
    for k in keys:
        v = runner.get(k)
        if v not in (None, "", []):
            return v
    if nested_keys:
        for parent, child_keys in nested_keys.items():
            nested = runner.get(parent)
            if isinstance(nested, dict):
                for ck in child_keys:
                    v = nested.get(ck)
                    if v not in (None, "", []):
                        return v
    return None


def _parse_race_from_next_data(data: dict, url: str) -> pd.DataFrame:
    """
    Extract horse data from the __NEXT_DATA__ JSON.
    Uses recursive walker since the exact path varies across builds.
    """
    props = data.get("props", {}).get("pageProps", {})
    initial_state = props.get("initialState") or props.get("pageData") or props

    # Find the runners list anywhere in the JSON tree
    runners, runners_path = _find_runners(initial_state)
    if not runners:
        # Fallback: search the entire pageProps
        runners, runners_path = _find_runners(props)
    if not runners:
        logger.debug(f"No runners list found in __NEXT_DATA__ for {url}")
        return pd.DataFrame()

    logger.debug(f"Found {len(runners)} runners at path: {runners_path}")

    # Race-level metadata — search anywhere in initialState
    race_id = str(
        props.get("raceId") or
        _find_value(initial_state, ["raceId", "race_id", "id"]) or
        url.rstrip("/").split("/")[-1]
    )
    racecourse = str(_find_value(initial_state, ["courseName", "course", "racecourseName"]) or "")
    race_date = str(_find_value(initial_state, ["raceDate", "date", "meetingDate"]) or "")
    race_time = str(_find_value(initial_state, ["raceTime", "time", "raceStartTime"]) or "")
    going = str(_find_value(initial_state, ["going", "goingDescription", "goingType"]) or "")
    prize_money = _safe_numeric(_find_value(initial_state, ["prizeMoney", "prize", "winnerPrize", "totalPrizeMoney"]))
    race_title = str(_find_value(initial_state, ["raceTitle", "title", "raceName"]) or "")
    num_runners = len(runners)

    # Derive race_date from URL if still missing
    if not race_date:
        m = re.search(r"\d{4}-\d{2}-\d{2}", url)
        if m:
            race_date = m.group()
    # Derive racecourse from URL if still missing
    if not racecourse:
        parts = url.rstrip("/").split("/")
        if len(parts) >= 3:
            racecourse = parts[-3].replace("-", " ").title()

    rows = []
    for runner in runners:
        horse_name = _runner_field(
            runner,
            ["horseName", "name"],
            nested_keys={"horse": ["horseName", "name"]},
        ) or ""

        horse_no = _safe_numeric(_runner_field(
            runner, ["saddleClothNo", "clothNumber", "number", "stallNumber"],
        ))
        draw = _safe_numeric(_runner_field(runner, ["draw", "stall", "stallDraw"]))
        weight = _safe_numeric(_runner_field(
            runner, ["weightValue", "weight", "weightCarried", "weightInPounds"],
            nested_keys={"weight": ["value", "pounds", "lbs"]},
        ))
        horse_age = _safe_numeric(_runner_field(
            runner, ["age", "horseAge"],
            nested_keys={"horse": ["age"]},
        ))
        last_run = _safe_numeric(_runner_field(
            runner, ["daysSinceLastRun", "lastRun", "daysSince"],
        ))

        or_val = _safe_numeric(_runner_field(
            runner, ["officialRating", "or", "OR"],
            nested_keys={"ratings": ["or", "officialRating"]},
        ))
        ts_val = _safe_numeric(_runner_field(
            runner, ["topSpeed", "ts", "TS"],
            nested_keys={"ratings": ["ts", "topSpeed"]},
        ))
        rpr_val = _safe_numeric(_runner_field(
            runner, ["rpr", "RPR", "racingPostRating"],
            nested_keys={"ratings": ["rpr", "RPR"]},
        ))

        trainer_rft = _safe_numeric(_runner_field(
            runner, ["trainerRFT", "trainerForm", "trainerRunToForm"],
            nested_keys={"trainer": ["runToForm", "rtf", "RTF"]},
        ))
        jockey_allowance = _runner_field(
            runner, ["jockeyAllowance", "allowance", "claim"],
            nested_keys={"jockey": ["allowance", "claim"]},
        )
        form = _runner_field(runner, ["form", "formString", "horseForm"]) or ""

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
            "horse_name": str(horse_name),
            "price_money": prize_money,
            "num_runners": num_runners,
            "race_terms": race_title,
            "going": going,
            "OR": or_val, "TS": ts_val, "RPR": rpr_val,
            "Flag1": flag1, "Flag2": flag2, "Flag3": flag3,
            "Flag4": flag4, "Flag5": flag5,
            "draw": draw,
            "past_performance": str(form),
            "last_run": last_run,
            "horse_age": horse_age,
            "weight": weight,
            "trainer_RFT": trainer_rft,
            "jockey_allowance": str(jockey_allowance) if jockey_allowance else None,
        })

    df = pd.DataFrame(rows)
    return df[df["horse_name"].astype(bool)].reset_index(drop=True)


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

    # Horse names — RacingPost now uses data-testid="Link__Horse" (stable across builds)
    # Deduplicate while preserving order — page has 116 such links (multiple per horse)
    horse_name_els = soup.select('a[data-testid="Link__Horse"]')
    seen_horse_urls = set()
    horse_names = []
    for el in horse_name_els:
        href = el.get("href", "")
        # Take only the canonical horse URL (first occurrence), strip the race fragment
        base = href.split("#")[0]
        if base in seen_horse_urls:
            continue
        seen_horse_urls.add(base)
        name = el.get_text(strip=True)
        if not name:
            # Fall back to slug from URL
            parts = base.rstrip("/").split("/")
            name = parts[-1].replace("-", " ").title() if parts else ""
        if name:
            horse_names.append(name)

    if not horse_names:
        # Fall back to older class patterns
        for sel in ["a.RC-runnerName", '[data-test-selector="link-horseName"]',
                    ".rp-horseTable__horse a"]:
            els = soup.select(sel)
            if els:
                horse_names = [el.get_text(strip=True) for el in els]
                break

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
                if race_id and TODAY_RACE_URL_RE.search(meeting_url):
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
            if TODAY_RACE_URL_RE.search(href) and href not in seen:
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

    # Filter to UK courses (England, Scotland, Wales)
    uk_records = [
        r for r in records
        if _is_uk_course(r.get("racecourse", ""), r.get("meeting_url", ""))
    ]
    if uk_records:
        logger.info(f"UK filter: keeping {len(uk_records)}/{len(records)} races (England/Scotland/Wales).")
        records = uk_records
    else:
        logger.warning(
            f"UK filter matched 0 of {len(records)} races — coursecountry lookup may be incomplete. "
            "Returning all races."
        )

    df = pd.DataFrame(records).drop_duplicates("meeting_url").reset_index(drop=True)
    logger.info(f"Found {len(df)} races in time order.")
    return df


def scrape_race_card(url: str, client: Optional[httpx.Client] = None) -> pd.DataFrame:
    """
    Scrape one race card page. Tries __NEXT_DATA__ JSON first, then HTML selectors.
    """
    # Only scrape today's races
    if not TODAY_RACE_URL_RE.search(url):
        logger.debug(f"Skipping non-today URL: {url}")
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

    # Filter to only today's race card URLs
    url_list = [u for u in url_list if TODAY_RACE_URL_RE.search(u)]
    logger.info(f"Scraping {len(url_list)} races for {TODAY}")

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

    # Align all frames to the same column set before concat so pandas dtype
    # inference is consistent (avoids FutureWarning for all-NA columns).
    all_cols = list(dict.fromkeys(col for df in all_dfs for col in df.columns))
    all_dfs = [df.reindex(columns=all_cols) for df in all_dfs if not df.empty]
    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined.reset_index(drop=True)
