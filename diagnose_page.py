"""
Diagnostic script — dumps the structure of a RacingPost race card so we can
work out the correct selectors / data extraction pattern.

Usage:
    python diagnose_page.py

Writes:
    diagnose_output.txt — human-readable analysis
    diagnose_page.html — raw HTML for inspection
"""

import re
import json
import sys
import httpx
from bs4 import BeautifulSoup

URL = "https://www.racingpost.com/racecards/21/goodwood/2026-05-23/918885/"

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


def main():
    out = []

    def log(msg):
        print(msg)
        out.append(msg)

    log(f"Fetching {URL}\n")
    resp = httpx.get(URL, headers=HEADERS, timeout=30, follow_redirects=True)
    log(f"Status: {resp.status_code}")
    log(f"Final URL: {resp.url}")
    log(f"Content-Type: {resp.headers.get('content-type')}")
    log(f"Body length: {len(resp.text)} chars\n")

    # Save raw HTML
    with open("diagnose_page.html", "w", encoding="utf-8") as f:
        f.write(resp.text)
    log("Raw HTML saved to: diagnose_page.html\n")

    soup = BeautifulSoup(resp.text, "lxml")

    # 1. Check for __NEXT_DATA__
    log("=== 1. Next.js __NEXT_DATA__ ===")
    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data and next_data.string:
        log(f"  ✓ Found __NEXT_DATA__ ({len(next_data.string)} chars)")
        try:
            data = json.loads(next_data.string)
            log(f"  ✓ Valid JSON")
            log(f"  Top-level keys: {list(data.keys())}")
            props = data.get("props", {}).get("pageProps", {})
            log(f"  pageProps keys: {list(props.keys())[:30]}")
        except Exception as e:
            log(f"  ✗ Parse error: {e}")
    else:
        log("  ✗ Not found")

    # 2. Other framework markers
    log("\n=== 2. Other JS framework markers ===")
    for pattern, name in [
        (r'window\.__INITIAL_STATE__\s*=', "Redux __INITIAL_STATE__"),
        (r'window\.__NUXT__\s*=', "Nuxt.js __NUXT__"),
        (r'window\.__APOLLO_STATE__\s*=', "Apollo __APOLLO_STATE__"),
        (r'window\.__PRELOADED_STATE__\s*=', "__PRELOADED_STATE__"),
        (r'<script[^>]+type="application/json"', "JSON script tag"),
        (r'<script[^>]+id="serverApp-state"', "Angular Universal"),
    ]:
        if re.search(pattern, resp.text):
            log(f"  ✓ {name}")
        else:
            log(f"  ✗ {name}")

    # 3. All script tags with JSON-ish content
    log("\n=== 3. JSON-like scripts ===")
    scripts = soup.find_all("script")
    log(f"  Total scripts: {len(scripts)}")
    json_scripts = 0
    for i, s in enumerate(scripts):
        content = s.string or ""
        if content and (content.strip().startswith("{") or s.get("type") == "application/json"):
            json_scripts += 1
            if json_scripts <= 5:
                log(f"  Script {i}: type={s.get('type')}, id={s.get('id')}, length={len(content)}")
                log(f"    Preview: {content[:200]}")

    # 4. API endpoint hints in JS
    log("\n=== 4. API endpoint hints ===")
    api_patterns = re.findall(r'["\'](/api/[^"\']{5,80})["\']', resp.text)
    unique_apis = sorted(set(api_patterns))
    log(f"  Found {len(unique_apis)} unique /api/ paths")
    for api in unique_apis[:20]:
        log(f"    {api}")

    # Also check for racecards API patterns
    rc_patterns = re.findall(r'["\'](/racecard[^"\']{5,80})["\']', resp.text)
    unique_rc = sorted(set(rc_patterns))
    log(f"  Found {len(unique_rc)} unique /racecard* paths")
    for rc in unique_rc[:10]:
        log(f"    {rc}")

    # 5. Element class probing
    log("\n=== 5. Common selector probes ===")
    selectors = [
        # New attempts
        "[data-test-selector]", "[data-testid]",
        "[class*='runner']", "[class*='Runner']",
        "[class*='horse']", "[class*='Horse']",
        "[class*='card']",
        # Old patterns
        ".RC-runnerName", ".RC-runnerRow",
        ".rp-horseTable", ".rp-horseTable__horse",
        "section[data-card-race-id]",
        # Generic
        "table", "main", "article",
        # All a tags
        "a[href*='/horse/']", "a[href*='/jockey/']",
    ]
    for sel in selectors:
        try:
            count = len(soup.select(sel))
            if count > 0:
                example = soup.select_one(sel)
                preview = str(example)[:150].replace("\n", " ")
                log(f"  ✓ {sel}: {count}    {preview}")
            else:
                log(f"  ✗ {sel}: 0")
        except Exception as e:
            log(f"  ! {sel}: error {e}")

    # 6. Distinct class names sample (most common)
    log("\n=== 6. Most common class names in page ===")
    all_classes = []
    for el in soup.find_all(class_=True):
        cls = el.get("class")
        if isinstance(cls, list):
            all_classes.extend(cls)
    from collections import Counter
    common = Counter(all_classes).most_common(30)
    for cls, count in common:
        log(f"  {count}× {cls}")

    # 7. Body text length (clue to whether content is server-rendered)
    log("\n=== 7. Rendered text ===")
    body = soup.find("body")
    if body:
        text = body.get_text(" ", strip=True)
        log(f"  Body text length: {len(text)} chars")
        log(f"  First 500 chars: {text[:500]}")

    # 8. Try the new scraper directly
    log("\n=== 8. New scraper test ===")
    try:
        import sys
        sys.path.insert(0, ".")
        from src.scraper import _extract_next_data, _parse_race_from_next_data, _parse_race_from_html
        data = _extract_next_data(resp.text)
        if data:
            df = _parse_race_from_next_data(data, URL)
            log(f"  JSON parser: {len(df)} horses")
            if not df.empty:
                log(f"  Columns: {list(df.columns)}")
                log(f"  First horse: {df.iloc[0].to_dict()}")
        else:
            log("  No __NEXT_DATA__")
        df2 = _parse_race_from_html(soup, URL)
        log(f"  HTML parser: {len(df2)} horses")
        if not df2.empty:
            log(f"  First horse: {df2.iloc[0].to_dict()}")
    except Exception as e:
        log(f"  Error: {e}")
        import traceback
        log(traceback.format_exc())

    # 9. Dump first runner's raw structure — confirms which keys hold OR/TS/RPR
    log("\n=== 9. First runner raw structure ===")
    if data:
        try:
            from src.scraper import _find_runners
            props = data.get("props", {}).get("pageProps", {})
            initial_state = props.get("initialState") or props.get("pageData") or props
            runners, rpath = _find_runners(initial_state)
            if not runners:
                runners, rpath = _find_runners(props)
            if runners:
                log(f"  Runners path: {rpath}   count: {len(runners)}")
                r = runners[0]
                log(f"  Top-level keys: {list(r.keys())}")
                for k, v in r.items():
                    if isinstance(v, dict):
                        log(f"    [{k}] -> {list(v.keys())[:15]}")
                        for kk, vv in list(v.items())[:8]:
                            if isinstance(vv, dict):
                                log(f"      [{kk}] -> {list(vv.keys())[:10]}")
                            elif not isinstance(vv, list):
                                log(f"      {kk}: {vv}")
                    elif isinstance(v, list) and v:
                        log(f"    {k} (list[{len(v)}]): {str(v[0])[:60]}")
                    else:
                        log(f"    {k}: {v}")
            else:
                log("  No runners found in JSON")
        except Exception as e:
            import traceback as _tb
            log(f"  Error: {e}\n{_tb.format_exc()}")

    # 10. initialState structure
    if data:
        log("\n=== 10. initialState top-level structure ===")
        try:
            props = data.get("props", {}).get("pageProps", {})
            initial_state = props.get("initialState", {})
            if isinstance(initial_state, dict):
                log(f"  initialState top keys: {list(initial_state.keys())}")
                for k, v in list(initial_state.items())[:5]:
                    log(f"    {k}: {type(v).__name__} {'(empty)' if not v else ''}")
                    if isinstance(v, dict):
                        log(f"      sub-keys: {list(v.keys())[:15]}")
                    elif isinstance(v, list) and v:
                        log(f"      list len={len(v)}, first item keys: {list(v[0].keys())[:15] if isinstance(v[0], dict) else type(v[0]).__name__}")
        except Exception as e:
            log(f"  Error: {e}")

    # Save analysis
    with open("diagnose_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\nFull analysis written to: diagnose_output.txt")
    print("Raw page saved to: diagnose_page.html")


if __name__ == "__main__":
    main()
