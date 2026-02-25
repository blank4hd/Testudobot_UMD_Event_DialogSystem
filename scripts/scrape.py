import re
import time
import json
import argparse
import logging
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime, date, timedelta

try:
    import cloudscraper
except ImportError:
    cloudscraper = None

logger = logging.getLogger(__name__)

# --- Optional progress bar (tqdm). Falls back to simple prints if not installed.
try:
    from tqdm import tqdm
    def pbar(iterable, desc="", unit="it"):
        return tqdm(iterable, desc=desc, unit=unit, leave=False)
except Exception:
    def pbar(iterable, desc="", unit="it"):
        total = len(iterable) if hasattr(iterable, "__len__") else None
        count = 0
        print(f"{desc} ...")
        for x in iterable:
            yield x
            count += 1
            if total:
                pct = int((count / total) * 100)
                if pct % 10 == 0:
                    print(f"  {desc}: {pct}%")
        print(f"{desc}: done")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) "
        "Gecko/20100101 Firefox/120.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://calendar.umd.edu/",
    "Origin": "https://calendar.umd.edu",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

BASE_SITE_URL = "https://calendar.umd.edu/"
BASE_DAY_URL = "https://calendar.umd.edu/events/{yyyy}/{mm}/{dd}"
DEFAULT_OUTPUT = "umd_events.json"
REQUEST_DELAY_SEC = 0.25  # be polite
MAX_RETRIES = 3
RETRY_DELAY_SEC = 5


def configure_http_session(use_cloudscraper: bool = False) -> None:
    """Configure the global HTTP session implementation."""
    global SESSION

    if use_cloudscraper:
        if cloudscraper is None:
            logger.warning(
                "--use-cloudscraper requested but package is not installed; falling back to requests.Session()"
            )
            SESSION = requests.Session()
        else:
            SESSION = cloudscraper.create_scraper(browser={"browser": "firefox", "platform": "darwin"})
            logger.info("Using cloudscraper HTTP client")
    else:
        SESSION = requests.Session()

    SESSION.headers.update(HEADERS)


def clean(txt: str) -> str:
    return " ".join((txt or "").replace("\xa0", " ").split())


def warm_up_session() -> None:
    """Prime session cookies with a lightweight homepage request."""
    try:
        SESSION.get(BASE_SITE_URL, timeout=30)
        time.sleep(REQUEST_DELAY_SEC)
        logger.info("Session warm-up complete")
    except requests.RequestException as e:
        logger.warning("Session warm-up failed: %s", e)


def fetch_soup(url: str, retries: int = MAX_RETRIES) -> BeautifulSoup:
    """Fetch a URL and return a BeautifulSoup object, with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, timeout=30)

            if r.status_code == 403 and not url.endswith("/"):
                retry_url = f"{url}/"
                logger.info("403 for %s, retrying once with trailing slash", url)
                r = SESSION.get(retry_url, timeout=30)

            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException as e:
            if attempt < retries:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s — retrying in %ds",
                    attempt, retries, url, e, RETRY_DELAY_SEC,
                )
                time.sleep(RETRY_DELAY_SEC)
            else:
                logger.error("All %d attempts failed for %s: %s", retries, url, e)
                raise


def extract_umd_description(soup: BeautifulSoup) -> str:
    """
    Description lives under <umd-event-description> on the detail page.
    """
    tag = soup.find("umd-event-description")
    if not tag:
        return "N/A"
    ps = tag.find_all("p")
    if ps:
        paras = [clean(p.get_text(" ", strip=True)) for p in ps if clean(p.get_text(" ", strip=True))]
        if paras:
            return "\n\n".join(paras)
    txt = clean(tag.get_text(" ", strip=True))
    return txt if txt else "N/A"


def extract_location(soup: BeautifulSoup) -> str:
    """
    Find location on detail page using <umd-event-location> and common patterns.
    """
    # 1) Preferred: <umd-event-location>
    loc_tag = soup.find("umd-event-location")
    if loc_tag:
        txt = clean(loc_tag.get_text(" ", strip=True))
        if txt:
            return txt

    # 2) dl/dt/dd with labels like Where/Location
    for dt_tag in soup.select("dt"):
        if clean(dt_tag.get_text()).rstrip(":").lower() in {"where", "location", "place", "venue"}:
            dd = dt_tag.find_next_sibling("dd")
            if dd:
                txt = clean(dd.get_text(" ", strip=True))
                if txt:
                    return txt

    # 3) strong/b labels
    for strong in soup.select("strong, b"):
        label = clean(strong.get_text()).rstrip(":").lower()
        if label in {"where", "location", "place", "venue"}:
            txt = clean(strong.parent.get_text(" ", strip=True).replace(strong.get_text(), "", 1))
            if txt:
                return txt

    # 4) Generic classes
    for sel in [".event__location", ".event-location", ".field--name-field-location", '[class*="location"]']:
        el = soup.select_one(sel)
        if el:
            txt = clean(el.get_text(" ", strip=True))
            if txt:
                return txt

    return "N/A"


def _parse_dt(val: str):
    """Parse 'YYYY-MM-DD HH:MM:SS' or ISO 'YYYY-MM-DDTHH:MM:SS±ZZ:ZZ'."""
    val = val.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(val, fmt)
        except Exception:
            continue
    return None


def extract_date_time_from_flags(soup: BeautifulSoup):
    """
    Extract date and time from <umd-calendar-flags>.

    For range dates like:
      "Mon Sep 15 – Wed Oct 01" (often with a hidden "To"),
    we keep the whole human-readable range but normalize it to use only "-",
    e.g. "Mon Sep 15 - Wed Oct 01".
    """
    flags = soup.find("umd-calendar-flags")
    if not flags:
        return "N/A", "N/A"

    lis = flags.find_all("li")
    date_str, time_str = "N/A", "N/A"

    # --- DATE from first <li> ---
    if lis:
        date_li = lis[0]
        date_times = date_li.find_all("time")

        if len(date_times) > 1:
            # Date RANGE case: keep full text, normalize it
            txt_raw = date_li.get_text(" ", strip=True)

            # Replace en/em dash with regular dash
            txt = txt_raw.replace("–", "-").replace("—", "-")

            # Remove the word "to"/"To"/"TO" etc.
            txt = re.sub(r"\bto\b", "", txt, flags=re.IGNORECASE)

            # Normalize spaces around dash to " - "
            txt = re.sub(r"\s*-\s*", " - ", txt)

            # Final whitespace cleanup
            txt = clean(txt)

            if txt:
                date_str = txt
        else:
            # Single date: as before, prefer datetime attr → ISO, else visible text
            t_date = date_li.find("time")
            if t_date:
                dt_attr = t_date.get("datetime")
                if dt_attr:
                    dt = _parse_dt(dt_attr)
                    if dt:
                        date_str = dt.date().isoformat()
                if date_str == "N/A":
                    txt = clean(t_date.get_text(" ", strip=True))
                    if txt:
                        date_str = txt

    # --- TIME from second <li> (time range) ---
    if len(lis) > 1:
        time_li = lis[1]
        time_tags = time_li.find_all("time")
        if time_tags:
            texts = [clean(t.get_text(" ", strip=True)) for t in time_tags if clean(t.get_text(" ", strip=True))]
            if len(texts) == 1:
                time_str = texts[0]
            elif len(texts) >= 2:
                time_str = f"{texts[0]} - {texts[1]}"

            # If still empty, try datetime attributes
            if time_str == "N/A":
                parsed = []
                for t in time_tags:
                    t_attr = t.get("datetime")
                    tdt = _parse_dt(t_attr) if t_attr else None
                    if tdt:
                        parsed.append(tdt.strftime("%-I:%M%p").lower())
                if parsed:
                    if len(parsed) == 1:
                        time_str = parsed[0]
                    else:
                        time_str = f"{parsed[0]} - {parsed[1]}"

    return date_str, time_str


def day_url(d: date) -> str:
    return BASE_DAY_URL.format(yyyy=d.year, mm=f"{d.month:02d}", dd=f"{d.day:02d}")


def _extract_day_event_links(day_soup: BeautifulSoup, page_url: str) -> list[str]:
    """Extract event detail links from a day listing page.

    Supports both legacy listing markup and current custom-element markup.
    """
    links: list[str] = []
    seen: set[str] = set()

    def add_link(href: str | None) -> None:
        if not href:
            return
        absolute = urljoin(page_url, href)
        parsed = urlparse(absolute)

        if parsed.netloc and "calendar.umd.edu" not in parsed.netloc:
            return

        path = (parsed.path or "").rstrip("/")
        if not path or path in {"", "/", "/submission", "/humans.txt", "/search"}:
            return

        if path.startswith("/events/") or path.startswith("/place-list/") or path.startswith("/uploads/"):
            return

        if path.startswith("/category/"):
            return

        if absolute in seen:
            return

        seen.add(absolute)
        links.append(absolute)

    # Current markup: event cards rendered as <umd-element-event><a href="..."></a></umd-element-event>
    for a in day_soup.select("umd-element-event a[href]"):
        add_link(a.get("href"))

    # Legacy fallback
    if not links:
        for h2 in day_soup.select("h2.event-title.heading-san-four"):
            a = h2.find("a", href=True)
            if a:
                add_link(a.get("href"))

    return links


def _extract_next_page_url(day_soup: BeautifulSoup, page_url: str) -> str | None:
    """Extract next-page URL from day listing pagination controls, if present."""
    candidates: list[str] = []

    for selector in ("a[rel='next'][href]", "link[rel='next'][href]", "a[aria-label*='next' i][href]"):
        for element in day_soup.select(selector):
            href = element.get("href")
            if href:
                candidates.append(urljoin(page_url, href))

    for a in day_soup.select("a[href]"):
        href = a.get("href") or ""
        text = clean(a.get_text(" ", strip=True)).lower()
        aria_label = (a.get("aria-label") or "").lower()
        if "page=" in href or text in {"next", "next >", "›", "→"} or "next" in aria_label:
            candidates.append(urljoin(page_url, href))

    for candidate in candidates:
        parsed = urlparse(candidate)
        if parsed.netloc and "calendar.umd.edu" not in parsed.netloc:
            continue
        if "page=" in (parsed.query or ""):
            return candidate

    return None


def scrape_day(d: date):
    """
    For a single day:
      - Load the day listing page
      - For each event card on the day page, grab the detail href
      - Go to that event link (detail page)
      - Extract title, date, time, location, description
    """
    page_url = day_url(d)
    visited_pages: set[str] = set()
    seen_detail_urls: set[str] = set()

    while page_url and page_url not in visited_pages:
        visited_pages.add(page_url)

        day_soup = fetch_soup(page_url)
        time.sleep(REQUEST_DELAY_SEC)

        detail_urls = _extract_day_event_links(day_soup, page_url)
        if not detail_urls:
            logger.info("No event links found on day page %s", page_url)

        for detail_url in pbar(detail_urls, desc=f"{d.isoformat()} events", unit="evt"):
            if detail_url in seen_detail_urls:
                continue
            seen_detail_urls.add(detail_url)

            try:
                detail = fetch_soup(detail_url)
            except Exception as e:
                logger.warning("Failed %s: %s", detail_url, e)
                continue

            # Title from detail page (prefer h1)
            title_tag = detail.find("h1")
            if title_tag:
                title = clean(title_tag.get_text(" ", strip=True))
            else:
                title = "N/A"

            desc = extract_umd_description(detail)
            loc = extract_location(detail)
            ev_date, ev_time = extract_date_time_from_flags(detail)

            yield {
                "event": title or "N/A",
                "date": ev_date or "N/A",
                "time": ev_time or "N/A",
                "url": detail_url or "N/A",
                "location": loc or "N/A",
                "description": desc or "N/A",
            }

            time.sleep(REQUEST_DELAY_SEC)

        next_page = _extract_next_page_url(day_soup, page_url)
        page_url = next_page if next_page and next_page not in visited_pages else None


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text)


def _parse_human_date(text: str, range_start: date | None = None, range_end: date | None = None) -> str | None:
    """Try to parse a human-readable date like 'Thu Oct 23' to YYYY-MM-DD.

    Uses date range context for year inference when available.
    """
    for fmt in ("%a %b %d", "%b %d", "%B %d", "%a %B %d"):
        try:
            parsed = datetime.strptime(text.strip(), fmt)
            month = parsed.month
            day_num = parsed.day

            candidate_years: list[int] = []
            if range_start and range_end:
                start_year = range_start.year
                end_year = range_end.year
                candidate_years = sorted({
                    start_year - 1, start_year, start_year + 1,
                    end_year - 1, end_year, end_year + 1,
                })
            else:
                current_year = date.today().year
                candidate_years = [current_year - 1, current_year, current_year + 1]

            candidates: list[date] = []
            for year in candidate_years:
                try:
                    candidates.append(date(year, month, day_num))
                except ValueError:
                    continue

            if not candidates:
                return None

            if range_start and range_end:
                in_range = [candidate for candidate in candidates if range_start <= candidate <= range_end]
                if in_range:
                    candidate = min(in_range, key=lambda value: abs((value - range_start).days))
                else:
                    def boundary_distance(value: date) -> int:
                        if value < range_start:
                            return (range_start - value).days
                        return (value - range_end).days

                    candidate = min(candidates, key=boundary_distance)
            else:
                today = date.today()
                candidate = min(candidates, key=lambda value: abs((value - today).days))

            return candidate.isoformat()
        except ValueError:
            continue
    return None


def normalize_event(event: dict, range_start: date | None = None, range_end: date | None = None) -> dict:
    """Clean up event data:

    - Strip leading/trailing whitespace from all string fields.
    - Normalize the date field to YYYY-MM-DD (take start date for ranges).
    - Replace 'N/A' or empty location with None.
    - Remove any HTML tags from description.
    """
    normalized = {}
    for key, value in event.items():
        if isinstance(value, str):
            value = value.strip()
        normalized[key] = value

    # --- Date normalization ---
    date_val = normalized.get("date", "")
    if date_val and date_val != "N/A":
        # Already in YYYY-MM-DD? Keep it.
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_val):
            # Range like "Thu Oct 23 - Fri Oct 24" → take first part
            first_part = date_val.split(" - ")[0].strip()
            parsed = _parse_human_date(first_part, range_start=range_start, range_end=range_end)
            if parsed:
                normalized["date"] = parsed

    # --- Location: replace N/A / empty with None ---
    loc = normalized.get("location", "")
    if not loc or loc == "N/A":
        normalized["location"] = None

    # --- Description: strip HTML tags ---
    desc = normalized.get("description", "")
    if desc and desc != "N/A":
        normalized["description"] = _strip_html(desc).strip()

    return normalized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_events(start_date: str, end_date: str) -> list[dict]:
    """Scrape all events from calendar.umd.edu in the given date range.

    Args:
        start_date: Start date as YYYY-MM-DD string.
        end_date:   End date as YYYY-MM-DD string.

    Returns:
        List of event dicts with keys:
        event, date, time, url, location, description.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if end < start:
        raise ValueError(f"end_date {end_date} is earlier than start_date {start_date}")

    logger.info("Scraping events from %s to %s", start_date, end_date)
    warm_up_session()

    seen_urls: set[str] = set()
    collected: list[dict] = []
    failed_days: list[str] = []

    # Build list of dates so we can show a progress bar
    days: list[date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)

    for d in pbar(days, desc="Days", unit="day"):
        try:
            for ev in scrape_day(d):
                if ev["url"] in seen_urls:
                    continue
                seen_urls.add(ev["url"])
                collected.append(normalize_event(ev, range_start=start, range_end=end))
        except Exception as e:
            logger.error("Day %s failed: %s", d.isoformat(), e)
            failed_days.append(d.isoformat())

    if failed_days:
        preview = ", ".join(failed_days[:5])
        logger.warning(
            "Failed days: %d of %d (%s%s)",
            len(failed_days),
            len(days),
            preview,
            "..." if len(failed_days) > 5 else "",
        )

    logger.info("Collected %d events", len(collected))
    return collected


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape UMD calendar events for a configurable date range.",
    )
    parser.add_argument(
        "--start-date",
        default=date.today().isoformat(),
        help="Start date in YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--end-date",
        default=(date.today() + timedelta(days=90)).isoformat(),
        help="End date in YYYY-MM-DD (default: 3 months from today)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--use-cloudscraper",
        action="store_true",
        help="Use cloudscraper HTTP client for anti-bot protected pages (optional)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    configure_http_session(use_cloudscraper=args.use_cloudscraper)

    try:
        events = scrape_events(args.start_date, args.end_date)
    except ValueError as e:
        logger.error("Invalid date input: %s", e)
        sys.exit(2)

    if not events:
        logger.info("No events collected for the range.")
        return

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d events to '%s'", len(events), args.output)


if __name__ == "__main__":
    main()
