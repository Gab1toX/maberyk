"""Download plain-text Spanish Wikipedia articles into datos_lectura/raw/.

Pure stdlib (urllib only). Run as: python -m lectura.wiki_fetch
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://es.wikipedia.org/w/api.php"
USER_AGENT = "MaberykCorpusBot/1.0 (educational research project)"
REQUEST_DELAY_SECONDS = 1
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TITLES_PATH = PROJECT_ROOT / "datos_lectura" / "wiki_titles.txt"
RAW_DIR = PROJECT_ROOT / "datos_lectura" / "raw"
OUTPUT_PATH = RAW_DIR / "wikipedia_es.txt"
FETCHED_LOG_PATH = PROJECT_ROOT / "datos_lectura" / "wiki_fetched.json"

DROP_HEADINGS = {
    "véase también",
    "referencias",
    "bibliografía",
    "enlaces externos",
    "notas",
}

HEADING_RE = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$", re.MULTILINE)


def read_titles(path):
    titles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            titles.append(line)
    return titles


def load_fetched(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_fetched(path, fetched):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fetched, f, ensure_ascii=False, indent=2, sort_keys=True)


def fetch_extract(title):
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < RATE_LIMIT_RETRIES:
                wait = RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1)
                print(f"  rate limited, retrying {title!r} in {wait}s...")
                time.sleep(wait)
                attempt += 1
                continue
            raise

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    if "missing" in page:
        return None
    return page.get("extract", "")


def clean_extract(text):
    cut_pos = None
    for match in HEADING_RE.finditer(text):
        heading_text = match.group(2).strip().lower()
        if heading_text in DROP_HEADINGS:
            cut_pos = match.start()
            break
    if cut_pos is not None:
        text = text[:cut_pos]

    text = HEADING_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    titles = read_titles(TITLES_PATH)
    fetched = load_fetched(FETCHED_LOG_PATH)

    missing_titles = []
    char_counts = []
    total_chars = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_file:
        for title in titles:
            if title in fetched:
                print(f"skip (already fetched): {title}")
                continue

            try:
                extract = fetch_extract(title)
            except urllib.error.URLError as exc:
                print(f"ERROR fetching {title!r}: {exc}")
                missing_titles.append(title)
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            if extract is None:
                print(f"missing page: {title}")
                missing_titles.append(title)
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            cleaned = clean_extract(extract)
            char_count = len(cleaned)
            char_counts.append((title, char_count))
            total_chars += char_count

            out_file.write(cleaned)
            out_file.write("\n\n")

            fetched[title] = {"chars": char_count}
            save_fetched(FETCHED_LOG_PATH, fetched)

            print(f"{title}: {char_count} chars")
            time.sleep(REQUEST_DELAY_SECONDS)

    print()
    print(f"Total articles fetched: {len(char_counts)}")
    print(f"Total chars: {total_chars}")
    if missing_titles:
        print(f"Missing/skipped ({len(missing_titles)}):")
        for title in missing_titles:
            print(f"  - {title}")


if __name__ == "__main__":
    main()
