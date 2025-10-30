import os
import sys
import csv
import statistics
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from urllib.request import urlopen
from urllib.error import URLError, HTTPError


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = os.path.join(BASE_DIR, "../stats")
COMPANY_XLSX = os.path.join(
    STATS_DIR, "Leetcode problem set (company tag, sorted by freq).xlsx"
)
SLUG_DATA_CSV = os.path.join(STATS_DIR, "slug_data.csv")
RATINGS_URL = "https://raw.githubusercontent.com/zerotrac/leetcode_problem_rating/main/ratings.txt"


def parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def extract_slug_from_url(url: str) -> Optional[str]:
    if not url or not isinstance(url, str):
        return None
    try:
        anchor = "/problems/"
        i = url.find(anchor)
        if i == -1:
            return None
        rest = url[i + len(anchor):]
        for sep in ["/", "?", "#"]:
            j = rest.find(sep)
            if j != -1:
                rest = rest[:j]
        slug = rest.strip().lower()
        return slug or None
    except Exception:
        return None


def fetch_ratings_by_slug() -> Dict[str, float]:
    """Fetch ratings.txt and map Title Slug -> Rating."""
    with urlopen(RATINGS_URL, timeout=30) as resp:
        data = resp.read()
        text = data.decode("utf-8", errors="replace")

    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    slug_to_rating: Dict[str, float] = {}
    for row in reader:
        rating = parse_float(row.get("Rating", ""))
        slug = (row.get("Title Slug", "") or "").strip().lower()
        if rating is None or not slug:
            continue
        slug_to_rating[slug] = rating
    return slug_to_rating


def load_title_to_slug_map() -> Dict[str, str]:
    """Build a normalized title -> slug map from stats/slug_data.csv."""
    mapping: Dict[str, str] = {}
    try:
        df = pd.read_csv(SLUG_DATA_CSV)
    except Exception:
        return mapping
    if "title" in df.columns and "titleSlug" in df.columns:
        for _, row in df[["title", "titleSlug"]].dropna().iterrows():
            title = str(row["title"]).strip().lower()
            slug = str(row["titleSlug"]).strip().lower()
            if title and slug:
                mapping[title] = slug
    return mapping


def pick_slugs_from_sheet(df: pd.DataFrame, title_to_slug: Dict[str, str]) -> List[str]:
    cols = {c.lower(): c for c in df.columns}

    # Prefer explicit slug column if present
    for cand in ["titleslug", "title_slug", "slug", "title slug"]:
        if cand in cols:
            s = (
                df[cols[cand]]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": np.nan})
                .dropna()
            )
            return [x for x in s.tolist() if x]

    # Try to parse from URL if present (known URL-ish column names)
    for cand in ["url", "link", "leetcode link", "problem url", "question link"]:
        if cand in cols:
            slugs = []
            for v in df[cols[cand]].astype(str).tolist():
                slug = extract_slug_from_url(v)
                if slug:
                    slugs.append(slug)
            if slugs:
                return slugs

    # Fallback: map from Title via slug_data.csv
    for cand in ["title", "problem", "question", "name"]:
        if cand in cols:
            titles = (
                df[cols[cand]].astype(str).str.strip().str.lower().replace({"nan": np.nan}).dropna()
            )
            slugs = [title_to_slug.get(t, None) for t in titles.tolist()]
            slugs = [s for s in slugs if s]
            if slugs:
                return slugs

    # Last-resort: scan all headers and all cells for any LeetCode URLs.
    # Some sheets have the first row mistakenly set as header, with the header itself being a URL.
    found: List[str] = []

    # Scan headers themselves
    for col in df.columns:
        slug = extract_slug_from_url(str(col))
        if slug:
            found.append(slug)

    # Scan all cells
    for col in df.columns:
        series = df[col].astype(str)
        for v in series.tolist():
            slug = extract_slug_from_url(v)
            if slug:
                found.append(slug)

    # Preserve order but drop duplicates
    seen = set()
    deduped: List[str] = []
    for s in found:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    return deduped


def compute_median(nums: List[float]) -> Optional[float]:
    if not nums:
        return None
    return float(statistics.median(nums))


def main() -> int:
    if not os.path.isfile(COMPANY_XLSX):
        print(f"Missing Excel file: {COMPANY_XLSX}", file=sys.stderr)
        return 1

    try:
        slug_to_rating = fetch_ratings_by_slug()
    except (URLError, HTTPError, Exception) as e:
        print(f"Failed to fetch remote ratings: {e}", file=sys.stderr)
        return 2

    title_to_slug = load_title_to_slug_map()

    try:
        xls = pd.ExcelFile(COMPANY_XLSX)
    except Exception as e:
        print(
            "Failed to open Excel. Install 'openpyxl' (e.g., pip install openpyxl).",
            file=sys.stderr,
        )
        print(str(e), file=sys.stderr)
        return 3

    results: List[Tuple[str, float, float, int, int, float]] = []  # (company, median, p90, rated_cnt, total_cnt, pct)

    for sheet_name in xls.sheet_names:
        try:
            df = xls.parse(sheet_name)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        slugs = pick_slugs_from_sheet(df, title_to_slug)
        if not slugs:
            continue

        total_cnt = len(slugs)
        ratings = {s: slug_to_rating[s] for s in slugs if s in slug_to_rating}
        rating_values = list(ratings.values())
        median = compute_median(rating_values)
        if median is None:
            continue
        # 90th percentile of ratings for this company
        try:
            p90 = float(np.percentile(rating_values, 90)) if rating_values else None
        except Exception:
            p90 = None
        rated_cnt = len(ratings)
        pct = (100.0 * rated_cnt / total_cnt) if total_cnt else 0.0
        results.append((sheet_name, median, p90 if p90 is not None else float('nan'), rated_cnt, total_cnt, pct))

    # Rank companies by median rating (higher -> harder) then by name
    results.sort(key=lambda x: (-x[1], x[0].lower()))

    print("Company,MedianRating,P90Rating,QuestionCount,RatedPct")
    for company, med, p90, rated_cnt, total_cnt, pct in results:
        if rated_cnt < 15:
            continue
        p90_str = "" if p90 != p90 else str(round(p90, 1))  # NaN check
        print(f"{company},{round(med, 1)},{p90_str},{rated_cnt}/{total_cnt},{round(pct, 1)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
