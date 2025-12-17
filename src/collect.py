# src/collect.py
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

GITHUB_API = "https://api.github.com"
DEFAULT_ACCEPT = "application/vnd.github+json"
DEFAULT_API_VERSION = "2022-11-28"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_REPOS = [
    "jupyter/notebook",
    "jupyterlab/jupyterlab",
    "microsoft/vscode",
]

@dataclass(frozen=True)
class CollectorConfig:
    out_raw: Path
    out_processed: Path
    token: Optional[str]
    per_page: int
    max_pages: int
    period_days: int
    timeout_s: int
    verbose: bool


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def slugify_repo(repo: str) -> str:
    return repo.replace("/", "__")


def safe_write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def make_session(cfg: CollectorConfig) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Accept": DEFAULT_ACCEPT,
            "X-GitHub-Api-Version": DEFAULT_API_VERSION,
            "User-Agent": "itmo-se-oss-analysis/1.0",
        }
    )
    if cfg.token:
        s.headers["Authorization"] = f"Bearer {cfg.token}"
    return s


def parse_link_header(link_header: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not link_header:
        return out
    parts = [p.strip() for p in link_header.split(",")]
    for part in parts:
        if ";" not in part:
            continue
        url_part, rel_part = part.split(";", 1)
        url = url_part.strip().lstrip("<").rstrip(">")
        rel = rel_part.strip()
        if 'rel="' in rel:
            rel_name = rel.split('rel="', 1)[1].split('"', 1)[0]
            out[rel_name] = url
    return out


def handle_rate_limit(resp: requests.Response) -> None:
    """
    если рейтлимит то мы ждем спада
    403 ошибка то мы спим немного
    """
    remaining = resp.headers.get("X-RateLimit-Remaining")
    reset = resp.headers.get("X-RateLimit-Reset")

    if remaining == "0" and reset:
        try:
            reset_ts = int(reset)
            now_ts = int(time.time())
            wait_s = max(0, reset_ts - now_ts) + 2
            logging.warning("Rate limit reached. Sleeping %ss...", wait_s)
            time.sleep(wait_s)
        except Exception:
            pass

    if resp.status_code == 403:
        try:
            msg = (resp.json() or {}).get("message", "")
        except Exception:
            msg = resp.text
        if "secondary rate limit" in msg.lower():
            logging.warning("403. ждем 20с")
            time.sleep(20)


def request_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]],
    cfg: CollectorConfig,
    retries: int = 6,
) -> Tuple[Any, requests.Response]:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=cfg.timeout_s)
            if resp.status_code in (429, 500, 502, 503, 504):
                handle_rate_limit(resp)
                backoff = min(2 ** attempt, 30)
                logging.warning("HTTP %s. Retry in %ss (%s/%s).", resp.status_code, backoff, attempt, retries)
                time.sleep(backoff)
                continue

            if resp.status_code == 403:
                handle_rate_limit(resp)

            resp.raise_for_status()
            return resp.json(), resp

        except Exception as e:
            last_err = e
            backoff = min(2 ** attempt, 30)
            logging.warning("Request failed: %s. Retry in %ss (%s/%s).", e, backoff, attempt, retries)
            time.sleep(backoff)

    raise RuntimeError(f"Failed after {retries} retries: {url}") from last_err


def save_envelope(path: Path, *, data: Any, url: str, params: Optional[Dict[str, Any]], resp: requests.Response) -> None:
    envelope = {
        "meta": {
            "collected_at": iso_z(utc_now()),
            "url": url,
            "params": params,
            "status_code": resp.status_code,
            "rate_limit": {
                "limit": resp.headers.get("X-RateLimit-Limit"),
                "remaining": resp.headers.get("X-RateLimit-Remaining"),
                "reset": resp.headers.get("X-RateLimit-Reset"),
                "resource": resp.headers.get("X-RateLimit-Resource"),
            },
        },
        "data": data,
    }
    safe_write_json(path, envelope)


def paginated_get(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    cfg: CollectorConfig,
) -> Iterable[Tuple[int, Any, requests.Response, str]]:
    page = 1
    next_url = url
    next_params: Optional[Dict[str, Any]] = dict(params)

    while True:
        if page > cfg.max_pages:
            logging.info("Reached max_pages=%s for %s", cfg.max_pages, url)
            return

        data, resp = request_json(session, next_url, next_params, cfg)
        yield page, data, resp, resp.url

        link = parse_link_header(resp.headers.get("Link", ""))
        if "next" in link:
            next_url = link["next"]
            next_params = None  # already encoded in URL
        else:
            if next_params is None:
                return
            page += 1
            next_params["page"] = page
            continue

        page += 1


# -----------------------------
# GitHub “stats” (commit activity)
# -----------------------------
def get_commit_activity_52w(session: requests.Session, repo: str, cfg: CollectorConfig) -> List[Dict[str, Any]]:
    """
    GET /repos/{owner}/{repo}/stats/commit_activity
    Can return 202 Accepted while GitHub generates stats. We poll.
    """
    owner, name = repo.split("/", 1)
    url = f"{GITHUB_API}/repos/{owner}/{name}/stats/commit_activity"

    max_polls = 12
    sleep_s = 3

    for i in range(max_polls):
        resp = session.get(url, timeout=cfg.timeout_s)
        if resp.status_code == 202:
            # GitHub is computing statistics
            logging.info("Stats not ready (202) for %s. Poll %s/%s, sleep %ss...", repo, i + 1, max_polls, sleep_s)
            time.sleep(sleep_s)
            sleep_s = min(sleep_s * 2, 30)
            continue

        if resp.status_code == 403:
            handle_rate_limit(resp)

        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return []

    logging.warning("Commit activity stats not ready after polls for %s; returning empty.", repo)
    return []


# -----------------------------
# Search API counts (issues / PR)
# -----------------------------
def search_total_count(session: requests.Session, q: str, cfg: CollectorConfig) -> int:
    """
    GET /search/issues?q=...
    Returns JSON with total_count.
    """
    url = f"{GITHUB_API}/search/issues"
    data, resp = request_json(session, url, {"q": q, "per_page": 1}, cfg)
    save_envelope(
        cfg.out_raw / f"search__{hash(q)}.json",
        data=data,
        url=resp.url,
        params=None,
        resp=resp,
    )
    return int((data or {}).get("total_count", 0))


def period_range_days(cfg: CollectorConfig) -> Tuple[str, str]:
    end = utc_now().date()
    start = (utc_now() - timedelta(days=cfg.period_days)).date()
    return start.isoformat(), end.isoformat()


def collect_issue_pr_counts(session: requests.Session, repo: str, cfg: CollectorConfig) -> Dict[str, int]:
    """
    For last N days:
    - issues_opened: created in range
    - issues_closed: closed in range
    - issues_open_now: current open issues (no time range)
    Optional:
    - prs_opened: PR created in range
    - prs_merged: PR merged in range
    """
    start, end = period_range_days(cfg)

    base_repo = f"repo:{repo}"

    # Issues
    q_issues_opened = f"{base_repo} type:issue created:{start}..{end}"
    q_issues_closed = f"{base_repo} type:issue closed:{start}..{end}"
    q_issues_open_now = f"{base_repo} type:issue is:open"

    # PRs (optional but useful)
    q_prs_opened = f"{base_repo} type:pr created:{start}..{end}"
    q_prs_merged = f"{base_repo} type:pr is:merged merged:{start}..{end}"

    counts = {
        "issues_opened_period": search_total_count(session, q_issues_opened, cfg),
        "issues_closed_period": search_total_count(session, q_issues_closed, cfg),
        "issues_open_now": search_total_count(session, q_issues_open_now, cfg),
        "prs_opened_period": search_total_count(session, q_prs_opened, cfg),
        "prs_merged_period": search_total_count(session, q_prs_merged, cfg),
    }
    return counts


# -----------------------------
# Contributors + releases
# -----------------------------
def collect_contributors(session: requests.Session, repo: str, cfg: CollectorConfig) -> List[Dict[str, Any]]:
    owner, name = repo.split("/", 1)
    repo_slug = slugify_repo(repo)

    url = f"{GITHUB_API}/repos/{owner}/{name}/contributors"
    params = {"per_page": cfg.per_page, "page": 1, "anon": "true"}

    all_items: List[Dict[str, Any]] = []
    for page, data, resp, final_url in paginated_get(session, url, params, cfg):
        if not isinstance(data, list):
            break
        all_items.extend(data)
        save_envelope(
            cfg.out_raw / f"contributors__{repo_slug}__page{page:03d}.json",
            data=data,
            url=final_url,
            params=None,
            resp=resp,
        )
        if len(data) < cfg.per_page:
            break

    return all_items


def collect_releases(session: requests.Session, repo: str, cfg: CollectorConfig) -> List[Dict[str, Any]]:
    owner, name = repo.split("/", 1)
    repo_slug = slugify_repo(repo)

    url = f"{GITHUB_API}/repos/{owner}/{name}/releases"
    params = {"per_page": cfg.per_page, "page": 1}

    all_items: List[Dict[str, Any]] = []
    for page, data, resp, final_url in paginated_get(session, url, params, cfg):
        if not isinstance(data, list):
            break
        all_items.extend(data)
        save_envelope(
            cfg.out_raw / f"releases__{repo_slug}__page{page:03d}.json",
            data=data,
            url=final_url,
            params=None,
            resp=resp,
        )
        if len(data) < cfg.per_page:
            break

    return all_items


def release_frequency_summary(releases: List[Dict[str, Any]], cfg: CollectorConfig) -> Dict[str, Any]:
    """
    Quick summary for last N days:
    - releases_in_period
    - avg_days_between_releases (approx)
    """
    start_dt = utc_now() - timedelta(days=cfg.period_days)
    times: List[datetime] = []
    for r in releases:
        ts = r.get("published_at") or r.get("created_at")
        if not ts:
            continue
        # GitHub timestamps: "YYYY-MM-DDTHH:MM:SSZ"
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if dt >= start_dt:
            times.append(dt)

    times.sort()
    n = len(times)
    if n < 2:
        return {"releases_in_period": n, "avg_days_between_releases": None}

    deltas = [(times[i] - times[i - 1]).total_seconds() / 86400 for i in range(1, n)]
    avg = sum(deltas) / len(deltas)
    return {"releases_in_period": n, "avg_days_between_releases": round(avg, 2)}


# -----------------------------
# Core collection per repo
# -----------------------------
def collect_repo(session: requests.Session, repo: str, cfg: CollectorConfig) -> Dict[str, Any]:
    owner, name = repo.split("/", 1)
    repo_slug = slugify_repo(repo)

    # Repo meta
    url = f"{GITHUB_API}/repos/{owner}/{name}"
    repo_meta, resp = request_json(session, url, None, cfg)
    save_envelope(cfg.out_raw / f"repo__{repo_slug}.json", data=repo_meta, url=url, params=None, resp=resp)

    # Weekly commits (52w)
    commit_activity = get_commit_activity_52w(session, repo, cfg)
    safe_write_json(cfg.out_raw / f"commit_activity_52w__{repo_slug}.json", {"collected_at": iso_z(utc_now()), "data": commit_activity})

    # Issues/PR counts over period (Search API)
    counts = collect_issue_pr_counts(session, repo, cfg)
    safe_write_json(cfg.out_raw / f"counts__{repo_slug}.json", {"collected_at": iso_z(utc_now()), "period_days": cfg.period_days, "data": counts})

    # Contributors list
    contributors = collect_contributors(session, repo, cfg)
    safe_write_json(
        cfg.out_raw / f"contributors__{repo_slug}__ALL.json",
        {"collected_at": iso_z(utc_now()), "data": contributors},
    )

    # Releases list (+ frequency quick summary)
    releases = collect_releases(session, repo, cfg)
    safe_write_json(
        cfg.out_raw / f"releases__{repo_slug}__ALL.json",
        {"collected_at": iso_z(utc_now()), "data": releases},
    )
    rel_freq = release_frequency_summary(releases, cfg)

    # Small processed summary row
    start, end = period_range_days(cfg)
    weekly_total = sum((w.get("total", 0) for w in commit_activity if isinstance(w, dict)))
    return {
        "repo": repo,
        "period_start": start,
        "period_end": end,
        "stars": repo_meta.get("stargazers_count"),
        "forks": repo_meta.get("forks_count"),
        "weekly_commits_52w_total": weekly_total,
        "contributors_collected": len(contributors),
        **counts,
        **rel_freq,
    }


def write_summary_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect OSS repo data for analysis (ITMO SE × Yandex Practice).")
    p.add_argument("--repos", nargs="*", default=DEFAULT_REPOS, help='List of "owner/repo".')
    p.add_argument("--period-days", type=int, default=364, help="Period length in days (default ~52 weeks).")
    p.add_argument("--per-page", type=int, default=100, help="per_page (max 100).")
    p.add_argument("--max-pages", type=int, default=10, help="Max pages per paginated endpoint (contributors/releases).")
    p.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logs.")
    p.add_argument("--raw-dir", default=str(DEFAULT_DATA_DIR / "raw"))
    p.add_argument("--processed-dir", default=str(DEFAULT_DATA_DIR / "processed"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        logging.warning("GITHUB_TOKEN is not set. You may hit rate limits.")

    cfg = CollectorConfig(
        out_raw=Path(args.raw_dir),
        out_processed=Path(args.processed_dir),
        token=token,
        per_page=min(max(args.per_page, 1), 100),
        max_pages=max(args.max_pages, 1),
        period_days=max(args.period_days, 1),
        timeout_s=max(args.timeout, 5),
        verbose=args.verbose,
    )
    ensure_dir(cfg.out_raw)
    ensure_dir(cfg.out_processed)

    session = make_session(cfg)

    summary_rows: List[Dict[str, Any]] = []
    for repo in args.repos:
        logging.info("=== Collecting %s ===", repo)
        row = collect_repo(session, repo, cfg)
        summary_rows.append(row)

    # quick processed summary
    write_summary_csv(summary_rows, cfg.out_processed / "summary.csv")
    safe_write_json(cfg.out_processed / "summary.json", {"collected_at": iso_z(utc_now()), "data": summary_rows})

    logging.info("Done. Raw -> %s | Processed -> %s", cfg.out_raw.resolve(), cfg.out_processed.resolve())


if __name__ == "__main__":
    main()
