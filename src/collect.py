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
from dotenv import load_dotenv

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
    sample_size: int = 0
    sample_query: str = "is:public"
    sample_min_stars: int = 1
    sample_max_stars: int = 1_000_000


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
            next_params = None #в url уже есть
        else:
            if next_params is None:
                return
            page += 1
            next_params["page"] = page
            continue

        page += 1


# -----------------------------
# статы
# -----------------------------
def get_commit_activity_52w(session: requests.Session, repo: str, cfg: CollectorConfig) -> List[Dict[str, Any]]:
    """
    GET /repos/{owner}/{repo}/stats/commit_activity
    """
    owner, name = repo.split("/", 1)
    url = f"{GITHUB_API}/repos/{owner}/{name}/stats/commit_activity"

    max_polls = 12
    sleep_s = 3

    for i in range(max_polls):
        resp = session.get(url, timeout=cfg.timeout_s)
        if resp.status_code == 202:
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
#pr/issueses
# -----------------------------
def search_total_count(session: requests.Session, q: str, cfg: CollectorConfig) -> int:
    """
    GET /search/issues?q=...
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
    за последние n дней:
     issues_opened: created in range
     issues_closed
     issues_open_now
    """
    start, end = period_range_days(cfg)

    base_repo = f"repo:{repo}"

    # Issues
    q_issues_opened = f"{base_repo} type:issue created:{start}..{end}"
    q_issues_closed = f"{base_repo} type:issue closed:{start}..{end}"
    q_issues_open_now = f"{base_repo} type:issue is:open"
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
# контрибьюторы и релизы
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
    релизы за последние N дней и средний перерыв между ними
    """
    start_dt = utc_now() - timedelta(days=cfg.period_days)
    times: List[datetime] = []
    for r in releases:
        ts = r.get("published_at") or r.get("created_at")
        if not ts:
            continue
        # гитхаб стандарттайм: "YYYY-MM-DDTHH:MM:SSZ"
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
# массовый сэмплинг репозиториев (для большого датасета, напр. 10000 сэмплов)
# -----------------------------
SEARCH_PER_PAGE = 100
SEARCH_MAX_PAGES = 10  # GitHub Search API отдаёт максимум 1000 результатов (10 * 100) на один запрос
SEARCH_MAX_RESULTS_PER_QUERY = 1000
SEARCH_SLEEP_S = 1.2  # Search API лимитирован жёстче основного (30/мин с токеном), не спамим


def repo_search_request(
    session: requests.Session, cfg: CollectorConfig, params: Dict[str, Any]
) -> Tuple[Any, requests.Response]:
    url = f"{GITHUB_API}/search/repositories"
    data, resp = request_json(session, url, params, cfg)
    time.sleep(SEARCH_SLEEP_S)
    return data, resp


def repo_search_count(session: requests.Session, cfg: CollectorConfig, query: str) -> int:
    data, resp = repo_search_request(session, cfg, {"q": query, "per_page": 1})
    return int((data or {}).get("total_count", 0))


def iter_star_range_buckets(
    session: requests.Session,
    cfg: CollectorConfig,
    base_query: str,
    lo: int,
    hi: int,
    cap: int = SEARCH_MAX_RESULTS_PER_QUERY,
) -> Iterable[Tuple[int, int, int]]:
    """
    Search API отдаёт не больше 1000 результатов на запрос, поэтому диапазон
    звёзд рекурсивно делим пополам, пока в каждом куске не станет <= cap репозиториев.

    Лениво (генератор): всегда сначала до конца раскрывает "верхнюю" (более
    звёздную) половину текущего диапазона, прежде чем трогать нижнюю — поэтому
    бакеты приходят примерно от самых популярных репо к менее популярным.
    Вызывающий код может прекратить обход (break) в любой момент, как только
    наберёт нужное число сэмплов — оставшийся диапазон вообще не будет запрошен.
    """
    stack: List[Tuple[int, int]] = [(lo, hi)]

    while stack:
        cur_lo, cur_hi = stack.pop()
        if cur_lo > cur_hi:
            continue
        q = f"{base_query} stars:{cur_lo}..{cur_hi}"
        count = repo_search_count(session, cfg, q)
        if count == 0:
            continue
        if count <= cap or cur_lo == cur_hi:
            yield (cur_lo, cur_hi, count)
            continue
        mid = (cur_lo + cur_hi) // 2
        stack.append((cur_lo, mid))       # нижняя половина - на потом
        stack.append((mid + 1, cur_hi))   # верхняя половина - следующей (LIFO)


def flatten_search_repo(item: Dict[str, Any]) -> Dict[str, Any]:
    owner = item.get("owner") or {}
    license_info = item.get("license") or {}
    topics = item.get("topics") or []
    return {
        "id": item.get("id"),
        "full_name": item.get("full_name"),
        "owner_login": owner.get("login"),
        "owner_type": owner.get("type"),
        "description": (item.get("description") or "").replace("\n", " ").strip(),
        "html_url": item.get("html_url"),
        "language": item.get("language"),
        "stargazers_count": item.get("stargazers_count"),
        "forks_count": item.get("forks_count"),
        "watchers_count": item.get("watchers_count"),
        "open_issues_count": item.get("open_issues_count"),
        "size_kb": item.get("size"),
        "default_branch": item.get("default_branch"),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "pushed_at": item.get("pushed_at"),
        "archived": item.get("archived"),
        "disabled": item.get("disabled"),
        "is_fork": item.get("fork"),
        "license_key": license_info.get("key"),
        "topics": ";".join(topics),
        "visibility": item.get("visibility"),
        "has_issues": item.get("has_issues"),
        "has_projects": item.get("has_projects"),
        "has_wiki": item.get("has_wiki"),
        "has_pages": item.get("has_pages"),
    }


def collect_repo_sample(
    session: requests.Session,
    cfg: CollectorConfig,
    target_count: int,
    base_query: str,
    min_stars: int,
    max_stars: int,
) -> List[Dict[str, Any]]:
    """
    Лёгкий сбор метаданных большого числа репозиториев через /search/repositories.
    В отличие от collect_repo (глубокая статистика по паре репо), тут только
    базовые метаданные, зато можно набрать 10000+ сэмплов в разумное время/лимиты.
    """
    logging.info(
        "Sampling repos: target=%s query=%r stars=[%s..%s]",
        target_count, base_query, min_stars, max_stars,
    )

    sample_dir = cfg.out_raw / "sample"
    ensure_dir(sample_dir)

    seen_ids: set = set()
    rows: List[Dict[str, Any]] = []
    page_counter = 0

    bucket_iter = iter_star_range_buckets(session, cfg, base_query, min_stars, max_stars)
    while len(rows) < target_count:
        try:
            lo, hi, count = next(bucket_iter)
        except StopIteration:
            break
        capped = min(count, SEARCH_MAX_RESULTS_PER_QUERY)
        pages = min(SEARCH_MAX_PAGES, (capped + SEARCH_PER_PAGE - 1) // SEARCH_PER_PAGE)
        query = f"{base_query} stars:{lo}..{hi}"

        for page in range(1, pages + 1):
            if len(rows) >= target_count:
                break
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": SEARCH_PER_PAGE,
                "page": page,
            }
            data, resp = repo_search_request(session, cfg, params)
            items = (data or {}).get("items", [])
            if not items:
                break

            page_counter += 1
            save_envelope(
                sample_dir / f"sample__page{page_counter:04d}.json",
                data=items,
                url=resp.url,
                params=None,
                resp=resp,
            )

            for item in items:
                rid = item.get("id")
                if rid is None or rid in seen_ids:
                    continue
                seen_ids.add(rid)
                rows.append(flatten_search_repo(item))
                if len(rows) >= target_count:
                    break

            if len(items) < SEARCH_PER_PAGE:
                break

        logging.info(
            "Bucket stars:%s..%s -> collected so far: %s/%s", lo, hi, len(rows), target_count
        )

    logging.info("Collected %s unique repositories via search sampling.", len(rows))
    return rows


#метаданные

def collect_repo(session: requests.Session, repo: str, cfg: CollectorConfig) -> Dict[str, Any]:
    owner, name = repo.split("/", 1)
    repo_slug = slugify_repo(repo)
    url = f"{GITHUB_API}/repos/{owner}/{name}"
    repo_meta, resp = request_json(session, url, None, cfg)
    save_envelope(cfg.out_raw / f"repo__{repo_slug}.json", data=repo_meta, url=url, params=None, resp=resp)

    # коммиты за последние 52 недели
    commit_activity = get_commit_activity_52w(session, repo, cfg)
    safe_write_json(cfg.out_raw / f"commit_activity_52w__{repo_slug}.json", {"collected_at": iso_z(utc_now()), "data": commit_activity})

    # Issues/PR колво
    counts = collect_issue_pr_counts(session, repo, cfg)
    safe_write_json(cfg.out_raw / f"counts__{repo_slug}.json", {"collected_at": iso_z(utc_now()), "period_days": cfg.period_days, "data": counts})

    # контрибьюторы
    contributors = collect_contributors(session, repo, cfg)
    safe_write_json(
        cfg.out_raw / f"contributors__{repo_slug}__ALL.json",
        {"collected_at": iso_z(utc_now()), "data": contributors},
    )

    # рилиз листы с статой
    releases = collect_releases(session, repo, cfg)
    safe_write_json(
        cfg.out_raw / f"releases__{repo_slug}__ALL.json",
        {"collected_at": iso_z(utc_now()), "data": releases},
    )
    rel_freq = release_frequency_summary(releases, cfg)

    #
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
    p.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Собрать N репозиториев через Search API для широкого датасета (0 = выкл, напр. 10000).",
    )
    p.add_argument(
        "--sample-query",
        default="is:public",
        help="Базовый поисковый запрос для сэмплинга (в дополнение к stars:LO..HI).",
    )
    p.add_argument("--sample-min-stars", type=int, default=1)
    p.add_argument("--sample-max-stars", type=int, default=1_000_000)
    return p.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
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
        sample_size=max(args.sample_size, 0),
        sample_query=args.sample_query,
        sample_min_stars=max(args.sample_min_stars, 1),
        sample_max_stars=max(args.sample_max_stars, args.sample_min_stars),
    )
    ensure_dir(cfg.out_raw)
    ensure_dir(cfg.out_processed)

    session = make_session(cfg)

    summary_rows: List[Dict[str, Any]] = []
    for repo in args.repos:
        logging.info("=== Collecting %s ===", repo)
        row = collect_repo(session, repo, cfg)
        summary_rows.append(row)

    write_summary_csv(summary_rows, cfg.out_processed / "summary.csv")
    safe_write_json(cfg.out_processed / "summary.json", {"collected_at": iso_z(utc_now()), "data": summary_rows})

    if cfg.sample_size > 0:
        logging.info("=== Collecting repo sample (target=%s) ===", cfg.sample_size)
        sample_rows = collect_repo_sample(
            session,
            cfg,
            cfg.sample_size,
            cfg.sample_query,
            cfg.sample_min_stars,
            cfg.sample_max_stars,
        )
        write_summary_csv(sample_rows, cfg.out_processed / "sample.csv")
        safe_write_json(
            cfg.out_processed / "sample.json",
            {"collected_at": iso_z(utc_now()), "count": len(sample_rows), "data": sample_rows},
        )
        logging.info("Sample dataset -> %s rows (%s)", len(sample_rows), cfg.out_processed / "sample.csv")

    logging.info("Done. Raw -> %s | Processed -> %s", cfg.out_raw.resolve(), cfg.out_processed.resolve())


if __name__ == "__main__":
    main()
