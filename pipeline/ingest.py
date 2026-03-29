"""
ingest.py
─────────────────────────────────────────────────────────────────────────────
Loads Pushshift Reddit data (JSON / JSONL / zst) into a DuckDB database.

Usage:
    python pipeline/ingest.py --input data/raw/ --db data/processed/reddit.duckdb

Supports:
    - .json         single JSON array file
    - .jsonl        newline-delimited JSON (most common Pushshift format)
    - .zst          zstandard-compressed JSONL (native Pushshift format)
    - directory     processes all supported files in the folder recursively

Subreddits filtered to: r/technology, r/gadgets, r/apple
(mirrors a content-feed product surface analogous to Meta's social feeds)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Generator

import duckdb
import zstandard as zstd

# ─── Config ───────────────────────────────────────────────────────────────────

TARGET_SUBREDDITS = {"technology", "gadgets", "apple"}

# Only columns we need — keeps DuckDB lean
REQUIRED_FIELDS = {
    "id",
    "subreddit",
    "title",
    "selftext",
    "score",
    "num_comments",
    "upvote_ratio",
    "url",
    "is_video",
    "created_utc",
}

BATCH_SIZE = 5_000   # rows per DuckDB insert batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Schema ───────────────────────────────────────────────────────────────────

DDL_POSTS = """
CREATE TABLE IF NOT EXISTS posts (
    post_id          VARCHAR PRIMARY KEY,
    subreddit        VARCHAR,
    title            VARCHAR,
    selftext         VARCHAR,
    score            INTEGER,
    num_comments     INTEGER,
    upvote_ratio     FLOAT,
    url              VARCHAR,
    is_video         BOOLEAN,
    created_utc      BIGINT,
    hours_since_post FLOAT
);
"""

DDL_NES = """
CREATE TABLE IF NOT EXISTS nes_scores (
    post_id          VARCHAR PRIMARY KEY,
    nes              FLOAT,
    engagement_tier  VARCHAR,
    week             DATE,
    computed_at      TIMESTAMP DEFAULT current_timestamp
);
"""

DDL_TRIBE = """
CREATE TABLE IF NOT EXISTS tribe_scores (
    post_id          VARCHAR PRIMARY KEY,
    ces              FLOAT,
    nes              FLOAT,
    computed_at      TIMESTAMP DEFAULT current_timestamp
);
"""


# ─── Readers ──────────────────────────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Generator[dict, None, None]:
    """Yield records from a plain .jsonl file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _iter_json(path: Path) -> Generator[dict, None, None]:
    """Yield records from a .json array file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        yield data


def _iter_zst(path: Path) -> Generator[dict, None, None]:
    """Yield records from a zstandard-compressed .jsonl file."""
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buffer = b""
            while True:
                chunk = reader.read(2**20)   # 1 MB chunks
                if not chunk:
                    break
                buffer += chunk
                lines = buffer.split(b"\n")
                buffer = lines[-1]           # keep incomplete line
                for line in lines[:-1]:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
            # flush remaining buffer
            if buffer.strip():
                try:
                    yield json.loads(buffer)
                except json.JSONDecodeError:
                    pass


def _get_reader(path: Path):
    """Return the appropriate reader for a file based on its suffix."""
    suffix = path.suffix.lower()
    if suffix == ".zst":
        return _iter_zst(path)
    elif suffix == ".jsonl":
        return _iter_jsonl(path)
    elif suffix == ".json":
        return _iter_json(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _collect_files(input_path: Path) -> list[Path]:
    """Return all supported data files under input_path."""
    supported = {".json", ".jsonl", ".zst"}
    if input_path.is_file():
        return [input_path]
    files = [
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
    ]
    return sorted(files)


# ─── Transform ────────────────────────────────────────────────────────────────

def _transform(raw: dict) -> dict | None:
    """
    Clean and filter a raw Pushshift record.

    Returns None if the record should be skipped (wrong subreddit,
    deleted post, missing fields).
    """
    subreddit = (raw.get("subreddit") or "").lower()
    if subreddit not in TARGET_SUBREDDITS:
        return None

    selftext = raw.get("selftext", "") or ""
    if selftext in ("[deleted]", "[removed]", ""):
        selftext = None

    created_utc = raw.get("created_utc")
    if not created_utc:
        return None

    # Convert epoch → hours since post (using now as reference for historical data
    # we treat created_utc as-is; downstream NES uses this for velocity)
    hours_since_post = max(
        (time.time() - float(created_utc)) / 3600,
        1.0   # floor at 1h to avoid division by zero
    )

    return {
        "post_id":          str(raw.get("id", "")),
        "subreddit":        subreddit,
        "title":            (raw.get("title") or "")[:500],   # cap length
        "selftext":         selftext,
        "score":            int(raw.get("score") or 0),
        "num_comments":     int(raw.get("num_comments") or 0),
        "upvote_ratio":     float(raw.get("upvote_ratio") or 0.5),
        "url":              (raw.get("url") or "")[:1000],
        "is_video":         bool(raw.get("is_video", False)),
        "created_utc":      int(created_utc),
        "hours_since_post": round(hours_since_post, 2),
    }


# ─── Ingest ───────────────────────────────────────────────────────────────────

def ingest(input_path: Path, db_path: Path) -> bool:
    """
    Main ingestion function.

    Reads all supported files from input_path, filters to target subreddits,
    transforms records, and bulk-inserts into DuckDB in batches.

    Returns False if no supported input files were found (no database file).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    files = _collect_files(input_path)

    if not files:
        return False

    log.info("Found %d file(s) to process", len(files))

    con = duckdb.connect(str(db_path))
    con.execute(DDL_POSTS)
    con.execute(DDL_NES)
    con.execute(DDL_TRIBE)
    log.info("DuckDB schema initialised at: %s", db_path)

    total_seen = 0
    total_inserted = 0
    total_skipped = 0

    for file_path in files:
        log.info("Processing: %s", file_path.name)
        batch: list[tuple] = []

        try:
            for raw in _get_reader(file_path):
                total_seen += 1
                record = _transform(raw)

                if record is None:
                    total_skipped += 1
                    continue

                batch.append((
                    record["post_id"],
                    record["subreddit"],
                    record["title"],
                    record["selftext"],
                    record["score"],
                    record["num_comments"],
                    record["upvote_ratio"],
                    record["url"],
                    record["is_video"],
                    record["created_utc"],
                    record["hours_since_post"],
                ))

                if len(batch) >= BATCH_SIZE:
                    _insert_batch(con, batch)
                    total_inserted += len(batch)
                    log.info(
                        "  Inserted %d rows (total: %d)",
                        len(batch), total_inserted
                    )
                    batch = []

        except Exception as exc:
            log.warning("Error reading %s: %s", file_path.name, exc)
            continue

        # flush remaining
        if batch:
            _insert_batch(con, batch)
            total_inserted += len(batch)

    con.close()

    log.info("─" * 60)
    log.info("Ingestion complete.")
    log.info("  Records seen     : %d", total_seen)
    log.info("  Records inserted : %d", total_inserted)
    log.info("  Records skipped  : %d", total_skipped)
    log.info("  Database         : %s", db_path)
    return True


def _insert_batch(con: duckdb.DuckDBPyConnection, batch: list[tuple]) -> None:
    """
    Insert a batch of records into the posts table.
    Uses INSERT OR IGNORE to handle duplicate post_ids gracefully.
    """
    con.executemany(
        """
        INSERT OR IGNORE INTO posts (
            post_id, subreddit, title, selftext,
            score, num_comments, upvote_ratio, url,
            is_video, created_utc, hours_since_post
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )


# ─── Verify ───────────────────────────────────────────────────────────────────

def verify(db_path: Path) -> None:
    """Print a quick summary of what was loaded."""
    if not db_path.exists():
        log.error(
            "Cannot verify: database does not exist at %s "
            "(nothing was ingested or the path is wrong).",
            db_path,
        )
        return

    con = duckdb.connect(str(db_path), read_only=True)

    total = con.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    by_sub = con.execute(
        "SELECT subreddit, COUNT(*) AS n FROM posts GROUP BY subreddit ORDER BY n DESC"
    ).fetchall()
    video_pct = con.execute(
        "SELECT ROUND(AVG(is_video::INT) * 100, 1) FROM posts"
    ).fetchone()[0]
    avg_score = con.execute("SELECT ROUND(AVG(score), 1) FROM posts").fetchone()[0]

    log.info("─" * 60)
    log.info("DATABASE SUMMARY")
    log.info("  Total posts  : %d", total)
    log.info("  Video posts  : %s%%", video_pct)
    log.info("  Avg score    : %s", avg_score)
    log.info("  By subreddit :")
    for sub, n in by_sub:
        log.info("    r/%-15s  %d", sub, n)

    con.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest Pushshift Reddit data into DuckDB."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a .json/.jsonl/.zst file or directory of files.",
    )
    parser.add_argument(
        "--db", "-d",
        default="data/processed/reddit.duckdb",
        help="Path to the DuckDB database file (created if missing).",
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Print a summary of the database after ingestion.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    input_path = Path(args.input)
    db_path = Path(args.db)

    ran = ingest(input_path, db_path)

    if not ran:
        base = (
            "No supported files (.json, .jsonl, .zst) under %s; "
            "no database at %s."
            % (input_path, db_path)
        )
        if args.verify:
            log.error("%s Add data under %s and run again.", base, input_path)
            sys.exit(1)
        log.error(base)
    elif args.verify:
        verify(db_path)