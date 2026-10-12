#!/usr/bin/env python3
"""Notion Observation Log watcher -- local-only glue, not part of CI.

Reads the Observation Log pages the pipeline already tracks (via
notion_sync_state, populated by the daily sync) and reports two things a
human would otherwise have to notice by eye:

1. Outcome change: a tracked page's "Actual Outcome" select differs from
   what we saw last time we checked. Catches a hand-edit in Notion (someone
   resolved a market) between syncs.
2. Forecast due: reuses src.health.check_forecast_due -- any tracked market
   resolving within --horizon-hours (default 48) with no revision-0 forecast
   on record.

No LLM, no edits -- read-only against both Notion and the local DB. State
(last-seen outcome per page) lives in ops/.notion_watch_state.json, which is
gitignored; losing it just means the next run treats every page's current
outcome as a fresh baseline instead of a diff.

Usage:
    python -m ops.notion_watch               # print only
    python -m ops.notion_watch --horizon-hours 48
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import db, health, notion_sync  # noqa: E402

STATE_PATH = REPO_ROOT / "ops" / ".notion_watch_state.json"


def _load_state() -> dict[str, str]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict[str, str]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def _outcome_of(page_id: str) -> str | None:
    page = notion_sync._call("GET", f"/pages/{page_id}")
    prop = (page.get("properties") or {}).get("Actual Outcome") or {}
    sel = prop.get("select")
    return sel.get("name") if sel else None


def _tracked_page_ids(conn) -> dict[str, dict[str, str]]:
    """market_id -> {page_id, question}, from config/markets.yml rules ∪ notion_sync_state.

    A rule's ``notion_page_id`` is known the moment it's added to config --
    don't wait for the next ``notion_sync`` run (which only populates
    ``notion_sync_state``) before a freshly-onboarded market's Notion page
    is checkable. Falls back to notion_sync_state for rows synced before
    they had an explicit notion_page_id in config.
    """
    rules = notion_sync.tracked_config()
    rollup = notion_sync.build_rollup(conn)
    out: dict[str, dict[str, str]] = {}
    for row in rollup:
        rule = notion_sync.match_tracked(row, rules)
        if rule is None or not rule.get("notion_page_id"):
            continue
        out[row["market_id"]] = {
            "page_id": rule["notion_page_id"],
            "question": row.get("question") or "?",
        }
    state_rows = conn.execute(
        "SELECT market_id, notion_page_id FROM notion_sync_state"
    ).fetchall()
    questions = {r["market_id"]: r["question"] for r in
                 conn.execute("SELECT market_id, question FROM markets")}
    for r in state_rows:
        out.setdefault(r["market_id"], {
            "page_id": r["notion_page_id"],
            "question": questions.get(r["market_id"], "?"),
        })
    return out


def check_outcome_changes() -> list[dict[str, str]]:
    """Diff each tracked page's current Actual Outcome against last-seen state."""
    db.init_db()
    changes: list[dict[str, str]] = []
    state = _load_state()
    with db.get_connection() as conn:
        tracked = _tracked_page_ids(conn)

    new_state = dict(state)
    for market_id, info in tracked.items():
        page_id = info["page_id"]
        try:
            current = _outcome_of(page_id)
        except notion_sync.NotionError as exc:
            print(f"  warn: could not read page {page_id} ({market_id[:12]}...): {exc}",
                  file=sys.stderr)
            continue
        previous = state.get(page_id)
        new_state[page_id] = current or ""
        if previous is not None and previous != (current or ""):
            changes.append({
                "market_id": market_id,
                "question": info["question"],
                "from": previous or "(none)",
                "to": current or "(none)",
            })
    _save_state(new_state)
    return changes


def check_notion_forecast_due(horizon_hours: float = 48.0) -> list[dict[str, Any]]:
    """Tracked pages whose own ``Forecast Due`` date (set by hand in Notion,
    e.g. the data-release time -- which can fall *before* the market's formal
    ``resolution_date``) is inside the horizon and ``My Independent Read`` is
    still blank.

    src.health.check_forecast_due deliberately checks a different thing: it
    derives its deadline from markets.resolution_date (the tape's own close),
    which is right for FOMC/election markets but wrong for data-release
    markets (CPI, jobs) where Notion's Forecast Due is the actual release
    time and is earlier than resolution_date. Checking only the DB-derived
    date misses rows where this hand-set field is already past or imminent.
    This function is the data-release-aware complement, not a replacement.
    """
    db.init_db()
    due: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    with db.get_connection() as conn:
        tracked = _tracked_page_ids(conn)

    for market_id, info in tracked.items():
        page_id = info["page_id"]
        try:
            page = notion_sync._call("GET", f"/pages/{page_id}")
        except notion_sync.NotionError as exc:
            print(f"  warn: could not read page {page_id} ({market_id[:12]}...): {exc}",
                  file=sys.stderr)
            continue
        props = page.get("properties") or {}
        fd_raw = ((props.get("Forecast Due") or {}).get("date") or {}).get("start")
        read = (props.get("My Independent Read") or {}).get("number")
        if not fd_raw or read is not None:
            continue
        fd_dt = datetime.fromisoformat(fd_raw.replace("Z", "+00:00"))
        if fd_dt.tzinfo is None:
            fd_dt = fd_dt.replace(tzinfo=timezone.utc)
        hours_left = (fd_dt - now).total_seconds() / 3600.0
        if hours_left > horizon_hours:
            continue  # not yet inside the window -- still fine to be silent
        due.append({
            "market_id": market_id,
            "question": info["question"],
            "forecast_due": fd_raw,
            "hours_left": round(hours_left, 1),
        })
    return due


def _main() -> None:
    ap = argparse.ArgumentParser(prog="python -m ops.notion_watch")
    ap.add_argument("--horizon-hours", type=float, default=48.0)
    args = ap.parse_args()

    out_lines: list[str] = []

    changes = check_outcome_changes()
    for c in changes:
        out_lines.append(
            f"OUTCOME CHANGED: {c['question'][:70]} -- {c['from']} -> {c['to']}"
        )

    db.init_db()
    with db.get_connection() as conn:
        due = health.check_forecast_due(conn, horizon_hours=args.horizon_hours)
    if not due["ok"]:
        for d in due["due"]:
            out_lines.append(
                f"FORECAST DUE: {d['question'][:70]} resolves {d['resolution_date']} "
                f"({d['hours_left']}h left)"
            )

    notion_due = check_notion_forecast_due(horizon_hours=args.horizon_hours)
    for d in notion_due:
        out_lines.append(
            f"FORECAST DUE (Notion): {d['question'][:70]} due {d['forecast_due']} "
            f"({d['hours_left']}h left)"
        )

    if out_lines:
        print("Polymarket Observation Log -- changes since last check:")
        for line in out_lines:
            print(f"  - {line}")
    else:
        print("Polymarket Observation Log: no outcome changes, no forecasts due.")


if __name__ == "__main__":
    _main()
