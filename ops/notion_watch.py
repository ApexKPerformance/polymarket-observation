#!/usr/bin/env python3
"""Notion Observation Log watcher -- local-only glue, not part of CI.

Reads the Observation Log pages the pipeline already tracks and reports:

1. Outcome change: a tracked page's "Actual Outcome" select differs from
   what we saw last time we checked. Catches a hand-edit in Notion (someone
   resolved a market) between syncs.

Forecast-due alerting was retired (2026-10-12): the experiment no longer
logs independent forecasts, so "forecast is missing before resolution" is
no longer a meaningful alarm. See git history for the removed
check_notion_forecast_due / src.health.check_forecast_due call if that ever
needs to come back.

No LLM, no edits -- read-only against both Notion and the local DB. State
(last-seen outcome per page) lives in ops/.notion_watch_state.json, which is
gitignored; losing it just means the next run treats every page's current
outcome as a fresh baseline instead of a diff.

Usage:
    python -m ops.notion_watch               # print only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import db, notion_sync  # noqa: E402

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


def _main() -> None:
    ap = argparse.ArgumentParser(prog="python -m ops.notion_watch")
    args = ap.parse_args()

    out_lines: list[str] = []

    changes = check_outcome_changes()
    for c in changes:
        out_lines.append(
            f"OUTCOME CHANGED: {c['question'][:70]} -- {c['from']} -> {c['to']}"
        )

    if out_lines:
        print("Polymarket Observation Log -- changes since last check:")
        for line in out_lines:
            print(f"  - {line}")
    else:
        print("Polymarket Observation Log: no outcome changes.")


if __name__ == "__main__":
    _main()
