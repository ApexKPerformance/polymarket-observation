"""Offline tests for tracking, link resolution and resolution logging.

No network, no secrets.  python -m pytest -q
"""
from __future__ import annotations

import pytest

from src import config, db, notion_sync, resolve


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATIONS_DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    with db.get_connection() as c:
        c.executescript(notion_sync.SYNC_STATE_SQL)
        yield c


# --------------------------------------------------------------------------
# Track rules from config/markets.yml
# --------------------------------------------------------------------------

def test_nine_observation_markets_are_tracked_and_bound():
    """Every active event must name a sub-market AND bind it to the row that
    already exists in the log, or the sync would duplicate the manual rows."""
    rules = notion_sync.tracked_config()
    assert len(rules) == 9
    assert all(r["notion_page_id"] for r in rules), \
        "a track rule without notion_page_id would create a second row"
    assert len({r["notion_page_id"] for r in rules}) == 9, \
        "two markets bound to the same Notion row"


def test_retired_scaffold_events_are_not_tracked():
    slugs = {r["event_slug"] for r in notion_sync.tracked_config()}
    assert "fed-decision-in-june-825" not in slugs
    assert "how-many-fed-rate-cuts-in-2026" not in slugs


@pytest.mark.parametrize("outcome_name,rule_outcome,should_match", [
    ("0.2%", "0.2%", True),
    ("0.2 %", "0.2%", True),          # spacing varies on the live board
    ("0.3%", "0.2%", False),
    ("No change", "No change", True),
    ("no change", "No change", True),
    ("add 0-50k", "0-50k", True),     # substring fallback
    ("0 - 50K", "0-50k", True),
    ("50-100k", "0-50k", False),
    ("Democrat", "Democrat", True),
    ("Republican", "Democrat", False),
])
def test_outcome_matching_tolerates_label_variation(outcome_name, rule_outcome, should_match):
    rules = [{"event_slug": "e", "outcome": rule_outcome, "notion_page_id": "p"}]
    row = {"event_slug": "e", "outcome_name": outcome_name, "question": ""}
    assert (notion_sync.match_tracked(row, rules) is not None) is should_match


def test_untracked_submarket_returns_none():
    """The rest of a board is polled but not logged."""
    rules = notion_sync.tracked_config()
    row = {"event_slug": "core-cpi-mom-august-2026-1786474662954",
           "outcome_name": "0.4%", "question": "Will Core CPI MoM be 0.4%?"}
    assert notion_sync.match_tracked(row, rules) is None


def test_market_from_an_unlisted_event_is_untracked():
    rules = notion_sync.tracked_config()
    row = {"event_slug": "some-sports-thing", "outcome_name": "Yes", "question": "?"}
    assert notion_sync.match_tracked(row, rules) is None


# --------------------------------------------------------------------------
# Settlement detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("closed,prices,expected", [
    (True, '["1", "0"]', "YES"),
    (True, '["0", "1"]', "NO"),
    (True, '["0.995", "0.005"]', "YES"),
    (True, '["0.62", "0.38"]', None),   # closed but not settled
    (False, '["1", "0"]', None),        # still trading
])
def test_settled_outcome(closed, prices, expected):
    market = {"closed": closed, "outcomes": '["Yes", "No"]', "outcomePrices": prices}
    assert resolve.settled_outcome(market) == expected


def test_closing_probability_ignores_the_settlement_tick(conn):
    """Once settled the price is 1.0, which says nothing. We want the market's
    last real opinion."""
    db.upsert_market(
        conn, market_id="0x1", event_slug="e", event_name="E", category="Economic",
        question="Q", outcome_name="0.2%", yes_token_id="t", no_token_id=None,
        polymarket_url=None, resolution_date="2026-09-11",
    )
    for ts, price in [
        ("2026-09-09T12:00:00Z", 0.61),
        ("2026-09-10T12:00:00Z", 0.68),
        ("2026-09-11T13:00:00Z", 1.0),    # settlement
    ]:
        db.save_price_snapshot(conn, market_id="0x1", yes_price=price, timestamp=ts)

    price, ts = resolve.last_live_price(conn, "0x1")
    assert price == pytest.approx(0.68)
    assert ts == "2026-09-10T12:00:00Z"


def test_closing_probability_falls_back_when_only_settled_ticks_exist(conn):
    db.upsert_market(
        conn, market_id="0x2", event_slug="e", event_name="E", category="Economic",
        question="Q", outcome_name="0.2%", yes_token_id="t", no_token_id=None,
        polymarket_url=None, resolution_date="2026-09-11",
    )
    db.save_price_snapshot(conn, market_id="0x2", yes_price=1.0,
                           timestamp="2026-09-11T13:00:00Z")
    price, _ = resolve.last_live_price(conn, "0x2")
    assert price == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Hypothetical P&L -- the part that would silently produce wrong numbers
# --------------------------------------------------------------------------

def test_pnl_is_none_without_an_independent_read():
    """The side of the trade is KeAndre's judgment. Never guess it."""
    pnl, why = resolve.hypothetical_pnl(None, 0.655, "YES", 100)
    assert pnl is None
    assert "no independent read" in why


def test_pnl_read_above_price_buys_yes_and_wins():
    # Read 0.80 vs price 0.50 -> buy YES at 0.50 -> 200 shares -> settles YES.
    pnl, why = resolve.hypothetical_pnl(0.80, 0.50, "YES", 100)
    assert pnl == pytest.approx(100.0)
    assert "on YES at 0.500" in why


def test_pnl_read_above_price_buys_yes_and_loses():
    pnl, _ = resolve.hypothetical_pnl(0.80, 0.50, "NO", 100)
    assert pnl == pytest.approx(-100.0)


def test_pnl_read_below_price_buys_no():
    # Read 0.20 vs price 0.50 -> buy NO at 0.50 -> 200 shares -> settles NO.
    pnl, why = resolve.hypothetical_pnl(0.20, 0.50, "NO", 100)
    assert pnl == pytest.approx(100.0)
    assert "on NO at 0.500" in why


def test_pnl_on_a_long_shot_pays_more():
    """Buying YES at 0.10 and winning returns 900, not 100."""
    pnl, _ = resolve.hypothetical_pnl(0.60, 0.10, "YES", 100)
    assert pnl == pytest.approx(900.0)


def test_pnl_on_a_favourite_pays_little():
    pnl, _ = resolve.hypothetical_pnl(0.99, 0.90, "YES", 100)
    assert pnl == pytest.approx(11.11, abs=0.01)


def test_pnl_loss_is_capped_at_the_stake():
    """A paper position cannot lose more than it staked."""
    for read, price, outcome in [(0.9, 0.1, "NO"), (0.1, 0.9, "YES"), (0.7, 0.65, "NO")]:
        pnl, _ = resolve.hypothetical_pnl(read, price, outcome, 100)
        assert pnl >= -100.0


def test_pnl_zero_when_read_matches_the_market():
    pnl, why = resolve.hypothetical_pnl(0.655, 0.655, "YES", 100)
    assert pnl == pytest.approx(0.0)
    assert "no position" in why


def test_pnl_respects_the_configured_stake():
    small, _ = resolve.hypothetical_pnl(0.80, 0.50, "YES", 25)
    assert small == pytest.approx(25.0)


@pytest.mark.parametrize("price", [0.0, 1.0, None, -0.2])
def test_pnl_rejects_unusable_entry_prices(price):
    pnl, why = resolve.hypothetical_pnl(0.6, price, "YES", 100)
    assert pnl is None and why


def test_configured_stake_matches_the_notion_property_description():
    """The Notion "Hypothetical P&L" description says $25. Config must agree,
    or the log documents a stake it does not use."""
    assert config.section("resolution").get("paper_stake") == 25
