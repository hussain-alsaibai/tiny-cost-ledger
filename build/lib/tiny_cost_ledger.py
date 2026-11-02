"""
tiny-cost-ledger — zero-dependency LLM cost ledger with budget alerts.

A single-file, stdlib-only library for tracking per-call LLM spend (tokens
in, tokens out, USD cost), enforcing per-period and per-task budgets, and
emitting structured alerts when a budget threshold is crossed. Designed for
autonomous agents that need to keep an itemized cost trail for every
external model call, while staying inside a daily/weekly/overall budget.

Design goals
------------
- Zero deps (stdlib only).
- One file. ~600 LOC. Readable in one sitting.
- Provider-agnostic: bring your own pricing table.
- Append-only ledger: entries are immutable; corrections are reversing entries.
- Concurrent-safe via a re-entrant lock for callers that share a ledger.
- Alerting hooks: callback-based; no global side effects.
- Persistence: JSONL on disk, queryable by task / model / time window.

Core concepts
-------------
- PricingTable: model -> (input_usd_per_1k, output_usd_per_1k).
- CostEntry: a single call record (timestamp, model, tokens_in, tokens_out, cost_usd, task_id, ...).
- Ledger: append-only log with thread-safe writes; query() and summarize().
- Budget: per-period (hourly, daily, weekly, monthly) USD ceiling + callback alerts.
- CostTracker: convenience facade — record() does the arithmetic for you.

Usage
-----
    from tiny_cost_ledger import (
        PricingTable, Ledger, Budget, CostTracker, AlertLevel,
    )

    pricing = PricingTable()
    pricing.add("gpt-4o",           input_usd_per_1k=0.0025,  output_usd_per_1k=0.0100)
    pricing.add("gpt-4o-mini",      input_usd_per_1k=0.00015, output_usd_per_1k=0.0006)
    pricing.add("claude-3-5-sonnet",input_usd_per_1k=0.003,  output_usd_per_1k=0.015)

    ledger = Ledger(path="cost_ledger.jsonl")

    def alert(level, period, spent_usd, budget_usd):
        print(f"[{level.name}] {period}: ${spent_usd:.4f} / ${budget_usd:.2f}")

    budget = Budget(
        daily_usd=10.0,
        weekly_usd=50.0,
        monthly_usd=200.0,
        threshold_alerts=(0.5, 0.8, 0.95, 1.0),
        on_alert=alert,
    )

    tracker = CostTracker(pricing=pricing, ledger=ledger, budget=budget)

    # Track a call.
    tracker.record(
        model="gpt-4o",
        tokens_in=2400,
        tokens_out=380,
        task_id="bounty-edgechains-273",
        note="first attempt",
    )

    # Summaries.
    print(tracker.summarize(period="day"))
    print(tracker.summarize(period="task", task_id="bounty-edgechains-273"))

    # Hard-block a call that would exceed budget.
    if tracker.would_exceed("gpt-4o", 8000, 1200):
        raise RuntimeError("cost guard: budget exceeded")

License: MIT
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import IO, Any


# ---------------------------------------------------------------------------
# Enums / dataclasses
# ---------------------------------------------------------------------------

class AlertLevel(Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"


class Period(Enum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass(frozen=True)
class CostEntry:
    timestamp: float               # unix seconds
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    task_id: str = ""
    note: str = ""
    reversal_of: str | None = None # entry_id this reverses, if any

    @property
    def entry_id(self) -> str:
        return f"{int(self.timestamp * 1000):013d}-{self.model}-{self.tokens_in}-{self.tokens_out}"

    @property
    def period_day(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(self.timestamp))

    @property
    def period_hour(self) -> str:
        return time.strftime("%Y-%m-%dT%H", time.gmtime(self.timestamp))

    @property
    def period_week(self) -> str:
        return time.strftime("%Y-G%V", time.gmtime(self.timestamp))  # ISO-ish week

    @property
    def period_month(self) -> str:
        return time.strftime("%Y-%m", time.gmtime(self.timestamp))


@dataclass(frozen=True)
class ModelPrice:
    name: str
    input_usd_per_1k: float
    output_usd_per_1k: float

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (
            (max(0, tokens_in) / 1000.0) * self.input_usd_per_1k
            + (max(0, tokens_out) / 1000.0) * self.output_usd_per_1k
        )


class PricingTable:
    """Provider-agnostic price list. Add/remove models; lookup() raises on miss."""
    def __init__(self) -> None:
        self._prices: dict[str, ModelPrice] = {}

    def add(self, name: str, input_usd_per_1k: float, output_usd_per_1k: float) -> None:
        if input_usd_per_1k < 0 or output_usd_per_1k < 0:
            raise ValueError("pricing must be non-negative")
        self._prices[name] = ModelPrice(name, input_usd_per_1k, output_usd_per_1k)

    def remove(self, name: str) -> None:
        self._prices.pop(name, None)

    def get(self, name: str) -> ModelPrice:
        if name not in self._prices:
            raise KeyError(f"unknown model: {name}")
        return self._prices[name]

    def has(self, name: str) -> bool:
        return name in self._prices

    @classmethod
    def defaults(cls) -> "PricingTable":
        """A reasonable default pricing table for 2026 frontier + budget models."""
        t = cls()
        # OpenAI
        t.add("gpt-4o",            input_usd_per_1k=0.00250, output_usd_per_1k=0.01000)
        t.add("gpt-4o-mini",       input_usd_per_1k=0.00015, output_usd_per_1k=0.00060)
        t.add("o1",                input_usd_per_1k=0.01500, output_usd_per_1k=0.06000)
        t.add("o1-mini",           input_usd_per_1k=0.00300, output_usd_per_1k=0.01200)
        # Anthropic
        t.add("claude-3-5-sonnet", input_usd_per_1k=0.00300, output_usd_per_1k=0.01500)
        t.add("claude-3-5-haiku",  input_usd_per_1k=0.00080, output_usd_per_1k=0.00400)
        t.add("claude-3-opus",     input_usd_per_1k=0.01500, output_usd_per_1k=0.07500)
        # Google
        t.add("gemini-1.5-pro",    input_usd_per_1k=0.00125, output_usd_per_1k=0.00500)
        t.add("gemini-1.5-flash",  input_usd_per_1k=0.000075,output_usd_per_1k=0.00030)
        # Synthetic / open-weight providers (often $0/token)
        t.add("kimi-k2",           input_usd_per_1k=0.0,     output_usd_per_1k=0.0)
        t.add("nemotron-3-super",  input_usd_per_1k=0.0,     output_usd_per_1k=0.0)
        t.add("qwen3-coder",       input_usd_per_1k=0.0,     output_usd_per_1k=0.0)
        t.add("local-default",     input_usd_per_1k=0.0,     output_usd_per_1k=0.0)
        return t


# ---------------------------------------------------------------------------
# Ledger — append-only JSONL with re-entrant lock
# ---------------------------------------------------------------------------

@dataclass
class Ledger:
    """Append-only JSONL ledger.

    Each write is one line. On open(path) we replay the file into memory;
    subsequent writes append. Queries return frozen entries.
    """
    path: str | None = None
    _entries: list[CostEntry] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        if self.path and os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._entries.append(_entry_from_dict(data))

    def append(self, entry: CostEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(entry), sort_keys=True) + "\n")

    def entries(self) -> tuple[CostEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def query(
        self,
        since: float | None = None,
        until: float | None = None,
        model: str | None = None,
        task_id: str | None = None,
        include_reversals: bool = True,
    ) -> list[CostEntry]:
        with self._lock:
            out: list[CostEntry] = []
            for e in self._entries:
                if since is not None and e.timestamp < since:
                    continue
                if until is not None and e.timestamp > until:
                    continue
                if model is not None and e.model != model:
                    continue
                if task_id is not None and e.task_id != task_id:
                    continue
                if not include_reversals and e.reversal_of is not None:
                    continue
                out.append(e)
            return out

    def reverse(self, entry: CostEntry, note: str = "reversal") -> CostEntry:
        """Append a reversing entry that zeroes out the original cost in summaries."""
        rev = CostEntry(
            timestamp=time.time(),
            model=entry.model,
            tokens_in=-entry.tokens_in,
            tokens_out=-entry.tokens_out,
            cost_usd=-entry.cost_usd,
            task_id=entry.task_id,
            note=note,
            reversal_of=entry.entry_id,
        )
        self.append(rev)
        return rev

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            if self.path and os.path.exists(self.path):
                os.remove(self.path)


def _entry_from_dict(d: dict[str, Any]) -> CostEntry:
    return CostEntry(
        timestamp=float(d["timestamp"]),
        model=str(d["model"]),
        tokens_in=int(d["tokens_in"]),
        tokens_out=int(d["tokens_out"]),
        cost_usd=float(d["cost_usd"]),
        task_id=str(d.get("task_id", "")),
        note=str(d.get("note", "")),
        reversal_of=d.get("reversal_of"),
    )


# ---------------------------------------------------------------------------
# Budget — multi-period thresholds + alert hooks
# ---------------------------------------------------------------------------

AlertFn = Callable[[AlertLevel, Period, float, float], None]


@dataclass
class Budget:
    """Multi-period USD budget with threshold alerts.

    Each period has its own ceiling. Alerts fire when a recording pushes the
    spend across a threshold (e.g. 0.5, 0.8, 0.95, 1.0). The same callback
    receives (level, period, spent, ceiling). Levels:
      - INFO       crossed a low threshold
      - WARN       crossed mid threshold
      - CRITICAL   crossed 95% threshold
      - EXCEEDED   crossed 100% (spend > ceiling)
    """
    hourly_usd: float = float("inf")
    daily_usd: float = float("inf")
    weekly_usd: float = float("inf")
    monthly_usd: float = float("inf")
    overall_usd: float = float("inf")
    threshold_alerts: tuple[float, ...] = (0.5, 0.8, 0.95, 1.0)
    on_alert: AlertFn | None = None
    _fired: dict[tuple[Period, float], bool] = field(default_factory=dict)

    def ceiling_for(self, period: Period) -> float:
        return {
            Period.HOUR:  self.hourly_usd,
            Period.DAY:   self.daily_usd,
            Period.WEEK:  self.weekly_usd,
            Period.MONTH: self.monthly_usd,
        }[period]

    def check(self, ledger: Ledger) -> None:
        """Walk thresholds across all periods; fire alerts on transitions."""
        if self.on_alert is None:
            return
        for p in Period:
            ceiling = self.ceiling_for(p)
            if not math.isfinite(ceiling) or ceiling <= 0:
                continue
            spent = _sum_cost(_entries_for_period(ledger, p))
            for thr in sorted(self.threshold_alerts):
                key = (p, thr)
                if spent >= thr * ceiling and not self._fired.get(key, False):
                    level = _level_for(thr)
                    try:
                        self.on_alert(level, p, spent, ceiling)
                    finally:
                        self._fired[key] = True

    def reset_alerts(self) -> None:
        self._fired.clear()


def _entries_for_period(ledger: Ledger, period: Period) -> Iterable[CostEntry]:
    now = time.time()
    if period is Period.HOUR:
        return ledger.query(since=now - 3600)
    if period is Period.DAY:
        return ledger.query(since=now - 86400)
    if period is Period.WEEK:
        return ledger.query(since=now - 86400 * 7)
    if period is Period.MONTH:
        return ledger.query(since=now - 86400 * 30)
    return ledger.query()


def _level_for(threshold: float) -> AlertLevel:
    if threshold >= 1.0:
        return AlertLevel.EXCEEDED
    if threshold >= 0.95:
        return AlertLevel.CRITICAL
    if threshold >= 0.8:
        return AlertLevel.WARN
    return AlertLevel.INFO


def _sum_cost(entries: Iterable[CostEntry]) -> float:
    return sum(max(0.0, e.cost_usd) for e in entries if e.reversal_of is None) - sum(
        max(0.0, -e.cost_usd) for e in entries if e.reversal_of is not None
    )


# ---------------------------------------------------------------------------
# CostTracker — facade
# ---------------------------------------------------------------------------

import math


@dataclass
class CostTracker:
    pricing: PricingTable
    ledger: Ledger
    budget: Budget | None = None
    strict: bool = False  # if True, raise on over-budget

    def record(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        task_id: str = "",
        note: str = "",
        timestamp: float | None = None,
    ) -> CostEntry:
        price = self.pricing.get(model)
        cost = price.cost(tokens_in, tokens_out)
        entry = CostEntry(
            timestamp=timestamp if timestamp is not None else time.time(),
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=round(cost, 6),
            task_id=task_id,
            note=note,
        )
        if self.budget is not None and self.strict and self._would_overflow(entry):
            raise RuntimeError(f"cost guard: recording {entry} would exceed budget")
        self.ledger.append(entry)
        if self.budget is not None:
            self.budget.check(self.ledger)
        return entry

    def would_exceed(self, model: str, tokens_in: int, tokens_out: int) -> bool:
        price = self.pricing.get(model)
        cost = price.cost(tokens_in, tokens_out)
        ghost = CostEntry(
            timestamp=time.time(), model=model,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=round(cost, 6),
        )
        return self._would_overflow(ghost)

    def _would_overflow(self, ghost: CostEntry) -> bool:
        if self.budget is None:
            return False
        for p in Period:
            ceiling = self.budget.ceiling_for(p)
            if not math.isfinite(ceiling):
                continue
            existing = _sum_cost(_entries_for_period(self.ledger, p))
            if existing + ghost.cost_usd > ceiling:
                return True
        return False

    def summarize(self, period: Period | str = Period.DAY, task_id: str | None = None) -> dict[str, Any]:
        p = Period(period) if isinstance(period, str) else period
        if task_id is not None:
            entries = self.ledger.query(task_id=task_id)
        else:
            entries = list(_entries_for_period(self.ledger, p))
        by_model: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
        total_cost = 0.0
        total_in = 0
        total_out = 0
        for e in entries:
            if e.reversal_of is not None:
                total_cost -= max(0.0, -e.cost_usd)
            else:
                total_cost += max(0.0, e.cost_usd)
            total_in += e.tokens_in
            total_out += e.tokens_out
            by_model[e.model]["calls"] += 1
            by_model[e.model]["tokens_in"] += e.tokens_in
            by_model[e.model]["tokens_out"] += e.tokens_out
            by_model[e.model]["cost_usd"] += e.cost_usd
        ceiling = self.budget.ceiling_for(p) if self.budget else float("inf")
        return {
            "period": p.value,
            "task_id": task_id,
            "calls": len(entries),
            "tokens_in": total_in,
            "tokens_out": total_out,
            "cost_usd": round(total_cost, 6),
            "by_model": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in by_model.items()},
            "ceiling_usd": ceiling if math.isfinite(ceiling) else None,
            "remaining_usd": (ceiling - total_cost) if self.budget and math.isfinite(ceiling) else None,
        }


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    pricing = PricingTable.defaults()
    ledger = Ledger(path=None)

    def alert(level, period, spent, ceiling):
        print(f"  ALERT [{level.name}] {period.value}: ${spent:.4f} / ${ceiling:.2f}")

    budget = Budget(
        daily_usd=2.0,
        threshold_alerts=(0.5, 0.8, 1.0),
        on_alert=alert,
    )
    tracker = CostTracker(pricing=pricing, ledger=ledger, budget=budget)

    print("Recording sample calls…")
    for i, (m, t_in, t_out, task) in enumerate([
        ("gpt-4o-mini",     1200,  220, "bounty-1"),
        ("claude-3-5-haiku", 3500,  500, "bounty-1"),
        ("gpt-4o",          8000, 1500, "bounty-2"),
        ("claude-3-5-sonnet",4200,  900, "bounty-2"),
        ("gpt-4o-mini",     2200,  340, "bounty-3"),
    ]):
        e = tracker.record(model=m, tokens_in=t_in, tokens_out=t_out, task_id=task)
        print(f"  {m:24s} in={t_in:5d} out={t_out:5d} cost=${e.cost_usd:.4f}")

    print()
    print("Daily summary:", json.dumps(tracker.summarize(Period.DAY), indent=2, default=str))
    print()
    print("Per-task summary (bounty-2):", json.dumps(tracker.summarize(task_id="bounty-2"), indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    _demo()
