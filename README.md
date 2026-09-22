# tiny-cost-ledger

> **Itemized LLM cost ledger with multi-period budgets and alert hooks.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](tiny_cost_ledger.py)
[![Part of the tiny-* ecosystem](https://img.shields.io/badge/tiny--*-ecosystem-purple.svg)](#ecosystem)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](tiny_cost_ledger.py)

`tiny-cost-ledger` is a single-file, zero-dependency library for **tracking
per-call LLM spend**, enforcing per-period budgets, and emitting structured
alerts when a threshold is crossed. Designed for autonomous agents that
need an itemized cost trail for every external model call while staying
inside a daily / weekly / monthly envelope.

## Why this exists

Every autonomous agent that talks to closed-source LLMs eventually hits
the same wall: **the bill**. By the time you notice you've overspent,
the agent has made 4,000 calls and you have no idea which task or
which model was responsible.

`tiny-cost-ledger` answers three questions on demand:

1. **How much have I spent today?** Per-model, per-task, per-period.
2. **Am I about to exceed my budget?** Strict mode raises before each
   over-budget record.
3. **Which calls were the expensive ones?** Append-only JSONL gives you
   an audit log for every model invocation, with reversal support for
   corrections.

## ✨ Features

- **Provider-agnostic pricing** — `PricingTable` ships with sensible
  2026 defaults (OpenAI, Anthropic, Google, Synthetic/open-weight) and
  accepts custom entries.
- **Multi-period budgets** — hourly, daily, weekly, monthly, overall —
  each with its own USD ceiling.
- **Threshold alerts** — fire callbacks at 50%, 80%, 95%, 100% (or any
  threshold you set). Each threshold fires once per period; resets are
  explicit.
- **Strict mode** — `CostTracker(strict=True)` raises a `RuntimeError`
  before any recording that would overflow the budget.
- **Append-only JSONL persistence** — replayable on load, human-readable,
  grep-friendly.
- **Reversal entries** — correct mistakes without rewriting history.
- **Thread-safe** — `Ledger` uses a re-entrant lock; safe to share
  across threads.
- **Zero deps** — 14 KB single file.

## 🚀 Quick Start

```python
from tiny_cost_ledger import (
    PricingTable, Ledger, Budget, CostTracker, AlertLevel,
)

# 1. Pick a pricing model.
pricing = PricingTable.defaults()      # sensible 2026 defaults
# Or build your own:
# pricing = PricingTable()
# pricing.add("my-local-llm", input_usd_per_1k=0.0, output_usd_per_1k=0.0)

# 2. Open a ledger (omit path for in-memory).
ledger = Ledger(path="cost_ledger.jsonl")

# 3. Define a budget with alerts.
def alert(level: AlertLevel, period: Period, spent: float, ceiling: float) -> None:
    print(f"  [{level.name}] {period.value}: ${spent:.4f} / ${ceiling:.2f}")

budget = Budget(
    daily_usd=2.00, weekly_usd=10.00, monthly_usd=40.00,
    threshold_alerts=(0.5, 0.8, 0.95, 1.0),
    on_alert=alert,
)

# 4. Wire a tracker.
tracker = CostTracker(pricing=pricing, ledger=ledger, budget=budget)

# 5. Record every external call.
tracker.record(model="gpt-4o-mini", tokens_in=1200, tokens_out=220, task_id="bounty-1")
tracker.record(model="claude-3-5-haiku", tokens_in=3500, tokens_out=500, task_id="bounty-1")

# 6. Query.
print(tracker.summarize(period="day"))
# -> {"period": "day", "calls": 2, "tokens_in": 4700, "tokens_out": 720, "cost_usd": 0.0018, ...}

# 7. Hard-block before going over budget.
if tracker.would_exceed("gpt-4o", 8000, 1200):
    raise RuntimeError("cost guard: budget would be exceeded")
```

## 📊 Output example

```python
{
  "period": "day",
  "calls": 5,
  "tokens_in": 17100,
  "tokens_out": 3460,
  "cost_usd": 0.0143,
  "by_model": {
    "gpt-4o-mini":        {"calls": 2, "tokens_in": 3400, "tokens_out": 560, "cost_usd": 0.0009},
    "claude-3-5-haiku":   {"calls": 1, "tokens_in": 3500, "tokens_out": 500, "cost_usd": 0.0048},
    "gpt-4o":             {"calls": 1, "tokens_in": 8000, "tokens_out": 1500, "cost_usd": 0.0350},
    "claude-3-5-sonnet":  {"calls": 1, "tokens_in": 4200, "tokens_out": 900, "cost_usd": 0.0261}
  },
  "ceiling_usd": 2.0,
  "remaining_usd": 1.9857
}
```

## 🔒 Strict mode

```python
tracker = CostTracker(pricing=pricing, ledger=ledger, budget=budget, strict=True)

try:
    tracker.record("gpt-4o", tokens_in=10000, tokens_out=5000)
except RuntimeError as e:
    print(f"blocked: {e}")
```

## 🧮 Default pricing (2026 Q3)

| Model | Input $/1K | Output $/1K |
|---|---|---|
| gpt-4o | 0.00250 | 0.01000 |
| gpt-4o-mini | 0.00015 | 0.00060 |
| o1 | 0.01500 | 0.06000 |
| o1-mini | 0.00300 | 0.01200 |
| claude-3-5-sonnet | 0.00300 | 0.01500 |
| claude-3-5-haiku | 0.00080 | 0.00400 |
| claude-3-opus | 0.01500 | 0.07500 |
| gemini-1.5-pro | 0.00125 | 0.00500 |
| gemini-1.5-flash | 0.000075 | 0.00030 |
| kimi-k2 / nemotron-3-super / qwen3-coder | 0.0 | 0.0 |

Override any model with `pricing.add(name, input_usd_per_1k, output_usd_per_1k)` or `pricing.remove(name)`.

## 🛡️ Why a separate ledger?

Most "LLM cost tracker" tools conflate the recorder with the dashboard.
That breaks down when you need to:

- **Audit.** "What did the bounty-hunting agent spend last Tuesday?"
  → JSONL is greppable.
- **Reverse.** "That call was double-counted by the retry."
  → `ledger.reverse(entry)` appends a -cost entry.
- **Replay.** "Re-run my budget against historical calls."
  → load the JSONL, run `summarize(...)`.
- **Stream.** "Send each call to a remote sink."
  → pass a different `Ledger` or wrap `append()`.

## 🧪 Testing

```bash
cd tiny-cost-ledger
python -m unittest tests.test_tiny_cost_ledger -v
```

15 unit tests covering pricing, ledger persistence, reversals, multi-period
budgets, threshold alerts, strict-mode overflow, and summarization.

## 🌱 Ecosystem

| Repo | Job |
|------|-----|
| [`tiny-budget`](https://github.com/hussain-alsaibai/tiny-budget) | token-budget governor (LLM-side) |
| [`tiny-context`](https://github.com/hussain-alsaibai/tiny-context) | LLM context window manager |
| [`tiny-prompt-cache`](https://github.com/hussain-alsaibai/tiny-prompt-cache) | prompt-response cache |
| [`tiny-circuit`](https://github.com/hussain-alsaibai/tiny-circuit) | circuit breaker |
| [`tiny-rate`](https://github.com/hussain-alsaibai/tiny-rate) | rate limiter |
| [`tiny-cost`](https://github.com/hussain-alsaibai/tiny-cost) | higher-level cost attribution |

## License

MIT © 2026 Hussain Al-Saibai
