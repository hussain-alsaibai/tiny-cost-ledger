"""Tests for tiny-cost-ledger."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest

from tiny_cost_ledger import (
    AlertLevel, Budget, CostEntry, CostTracker, Ledger, Period, PricingTable,
)


class TestPricing(unittest.TestCase):
    def test_cost_basic(self):
        p = PricingTable()
        p.add("test", input_usd_per_1k=0.001, output_usd_per_1k=0.002)
        cost = p.get("test").cost(2000, 500)
        # 2 * 0.001 + 0.5 * 0.002 = 0.002 + 0.001 = 0.003
        self.assertAlmostEqual(cost, 0.003, places=6)

    def test_unknown_model_raises(self):
        p = PricingTable()
        with self.assertRaises(KeyError):
            p.get("missing")

    def test_negative_pricing_rejected(self):
        p = PricingTable()
        with self.assertRaises(ValueError):
            p.add("bad", input_usd_per_1k=-1.0, output_usd_per_1k=0.0)

    def test_defaults_have_common_models(self):
        p = PricingTable.defaults()
        for m in ("gpt-4o", "claude-3-5-sonnet", "gemini-1.5-flash", "kimi-k2"):
            self.assertTrue(p.has(m), m)


class TestLedger(unittest.TestCase):
    def test_append_and_query(self):
        l = Ledger()
        for i in range(3):
            l.append(CostEntry(timestamp=time.time(), model="m", tokens_in=10,
                               tokens_out=5, cost_usd=0.001))
        self.assertEqual(len(l.entries()), 3)

    def test_persistence_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.jsonl")
            l1 = Ledger(path=path)
            l1.append(CostEntry(timestamp=1000.0, model="x", tokens_in=100, tokens_out=50, cost_usd=0.01))
            l1.append(CostEntry(timestamp=2000.0, model="y", tokens_in=200, tokens_out=80, cost_usd=0.02))
            l2 = Ledger(path=path)
            self.assertEqual(len(l2.entries()), 2)
            self.assertEqual(l2.entries()[0].model, "x")

    def test_reverse_zeros_cost_in_summary(self):
        l = Ledger()
        e = CostEntry(timestamp=time.time(), model="m", tokens_in=10, tokens_out=5, cost_usd=0.5)
        l.append(e)
        l.reverse(e, note="refund")
        # Total should be 0 (original 0.5 - reversal 0.5).
        from tiny_cost_ledger import _sum_cost, _entries_for_period
        self.assertAlmostEqual(_sum_cost(_entries_for_period(l, Period.DAY)), 0.0)

    def test_query_filter_by_model(self):
        l = Ledger()
        l.append(CostEntry(timestamp=time.time(), model="a", tokens_in=10, tokens_out=5, cost_usd=0.1))
        l.append(CostEntry(timestamp=time.time(), model="b", tokens_in=10, tokens_out=5, cost_usd=0.2))
        self.assertEqual(len(l.query(model="a")), 1)


class TestBudget(unittest.TestCase):
    def test_alert_fires_once_per_threshold(self):
        fired: list[tuple[AlertLevel, Period]] = []
        b = Budget(
            daily_usd=1.0,
            threshold_alerts=(0.5,),
            on_alert=lambda lvl, p, s, c: fired.append((lvl, p)),
        )
        l = Ledger()
        l.append(CostEntry(timestamp=time.time(), model="m", tokens_in=10, tokens_out=5, cost_usd=0.4))
        b.check(l)  # below 0.5
        self.assertEqual(fired, [])
        l.append(CostEntry(timestamp=time.time(), model="m", tokens_in=10, tokens_out=5, cost_usd=0.2))
        b.check(l)  # 0.6, crosses 0.5
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0], (AlertLevel.INFO, Period.DAY))  # 0.5 -> INFO per _level_for
        # Idempotent
        b.check(l)
        self.assertEqual(len(fired), 1)

    def test_exceeded_alert_at_full_threshold(self):
        fired: list[AlertLevel] = []
        b = Budget(daily_usd=1.0, threshold_alerts=(1.0,),
                   on_alert=lambda lvl, *_a: fired.append(lvl))
        l = Ledger()
        l.append(CostEntry(timestamp=time.time(), model="m", tokens_in=10, tokens_out=5, cost_usd=1.1))
        b.check(l)
        self.assertEqual(fired, [AlertLevel.EXCEEDED])


class TestCostTracker(unittest.TestCase):
    def test_record_computes_cost(self):
        t = CostTracker(pricing=PricingTable.defaults(), ledger=Ledger())
        e = t.record(model="gpt-4o-mini", tokens_in=1000, tokens_out=500, task_id="t")
        # 1 * 0.00015 + 0.5 * 0.0006 = 0.00045
        self.assertAlmostEqual(e.cost_usd, 0.00045, places=6)

    def test_summarize_by_model_and_task(self):
        t = CostTracker(pricing=PricingTable.defaults(), ledger=Ledger())
        t.record("gpt-4o-mini", 1000, 200, task_id="a")
        t.record("gpt-4o-mini", 1000, 200, task_id="a")
        t.record("claude-3-5-haiku", 500, 100, task_id="b")
        s = t.summarize(task_id="a")
        self.assertEqual(s["calls"], 2)
        self.assertIn("gpt-4o-mini", s["by_model"])

    def test_would_exceed(self):
        b = Budget(daily_usd=0.001)
        t = CostTracker(pricing=PricingTable.defaults(), ledger=Ledger(), budget=b)
        self.assertTrue(t.would_exceed("gpt-4o", 10000, 5000))

    def test_strict_mode_blocks(self):
        b = Budget(daily_usd=0.001)
        t = CostTracker(pricing=PricingTable.defaults(), ledger=Ledger(), budget=b, strict=True)
        with self.assertRaises(RuntimeError):
            t.record("gpt-4o", 10000, 5000)


if __name__ == "__main__":
    unittest.main()
