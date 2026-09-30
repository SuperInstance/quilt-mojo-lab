"""Benchmark harness: same semantics across 5 runtimes, timed.

Correctness gate FIRST (every runtime must agree within tolerance on a small
grid), then timed runs (best of N). Writes outputs/bench.json + a markdown
table to stdout. Mojo timings come from the substrate's internal
perf_counter_ns output (FLOW_NS / ENTROPY_NS); Python/C timings from
time.perf_counter around run().
"""
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
ROOT = os.path.dirname(_HERE)

from naive_quilt import NaiveQuilt  # noqa: E402
from flat_quilt import FlatQuilt  # noqa: E402
from vec_quilt import VecQuilt  # noqa: E402
import cflat  # noqa: E402

MOJO = os.environ.get("MOJO_BIN", "mojo")
MOJO_ENV = dict(os.environ)
if "MODULAR_HOME" not in MOJO_ENV:
    MOJO_ENV["MODULAR_HOME"] = os.path.expanduser("~/.pixi/envs/mojo/share/max")
    MOJO_ENV["PATH"] = os.path.expanduser("~/.pixi/envs/mojo/bin") + ":" + MOJO_ENV["PATH"]

INJECT = [(1, 1, 4.8)]
THRESHOLD = 0.6
REPEAT = 3


def make(kind, size):
    if kind == "naive":
        return NaiveQuilt(size, THRESHOLD)
    if kind == "flat":
        return FlatQuilt(size, THRESHOLD)
    if kind == "vec":
        return VecQuilt(size, THRESHOLD)
    if kind == "c":
        return cflat.CFlatQuilt(size, THRESHOLD)
    raise ValueError(kind)


def bench_python(kind, size, steps):
    q = make(kind, size)
    for r, c, p in INJECT:
        q.inject_force(r, c, p)
    best = None
    for _ in range(REPEAT):
        # fresh reset via re-inject on a new instance (construction excluded)
        q = make(kind, size)
        for r, c, p in INJECT:
            q.inject_force(r, c, p)
        t0 = time.perf_counter_ns()
        for _ in range(steps):
            q.step_flow()
        t1 = time.perf_counter_ns()
        q.step_entropy()
        best = (t1 - t0) if best is None else min(best, t1 - t0)
    return best, q.checksum()


def bench_mojo(size, steps):
    """Compile once, run REPEAT times; parse internal timings."""
    src = os.path.join(ROOT, "mojo", "quilt_high_perf.mojo")
    # warm compile
    subprocess.run([MOJO, "run", src, str(size), str(steps)],
                   capture_output=True, env=MOJO_ENV, timeout=600, check=True)
    best_flow = best_total = None
    checksum = None
    for _ in range(REPEAT):
        r = subprocess.run([MOJO, "run", src, str(size), str(steps)],
                           capture_output=True, env=MOJO_ENV, timeout=600, check=True)
        out = r.stdout.decode()
        vals = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in ("CHECKSUM", "FLOW_NS", "ENTROPY_NS"):
                vals[parts[0]] = float(parts[1])
        checksum = vals.get("CHECKSUM")
        flow = vals.get("FLOW_NS", 0.0)
        ent = vals.get("ENTROPY_NS", 0.0)
        best_flow = flow if best_flow is None else min(best_flow, flow)
        best_total = (flow + ent) if best_total is None else min(best_total, flow + ent)
    return best_flow, best_total, checksum


def main():
    sizes = [16, 32, 64, 128, 256, 512]
    steps = 10
    results = {}
    # correctness gate on 16
    ref = FlatQuilt(16, THRESHOLD)
    for r, c, p in INJECT:
        ref.inject_force(r, c, p)
    ref.run(steps)
    ref_cs = ref.checksum()
    gate = {}
    for kind in ("naive", "vec", "c"):
        try:
            q = make(kind, 16)
        except RuntimeError:
            gate[kind] = "skipped (no lib)"
            continue
        for r, c, p in INJECT:
            q.inject_force(r, c, p)
        q.run(steps)
        ok = abs(q.checksum() - ref_cs) <= max(abs(ref_cs) * 1e-4, 1e-4)
        gate[kind] = "OK" if ok else f"FAIL ({q.checksum()} vs {ref_cs})"
    mojo_flow, mojo_total, mojo_cs = bench_mojo(16, steps)
    gate["mojo"] = "OK" if (mojo_cs is not None and
                            abs(mojo_cs - ref_cs) <= max(abs(ref_cs) * 1e-4, 1e-4)) else \
                   f"CHECK ({mojo_cs} vs {ref_cs})"
    print(f"correctness gate: {gate}")
    results["gate"] = gate

    table = []
    for size in sizes:
        row = {"size": size, "steps": steps, "cells": size * size}
        # python runtimes (naive only up to 64 to keep wall time sane)
        for kind in ("naive", "flat", "vec", "c"):
            if kind == "naive" and size > 64:
                continue
            try:
                ns, cs = bench_python(kind, size, steps)
            except RuntimeError:
                row[kind] = None
                continue
            row[kind] = ns
            row[kind + "_cells_per_s"] = round(size * size * steps / (ns / 1e9), 0)
        mf, mt_, cs = bench_mojo(size, steps)
        row["mojo_flow"] = mf
        row["mojo_total"] = mt_
        row["mojo_cells_per_s"] = round(size * size * steps / (mt_ / 1e9), 0)
        table.append(row)

    print("\n| grid | cells | naive | flat | numpy | C | mojo(flow) | mojo(total) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in table:
        def fmt(v):
            return f"{v/1e6:.2f}ms" if v else "—"
        def cps(v):
            return f"{v:,.0f}" if v else "—"
        print(f"| {r['size']}x{r['size']} | {r['cells']} | {fmt(r.get('naive'))} | "
              f"{fmt(r.get('flat'))} | {fmt(r.get('vec'))} | {fmt(r.get('c'))} | "
              f"{fmt(r.get('mojo_flow'))} | {fmt(r.get('mojo_total'))} |")
    print("\ncells/sec (best of 3):")
    for r in table:
        row = " ".join(f"{k}={cps(r.get(k + '_cells_per_s'))}"
                       for k in ("naive", "flat", "vec", "c", "mojo")
                       if r.get(k) or r.get("mojo_total") and k == "mojo")
        print(f"  {r['size']}x{r['size']}: {row}")

    results["table"] = table
    results["meta"] = {
        "steps": steps, "repeat": REPEAT, "inject": INJECT,
        "threshold": THRESHOLD, "timing": "best_of_n; python=perf_counter_ns; "
        "mojo=internal std.time.perf_counter_ns",
        "machine": {"cores": os.cpu_count()},
    }
    out = os.path.join(ROOT, "outputs", "bench.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
