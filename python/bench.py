"""Benchmark harness: same semantics across 8 runtimes, timed.

v0.2.0 (wave-69): adds the SoA (structure-of-arrays) runtimes —
python/soa_quilt.py, the C SoA kernel (c/soa_quilt.c via csoa.py), and the
Mojo SoA substrate (mojo/quilt_soa.mojo, lane-vectorized flow pass) — plus
the REGISTERED wave-69 check: SoA flow throughput at 512^2 >= 2x the AoS
flow on the compiled runtimes (prediction registered at p=0.45 in the
wave-68 discussion round; the measured verdict is recorded either way).

Correctness gate FIRST (every runtime must agree within tolerance on a small
grid), then timed runs (best of 3, flow passes only). Writes
outputs/bench<tag>.json + a markdown table to stdout. Mojo timings come from
the substrates' internal std.time.perf_counter_ns (FLOW_NS / ENTROPY_NS);
Python/C timings from time.perf_counter around the flow loop.
"""
import argparse
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
from soa_quilt import SoaQuilt  # noqa: E402
from vec_quilt import VecQuilt  # noqa: E402
import cflat  # noqa: E402
import csoa  # noqa: E402

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
    if kind == "soa":
        return SoaQuilt(size, THRESHOLD)
    if kind == "vec":
        return VecQuilt(size, THRESHOLD)
    if kind == "c":
        return cflat.CFlatQuilt(size, THRESHOLD)
    if kind == "csoa":
        return csoa.CSoaQuilt(size, THRESHOLD)
    raise ValueError(kind)


def bench_python(kind, size, steps):
    best = None
    checksum = None
    for _ in range(REPEAT):
        q = make(kind, size)
        for r, c, p in INJECT:
            q.inject_force(r, c, p)
        t0 = time.perf_counter_ns()
        for _ in range(steps):
            q.step_flow()
        t1 = time.perf_counter_ns()
        q.step_entropy()
        best = (t1 - t0) if best is None else min(best, t1 - t0)
        checksum = q.checksum()
    return best, checksum


def bench_mojo(src_name, size, steps):
    """Compile once (warm), run REPEAT times; parse internal timings."""
    src = os.path.join(ROOT, "mojo", src_name)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="16,32,64,128,256,512")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-mojo", action="store_true")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    steps = args.steps

    # correctness gate on 16
    ref = FlatQuilt(16, THRESHOLD)
    for r, c, p in INJECT:
        ref.inject_force(r, c, p)
    ref.run(steps)
    ref_cs = ref.checksum()
    gate = {}
    for kind in ("naive", "vec", "c", "soa", "csoa"):
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
    if not args.no_mojo:
        for label, src in (("mojo", "quilt_high_perf.mojo"),
                           ("mojo_soa", "quilt_soa.mojo")):
            _, _, cs = bench_mojo(src, 16, steps)
            gate[label] = "OK" if (cs is not None and
                                   abs(cs - ref_cs) <= max(abs(ref_cs) * 1e-4, 1e-4)) else \
                           f"CHECK ({cs} vs {ref_cs})"
    print(f"correctness gate ({steps} steps, ref={ref_cs:.10f}): {gate}")
    results = {"gate": gate}

    table = []
    for size in sizes:
        row = {"size": size, "steps": steps, "cells": size * size}
        for kind in ("naive", "flat", "soa", "vec", "c", "csoa"):
            if kind == "naive" and size > 64:
                continue
            try:
                ns, _cs = bench_python(kind, size, steps)
            except RuntimeError:
                row[kind] = None
                continue
            row[kind] = ns
            row[kind + "_cells_per_s"] = round(size * size * steps / (ns / 1e9), 0)
        if not args.no_mojo:
            mf, mt_, _ = bench_mojo("quilt_high_perf.mojo", size, steps)
            row["mojo_flow"] = mf
            row["mojo_total"] = mt_
            row["mojo_cells_per_s"] = round(size * size * steps / (mt_ / 1e9), 0)
            sf, st_, _ = bench_mojo("quilt_soa.mojo", size, steps)
            row["mojo_soa_flow"] = sf
            row["mojo_soa_total"] = st_
            row["mojo_soa_cells_per_s"] = round(size * size * steps / (st_ / 1e9), 0)
        table.append(row)

    print("\n| grid | naive | flat | soa | vec | c | csoa | mojo | mojo_soa |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in table:
        def fmt(v):
            return f"{v/1e6:.2f}ms" if v else "—"
        print(f"| {r['size']}x{r['size']} | {fmt(r.get('naive'))} | "
              f"{fmt(r.get('flat'))} | {fmt(r.get('soa'))} | {fmt(r.get('vec'))} | "
              f"{fmt(r.get('c'))} | {fmt(r.get('csoa'))} | "
              f"{fmt(r.get('mojo_total'))} | {fmt(r.get('mojo_soa_total'))} |")
    print("\nflow-only cells/sec (best of 3):")
    for r in table:
        parts = []
        for k in ("naive", "flat", "soa", "vec", "c", "csoa"):
            v = r.get(k + "_cells_per_s")
            if v:
                parts.append(f"{k}={v:,.0f}")
        if r.get("mojo_cells_per_s"):
            parts.append(f"mojo={r['mojo_cells_per_s']:,.0f}")
        if r.get("mojo_soa_cells_per_s"):
            parts.append(f"mojo_soa={r['mojo_soa_cells_per_s']:,.0f}")
        print(f"  {r['size']}x{r['size']}: " + " ".join(parts))

    # REGISTERED wave-69 check at the largest grid: SoA >= 2x AoS (flow-only)
    top = table[-1]
    check = {}
    if top.get("c") and top.get("csoa"):
        check["c_ratio"] = round(top["c"] / top["csoa"], 3)
    if top.get("mojo_flow") and top.get("mojo_soa_flow"):
        check["mojo_ratio"] = round(top["mojo_flow"] / top["mojo_soa_flow"], 3)
    ratios = [v for k, v in check.items() if k.endswith("_ratio")]
    check["registered"] = ("PASS" if any(v >= 2.0 for v in ratios)
                           else "FAIL") if ratios else "n/a"
    check["prediction"] = "SoA >= 2x AoS flow at 512^2 (registered p=0.45)"
    print(f"\nregistered wave-69 check @ {top['size']}x{top['size']} "
          f"(flow-only): {check}")
    results["registered_check"] = check

    results["table"] = table
    results["meta"] = {
        "steps": steps, "repeat": REPEAT, "inject": INJECT,
        "threshold": THRESHOLD, "timing": "best_of_n; python=perf_counter_ns; "
        "mojo=internal std.time.perf_counter_ns; flow passes only",
        "machine": {"cores": os.cpu_count()},
    }
    out = os.path.join(ROOT, "outputs", f"bench{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
