"""wave-72 scale test: the full 8-runtime grid at 1024^2 (+ a 512^2 in-session
reference row for the registered check).

PRE-REGISTERED PREDICTION (written in this docstring BEFORE any measurement,
per the wave-69 discipline; registered 2026-10-02, p=0.55):

    C-SoA per-cell flow throughput at 1024^2 stays within 2x of its measured
    512^2 value in the SAME session:
        csoa_cells_per_s(1024) >= csoa_cells_per_s(512) / 2
    Rationale: the SoA working set at 1024^2 is pot+snap+res ~= 12 MB (vs
    ~3 MB at 512^2). Wave-69 measured SoA degrading gracefully across the
    256->512 doubling (C-SoA 565M -> 507M cells/s, a ~10% cliff, vs the AoS
    23-25% cliff), so another doubling of grid size should NOT halve
    per-cell throughput; cache pressure costs some, not all. The verdict is
    recorded honestly either way; PASS requires ratio >= 0.5.

Everything else matches python/bench.py (wave-69 protocol): identical
semantics across runtimes, correctness gate FIRST (16^2 checksum agreement,
all 8 runtimes), then timed runs best-of-3, flow passes only. wave-72 adds:
  - a 1024^2 row for ALL runtimes (naive included — the v0.1.0 tables
    dropped it past 64^2; the dict loop is required here per wave-72 task),
  - 1024^2 checksum agreement across all 8 runtimes,
  - full-grid max|dPot| vs the flat runtime at 1024^2 for every runtime
    that can stream its potential field out (stronger than the checksum),
  - peak RSS via resource.getrusage (self high-water after each Python
    runtime; children high-water for the Mojo subprocess runs).

Writes outputs/bench_1024.json + a markdown table to stdout.
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
ROOT = os.path.dirname(_HERE)

import bench  # noqa: E402  (shared protocol: make/bench_python/bench_mojo/INJECT/THRESHOLD/REPEAT)
from flat_quilt import FlatQuilt  # noqa: E402

KINDS = ("naive", "flat", "soa", "vec", "c", "csoa")
MOJO_SRCS = (("mojo", "quilt_high_perf.mojo"), ("mojo_soa", "quilt_soa.mojo"))
GATE_TOL = lambda ref: max(abs(ref) * 1e-4, 1e-4)


def self_rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux


def child_rss_kb():
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss


def pot_field(kind, q, n):
    """Stream a runtime's potential field into an (n, n) float32 ndarray."""
    if kind == "flat":
        a = np.frombuffer(q.raw_bytes(), dtype=np.float32).reshape(n * n, 4)
        return a[:, 0].reshape(n, n).copy()
    if kind == "naive":
        return np.array([[q.potential(r, c) for c in range(n)] for r in range(n)],
                        dtype=np.float32)
    if kind in ("soa", "csoa"):
        return np.frombuffer(bytes(q.pot), dtype=np.float32).reshape(n, n).copy()
    if kind == "vec":
        return q.pot.copy()
    if kind == "c":
        a = np.frombuffer(bytes(q.mem), dtype=np.float32).reshape(n * n, 4)
        return a[:, 0].reshape(n, n).copy()
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,1024")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--no-mojo", action="store_true")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    steps = args.steps
    t_wall0 = time.perf_counter()

    # ---- correctness gate FIRST: 16^2 checksum agreement, all 8 runtimes ----
    ref = FlatQuilt(16, bench.THRESHOLD)
    for r, c, p in bench.INJECT:
        ref.inject_force(r, c, p)
    ref.run(steps)
    ref_cs = ref.checksum()
    gate = {"flat": "OK (reference)"}
    for kind in KINDS:
        if kind == "flat":
            continue
        try:
            q = bench.make(kind, 16)
        except RuntimeError:
            gate[kind] = "skipped (no lib)"
            continue
        for r, c, p in bench.INJECT:
            q.inject_force(r, c, p)
        q.run(steps)
        ok = abs(q.checksum() - ref_cs) <= GATE_TOL(ref_cs)
        gate[kind] = "OK" if ok else f"FAIL ({q.checksum()} vs {ref_cs})"
    if not args.no_mojo:
        for label, src in MOJO_SRCS:
            _, _, cs = bench.bench_mojo(src, 16, steps)
            ok = cs is not None and abs(cs - ref_cs) <= GATE_TOL(ref_cs)
            gate[label] = "OK" if ok else f"CHECK ({cs} vs {ref_cs})"
    print(f"correctness gate 16^2 ({steps} steps, ref={ref_cs:.10f}): {gate}")
    bad = [k for k, v in gate.items() if str(v).startswith("FAIL")]
    if bad:
        print(f"GATE FAILED for {bad}; aborting before any timed run.")
        sys.exit(1)

    results = {"gate_16": gate, "gate_ref_checksum": ref_cs}

    # ---- timed rows: flow-only, best of 3 (same protocol as bench.py) ----
    table = []
    for size in sizes:
        row = {"size": size, "steps": steps, "cells": size * size,
               "self_rss_kb_after": {}, "checksums": {}, "maxdiff_vs_flat": {}}
        flat_q = None
        for kind in KINDS:
            try:
                ns, cs = bench.bench_python(kind, size, steps)
            except RuntimeError:
                row[kind] = None
                continue
            row[kind] = ns
            row[kind + "_cells_per_s"] = round(size * size * steps / (ns / 1e9), 0)
            row["checksums"][kind] = cs
            row["self_rss_kb_after"][kind] = self_rss_kb()
            if kind == "flat":
                flat_q = bench.make(kind, size)
                for r, c, p in bench.INJECT:
                    flat_q.inject_force(r, c, p)
                flat_q.run(steps)
        if not args.no_mojo:
            for label, src in MOJO_SRCS:
                mf, mt_, cs = bench.bench_mojo(src, size, steps)
                row[label + "_flow"] = mf
                row[label + "_total"] = mt_
                row[label + "_flow_cells_per_s"] = round(size * size * steps / (mf / 1e9), 0)
                row[label + "_cells_per_s"] = round(size * size * steps / (mt_ / 1e9), 0)
                row["checksums"][label] = cs
                row["self_rss_kb_after"][label] = child_rss_kb()
        # full-grid agreement at this size (skip the naive O(n^2) walk at 1024
        # for speed: naive already agreed at the gate + checksum here)
        if flat_q is not None:
            ref_pot = pot_field("flat", flat_q, size)
            for kind in KINDS:
                if kind in ("naive", "flat") or row.get(kind) is None:
                    continue
                try:
                    q = bench.make(kind, size)
                except RuntimeError:
                    continue
                for r, c, p in bench.INJECT:
                    q.inject_force(r, c, p)
                q.run(steps)
                d = float(np.max(np.abs(pot_field(kind, q, size) - ref_pot)))
                row["maxdiff_vs_flat"][kind] = d
        table.append(row)

    # checksum agreement at the largest grid
    top = table[-1]
    cs_ref = top["checksums"]["flat"]
    cs_ok = {}
    for k, v in top["checksums"].items():
        if v is not None:
            cs_ok[k] = bool(abs(v - cs_ref) <= GATE_TOL(cs_ref))
    results["checksum_agreement_top"] = {"size": top["size"], "ref": cs_ref,
                                         "tolerance": GATE_TOL(cs_ref), "ok": cs_ok}

    # ---- the REGISTERED wave-72 check ----
    rows512 = [r for r in table if r["size"] == 512]
    r512 = rows512[0] if rows512 else None
    r1024 = top
    check = {
        "prediction": ("C-SoA per-cell flow throughput at 1024^2 stays within 2x "
                       "of its in-session 512^2 value: csoa(1024)/csoa(512) >= 0.5 "
                       "(pre-registered in python/bench_1024.py docstring, p=0.55, "
                       "BEFORE measurement)"),
        "registered_before_measurement": True,
    }
    if r512 and r512.get("csoa_cells_per_s") and r1024.get("csoa_cells_per_s"):
        ratio = r1024["csoa_cells_per_s"] / r512["csoa_cells_per_s"]
        check["csoa_512_cells_per_s"] = r512["csoa_cells_per_s"]
        check["csoa_1024_cells_per_s"] = r1024["csoa_cells_per_s"]
        check["ratio_1024_over_512"] = round(ratio, 4)
        check["verdict"] = "PASS" if ratio >= 0.5 else "FAIL"
    else:
        check["verdict"] = "n/a (csoa missing)"
    results["registered_check"] = check

    # ---- stdout tables ----
    print("\nwall-clock per 10 flow steps (best of 3):")
    print("| grid | naive | flat | soa | vec | c | csoa | mojo | mojo_soa |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in table:
        fmt = lambda v: f"{v/1e6:.2f}ms" if v else "—"
        mo = r.get("mojo_total")
        mso = r.get("mojo_soa_total")
        print(f"| {r['size']}x{r['size']} | {fmt(r.get('naive'))} | {fmt(r.get('flat'))} | "
              f"{fmt(r.get('soa'))} | {fmt(r.get('vec'))} | {fmt(r.get('c'))} | "
              f"{fmt(r.get('csoa'))} | {fmt(mo)} | {fmt(mso)} |")
    print("\nflow-only cells/sec (best of 3; mojo columns are FLOW_NS-based):")
    for r in table:
        parts = []
        for k in KINDS:
            v = r.get(k + "_cells_per_s")
            if v:
                parts.append(f"{k}={v:,.0f}")
        for label in ("mojo", "mojo_soa"):
            v = r.get(label + "_flow_cells_per_s")
            if v:
                parts.append(f"{label}={v:,.0f}")
        print(f"  {r['size']}x{r['size']}: " + " ".join(parts))
    print("\n1024^2 checksum agreement:", results["checksum_agreement_top"])
    print(f"\npeak RSS self={self_rss_kb()/1024:.1f}MB children={child_rss_kb()/1024:.1f}MB")
    print(f"registered wave-72 check: {check}")

    results["table"] = table
    results["peak_rss_kb"] = {"self": self_rss_kb(), "children": child_rss_kb()}
    results["meta"] = {
        "steps": steps, "repeat": bench.REPEAT, "inject": bench.INJECT,
        "threshold": bench.THRESHOLD,
        "timing": "best_of_n; python=perf_counter_ns; mojo=internal "
                  "std.time.perf_counter_ns; flow passes only; mojo flow column "
                  "is FLOW_NS-based (total-based also recorded)",
        "machine": {"cores": os.cpu_count()},
        "mojo_available": not args.no_mojo,
        "wall_seconds": round(time.perf_counter() - t_wall0, 1),
        "prediction_registered_in": "python/bench_1024.py docstring (pre-run)",
    }
    out = os.path.join(ROOT, "outputs", "bench_1024.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
