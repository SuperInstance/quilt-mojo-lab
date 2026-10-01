"""RUNTIME 10 — fp16/bf16 precision-budget probe on the CuPy quilt substrate.

Pre-registered: docs/RUNTIME10-FP16-PREREG.md (owner: lucineer).
One question: how much precision can the substrate give up before the
CHECKSUM stops being trustworthy? The deliverable IS the divergence curve
(P1); a verdict is deferred by design. "Parity" language is banned here.

Rounding topology (declared): every per-cell binary op (-, *, compare) is
rounded to half precision (fp16 or bf16); the field store stays float32;
the checksum is accumulated HOST-side in the oracle's sequential float64
order — identical bookkeeping to runtime #9, only the per-cell temporaries
lose precision.

P2 speed bar: fp16 kernel >= 1.4x the #9 fp64-intermediate kernel at 1024^2,
measured under the RAMP LAW — a 0.6 s sustained synced ramp on a throwaway
instance before EVERY timed block (banked instrument law; ramp receipts are
embedded in the JSON).
"""
import json
import sys
import time
import traceback

import numpy as np
import cupy as cp

_FLOW_SRC = r"""
#if defined(FP16)
#include <cuda_fp16.h>
#endif
#if defined(BF16)
#include <cuda_bf16.h>
#endif

#if defined(FP16)
typedef __half math_t;
__device__ __forceinline__ math_t _cast(float x) { return __float2half(x); }
__device__ __forceinline__ math_t _sub(math_t a, math_t b) { return __hsub(a, b); }
__device__ __forceinline__ math_t _mul(math_t a, math_t b) { return __hmul(a, b); }
__device__ __forceinline__ bool _gt(math_t a, math_t b) { return __hgt(a, b); }
__device__ __forceinline__ float _out(math_t a) { return __half2float(a); }
#elif defined(BF16)
typedef __nv_bfloat16 math_t;
__device__ __forceinline__ math_t _cast(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ math_t _sub(math_t a, math_t b) { return __hsub(a, b); }
__device__ __forceinline__ math_t _mul(math_t a, math_t b) { return __hmul(a, b); }
__device__ __forceinline__ bool _gt(math_t a, math_t b) { return __hgt(a, b); }
__device__ __forceinline__ float _out(math_t a) { return __bfloat162float(a); }
#else
typedef double math_t;
__device__ __forceinline__ math_t _cast(float x) { return (double)x; }
__device__ __forceinline__ math_t _sub(math_t a, math_t b) { return a - b; }
__device__ __forceinline__ math_t _mul(math_t a, math_t b) { return a * b; }
__device__ __forceinline__ bool _gt(math_t a, math_t b) { return a > b; }
__device__ __forceinline__ float _out(math_t a) { return (float)a; }
#endif

extern "C" __global__ void flow_kernel(const float* snap, const float* res,
                                       float* pot, int n, long long total) {
    long long idx = (long long)blockIdx.x * (long long)blockDim.x
                  + (long long)threadIdx.x;
    if (idx >= total) return;
    int r = (int)(idx / (long long)n);
    int c = (int)(idx - (long long)r * (long long)n);
    math_t p = _cast(snap[idx]);
    math_t rr = _cast(res[idx]);
    math_t adj = p;
    math_t d;
    if (r + 1 < n) { d = _sub(p, _cast(snap[idx + n])); if (_gt(d, rr)) adj = _sub(adj, _mul(_sub(d, rr), _cast(0.22f))); }
    if (r - 1 >= 0) { d = _sub(p, _cast(snap[idx - n])); if (_gt(d, rr)) adj = _sub(adj, _mul(_sub(d, rr), _cast(0.22f))); }
    if (c + 1 < n) { d = _sub(p, _cast(snap[idx + 1])); if (_gt(d, rr)) adj = _sub(adj, _mul(_sub(d, rr), _cast(0.22f))); }
    if (c - 1 >= 0) { d = _sub(p, _cast(snap[idx - 1])); if (_gt(d, rr)) adj = _sub(adj, _mul(_sub(d, rr), _cast(0.22f))); }
    pot[idx] = _out(adj);
}
"""

# Provenance: the "f64" arm below is python/cupy_quilt.py _FLOW_SRC semantics
# (double temporaries, single float store) — rebuilt here from the same source
# so this file is self-contained; the -DFP16/-DBF16 arms share the code path.
_f64_kernel = cp.RawKernel(_FLOW_SRC, "flow_kernel", options=("--fmad=false",))
_fp16_kernel = cp.RawKernel(_FLOW_SRC, "flow_kernel",
                            options=("--fmad=false", "-DFP16"))
try:
    _bf16_kernel = cp.RawKernel(_FLOW_SRC, "flow_kernel",
                                options=("--fmad=false", "-DBF16"))
except Exception as _e:  # declared: bf16 needs nvrtc bf16 header support
    _bf16_kernel = None
    _bf16_err = str(_e)
_THREADS = 256


def _launch(kernel, args, total):
    blocks = (total + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,), args)


def _make_instance(size, injects):
    q = cp.zeros(size * size, dtype=cp.float32)
    q_res = cp.full(size * size, 0.4, dtype=cp.float32)
    for (r, c, v) in injects:
        q[r * size + c] = np.float32(v)
    return q, q_res


def _run_flow(kernel, q, q_res, steps):
    n = np.int32(q.size**0.5 if False else int(round(q.size**0.5)))
    total = np.int64(q.size)
    for _ in range(steps):
        snap = q.copy()
        _launch(kernel, (snap, q_res, q, n, total), q.size)
    cp.cuda.get_current_stream().synchronize()


def _checksum(q):
    host = cp.asnumpy(q)
    acc = 0.0
    for i in range(host.size):
        acc += host[i]
    return acc


def _ramp(seconds):
    """RAMP LAW: sustained synced load on a throwaway instance (0.6 s default)."""
    a = cp.random.rand(2048 * 2048, dtype=cp.float32)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a *= 1.000001
        cp.cuda.get_current_stream().synchronize()
    del a
    cp.get_default_memory_pool().free_all_blocks()


def main():
    free, _total = cp.cuda.runtime.memGetInfo()
    if free < 256 * 1024 * 1024:
        raise RuntimeError(f"preflight FAIL: only {free/2**20:.0f} MiB VRAM free")
    rng = np.random.RandomState(2718)
    injects = [(int(rng.randint(0, 512)), int(rng.randint(0, 512)),
                float(rng.uniform(-3, 3))) for _ in range(16)]

    arms = [("f64", _f64_kernel), ("fp16", _fp16_kernel)]
    arms.append(("bf16", _bf16_kernel) if _bf16_kernel is not None
                else ("bf16-unavailable", None))

    # ---- P1: divergence curves at 512^2 -------------------------------
    p1 = []
    for name, kern in arms:
        if kern is None:
            continue
        row = {"dtype": name, "size": 512, "curves": []}
        for T in (16, 64, 256, 1024):
            q_ref, r_ref = _make_instance(512, injects)
            q_h, r_h = _make_instance(512, injects)
            kern_ref = _f64_kernel if name != "f64" else _f64_kernel
            kern_cmp = _f64_kernel if name == "f64" else kern
            _run_flow(kern_ref, q_ref, r_ref, T)
            _run_flow(kern_cmp, q_h, r_h, T)
            diff = cp.abs(q_ref - q_h)
            row["curves"].append({
                "T": T,
                "checksum": _checksum(q_h),
                "checksum_ref": _checksum(q_ref),
                "abs_checksum_delta": abs(_checksum(q_h) - _checksum(q_ref)),
                "max_abs_pot_delta": float(diff.max().get()),
            })
        p1.append(row)

    # ---- P2: speed at 1024^2 under the ramp law -----------------------
    p2 = {}
    for name, kern in (("f64", _f64_kernel), ("fp16", _fp16_kernel)):
        q, r = _make_instance(1024, injects)   # throwaway timing instance
        _ramp(0.6)
        steps = 50
        t0 = time.perf_counter()
        for _ in range(steps):
            snap = q.copy()
            _launch(kern, (snap, r, q, np.int32(1024), np.int64(q.size)), q.size)
        cp.cuda.get_current_stream().synchronize()
        p2[name] = {"ramp_s": 0.6, "steps": steps,
                    "wall_s": round(time.perf_counter() - t0, 4)}
    ratio = p2["f64"]["wall_s"] / max(p2["fp16"]["wall_s"], 1e-9)
    p2_pass = ratio >= 1.4

    result = {
        "experiment": "runtime10_fp16_precision_budget",
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "rounding_topology": "every per-cell binary op rounded to half; float32 field store; float64 host checksum",
        "P1_divergence_curves": p1,
        "P2_speed": {**p2, "f64_over_fp16": round(ratio, 3),
                     "gate_ge_1p4": p2_pass},
        "verdict": "MEASUREMENT-PROBE (P1 curve is the deliverable; verdict deferred by design). "
                   f"P2 {'PASS' if p2_pass else 'FAIL'} — on FAIL: book, no in-flight tuning (P3 trap clause).",
        "pre_registered": "docs/RUNTIME10-FP16-PREREG.md",
    }
    with open("outputs/runtime10_fp16_curves.json", "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    print(json.dumps({"P2_ratio": round(ratio, 3), "bf16": _bf16_kernel is not None},
                     indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        kill_path = "outputs/runtime10_fp16_curves.harness-invalid-%d.json" % int(time.time())
        with open(kill_path, "w") as fh:
            json.dump({"experiment": "runtime10_fp16_precision_budget",
                       "kill_receipt": kill_path,
                       "verdict": "KILL-harness",
                       "error": traceback.format_exc(),
                       "python": sys.executable}, fh, indent=2)
        raise
