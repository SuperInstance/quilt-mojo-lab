# Mojo 1.2.0 Field Guide — earned the hard way, overnight 2026-09-30→10-01

**How this was earned:** 7 SuperInstance repos verified on-box (quilt-mojo-lab, kev-substrate,
quilt-mojo, grand-pattern, exocortex-embed, constraint-theory, conservation-spectral) —
~53 verified trap hits across ~6 hours of mechanical ports, every one reproduced and banked.
Toolchain: Mojo 1.2.0.dev2026100105 via `/home/eileen/.pixi/bin/pixi run mojo`.

**Doctrine that made it survivable:** mechanical drift FIXED (originals archived
`*.orig-YYYYMMDD`, never deleted); semantic defects BOOKED with full diagnosis (never
silently patched); every test's hardcoded constants verified against a live oracle —
**two of seven repos had wrong constants**; cross-language parity caught real content bugs
twice (nibble-reversal; dict-order serialization).

## Invocation & flags

| Trap | Truth |
|------|-------|
| `-D ASSERT=all` after filename | **Silently ignored** — produced a false PASS in a negative control. Flags ALWAYS before filename. |
| `assert` without `-D ASSERT=all` | Compile-clean **no-op**. |
| Builtin `assert(cond, msg)` | **Uncallable in 1.2** — args coerce to a Tuple. Use the fleet Counters pass/fail pattern. |
| `pixi run mojo` from non-manifest dir | Fails. `mojo.modular` is NOT a pixi manifest (needs pixi.toml/pyproject.toml). Cross-repo: `export MODULAR_HOME=/home/eileen/.pixi` + PATH, or run via quilt-mojo-lab's manifest with `-I` back-pointers. |
| Makefile env | Bare `MODULAR_HOME := ...` is not exported — needs `export MODULAR_HOME`. |

## Imports & layout

- Imports resolve from the **MAIN FILE's dir**, not cwd → `-I src` (src/-layouts) / `-I .` (repo-root packages). Wrong `-I` = silent resolution failures.
- Packages need `__init__.mojo`; relative intra-package imports (`from .graph import`) **don't resolve** → absolute (`from conservation_spectral.graph import`) + `-I .`.
- Prelude needs **≥1 `from std import ...`** line or prelude symbols (Pointer, alloc…) don't resolve.
- Module-level `var` AND `alias` both banned (file-scope expressions rejected) — state lives in mut-passed structs.

## Removed / renamed constructs (1.2 nightly drift)

- `fn` **REMOVED** (→def) · `let` (→var) · `owned` keyword (parse error) · `@value` (plain struct + explicit `__init__(out self, ...)`).
- `inout` → `out` / `mut self` · `byte_length()` (not `len()` on String — compile error) · `StringRef` → String · `case` is a keyword · `from collections import` gone.
- `DynamicVector` → builtin `List` · `UnsafePointer` → `Pointer[T, MutUntrackedOrigin]` · `__del__` → `__deinit__(deinit self)` · params are immutable borrows (call-site `x^` works).
- `from std import time` + `time.perf_counter_ns()` — all `now()` variants dead. `math` → `from std import math`. `bitcast[DType.uint64](x)` from std.memory — free function taking a **DType parameter** (the `.bitcast()` method doesn't exist).
- Tuple `.get[i,i]()` → `t[i]`; Tuples not implicitly copyable → rebuild from elements. No-self methods need `@staticmethod`. `print(True)` → `True` (no `True`/`False` lowercase mess).
- f-string format specs (`{x:.3f}`) are **parse errors** — round manually. `String(x, 3)` / `.align_right()` gone.

## Ownership & List semantics (the deep water)

- `List` is **NOT ImplicitlyCopyable** → List-holding structs can't be either; explicit `__copyinit__` fails; design **zero-copy** (mutate via refs).
- Same-List `(read, mut)` arg aliasing = **compile error** → factor helpers that copy small values.
- Move-out-of-subscript rejected (`list[i]^` won't compile) — and `var x = list[i]` of List element type is rejected too; index directly.
- `^` transfer then read = **uninitialized** value (compute `len()` before the move).
- Whole-struct field overwrite through mut ref (`room.vibe = Vibe()`) rejected — assign fields individually.
- `__init__` calling a mut method needs ALL fields pre-initialized first.
- Implicit field-wise ctor demands a `move` arg. Returns of List-holders need `return x^`.
- **Pointer:** `Pointer[T].alloc` doesn't exist → free function `alloc[T](n)`; `load/store/free/+` → `unsafe_*` spellings; **Pointer truthiness is a compile error** ("non-null by design") → `alloc[T](1)` dummies + flag. `unsafe_alloc` not prelude-resolved.

## SIMD & numerics

- SIMD comparisons **reduce to scalar Bool** (no lane masks). Elementwise max/min **GONE** (free function AND method). `SIMD[DType.bool, N](False)` won't instantiate — splat `(0)`.
- `hex(UInt64)` = 0x-prefixed, no sign-extension (safe for hash receipts).
- Runtime Int→Int32 not implicit AND instantiation is **LAZY** — bad calls can hide from an entire test suite until some caller triggers them. (Check every Int-typed param call site.)
- `StringLiteral` not concrete as a struct field type → use `String`.

## Perf / behavioral traps

- Broken QR never converges → eigendecompose burns the full iteration budget (~10s @ n=256). Silent perf poison: always time your tests.
- The nightly moves FAST: `fn`/`@value` removed and SIMD max/min vanished **between adjacent nightlies** — pin the version in every receipt, re-run parity after any toolchain bump.

## The parity discipline (fleet standard)

1. Oracle first: write `python/oracle.py` from the same formulas if none exists.
2. Bit-for-bit on ≥5–7 cases **including degenerates** (empty, zero-vector, single element, exact-conservation).
3. Verify hardcoded test constants against the live oracle — never trust them.
4. Conservation laws / invariants are the natural oracle where they exist (L·1=0, symmetry, unit diagonal → machine epsilon).
5. Divergence? Diagnose fully, decide canon from the spec, book it. Semantic fixes are BOOKED, not made.
6. IEEE754 note: `reduce_add` over 8 SIMD lanes matched Python sequential accumulation exactly on this box — float determinism receipts are possible.
