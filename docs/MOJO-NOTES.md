# MOJO-NOTES — API archaeology, Mojo 1.2.0-dev (2026-09-30 nightly)

The original draft (`mojo/original_draft.mojo`) targets the 24.x-era API and
does **not** compile on the current nightly. Every difference below was hit,
diagnosed, and fixed during this wave's build — recorded so the next porter
does not re-derive them. Toolchain: `pixi global install mojo -c
https://conda.modular.com/max-nightly -c conda-forge`; **`MODULAR_HOME` must
point at `<env>/share/max` or the compiler reports `unable to locate module
'std'`** (the conda activation script sets it; raw PATH export does not).

| Draft (24.x) | 1.2.0-dev | Notes |
|---|---|---|
| `fn foo()` | `def foo()` | `fn` **removed** (error: "use 'def' instead") |
| `let x = ...` | `var x = ...` | `let` **removed** |
| `alias F32 = ...` at file scope | removed | file-scope alias is a parse error; inline full types |
| `DTypePointer[DType.float32]` | `Pointer[Scalar[DType.float32], MutUntrackedOrigin]` | two deprecations deep |
| `DTypePointer.alloc(n)` | free function `alloc[T](n)` | deprecated; Layout-based `alloc` is the migration target |
| `ptr.free()` | `ptr.unsafe_free()` | deprecation warning only |
| `ptr.load(i)` / `ptr.store(i, v)` | `ptr[i]` (deprecated positional) or `ptr.load[width=k](off)` / `ptr.store[width=k](off, v)` | SIMD width variants work and are the fast path |
| `fn __init__(inout self, ...)` | `def __init__(out self, ...)` | `out` convention |
| `fn __del__(owned self)` | `def __deinit__(deinit self)` | `__del__` deprecated; `deinit` self-convention |
| `fn __moveinit__` | not needed | ownership semantics moved into the language defaults |
| `@value struct` | plain `struct` | decorator unnecessary for this pattern |
| `from time import Instant` | `from std import time` → `time.perf_counter_ns()` | `time`/`sys` live under the `std` package; `Instant` gone from the prelude surface |
| `sys.argv` list | `from std import sys` → `sys.argv()` → `len(args)`, `args[i]` | returns a Span; no `.size`/`.len` method — use builtin `len()` |
| `atol(s)` bare | `atol(s)` **raises** | wrap in `try/except e:` (new syntax works as expected) |
| `VariadicList[Tuple[...]]` | unrolled conditionals | dropped for the 4-neighbor gather anyway |

## Discovery notes

- The prelude resolves common symbols (`Pointer`, `SIMD`, `Scalar`, `alloc`,
  `atol`) **only when at least one `from std import ...` line is present**.
  With zero imports the same file reports `unknown declaration`. Quirk
  observed reproducibly; harmless once known.
- Finding the origin spelling: the compiler refuses to infer `origin` for
  struct *fields*. Instead of guessing aliases, force the error to print the
  concrete type: assign the inferred pointer to a wrongly-typed variable
  (`var z: Int = p`) and read the message —
  `cannot implicitly convert 'Pointer[Float32, MutUntrackedOrigin]' to 'Int'`.
  This trick beats trial-and-error on undocumented signatures.
- The `Pointer` synthetic signature renders garbled in error notes
  (`Pointerut: Bool, //, T: AnyType, origin: Originut=mut]`); `T` is the
  first parameter, `origin` the second.
- Modular.cfg looked corrupted (`ax]` for `[max]`) in terminal output —
  it was byte-perfect on inspection. Same ANSI-mangling false-positive class
  the fleet already receipted in wave 67 (L9): **verify bytes before filing
  defects.**

## Honest engineering deltas vs the draft (beyond API fixes)

1. **Snapshot buffer allocated once** in `__init__`, not per flow pass (the
   draft allocated and freed every step — an allocation in the hot loop is
   exactly what the architecture claims to eliminate).
2. **Whole-cell SIMD register I/O**: the 4-float-per-cell layout means one
   `load[width=4]`/`store[width=4]` per cell in the entropy pass. The flow
   pass stays a scalar gather (neighbor reads at ±1, ±N are data-dependent);
   the results page is explicit about what vectorized and what did not.
3. **Neighbor checks unrolled** S/N/E/W with identical order to the Python/C
   runtimes, so the cross-runtime checksum stays bit-comparable.
