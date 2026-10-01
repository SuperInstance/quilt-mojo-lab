"""export_soa.py — wave-72 four-block C-ABI interface v2 (Python bridge).

v1 (python/interface_probe.py) streamed the substrate as ONE interleaved
AoS block: float32[cells*4], [potential, resistance, entropy, split] per
cell. v2 dumps the SAME substrate as FOUR separate flat blocks (one per
field, structure-of-arrays layout) — the layout the wave-69 SoA kernels
already run in memory — wrapped in a self-describing container with a JSON
header (magic, version, grid size, dtype, block offsets, sha256 per block):

    offset 0      magic "QS2" (4 bytes)
    offset 4      uint32 header_len, little-endian (authoritative length)
    offset 8      header_len bytes UTF-8 JSON (space-padded)
    8+header_len  block 0..3, each cells*4 bytes, field order
                  potential, resistance, entropy, split

The heavy lifting (AoS<->SoA de-interleave, block hashing, container
writing) lives in C: c/export_soa.c, compiled -O3. This bridge:

  - export_v2(...): write a v2 container from either four field buffers
    (SoA runtimes: SoaQuilt / CSoaQuilt) or one AoS dump (v1 runtimes:
    FlatQuilt / CFlatQuilt raw_bytes) via the C de-interleaver,
  - read_v2(path): parse + verify (magic, version, dtype, offsets, and
    every block's sha256 against hashlib — cross-implementation agreement
    with the C hasher), returning one float32 ndarray per field,
  - to_aos(...): reconstitute a v1 interleaved block from v2 fields via
    the C interleaver.

Demo main: export a 256^2 grid, re-import, assert field-identical, assert
the four blocks are byte-identical to the corresponding strided slices of
the AoS dump. Receipts land in outputs/interface/ (v1 receipts kept).
"""
import ctypes
import hashlib
import json
import os
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(_HERE, "..", "c", "libexportsoa.so")
ROOT = os.path.dirname(_HERE)
OUT = os.path.join(ROOT, "outputs", "interface")

MAGIC = b"QS2"
VERSION = 2
FIELDS = ("potential", "resistance", "entropy", "split")


def _load():
    if not os.path.exists(_LIB):
        raise RuntimeError("c/libexportsoa.so missing — run c/build.sh first")
    lib = ctypes.CDLL(_LIB)
    lib.qml_soa_deinterleave.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_long] + \
        [ctypes.POINTER(ctypes.c_float)] * 4
    lib.qml_soa_interleave.argtypes = (
        [ctypes.POINTER(ctypes.c_float)] * 4 +
        [ctypes.c_long, ctypes.POINTER(ctypes.c_float)])
    lib.qml_soa_export.argtypes = (
        [ctypes.POINTER(ctypes.c_float)] * 4 +
        [ctypes.c_int, ctypes.c_char_p])
    lib.qml_soa_export.restype = ctypes.c_long
    return lib


def _f32_ptr(buf):
    """Zero-copy ctypes float* over any writable float32 buffer (ctypes
    array, array('f'), bytearray, writable ndarray)."""
    n = memoryview(buf).nbytes // 4
    arr = (ctypes.c_float * n).from_buffer(buf)
    return ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))


def _f32_copy(blob):
    n = len(blob) // 4
    arr = (ctypes.c_float * n).from_buffer_copy(blob)
    return ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))


def export_v2(path, size, fields=None, aos_bytes=None):
    """Write a v2 container. Exactly one of `fields` (4 float32 buffers, in
    field order) or `aos_bytes` (one interleaved v1 block) must be given.
    Returns the parsed header dict."""
    lib = _load()
    cells = size * size
    if (fields is None) == (aos_bytes is None):
        raise ValueError("pass exactly one of fields= or aos_bytes=")
    if fields is not None:
        assert len(fields) == 4
        ptrs = [_f32_ptr(f) for f in fields]
        keepalive = list(fields)
    else:
        if len(aos_bytes) != cells * 16:
            raise ValueError(f"AoS dump must be {cells * 16} bytes, "
                             f"got {len(aos_bytes)}")
        blocks = [(ctypes.c_float * cells)() for _ in range(4)]
        lib.qml_soa_deinterleave(_f32_copy(bytes(aos_bytes)), cells, *blocks)
        ptrs = [ctypes.cast(b, ctypes.POINTER(ctypes.c_float)) for b in blocks]
        keepalive = blocks
    hlen = lib.qml_soa_export(*ptrs, size, path.encode())
    if hlen < 0:
        raise IOError(f"qml_soa_export failed: {hlen}")
    del keepalive
    hdr = parse_header(open(path, "rb").read())
    if hdr["header_len"] != hlen:
        raise AssertionError("header_len mismatch between C return and file")
    return hdr


def parse_header(blob):
    """Parse + structurally validate a v2 container (bytes). No hashing."""
    if blob[:3] != MAGIC or blob[3] != 0:
        raise ValueError(f"bad magic {blob[:4]!r} (want {MAGIC!r} + 0x00)")
    (hlen,) = struct.unpack_from("<I", blob, 4)
    if len(blob) < 8 + hlen:
        raise ValueError("truncated container (header region)")
    hdr = json.loads(blob[8:8 + hlen].decode("utf-8").rstrip(" \x00"))
    if hdr["magic"] != MAGIC.decode() or hdr["version"] != VERSION:
        raise ValueError(f"bad header identity: {hdr['magic']!r} v{hdr['version']}")
    if hdr["header_len"] != hlen or hdr["data_offset"] != 8 + hlen:
        raise ValueError("header_len/data_offset inconsistent with prefix")
    if hdr["dtype"] != "float32-le":
        raise ValueError(f"unsupported dtype {hdr['dtype']}")
    if [b["field"] for b in hdr["blocks"]] != list(FIELDS):
        raise ValueError("block field order mismatch")
    return hdr


def read_v2(path):
    """Read + fully verify a v2 container from disk. See read_v2_bytes."""
    return read_v2_bytes(open(path, "rb").read())


def read_v2_bytes(blob):
    """Read + fully verify a v2 container (bytes). Returns dict with header
    and one (size, size) float32 ndarray per field. Verifies every block's
    sha256 (hashlib) against the header — C writer vs Python reader
    agreement."""
    hdr = parse_header(blob)
    size = hdr["grid"]
    cells = hdr["cells"]
    if size * size != cells:
        raise ValueError("grid^2 != cells")
    bb = hdr["block_size_bytes"]
    if bb != cells * 4:
        raise ValueError("block_size_bytes != cells*4")
    fields = {}
    for b in hdr["blocks"]:
        off, end = b["offset"], b["offset"] + bb
        if off < 8 + hdr["header_len"] or end > len(blob):
            raise ValueError(f"block {b['field']} out of bounds")
        data = blob[off:end]
        got = hashlib.sha256(data).hexdigest()
        if got != b["sha256"]:
            raise ValueError(
                f"sha256 mismatch for {b['field']}: {got} != {b['sha256']}")
        # bytearray copy -> writable arrays (consumers may buffer-share)
        fields[b["field"]] = np.frombuffer(
            bytearray(data), dtype="<f4").reshape(size, size)
    return {"header": hdr, "fields": fields, "blob": blob}


def to_aos(fields, size):
    """v2 -> v1: interleave four field buffers back into one AoS block."""
    lib = _load()
    cells = size * size
    out = (ctypes.c_float * (cells * 4))()
    ptrs = [_f32_ptr(f) for f in fields]
    keepalive = list(fields)
    lib.qml_soa_interleave(*ptrs, cells, ctypes.cast(
        out, ctypes.POINTER(ctypes.c_float)))
    del keepalive
    return bytes(out)


def field_buffers(quilt):
    """The four field buffers of an SoA runtime (SoaQuilt or CSoaQuilt)."""
    return (quilt.pot, quilt.res, quilt.ent, quilt.spl)


def aos_bytes_of(quilt):
    """The v1 interleaved dump of an AoS runtime (FlatQuilt / CFlatQuilt)."""
    return quilt.raw_bytes()


def main():
    """256^2 roundtrip receipt: export -> re-import -> field identity +
    byte-identity of the four blocks vs the AoS dump's strided slices."""
    sys.path.insert(0, _HERE)
    from flat_quilt import FlatQuilt

    N, STEPS = 256, 5
    os.makedirs(OUT, exist_ok=True)
    q = FlatQuilt(N)
    for r, c, p in ((1, 1, 4.8), (40, 60, 3.3), (100, 12, 5.9), (250, 250, 2.0)):
        q.inject_force(r, c, p)
    for _ in range(STEPS):
        q.step_flow()
    q.step_entropy()

    dump = aos_bytes_of(q)
    qbin = os.path.join(OUT, f"quilt_soa_v2_{N}.qbin")
    hdr = export_v2(qbin, N, aos_bytes=dump)
    print(f"exported 4x{hdr['block_size_bytes']}B blocks "
          f"(header_len={hdr['header_len']}, data_offset={hdr['data_offset']})")
    print(f"  -> {qbin}")

    # --- check 1: the four blocks are byte-identical to the strided slices
    # of the v1 AoS dump (de-interleave correctness, C -O3 path)
    aos = np.frombuffer(dump, dtype="<f4").reshape(N * N, 4)
    got = read_v2(qbin)
    for j, name in enumerate(FIELDS):
        sl = aos[:, j].tobytes()  # strided slice of the AoS dump
        blk = got["fields"][name].tobytes()
        assert sl == blk, f"block {name} != AoS strided slice"
    print("  blocks == strided slices of the AoS dump: OK")

    # --- check 2: sha256 in the C-written header matches hashlib (the
    # reader already verified; assert the header is internally consistent)
    for b in hdr["blocks"]:
        assert len(b["sha256"]) == 64
    print("  header sha256 per block verified against hashlib: OK")

    # --- check 3: re-import is field-identical to the source substrate
    f = got["fields"]
    ref_pot = aos[:, 0].reshape(N, N)
    assert np.array_equal(f["potential"], ref_pot)
    assert np.all(f["resistance"] == 0.4)  # uniform: nothing mutates resistance
    assert np.array_equal(f["entropy"], aos[:, 2].reshape(N, N))
    assert np.array_equal(f["split"], aos[:, 3].reshape(N, N))
    # source-substrate spot checks (offsets the snapshot semantics touched)
    assert got["fields"]["potential"][1, 1] == np.float32(q.potential(1, 1))
    assert got["fields"]["entropy"][40, 60] == np.float32(q.entropy(40, 60))
    assert bool(got["fields"]["split"][100, 12]) == q.split(100, 12)
    print("  re-import field-identical to source substrate: OK")

    # --- check 4: v2 -> v1 roundtrip (interleave back) is byte-exact
    back = to_aos((f["potential"].reshape(-1), f["resistance"].reshape(-1),
                   f["entropy"].reshape(-1), f["split"].reshape(-1)), N)
    assert back == dump
    print("  v2 -> v1 interleave roundtrip byte-exact: OK")

    # human-readable header receipt next to the container
    with open(os.path.join(OUT, f"quilt_soa_v2_{N}_header.json"), "w") as fh:
        json.dump(hdr, fh, indent=1)
    print(f"  header receipt -> {os.path.join(OUT, f'quilt_soa_v2_{N}_header.json')}")
    print("v2 export roundtrip: PASS")


if __name__ == "__main__":
    main()
