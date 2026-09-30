"""ctypes bridge to the compiled C SoA quilt kernel (c/libsoaquilt.so).

Wave-69: the SoA counterpart of cflat.CFlatQuilt. Same Python API as the
other runtimes; the hot loops live in C compiled -O3 over four separate
field arrays.
"""
import ctypes
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(_HERE, "..", "c", "libsoaquilt.so")


class CSoaQuilt:
    def __init__(self, size: int, split_threshold: float = 0.6):
        if not os.path.exists(_LIB):
            raise RuntimeError("c/libsoaquilt.so missing — run c/build.sh first")
        self.lib = ctypes.CDLL(_LIB)
        self.size = size
        self.split_threshold = split_threshold
        self.total_cells = size * size
        self.pot = (ctypes.c_float * self.total_cells)()
        self.res = (ctypes.c_float * self.total_cells)()
        self.ent = (ctypes.c_float * self.total_cells)()
        self.spl = (ctypes.c_float * self.total_cells)()
        for i in range(self.total_cells):
            self.res[i] = 0.4
        self._snap = (ctypes.c_float * self.total_cells)()
        self.lib.qml_soa_flow.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self.lib.qml_soa_entropy.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]
        self.lib.qml_soa_checksum.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        self.lib.qml_soa_checksum.restype = ctypes.c_float

    def inject_force(self, r: int, c: int, potential: float):
        self.pot[r * self.size + c] = float(potential)

    def step_flow(self):
        self.lib.qml_soa_flow(self.pot, self.res, self.size, self._snap)

    def step_entropy(self):
        self.lib.qml_soa_entropy(
            self.pot, self.ent, self.spl, self.size, float(self.split_threshold))

    def potential(self, r: int, c: int) -> float:
        return self.pot[r * self.size + c]

    def entropy(self, r: int, c: int) -> float:
        return self.ent[r * self.size + c]

    def split(self, r: int, c: int) -> bool:
        return self.spl[r * self.size + c] == 1.0

    def checksum(self) -> float:
        return float(self.lib.qml_soa_checksum(self.pot, self.size))

    def run(self, steps: int):
        for _ in range(steps):
            self.step_flow()
        self.step_entropy()

    def raw_bytes(self) -> bytes:
        return bytes(self.pot) + bytes(self.res) + bytes(self.ent) + bytes(self.spl)
