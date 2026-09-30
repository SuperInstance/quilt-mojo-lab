/* flat_quilt.c — compiled flat-memory quilt kernel (C baseline).
 *
 * Layout: interleaved AoS, 4 float32 slots per cell:
 *   [potential, resistance, entropy, split]
 * Semantics identical to python/flat_quilt.py (see its header).
 * The caller owns all buffers; the kernel never allocates.
 */
#include <math.h>

void qml_inject(float *mem, int size, int r, int c, float potential) {
    long idx = ((long)r * size + c) * 4;
    mem[idx] = potential;
}

void qml_flow(float *mem, int size, float *snap) {
    /* snap: caller-provided buffer of size*size floats (potentials snapshot) */
    long total = (long)size * size;
    for (long i = 0; i < total; i++) snap[i] = mem[i * 4];
    for (int r = 0; r < size; r++) {
        long row = (long)r * size;
        for (int c = 0; c < size; c++) {
            long idx = row + c;
            float pot = snap[idx];
            float res = mem[idx * 4 + 1];
            float adj = pot;
            if (r + 1 < size) { float d = pot - snap[idx + size]; if (d > res) adj -= (d - res) * 0.22f; }
            if (r - 1 >= 0)   { float d = pot - snap[idx - size]; if (d > res) adj -= (d - res) * 0.22f; }
            if (c + 1 < size) { float d = pot - snap[idx + 1];    if (d > res) adj -= (d - res) * 0.22f; }
            if (c - 1 >= 0)   { float d = pot - snap[idx - 1];    if (d > res) adj -= (d - res) * 0.22f; }
            mem[idx * 4] = adj;
        }
    }
}

void qml_entropy(float *mem, int size, float threshold) {
    long total = (long)size * size;
    for (long i = 0; i < total; i++) {
        float ent = mem[i * 4] / 5.0f * 0.95f;
        if (ent > 1.0f) ent = 1.0f;
        mem[i * 4 + 2] = ent;
        mem[i * 4 + 3] = (ent > threshold) ? 1.0f : 0.0f;
    }
}

float qml_checksum(float *mem, int size) {
    long total = (long)size * size;
    float acc = 0.0f;
    for (long i = 0; i < total; i++) acc += mem[i * 4];
    return acc;
}
