/* soa_quilt.c — wave-69 SoA re-layout of the flat quilt kernel.
 *
 * Layout: four separate arrays (pot / res / ent / split), each size*size
 * float32. The flow-pass snapshot is now a contiguous memcpy and every
 * neighbor read is unit-stride — this is the layout fix for the 512^2 cache
 * cliff measured on the AoS kernel in v0.1.0 (docs/RESULTS.md).
 * Semantics identical to flat_quilt.c (AoS). The caller owns all buffers;
 * the kernel never allocates.
 */
#include <string.h>
#include <math.h>

void qml_soa_flow(float *pot, const float *res, int size, float *snap) {
    /* snap: caller-provided buffer of size*size floats */
    long total = (long)size * size;
    memcpy(snap, pot, total * sizeof(float));
    for (int r = 0; r < size; r++) {
        long row = (long)r * size;
        for (int c = 0; c < size; c++) {
            long idx = row + c;
            float p = snap[idx];
            float rr = res[idx];
            float adj = p;
            if (r + 1 < size) { float d = p - snap[idx + size]; if (d > rr) adj -= (d - rr) * 0.22f; }
            if (r - 1 >= 0)   { float d = p - snap[idx - size]; if (d > rr) adj -= (d - rr) * 0.22f; }
            if (c + 1 < size) { float d = p - snap[idx + 1];    if (d > rr) adj -= (d - rr) * 0.22f; }
            if (c - 1 >= 0)   { float d = p - snap[idx - 1];    if (d > rr) adj -= (d - rr) * 0.22f; }
            pot[idx] = adj;
        }
    }
}

void qml_soa_entropy(const float *pot, float *ent, float *split,
                     int size, float threshold) {
    long total = (long)size * size;
    for (long i = 0; i < total; i++) {
        float e = pot[i] / 5.0f * 0.95f;
        if (e > 1.0f) e = 1.0f;
        ent[i] = e;
        split[i] = (e > threshold) ? 1.0f : 0.0f;
    }
}

float qml_soa_checksum(const float *pot, int size) {
    long total = (long)size * size;
    float acc = 0.0f;
    for (long i = 0; i < total; i++) acc += pot[i];
    return acc;
}
