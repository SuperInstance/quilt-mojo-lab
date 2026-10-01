/* export_soa.c — wave-72 four-block C-ABI exporter (interface v2).
 *
 * v1 (interface_probe.py) streamed ONE interleaved block: float32[cells*4],
 * layout [pot,res,ent,split] per cell (AoS). v2 dumps the substrate as FOUR
 * separate flat blocks — one per field, structure-of-arrays layout — each
 * with its own sha256, wrapped in a self-describing container:
 *
 *   offset 0   : magic "QS2" (4 bytes)
 *   offset 4   : uint32 header_len, little-endian (the authoritative length)
 *   offset 8   : header_len bytes of UTF-8 JSON (space-padded):
 *                  magic, version, grid, cells, dtype, fields,
 *                  block_size_bytes, header_len, data_offset,
 *                  blocks[4]: {field, offset (absolute), sha256}
 *   8+header_len : block 0..3, each cells*4 bytes, field order
 *                  potential, resistance, entropy, split
 *
 * The header JSON is rendered with a fixed-point pass (data_offset depends on
 * header_len depends on the JSON length); a u32 prefix keeps the authoritative
 * length binary-exact for consumers that never parse JSON.
 *
 * Also provided: qml_soa_deinterleave (AoS block -> four blocks) and
 * qml_soa_interleave (four blocks -> AoS block), so a v2 container can be
 * produced from a v1 dump and reconstituted back. The caller owns all
 * buffers; nothing here allocates on behalf of the substrate hot path.
 * Compile: gcc -O3 -fPIC -shared -o libexportsoa.so export_soa.c -lm
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- sha256 (FIPS 180-4, compact, unoptimized is fine: it is
 * run once per block at export time, off the hot path) ---------------- */

static const uint32_t SHA_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha256(const uint8_t *msg, long len, uint8_t out[32]) {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    long total = ((len + 8) / 64 + 1) * 64; /* padded message length */
    uint8_t block[64];
    for (long off = 0; off < total; off += 64) {
        if (off + 64 <= len) {
            memcpy(block, msg + off, 64);
        } else {
            memset(block, 0, 64);
            long rem = len - off;
            if (rem < 0) rem = 0;
            if (rem > 64) rem = 64;
            if (rem) memcpy(block, msg + off, rem);
            if (rem < 64 && off + rem < len + 1) block[rem] = 0x80;
            /* 64-bit big-endian bit length in the final block */
            uint64_t bits = (uint64_t)len * 8;
            for (int i = 0; i < 8; i++) block[63 - i] = (uint8_t)(bits >> (8 * i));
        }
        uint32_t w[64];
        for (int t = 0; t < 16; t++) {
            w[t] = ((uint32_t)block[4 * t] << 24) | ((uint32_t)block[4 * t + 1] << 16) |
                   ((uint32_t)block[4 * t + 2] << 8) | (uint32_t)block[4 * t + 3];
        }
        for (int t = 16; t < 64; t++) {
            uint32_t s0 = ROTR(w[t - 15], 7) ^ ROTR(w[t - 15], 18) ^ (w[t - 15] >> 3);
            uint32_t s1 = ROTR(w[t - 2], 17) ^ ROTR(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16] + s0 + w[t - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int t = 0; t < 64; t++) {
            uint32_t S1 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + SHA_K[t] + w[t];
            uint32_t S0 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    for (int i = 0; i < 8; i++) {
        out[4 * i] = (uint8_t)(h[i] >> 24);
        out[4 * i + 1] = (uint8_t)(h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(h[i] >> 8);
        out[4 * i + 3] = (uint8_t)h[i];
    }
}

static void sha256_hex(const void *data, long len, char out[65]) {
    static const char HEX[] = "0123456789abcdef";
    uint8_t dig[32];
    sha256((const uint8_t *)data, len, dig);
    for (int i = 0; i < 32; i++) {
        out[2 * i] = HEX[dig[i] >> 4];
        out[2 * i + 1] = HEX[dig[i] & 0xf];
    }
    out[64] = '\0';
}

/* ---------------- AoS <-> SoA block conversion ---------------- */

void qml_soa_deinterleave(const float *aos, long cells, float *pot, float *res,
                          float *ent, float *spl) {
    for (long i = 0; i < cells; i++) {
        pot[i] = aos[4 * i];
        res[i] = aos[4 * i + 1];
        ent[i] = aos[4 * i + 2];
        spl[i] = aos[4 * i + 3];
    }
}

void qml_soa_interleave(const float *pot, const float *res, const float *ent,
                        const float *spl, long cells, float *aos) {
    for (long i = 0; i < cells; i++) {
        aos[4 * i] = pot[i];
        aos[4 * i + 1] = res[i];
        aos[4 * i + 2] = ent[i];
        aos[4 * i + 3] = spl[i];
    }
}

/* ---------------- v2 container export ---------------- */

#define HDR_CAP 4096

static long render_header(int size, long hlen, const char sha[4][65], char *buf) {
    long cells = (long)size * size;
    long bb = cells * 4;
    long data_off = 8 + hlen;
    return snprintf(
        buf, HDR_CAP,
        "{\"magic\": \"QS2\", \"version\": 2, \"grid\": %d, \"cells\": %ld, "
        "\"dtype\": \"float32-le\", \"byte_order\": \"little\", "
        "\"fields\": [\"potential\", \"resistance\", \"entropy\", \"split\"], "
        "\"block_size_bytes\": %ld, \"header_len\": %ld, \"data_offset\": %ld, "
        "\"blocks\": ["
        "{\"field\": \"potential\", \"offset\": %ld, \"sha256\": \"%s\"}, "
        "{\"field\": \"resistance\", \"offset\": %ld, \"sha256\": \"%s\"}, "
        "{\"field\": \"entropy\", \"offset\": %ld, \"sha256\": \"%s\"}, "
        "{\"field\": \"split\", \"offset\": %ld, \"sha256\": \"%s\"}]}",
        size, cells, bb, hlen, data_off, data_off, sha[0], data_off + bb,
        sha[1], data_off + 2 * bb, sha[2], data_off + 3 * bb, sha[3]);
}

/* Writes the v2 container. Returns header_len (>0) on success, negative on
 * error. Field order in the file: potential, resistance, entropy, split. */
long qml_soa_export(const float *pot, const float *res, const float *ent,
                    const float *spl, int size, const char *path) {
    long cells = (long)size * size;
    long bb = cells * 4;
    char sha[4][65];
    sha256_hex(pot, bb, sha[0]);
    sha256_hex(res, bb, sha[1]);
    sha256_hex(ent, bb, sha[2]);
    sha256_hex(spl, bb, sha[3]);

    char *json = malloc(HDR_CAP);
    if (!json) return -2;
    /* fixed point: data_offset (= 8 + header_len) is printed INSIDE the JSON */
    long hlen = 0;
    long jlen = render_header(size, hlen, sha, json);
    for (int it = 0; it < 8 && jlen > hlen; it++) {
        hlen = jlen;
        jlen = render_header(size, hlen, sha, json);
    }
    if (jlen > hlen || hlen + 1 > HDR_CAP) {
        free(json);
        return -3;
    }
    memset(json + jlen, ' ', hlen - jlen); /* pad to the authoritative length */

    FILE *f = fopen(path, "wb");
    if (!f) {
        free(json);
        return -1;
    }
    uint8_t pre[8] = {'Q', 'S', '2', 0, 0, 0, 0, 0};
    uint32_t hl = (uint32_t)hlen;
    for (int i = 0; i < 4; i++) pre[4 + i] = (uint8_t)(hl >> (8 * i)); /* LE */
    long rc = 0;
    if (fwrite(pre, 1, 8, f) != 8) rc = -4;
    if (!rc && fwrite(json, 1, hlen, f) != (size_t)hlen) rc = -4;
    if (!rc && fwrite(pot, 1, bb, f) != (size_t)bb) rc = -4;
    if (!rc && fwrite(res, 1, bb, f) != (size_t)bb) rc = -4;
    if (!rc && fwrite(ent, 1, bb, f) != (size_t)bb) rc = -4;
    if (!rc && fwrite(spl, 1, bb, f) != (size_t)bb) rc = -4;
    if (fclose(f) != 0 && !rc) rc = -5;
    free(json);
    return rc ? rc : hlen;
}
