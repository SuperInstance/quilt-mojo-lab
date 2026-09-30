/* quilt_layout.h — C ABI of the quilt flat-memory block (auto-generated) */
#ifndef QUILT_LAYOUT_H
#define QUILT_LAYOUT_H
#include <stdint.h>
#define QUILT_SIZE 16
#define QUILT_CELLS 256
#define QUILT_SLOTS_PER_CELL 4   /* [potential, resistance, entropy, split] */
typedef struct {
    float potential;
    float resistance;
    float entropy;
    float split;      /* 1.0 = ACTIVE, 0.0 = STATIC */
} quilt_cell_t;
/* The whole fabric is quilt_cell_t QUILT_CELLS, contiguous, row-major. */
typedef quilt_cell_t quilt_fabric_t[QUILT_CELLS];
#endif
