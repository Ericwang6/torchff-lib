#ifndef TORCHFF_PBC_CUH
#define TORCHFF_PBC_CUH

#include "vec3.cuh"

template <typename scalar_t>
__device__ void apply_pbc_triclinic(scalar_t* vec, scalar_t* box, scalar_t* box_inv, scalar_t* out) {
    // box in row major
    scalar_t sa = vec[0] * box_inv[0] + vec[1] * box_inv[3] + vec[2] * box_inv[6];
    scalar_t sb = vec[0] * box_inv[1] + vec[1] * box_inv[4] + vec[2] * box_inv[7];
    scalar_t sc = vec[0] * box_inv[2] + vec[1] * box_inv[5] + vec[2] * box_inv[8];
    sa -= round_(sa);
    sb -= round_(sb);
    sc -= round_(sc);
    out[0] = sa * box[0] + sb * box[3] + sc * box[6];
    out[1] = sa * box[1] + sb * box[4] + sc * box[7];
    out[2] = sa * box[2] + sb * box[5] + sc * box[8];
}


template <typename scalar_t>
__device__ void apply_pbc_orthorhombic(scalar_t* vec, scalar_t* box, scalar_t* out) {
    out[0] = vec[0] - round_(vec[0] / box[0]) * box[0];
    out[1] = vec[1] - round_(vec[1] / box[1]) * box[1];
    out[2] = vec[2] - round_(vec[2] / box[2]) * box[2];
}


template <typename scalar_t>
__device__ void apply_pbc_cubic(scalar_t* vec, scalar_t box, scalar_t* out) {
    out[0] = vec[0] - round_(vec[0] / box) * box;
    out[1] = vec[1] - round_(vec[1] / box) * box;
    out[2] = vec[2] - round_(vec[2] / box) * box;
}

#endif
