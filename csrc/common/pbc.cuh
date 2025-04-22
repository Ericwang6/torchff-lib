#ifndef TORCHFF_PBC_CUH
#define TORCHFF_PBC_CUH

#include "vec3.cuh"

template <typename scalar_t>
__device__ void apply_pbc_triclinic(scalar_t* vec, scalar_t* box, scalar_t* box_inv, scalar_t* out) {
    // box in column major
    scalar_t s[3];
    for (int i = 0; i < 3; ++i) {
        s[i] = dot_vec3(&box_inv[i * 3], vec);
        s[i] -= round(s[i]);
    }
    for (int i = 0; i < 3; ++i) {
        out[i] = dot_vec3(&box[i * 3], s);
    }
}

template <typename scalar_t>
__device__ void apply_pbc_orthorhombic(scalar_t* vec, scalar_t* box, scalar_t* out) {
    out[0] = vec[0] - round(vec[0] / box[0]) * box[0];
    out[1] = vec[1] - round(vec[1] / box[1]) * box[1];
    out[2] = vec[2] - round(vec[2] / box[2]) * box[2];
}

template <typename scalar_t>
__device__ void apply_pbc_cubic(scalar_t* vec, scalar_t box, scalar_t* out) {
    out[0] = vec[0] - round(vec[0] / box) * box;
    out[1] = vec[1] - round(vec[1] / box) * box;
    out[2] = vec[2] - round(vec[2] / box) * box;
}

#endif