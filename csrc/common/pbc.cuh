#ifndef TORCHFF_PBC_CUH
#define TORCHFF_PBC_CUH

#include "vec3.cuh"

template <typename scalar_t>
__device__ void apply_pbc_triclinic(scalar_t* vec, scalar_t* box, scalar_t* box_inv, scalar_t* out) {
    // box in column major
    scalar_t s[3];
    for (int i = 0; i < 3; ++i) {
        s[i] = dot_vec3(&box_inv[i * 3], vec);
        if ( s[i] > 0.5 ) {
            s[i] -= 1.0;
        }
        else if ( s[i] < -0.5 ) {
            s[i] += 1.0;
        }
    }
    for (int i = 0; i < 3; ++i) {
        out[i] = dot_vec3(&box[i * 3], s);
    }
}

template <typename scalar_t>
__device__ void apply_pbc_orthorhombic(scalar_t* vec, scalar_t* box, scalar_t* out) {
    scalar_t s;
    scalar_t half_box;
    for (int i = 0; i < 3; ++i) {
        half_box = box[i] / 2;
        if (vec[i] > half_box) {
            out[i] = vec[i] - box;
        }
        else if (vec[i] < -half_box) {
            out[i] = vec[i] + box;
        }
        else {
            out[i] = vec[i];
        }
    }
}

template <typename scalar_t>
__device__ void apply_pbc_cubic(scalar_t* vec, scalar_t box, scalar_t* out) {
    scalar_t s;
    scalar_t half_box = box / 2;
    for (int i = 0; i < 3; ++i) {
        if (vec[i] > half_box) {
            out[i] = vec[i] - box;
        }
        else if (vec[i] < -half_box) {
            out[i] = vec[i] + box;
        }
        else {
            out[i] = vec[i];
        }
    }
}

#endif