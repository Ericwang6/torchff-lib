#ifndef TORCHFF_CMM_DAMPS_CUH
#define TORCHFF_CMM_DAMPS_CUH

#include "common/vec3.cuh"


template <typename scalar_t>
__device__ void one_center_damps(scalar_t r, scalar_t b, scalar_t* damps) {
    scalar_t u = b * r;
    scalar_t u2 = u * u;
    scalar_t u4 = u2 * u2;
    scalar_t expu = exp_(-u);

    constexpr scalar_t c1_2 = scalar_t(1.0/2.0);
    constexpr scalar_t c1_6 = scalar_t(1.0/6.0);
    constexpr scalar_t c1_30 = scalar_t(1.0/30.0);
    constexpr scalar_t c4_105 = scalar_t(4.0/105.0);
    constexpr scalar_t c1_210 = scalar_t(1.0/210.0);
    constexpr scalar_t c2_315 = scalar_t(2.0/315.0);
    constexpr scalar_t c5_126 = scalar_t(5.0/126.0);
    constexpr scalar_t c1_1890 = scalar_t(1.0/1890.0);

    scalar_t p;
    p = 1 + u * c1_2; 
    damps[0] = expu * p;
    p = 1 + u + u2 * c1_2;
    damps[1] = expu * p;
    p += u2*u * c1_6;
    damps[2] = expu * p;
    damps[3] = expu * ( p + u4 * c1_30 );
    damps[4] = expu * ( p + u4 * c4_105 + u4*u * c1_210);
    damps[5] = expu * ( p + u4 * c5_126 + u4*u * c2_315 + u2*u4 * c1_1890);
}


template <typename scalar_t>
__device__ void two_center_damps(scalar_t r, scalar_t b, scalar_t* damps) {
    scalar_t u = b * r;
    scalar_t u2 = u * u;
    scalar_t u4 = u2 * u2;
    scalar_t expu = exp_(-u);

    constexpr scalar_t c1_2 = scalar_t(1.0/2.0);
    constexpr scalar_t c11_16 = scalar_t(11.0/16.0);
    constexpr scalar_t c3_16 = scalar_t(3.0/16.0);
    constexpr scalar_t c1_48 = scalar_t(1.0/48.0);
    constexpr scalar_t c3_48 = scalar_t(3.0/48.0);
    constexpr scalar_t c7_48 = scalar_t(7.0/48.0);
    constexpr scalar_t c1_6 = scalar_t(1.0/6.0);
    constexpr scalar_t c1_24 = scalar_t(1.0/24.0);
    constexpr scalar_t c1_144 = scalar_t(1.0/144.0);
    constexpr scalar_t c1_120 = scalar_t(1.0/120.0);
    constexpr scalar_t c1_720 = scalar_t(1.0/720.0);
    constexpr scalar_t c1_5040 = scalar_t(1.0/5040.0);
    constexpr scalar_t c1_45360 = scalar_t(1.0/45360.0);

    scalar_t p;
    p = 1 + u * c11_16 + u2 * c3_16 + u2*u * c1_48; 
    damps[0] = expu * p;
    p = 1 + u + u2 * c1_2;
    damps[1] = expu * ( p + u2*u * c7_48 + u4 * c1_48);
    p += u2*u * c1_6 + u4 * c1_24;
    damps[2] = expu * (p + u4*u * c1_144 );
    p += u4*u * c1_120 + u4*u2 * c1_720;
    damps[3] = expu * p;
    p += u4*u2*u * c1_5040;
    damps[4] = expu * p;
    p += u4*u4 * c1_45360;
    damps[5] = expu * p;
}


template <typename scalar_t>
__device__ void polarization_damps(scalar_t r, scalar_t b, scalar_t* damps) {
    scalar_t u = b * r;
    scalar_t u2 = u * u;
    scalar_t u4 = u2 * u2;
    scalar_t expu = exp_(-u);

    constexpr scalar_t c1_9 = scalar_t(1.0/9.0);
    constexpr scalar_t c1_11 = scalar_t(1.0/11.0);
    constexpr scalar_t c1_13 = scalar_t(1.0/13.0);
    constexpr scalar_t c1_15 = scalar_t(1.0/15.0);
    constexpr scalar_t c2_99 = scalar_t(2.0/99.0);
    constexpr scalar_t c9_143_m = scalar_t(-9.0/143.0);
    constexpr scalar_t c8_65_m = scalar_t(-8.0/65.0);
    constexpr scalar_t c2_297 = scalar_t(2.0/297.0);
    constexpr scalar_t c101_297 = scalar_t(101.0/297.0);
    constexpr scalar_t c43_2145 = scalar_t(43.0/2145.0);
    constexpr scalar_t c10_117_m = scalar_t(-10.0/117.0);
    constexpr scalar_t c1_45 = scalar_t(1.0/45.0);
    constexpr scalar_t c1_192 = scalar_t(1.0/192.0);

    scalar_t p;
    p = 1 + u * c1_9 + u2 * c1_11 + u2*u * c1_13 + u4 * c1_15; 
    damps[0] = expu * p;
    p = 1 + u;
    damps[1] = expu * ( p + u2 * c2_99 + u2*u * c9_143_m + u4 * c8_65_m + u4*u * c1_15);
    damps[2] = expu * ( p + u2 * c101_297 + u2*u * c2_297 + u4 * c43_2145 + u4*u * c10_117_m + u4*u2 * c1_45);
}

#endif