#pragma once

template <typename T, int RANK, int NUM_WARPS>
struct Smem;

template <typename T, int NUM_WARPS>
struct Smem<T,0,NUM_WARPS> {
  T ep[NUM_WARPS], fx[NUM_WARPS], fy[NUM_WARPS], fz[NUM_WARPS];
};

template <typename T, int NUM_WARPS>
struct Smem<T,1,NUM_WARPS> : Smem<T,0,NUM_WARPS> {
  T efx[NUM_WARPS], efy[NUM_WARPS], efz[NUM_WARPS];
};

template <typename T, int NUM_WARPS>
struct Smem<T,2,NUM_WARPS> : Smem<T,1,NUM_WARPS> {
  T egxx[NUM_WARPS], egxy[NUM_WARPS], egxz[NUM_WARPS],
    egyy[NUM_WARPS], egyz[NUM_WARPS], egzz[NUM_WARPS];
};


template <typename T, int RANK>
struct MultipoleAccumWithGrad;

template <typename T>
struct MultipoleAccumWithGrad<T,0> {
    T c0;
    T ep;
};

template <typename T>
struct MultipoleAccumWithGrad<T,1> {
    T c0, dx, dy, dz;
    T ep, efx, efy, efz;
};

template <typename T>
struct MultipoleAccumWithGrad<T,2> {
    T c0, dx, dy, dz, qxx, qxy, qxz, qyy, qyz, qzz;
    T ep, efx, efy, efz, egxx, egxy, egxz, egyy, egyz, egzz;
};


template <typename T, int RANK>
struct MultipoleAccum;

template <typename T>
struct MultipoleAccum<T,0> {
    T c0;
};

template <typename T>
struct MultipoleAccum<T,1> {
    T c0, dx, dy, dz;
};

template <typename T>
struct MultipoleAccum<T,2> {
    T c0, dx, dy, dz, qxx, qxy, qxz, qyy, qyz, qzz;
};