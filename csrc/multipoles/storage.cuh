#ifndef MULTIPOLES_STORAGE_CUH
#define MULTIPOLES_STORAGE_CUH

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

#endif