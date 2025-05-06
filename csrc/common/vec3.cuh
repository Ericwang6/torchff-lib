#ifndef TORCHFF_VEC3_CUH
#define TORCHFF_VEC3_CUH

// sqrt
template <typename scalar_t> __device__ __forceinline__ scalar_t sqrt_(scalar_t x) {};
template<> __device__ __forceinline__ float sqrt_(float x) { return ::sqrtf(x); };
template<> __device__ __forceinline__ double sqrt_(double x) { return ::sqrt(x); };

// arccos
template <typename scalar_t> __device__ __forceinline__ scalar_t acos_(scalar_t x) {};
template<> __device__ __forceinline__ float acos_(float x) { return ::acosf(x); };
template<> __device__ __forceinline__ double acos_(double x) { return ::acos(x); };

// cos
template <typename scalar_t> __device__ __forceinline__ scalar_t cos_(scalar_t x) {};
template<> __device__ __forceinline__ float cos_(float x) { return ::cos(x); };
template<> __device__ __forceinline__ double cos_(double x) { return ::cosf(x); };

// sin
template <typename scalar_t> __device__ __forceinline__ scalar_t sin_(scalar_t x) {};
template<> __device__ __forceinline__ float sin_(float x) { return ::sin(x); };
template<> __device__ __forceinline__ double sin_(double x) { return ::sinf(x); };

// clamp
template <typename scalar_t> __device__ __forceinline__ scalar_t clamp_(scalar_t x, scalar_t lo, scalar_t hi) {};
template<> __device__ __forceinline__ float clamp_(float x, float lo, float hi) { return ::fminf( ::fmaxf(x, lo), hi ); };
template<> __device__ __forceinline__ double clamp_(double x, double lo, double hi) { return ::fmin( ::fmax(x, lo), hi ); };


template <typename scalar_t>
__device__ __forceinline__ void cross_vec3(scalar_t* a, scalar_t* b, scalar_t* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

template <typename scalar_t>
__device__ __forceinline__ void diff_vec3(scalar_t* a, scalar_t* b, scalar_t* out) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t dot_vec3(scalar_t* a, scalar_t* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t norm_vec3(scalar_t* a) {
    return sqrt_(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

template <typename scalar_t>
__device__ __forceinline__ void scalar_mult_vec3(scalar_t* vec, scalar_t s, scalar_t* out) {
    out[0] = vec[0] * s;
    out[1] = vec[1] * s;
    out[2] = vec[2] * s;
}

template <typename scalar_t>
__device__ __forceinline__ void add_vec3(scalar_t* a, scalar_t* b, scalar_t* out) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
}

#endif
