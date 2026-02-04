#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>


// PI constant for different types
template <typename T>
__host__ __device__ constexpr T pi() {
    if constexpr (std::is_same_v<T, float>) {
        return 3.14159265358979323846f;
    } else if constexpr (std::is_same_v<T, double>) {
        return 3.14159265358979323846;
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, 
                     "Type T must be float or double");
    }
}

// 2*PI constant for different types
template <typename T>
__host__ __device__ constexpr T two_pi() {
    return T(2) * pi<T>();
}

// PI*PI constant for different types
template <typename T>
__host__ __device__ constexpr T pi_squared() {
    return pi<T>() * pi<T>();
}

// 2*PI/3 constant for different types
template <typename T>
__host__ __device__ constexpr T two_pi_over_3() {
    return two_pi<T>() / T(3);
}

// PI*PI/3 constant for different types
template <typename T>
__host__ __device__ constexpr T pi_squared_over_3() {
    return pi_squared<T>() / T(3);
}


template <typename T>
__host__ __device__ constexpr T inv_root_pi() {
    return T(0.56418958354);
}