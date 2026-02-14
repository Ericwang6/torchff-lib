#ifndef BSPLINES_CUH
#define BSPLINES_CUH

// ============================================================================
// B-Spline Evaluation (Includes optional 3rd Derivative)
// ============================================================================
template <typename T>
__device__ __forceinline__ void eval_b6_and_derivs(T u, T* val, T* d1, T* d2, T* d3 = nullptr) {
    T u2 = u * u;
    T u3 = u2 * u;
    T u4 = u3 * u;
    T u5 = u4 * u;

    if (u < 1.0) {
        *val = u5 * (T(1.0)/T(120.0));
        *d1  = u4 * (T(1.0)/T(24.0));
        *d2  = u3 * (T(1.0)/T(6.0));
        if (d3) {
            *d3  = u2 * T(0.5);
        }
    }
    else if (u < 2.0) {
        T u_1 = u - T(1.0);
        T u_1_2 = u_1 * u_1;
        T u_1_3 = u_1_2 * u_1;
        T u_1_4 = u_1_3 * u_1;
        T u_1_5 = u_1_4 * u_1;

        *val = u5 * (T(1.0)/T(120.0)) - u_1_5 * (T(1.0)/T(20.0));
        *d1  = u4 * (T(1.0)/T(24.0))  - u_1_4 * T(0.25);
        *d2  = u3 * (T(1.0)/T(6.0))   - u_1_3;
        if (d3) {
            *d3  = u2 * T(0.5)         - T(3.0) * u_1_2;
        }
    }
    else if (u < 3.0) {
        // Range [2, 3)
        T u_1 = u - T(1.0);
        T u_1_2 = u_1 * u_1;
        T u_1_4 = u_1_2 * u_1_2;
        T u_1_5 = u_1_4 * u_1;

        T u_2 = u - T(2.0);
        T u_2_2 = u_2 * u_2;
        T u_2_4 = u_2_2 * u_2_2;
        T u_2_5 = u_2_4 * u_2;

        *val = u5 * (T(1.0)/T(120.0)) + u_2_5 * T(0.125) - u_1_5 * (T(1.0)/T(20.0));
        *d1  = u4 * (T(1.0)/T(24.0))  + T(0.625) * u_2_4 - u_1_4 * T(0.25);
        *d2  = (T(5.0)/T(3.0))*u3 - T(12.0)*u2 + T(27.0)*u - T(19.0);
        if (d3) {
            *d3  = T(5.0)*u2 - T(24.0)*u + T(27.0);
        }
    }
    else if (u < 4.0) {
        // Range [3, 4)
        T u_1 = u - T(1.0);
        T u_1_2 = u_1 * u_1; T u_1_4 = u_1_2 * u_1_2; T u_1_5 = u_1_4 * u_1;

        T u_2 = u - T(2.0);
        T u_2_2 = u_2 * u_2; T u_2_4 = u_2_2 * u_2_2; T u_2_5 = u_2_4 * u_2;

        T u_3 = u - T(3.0);
        T u_3_2 = u_3 * u_3; T u_3_4 = u_3_2 * u_3_2; T u_3_5 = u_3_4 * u_3;

        *val = u5*(T(1.0)/T(120.0)) - u_3_5*(T(1.0)/T(6.0)) + u_2_5*T(0.125) - u_1_5*(T(1.0)/T(20.0));

        *d1 = (-T(5.0)/T(12.0))*u4 + T(6.0)*u3 - T(31.5)*u2 + T(71.0)*u - T(57.75);
        *d2 = (-T(5.0)/T(3.0))*u3 + T(18.0)*u2 - T(63.0)*u + T(71.0);
        if (d3) {
            *d3 = -T(5.0)*u2 + T(36.0)*u - T(63.0);
        }
    }
    else if (u < 5.0) {
        // Range [4, 5)
        *val = u5*(T(1.0)/T(24.0)) - u4 + T(9.5)*u3 - T(44.5)*u2 + T(102.25)*u - T(91.45);
        *d1  = (T(5.0)/T(24.0))*u4 - T(4.0)*u3 + T(28.5)*u2 - T(89.0)*u + T(102.25);
        *d2  = (T(5.0)/T(6.0))*u3 - T(12.0)*u2 + T(57.0)*u - T(89.0);
        if (d3) {
            *d3  = T(2.5)*u2 - T(24.0)*u + T(57.0);
        }
    }
    else if (u < 6.0) {
        // Range [5, 6) 
        *val = -u5*(T(1.0)/T(120.0)) + T(0.25)*u4 - T(3.0)*u3 + T(18.0)*u2 - T(54.0)*u + T(64.8);
        *d1  = -u4*(T(1.0)/T(24.0)) + u3 - T(9.0)*u2 + T(36.0)*u - T(54.0);
        *d2  = -u3*(T(1.0)/T(6.0)) + T(3.0)*u2 - T(18.0)*u + T(36.0);
        if (d3) {
            *d3  = -T(0.5)*u2 + T(6.0)*u - T(18.0);
        }
    }
    else {
        *val = T(0.0); *d1 = T(0.0); *d2 = T(0.0);
        if (d3) {
            *d3 = T(0.0);
        }
    }
}


template <typename T>
__device__ __forceinline__ T get_bspline_coeff_order6(int i) {
    switch(i) {
        case 1: return 1.0 / 120.0;
        case 2: return 26.0 / 120.0;
        case 3: return 66.0 / 120.0; 
        case 4: return 26.0 / 120.0;
        case 5: return 1.0 / 120.0;
        default: return 0.0; 
    }
}

template <typename T>
__device__ __forceinline__ T get_bspline_modulus_device(int k, int K, int order) {
    constexpr T TWOPI = two_pi<T>();
    T sum_val = (T)0.0;
    int half = order / 2; 
    for (int m = -half; m < half; m++) {
        T b_val = (order == 6) ? get_bspline_coeff_order6<T>(m + half) : (T)0.0;
        T arg = (TWOPI * (T)m * (T)k) / (T)K;
        sum_val += b_val * cos(arg);
    }
    return sum_val;
}

#endif // BSPLINES_CUH
