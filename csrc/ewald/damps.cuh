#ifndef TORCHFF_EWALD_DAMPS_CUH
#define TORCHFF_EWALD_DAMPS_CUH


template <typename scalar_t, int ORDER>
__device__ __forceinline__ void ewald_erfc_damps(scalar_t r, scalar_t b, scalar_t* damps) {
    if constexpr ( ORDER == 1 ) {
        damps[0] = erfc_(b*r);
        return;
    }
    else {
        constexpr scalar_t c2_sqrt_pi = scalar_t(1.1283791670955126);
        scalar_t u = b * r;
        scalar_t u2 = u * u;
        scalar_t u3 = u2 * u;
        scalar_t erfcu = erfc_(u);
        scalar_t prefac = exp_(-u2) * c2_sqrt_pi;
        damps[0] = erfcu;
        scalar_t p = u;
        damps[1] = erfcu + prefac * p;
        if constexpr ( ORDER >= 5 ) {
            constexpr scalar_t c2_3 = scalar_t(2.0/3.0);
            p += u3 * c2_3;
            damps[2] = erfcu + prefac * p;
        }
        if constexpr ( ORDER >= 7 ) {
            constexpr scalar_t c4_15 = scalar_t(4.0/15.0);
            p += u3*u2 * c4_15;
            damps[3] = erfcu + prefac * p;
        }
        if constexpr ( ORDER >= 9 ) {
            constexpr scalar_t c8_105 = scalar_t(8.0/105.0);
            p += u3*u2*u2 * c8_105;
            damps[4] = erfcu + prefac * p;
        }
        if constexpr ( ORDER >= 11 ) {
            constexpr scalar_t c16_945 = scalar_t(16.0/945.0);
            p += u3*u3*u3 * c16_945;
            damps[5] = erfcu + prefac * p;
        }    
    }
}


#endif