#ifndef TORCHFF_TT_CUH
#define TORCHFF_TT_CUH

#include "common/vec3.cuh"

template <typename scalar_t> __device__ __forceinline__
void tang_tonnies_6_dispersion(
    scalar_t c6, scalar_t b,
    scalar_t dr, scalar_t drx, scalar_t dry, scalar_t drz,
    scalar_t* ene, scalar_t* egrad_x, scalar_t* egrad_y, scalar_t* egrad_z
) 
{
    scalar_t rinv = 1 / dr;
    scalar_t rinv6 = rinv*rinv*rinv*rinv*rinv*rinv;
    scalar_t u = b * dr;
    scalar_t u6 = u * u * u * u * u * u;
    scalar_t expu = exp_(-u);
    scalar_t f6 = (1 - exp_(-u) * (1 + u * ( 1 + u/2 * (1 + u/3 * (1 + u/4 * (1 + u/5 * (1 + u/6)))))));
    scalar_t egrad = c6 * (6 * f6 - expu * u6 * u / 720) * rinv6 * rinv * rinv;
    *ene = -f6*c6*rinv6;
    *egrad_x = egrad * drx;
    *egrad_y = egrad * dry;
    *egrad_z = egrad * drz;
}


#endif
