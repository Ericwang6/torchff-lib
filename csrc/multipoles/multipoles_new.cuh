#ifndef TORCHFF_MULTIPOLES_CUH
#define TORCHFF_MULTIPOLES_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"
#include "storage.cuh"


template <typename scalar_t, int RANK=0, bool USE_DAMPS=true>
__device__ __forceinline__ void pairwise_multipole_kernel_with_grad(
    MultipoleAccumWithGrad<scalar_t, RANK>& mpi,
    MultipoleAccumWithGrad<scalar_t, RANK>& mpj,
    scalar_t drx, scalar_t dry, scalar_t drz, scalar_t dr,
    scalar_t* damps, // 1,3,5,7,9,11
    scalar_t& ene,
    scalar_t& drx_g, scalar_t& dry_g, scalar_t& drz_g
)
{

    scalar_t drinvs[RANK * 2 + 2];
    drinvs[0] = scalar_t(1.0) / dr;
    scalar_t drinv2 = drinvs[0] * drinvs[0];

    #pragma unroll
    for (int i = 1; i < RANK * 2 + 2; i++) {
        drinvs[i] = drinvs[i-1] * drinv2;
    }

    if constexpr (USE_DAMPS) {
        #pragma unroll
        for (int i = 0; i < RANK * 2 + 2; i++) {
            drinvs[i] *= damps[i];
        }
    }

    scalar_t& drinv = drinvs[0];
    scalar_t& drinv3 = drinvs[1];

    scalar_t tx = -drx * drinv3; 
    scalar_t ty = -dry * drinv3;
    scalar_t tz = -drz * drinv3;

    // energy
    scalar_t c0prod = mpi.c0 * mpj.c0;
    ene = c0prod*drinv;

    // charge gradient;
    mpi.ep = drinv * mpj.c0;
    mpj.ep = drinv * mpi.c0;
    drx_g = c0prod * tx;
    dry_g = c0prod * ty;
    drz_g = c0prod * tz;

    if constexpr (RANK >= 1) {
        scalar_t& drinv5 = drinvs[2];
        scalar_t& drinv7 = drinvs[3];

        scalar_t x2 = drx * drx;
        scalar_t xy = drx * dry;
        scalar_t xz = drx * drz;
        scalar_t y2 = dry * dry;
        scalar_t yz = dry * drz;
        scalar_t z2 = drz * drz;
        scalar_t xyz = drx * dry * drz;

        scalar_t txx = 3 * x2 * drinv5 - drinv3;
        scalar_t txy = 3 * xy * drinv5;
        scalar_t txz = 3 * xz * drinv5;
        scalar_t tyy = 3 * y2 * drinv5 - drinv3;
        scalar_t tyz = 3 * yz * drinv5;
        scalar_t tzz = 3 * z2 * drinv5 - drinv3;

        scalar_t txxx = -15 * x2 * drx * drinv7 + 9 * drx * drinv5;
        scalar_t txxy = -15 * x2 * dry * drinv7 + 3 * dry * drinv5;
        scalar_t txxz = -15 * x2 * drz * drinv7 + 3 * drz * drinv5;
        scalar_t tyyy = -15 * y2 * dry * drinv7 + 9 * dry * drinv5;
        scalar_t tyyx = -15 * y2 * drx * drinv7 + 3 * drx * drinv5;
        scalar_t tyyz = -15 * y2 * drz * drinv7 + 3 * drz * drinv5;
        scalar_t tzzz = -15 * z2 * drz * drinv7 + 9 * drz * drinv5;
        scalar_t tzzx = -15 * z2 * drx * drinv7 + 3 * drx * drinv5;
        scalar_t tzzy = -15 * z2 * dry * drinv7 + 3 * dry * drinv5;
        scalar_t txyz = -15 * xyz * drinv7;

        scalar_t tx_g = mpi.c0 * mpj.dx - mpj.c0 * mpi.dx;
        scalar_t ty_g = mpi.c0 * mpj.dy - mpj.c0 * mpi.dy;
        scalar_t tz_g = mpi.c0 * mpj.dz - mpj.c0 * mpi.dz;

        scalar_t txx_g = - mpi.dx * mpj.dx;
        scalar_t txy_g = - mpi.dx * mpj.dy - mpj.dx * mpi.dy;
        scalar_t txz_g = - mpi.dx * mpj.dz - mpj.dx * mpi.dz;
        scalar_t tyy_g = - mpi.dy * mpj.dy;
        scalar_t tyz_g = - mpi.dy * mpj.dz - mpj.dy * mpi.dz;
        scalar_t tzz_g = - mpi.dz * mpj.dz;

        if constexpr (RANK >= 2) {
            txx_g += mpi.c0 * mpj.qxx + mpj.c0 * mpi.qxx;
            txy_g += mpi.c0 * mpj.qxy + mpj.c0 * mpi.qxy;
            txz_g += mpi.c0 * mpj.qxz + mpj.c0 * mpi.qxz;
            tyy_g += mpi.c0 * mpj.qyy + mpj.c0 * mpi.qyy;
            tyz_g += mpi.c0 * mpj.qyz + mpj.c0 * mpi.qyz;
            tzz_g += mpi.c0 * mpj.qzz + mpj.c0 * mpi.qzz;
        }

        // energy (charge-dipole + dipole-dipole / charge-quad)
        ene += tx * tx_g + ty * ty_g + tz * tz_g;
        ene += txx * txx_g + txy * txy_g + txz * txz_g + tyy * tyy_g + tyz * tyz_g + tzz * tzz_g;

        // charge gradient - electric potential
        mpi.ep +=  tx * mpj.dx + ty * mpj.dy + tz * mpj.dz;
        mpj.ep += -tx * mpi.dx - ty * mpi.dy - tz * mpi.dz;

        // dipole gradient - electric field
        mpi.efx = -mpj.c0 * tx - txx * mpj.dx - txy * mpj.dy - txz * mpj.dz;
        mpi.efy = -mpj.c0 * ty - txy * mpj.dx - tyy * mpj.dy - tyz * mpj.dz;
        mpi.efz = -mpj.c0 * tz - txz * mpj.dx - tyz * mpj.dy - tzz * mpj.dz;

        mpj.efx = mpi.c0 * tx - txx * mpi.dx - txy * mpi.dy - txz * mpi.dz;
        mpj.efy = mpi.c0 * ty - txy * mpi.dx - tyy * mpi.dy - tyz * mpi.dz;
        mpj.efz = mpi.c0 * tz - txz * mpi.dx - tyz * mpi.dy - tzz * mpi.dz;

        // coordinate gradient - force
        drx_g += tx_g * txx + ty_g * txy + tz_g * txz + txxx * txx_g + txxy * txy_g + txxz * txz_g + tyyx * tyy_g + txyz * tyz_g + tzzx * tzz_g;
        dry_g += tx_g * txy + ty_g * tyy + tz_g * tyz + txxy * txx_g + tyyx * txy_g + txyz * txz_g + tyyy * tyy_g + tyyz * tyz_g + tzzy * tzz_g;
        drz_g += tx_g * txz + ty_g * tyz + tz_g * tzz + txxz * txx_g + txyz * txy_g + tzzx * txz_g + tyyz * tyy_g + tzzy * tyz_g + tzzz * tzz_g;

        if constexpr (RANK >= 2) {
            scalar_t& drinv9 = drinvs[4];

            scalar_t txxxx = 105 * x2 * x2 * drinv9 - 90 * x2 * drinv7 + 9 * drinv5;
            scalar_t txxxy = 105 * x2 * xy * drinv9 - 45 * xy * drinv7;
            scalar_t txxxz = 105 * x2 * xz * drinv9 - 45 * xz * drinv7;
            scalar_t txxyy = 105 * x2 * y2 * drinv9 - 15 * (x2 + y2) * drinv7 + 3 * drinv5;
            scalar_t txxzz = 105 * x2 * z2 * drinv9 - 15 * (x2 + z2) * drinv7 + 3 * drinv5;
            scalar_t txxyz = 105 * x2 * yz * drinv9 - 15 * yz * drinv7;

            scalar_t tyyyy = 105 * y2 * y2 * drinv9 - 90 * y2 * drinv7 + 9 * drinv5;
            scalar_t tyyyx = 105 * y2 * xy * drinv9 - 45 * xy * drinv7;
            scalar_t tyyyz = 105 * y2 * yz * drinv9 - 45 * yz * drinv7;
            scalar_t tyyzz = 105 * y2 * z2 * drinv9 - 15 * (y2 + z2) * drinv7 + 3 * drinv5;
            scalar_t tyyxz = 105 * y2 * xz * drinv9 - 15 * xz * drinv7;

            scalar_t tzzzz = 105 * z2 * z2 * drinv9 - 90 * z2 * drinv7 + 9 * drinv5;
            scalar_t tzzzx = 105 * z2 * xz * drinv9 - 45 * xz * drinv7;
            scalar_t tzzzy = 105 * z2 * yz * drinv9 - 45 * yz * drinv7;
            scalar_t tzzxy = 105 * z2 * xy * drinv9 - 15 * xy * drinv7;

            scalar_t txxx_g = mpi.qxx * mpj.dx - mpj.qxx * mpi.dx;
            scalar_t txxy_g = mpi.qxx * mpj.dy - mpj.qxx * mpi.dy + mpi.qxy * mpj.dx - mpj.qxy * mpi.dx;
            scalar_t txxz_g = mpi.qxx * mpj.dz - mpj.qxx * mpi.dz + mpi.qxz * mpj.dx - mpj.qxz * mpi.dx;
            scalar_t tyyy_g = mpi.qyy * mpj.dy - mpj.qyy * mpi.dy;
            scalar_t tyyx_g = mpi.qyy * mpj.dx - mpj.qyy * mpi.dx + mpi.qxy * mpj.dy - mpj.qxy * mpi.dy;
            scalar_t tyyz_g = mpi.qyy * mpj.dz - mpj.qyy * mpi.dz + mpi.qyz * mpj.dy - mpj.qyz * mpi.dy;
            scalar_t tzzz_g = mpi.qzz * mpj.dz - mpj.qzz * mpi.dz;
            scalar_t tzzx_g = mpi.qzz * mpj.dx - mpj.qzz * mpi.dx + mpi.qxz * mpj.dz - mpj.qxz * mpi.dz;
            scalar_t tzzy_g = mpi.qzz * mpj.dy - mpj.qzz * mpi.dy + mpi.qyz * mpj.dz - mpj.qyz * mpi.dz;
            scalar_t txyz_g = mpi.qxy * mpj.dz - mpj.qxy * mpi.dz + mpi.qxz * mpj.dy - mpj.qxz * mpi.dy + mpi.qyz * mpj.dx - mpj.qyz * mpi.dx;

            scalar_t txxxx_g = mpi.qxx * mpj.qxx;
            scalar_t txxxy_g = mpi.qxx * mpj.qxy + mpi.qxy * mpj.qxx;
            scalar_t txxxz_g = mpi.qxx * mpj.qxz + mpi.qxz * mpj.qxx;
            scalar_t txxyy_g = mpi.qxx * mpj.qyy + mpi.qyy * mpj.qxx + mpi.qxy * mpj.qxy;
            scalar_t txxzz_g = mpi.qxx * mpj.qzz + mpi.qzz * mpj.qxx + mpi.qxz * mpj.qxz;
            scalar_t txxyz_g = mpi.qxx * mpj.qyz + mpi.qyz * mpj.qxx + mpi.qxy * mpj.qxz + mpi.qxz * mpj.qxy;

            scalar_t tyyyy_g = mpi.qyy * mpj.qyy;
            scalar_t tyyyx_g = mpi.qyy * mpj.qxy + mpi.qxy * mpj.qyy;
            scalar_t tyyyz_g = mpi.qyy * mpj.qyz + mpi.qyz * mpj.qyy;
            scalar_t tyyzz_g = mpi.qyy * mpj.qzz + mpi.qzz * mpj.qyy + mpi.qyz * mpj.qyz;
            scalar_t tyyxz_g = mpi.qyy * mpj.qxz + mpi.qxz * mpj.qyy + mpi.qxy * mpj.qyz + mpi.qyz * mpj.qxy;

            scalar_t tzzzz_g = mpi.qzz * mpj.qzz;
            scalar_t tzzzx_g = mpi.qzz * mpj.qxz + mpi.qxz * mpj.qzz;
            scalar_t tzzzy_g = mpi.qzz * mpj.qyz + mpi.qyz * mpj.qzz;
            scalar_t tzzxy_g = mpi.qzz * mpj.qxy + mpi.qxy * mpj.qzz + mpi.qxz * mpj.qyz + mpi.qyz * mpj.qxz;

            // energy
            ene += txxx * txxx_g + txxy * txxy_g + txxz * txxz_g + tyyy * tyyy_g + tyyx * tyyx_g + tyyz * tyyz_g + tzzz * tzzz_g + tzzx * tzzx_g + tzzy * tzzy_g + txyz * txyz_g
                    + txxxx * txxxx_g
                    + txxxy * txxxy_g
                    + txxxz * txxxz_g
                    + txxyy * txxyy_g
                    + txxzz * txxzz_g
                    + txxyz * txxyz_g
                    + tyyyy * tyyyy_g
                    + tyyyx * tyyyx_g
                    + tyyyz * tyyyz_g
                    + tyyzz * tyyzz_g
                    + tyyxz * tyyxz_g
                    + tzzzz * tzzzz_g
                    + tzzzx * tzzzx_g
                    + tzzzy * tzzzy_g
                    + tzzxy * tzzxy_g;

            // charge gradient - electric potential
            mpi.ep += txx * mpj.qxx + txy * mpj.qxy + txz * mpj.qxz + tyy * mpj.qyy + tyz * mpj.qyz + tzz * mpj.qzz;
            mpj.ep += txx * mpi.qxx + txy * mpi.qxy + txz * mpi.qxz + tyy * mpi.qyy + tyz * mpi.qyz + tzz * mpi.qzz;

            // dipole gradient - electric field
            mpi.efx += - txxx * mpj.qxx - txxy * mpj.qxy - txxz * mpj.qxz - tyyx * mpj.qyy - tzzx * mpj.qzz - txyz * mpj.qyz;
            mpi.efy += - txxy * mpj.qxx - tyyx * mpj.qxy - txyz * mpj.qxz - tyyy * mpj.qyy - tzzy * mpj.qzz - tyyz * mpj.qyz;
            mpi.efz += - txxz * mpj.qxx - txyz * mpj.qxy - tzzx * mpj.qxz - tyyz * mpj.qyy - tzzz * mpj.qzz - tzzy * mpj.qyz;

            mpj.efx += txxx * mpi.qxx + txxy * mpi.qxy + txxz * mpi.qxz + tyyx * mpi.qyy + tzzx * mpi.qzz + txyz * mpi.qyz;
            mpj.efy += txxy * mpi.qxx + tyyx * mpi.qxy + txyz * mpi.qxz + tyyy * mpi.qyy + tzzy * mpi.qzz + tyyz * mpi.qyz;
            mpj.efz += txxz * mpi.qxx + txyz * mpi.qxy + tzzx * mpi.qxz + tyyz * mpi.qyy + tzzz * mpi.qzz + tzzy * mpi.qyz;

            // quadrupole gradient - electric field graident
            mpi.egxx = mpj.c0 * txx + txxx * mpj.dx + txxy * mpj.dy + txxz * mpj.dz + txxxx * mpj.qxx + txxxy * mpj.qxy + txxxz * mpj.qxz + txxyy * mpj.qyy + txxyz * mpj.qyz + txxzz * mpj.qzz;
            mpi.egxy = mpj.c0 * txy + txxy * mpj.dx + tyyx * mpj.dy + txyz * mpj.dz + txxxy * mpj.qxx + txxyy * mpj.qxy + txxyz * mpj.qxz + tyyyx * mpj.qyy + tyyxz * mpj.qyz + tzzxy * mpj.qzz;
            mpi.egxz = mpj.c0 * txz + txxz * mpj.dx + txyz * mpj.dy + tzzx * mpj.dz + txxxz * mpj.qxx + txxyz * mpj.qxy + txxzz * mpj.qxz + tyyxz * mpj.qyy + tzzxy * mpj.qyz + tzzzx * mpj.qzz;
            mpi.egyy = mpj.c0 * tyy + tyyx * mpj.dx + tyyy * mpj.dy + tyyz * mpj.dz + txxyy * mpj.qxx + tyyyx * mpj.qxy + tyyxz * mpj.qxz + tyyyy * mpj.qyy + tyyyz * mpj.qyz + tyyzz * mpj.qzz;
            mpi.egyz = mpj.c0 * tyz + txyz * mpj.dx + tyyz * mpj.dy + tzzy * mpj.dz + txxyz * mpj.qxx + tyyxz * mpj.qxy + tzzxy * mpj.qxz + tyyyz * mpj.qyy + tyyzz * mpj.qyz + tzzzy * mpj.qzz;
            mpi.egzz = mpj.c0 * tzz + tzzx * mpj.dx + tzzy * mpj.dy + tzzz * mpj.dz + txxzz * mpj.qxx + tzzxy * mpj.qxy + tzzzx * mpj.qxz + tyyzz * mpj.qyy + tzzzy * mpj.qyz + tzzzz * mpj.qzz;

            mpj.egxx = mpi.c0 * txx - txxx * mpi.dx - txxy * mpi.dy - txxz * mpi.dz + txxxx * mpi.qxx + txxxy * mpi.qxy + txxxz * mpi.qxz + txxyy * mpi.qyy + txxyz * mpi.qyz + txxzz * mpi.qzz;
            mpj.egxy = mpi.c0 * txy - txxy * mpi.dx - tyyx * mpi.dy - txyz * mpi.dz + txxxy * mpi.qxx + txxyy * mpi.qxy + txxyz * mpi.qxz + tyyyx * mpi.qyy + tyyxz * mpi.qyz + tzzxy * mpi.qzz;
            mpj.egxz = mpi.c0 * txz - txxz * mpi.dx - txyz * mpi.dy - tzzx * mpi.dz + txxxz * mpi.qxx + txxyz * mpi.qxy + txxzz * mpi.qxz + tyyxz * mpi.qyy + tzzxy * mpi.qyz + tzzzx * mpi.qzz;
            mpj.egyy = mpi.c0 * tyy - tyyx * mpi.dx - tyyy * mpi.dy - tyyz * mpi.dz + txxyy * mpi.qxx + tyyyx * mpi.qxy + tyyxz * mpi.qxz + tyyyy * mpi.qyy + tyyyz * mpi.qyz + tyyzz * mpi.qzz;
            mpj.egyz = mpi.c0 * tyz - txyz * mpi.dx - tyyz * mpi.dy - tzzy * mpi.dz + txxyz * mpi.qxx + tyyxz * mpi.qxy + tzzxy * mpi.qxz + tyyyz * mpi.qyy + tyyzz * mpi.qyz + tzzzy * mpi.qzz;
            mpj.egzz = mpi.c0 * tzz - tzzx * mpi.dx - tzzy * mpi.dy - tzzz * mpi.dz + txxzz * mpi.qxx + tzzxy * mpi.qxy + tzzzx * mpi.qxz + tyyzz * mpi.qyy + tzzzy * mpi.qyz + tzzzz * mpi.qzz;

            // dr gradient - forces
            scalar_t& drinv11 = drinvs[5];

            scalar_t c945dr11 = -945 * drinv11;
            scalar_t c105dr9 = 105 * drinv9;
            scalar_t c15dr7 = 15 * drinv7;

            scalar_t t5x = c945dr11 * x2 * x2 * drx + 10 * c105dr9 * x2 * drx - 15 * c15dr7 * drx;
            scalar_t t5y = c945dr11 * y2 * y2 * dry + 10 * c105dr9 * y2 * dry - 15 * c15dr7 * dry;
            scalar_t t5z = c945dr11 * z2 * z2 * drz + 10 * c105dr9 * z2 * drz - 15 * c15dr7 * drz;
            scalar_t t4x1y = c945dr11 * x2 * x2 * dry + 6 * c105dr9 * x2 * dry - 3 * c15dr7 * dry;
            scalar_t t4x1z = c945dr11 * x2 * x2 * drz + 6 * c105dr9 * x2 * drz - 3 * c15dr7 * drz;
            scalar_t t4y1x = c945dr11 * y2 * y2 * drx + 6 * c105dr9 * y2 * drx - 3 * c15dr7 * drx;
            scalar_t t4z1x = c945dr11 * z2 * z2 * drx + 6 * c105dr9 * z2 * drx - 3 * c15dr7 * drx;
            scalar_t t4y1z = c945dr11 * y2 * y2 * drz + 6 * c105dr9 * y2 * drz - 3 * c15dr7 * drz;
            scalar_t t4z1y = c945dr11 * z2 * z2 * dry + 6 * c105dr9 * z2 * dry - 3 * c15dr7 * dry;
            scalar_t t3x1y1z = c945dr11 * x2 * xyz + 3 * c105dr9 * xyz;
            scalar_t t3y1x1z = c945dr11 * y2 * xyz + 3 * c105dr9 * xyz;
            scalar_t t3z1x1y = c945dr11 * z2 * xyz + 3 * c105dr9 * xyz;
            scalar_t t3x2y = c945dr11 * x2 * drx * y2 + c105dr9 * drx * (3 * y2 + x2) - 3 * c15dr7 * drx;
            scalar_t t3y2x = c945dr11 * y2 * dry * x2 + c105dr9 * dry * (3 * x2 + y2) - 3 * c15dr7 * dry;
            scalar_t t3x2z = c945dr11 * x2 * drx * z2 + c105dr9 * drx * (3 * z2 + x2) - 3 * c15dr7 * drx;
            scalar_t t3z2x = c945dr11 * z2 * drz * x2 + c105dr9 * drz * (3 * x2 + z2) - 3 * c15dr7 * drz;
            scalar_t t3y2z = c945dr11 * y2 * dry * z2 + c105dr9 * dry * (3 * z2 + y2) - 3 * c15dr7 * dry;
            scalar_t t3z2y = c945dr11 * z2 * drz * y2 + c105dr9 * drz * (3 * y2 + z2) - 3 * c15dr7 * drz;
            scalar_t t2x2y1z = c945dr11 * xy * xyz + c105dr9 * drz * (x2 + y2) - c15dr7 * drz;
            scalar_t t2x2z1y = c945dr11 * xz * xyz + c105dr9 * dry * (x2 + z2) - c15dr7 * dry;
            scalar_t t2y2z1x = c945dr11 * yz * xyz + c105dr9 * drx * (y2 + z2) - c15dr7 * drx;

            drx_g += txxxx * txxx_g + txxxy * txxy_g + txxxz * txxz_g + tyyyx * tyyy_g + txxyy * tyyx_g + tyyxz * tyyz_g + tzzzx * tzzz_g + txxzz * tzzx_g + tzzxy * tzzy_g + txxyz * txyz_g
                    + t5x * txxxx_g
                    + t4x1y * txxxy_g
                    + t4x1z * txxxz_g
                    + t3x2y * txxyy_g
                    + t3x2z * txxzz_g
                    + t3x1y1z * txxyz_g
                    + t4y1x * tyyyy_g
                    + t3y2x * tyyyx_g
                    + t3y1x1z * tyyyz_g
                    + t2y2z1x * tyyzz_g
                    + t2x2y1z * tyyxz_g
                    + t4z1x * tzzzz_g
                    + t3z2x * tzzzx_g
                    + t3z1x1y * tzzzy_g
                    + t2x2z1y * tzzxy_g;

            dry_g += txxxy * txxx_g + txxyy * txxy_g + txxyz * txxz_g + tyyyy * tyyy_g + tyyyx * tyyx_g + tyyyz * tyyz_g + tzzzy * tzzz_g + tzzxy * tzzx_g + tyyzz * tzzy_g + tyyxz * txyz_g
                    + t4x1y * txxxx_g
                    + t3x2y * txxxy_g
                    + t3x1y1z * txxxz_g
                    + t3y2x * txxyy_g
                    + t2x2z1y * txxzz_g
                    + t2x2y1z * txxyz_g
                    + t5y * tyyyy_g
                    + t4y1x * tyyyx_g
                    + t4y1z * tyyyz_g
                    + t3y2z * tyyzz_g
                    + t3y1x1z * tyyxz_g
                    + t4z1y * tzzzz_g
                    + t3z1x1y * tzzzx_g
                    + t3z2y * tzzzy_g
                    + t2y2z1x * tzzxy_g;

            drz_g += txxxz * txxx_g + txxyz * txxy_g + txxzz * txxz_g + tyyyz * tyyy_g + tyyxz * tyyx_g + tyyzz * tyyz_g + tzzzz * tzzz_g + tzzzx * tzzx_g + tzzzy * tzzy_g + tzzxy * txyz_g
                    + t4x1z * txxxx_g
                    + t3x1y1z * txxxy_g
                    + t3x2z * txxxz_g
                    + t2x2y1z * txxyy_g
                    + t3z2x * txxzz_g
                    + t2x2z1y * txxyz_g
                    + t4y1z * tyyyy_g
                    + t3y1x1z * tyyyx_g
                    + t3y2z * tyyyz_g
                    + t3z2y * tyyzz_g
                    + t2y2z1x * tyyxz_g
                    + t5z * tzzzz_g
                    + t4z1x * tzzzx_g
                    + t4z1y * tzzzy_g
                    + t3z1x1y * tzzxy_g;
        }  // RANK >= 2

    }  // RANK >= 1
}

#endif