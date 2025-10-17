#ifndef TORCHFF_MORSE_BOND_CUH
#define TORCHFF_MORSE_BOND_CUH

#include "common/vec3.cuh"

template <typename scalar_t>
__device__ void morse_bond_from_kb_d(
    scalar_t r, scalar_t rx, scalar_t ry, scalar_t rz, scalar_t req, scalar_t kb, scalar_t d,
    scalar_t* ene, scalar_t* drx, scalar_t* dry, scalar_t* drz
) 
{
    scalar_t beta = sqrt_(kb / 2 / d);
    scalar_t tmp = (1 - exp_(-beta * (r - req)));
    *ene = d * tmp * tmp;
    tmp = 2 * d * beta * tmp * (1 - tmp);
    *drx = rx * tmp / r;
    *dry = ry * tmp / r;
    *drz = rz * tmp / r;
}


template <typename scalar_t>
__device__ void morse_bond_from_kb_d(
    scalar_t r, scalar_t rx, scalar_t ry, scalar_t rz, scalar_t req, scalar_t kb, scalar_t d,
    scalar_t* ene, scalar_t* drx, scalar_t* dry, scalar_t* drz, scalar_t* dreq, scalar_t* dkb, scalar_t* dd
) 
{
    scalar_t beta = sqrt_(kb / 2 / d);
    scalar_t tmp = (1 - exp_(-beta * (r - req)));
    *ene = d * tmp * tmp;
    *dd = tmp * tmp;
    tmp = 2 * d * beta * tmp * (1 - tmp);
    *dreq = -tmp;
    *dkb = tmp / beta * (r - req) / (2 * sqrt_(2 * kb * d));
    *drx = rx * tmp / r;
    *dry = ry * tmp / r;
    *drz = rz * tmp / r;
}


template <typename scalar_t>
__device__ __forceinline__ void fd_morse_bond_from_kb_d(
    scalar_t r, scalar_t rx, scalar_t ry, scalar_t rz, scalar_t req_0, scalar_t kb_0, scalar_t d,
    scalar_t efx, scalar_t efy, scalar_t efz, scalar_t dipole_1, scalar_t dipole_2,
    scalar_t* ene, scalar_t* drx, scalar_t* dry, scalar_t* drz,
    scalar_t* defx, scalar_t* defy, scalar_t* defz
) 
{
    scalar_t beta_0 = sqrt_(kb_0 / d / 2);
    scalar_t efp = (efx*rx + efy*ry + efz*rz) / r;
    scalar_t d_req = efp * dipole_1 / (kb_0 - efp * dipole_2);
    scalar_t d_kb = -3 * kb_0 * beta_0 * d_req - efp * dipole_2;

    scalar_t defp_drx = efx / r - efp * rx / (r * r);
    scalar_t defp_dry = efy / r - efp * ry / (r * r);
    scalar_t defp_drz = efz / r - efp * rz / (r * r);

    scalar_t req = req_0 + d_req;
    scalar_t kb = max_(kb_0 + d_kb, scalar_t(0.4)*kb_0);
    scalar_t beta = sqrt_(kb / 2 / d);
    scalar_t one_minus_exp = 1 - exp_(-beta * (r - req));
    *ene = d * one_minus_exp * one_minus_exp;

    scalar_t du_dr = 2 * d * beta * one_minus_exp * (1 - one_minus_exp);
    scalar_t dreq_defp = dipole_1 * (1 / (kb_0 - efp * dipole_2) + (efp * dipole_2) / (kb_0 - efp * dipole_2) / (kb_0 - efp * dipole_2));
    scalar_t du_dkb = 2 * d * one_minus_exp * (1-one_minus_exp) * (r - req) / (2 * sqrt_(2 * kb * d));
    scalar_t dkb_defp = -3 * kb_0 * beta_0 * dreq_defp - dipole_2;

    *drx = du_dr * rx / r - du_dr * dreq_defp * defp_drx + du_dkb * dkb_defp * defp_drx;
    *dry = du_dr * ry / r - du_dr * dreq_defp * defp_dry + du_dkb * dkb_defp * defp_dry;
    *drz = du_dr * rz / r - du_dr * dreq_defp * defp_drz + du_dkb * dkb_defp * defp_drz;

    *defx = (du_dkb * dkb_defp - du_dr * dreq_defp) * rx / r;
    *defy = (du_dkb * dkb_defp - du_dr * dreq_defp) * ry / r;
    *defz = (du_dkb * dkb_defp - du_dr * dreq_defp) * rz / r;
}

#endif