#ifndef TORCHFF_SWITCH_CUH
#define TORCHFF_SWITCH_CUH

#include "vec3.cuh"

#define SWITCH(x) ( 1 - (x)*(x)*(x) * (10 - (x) * (15 - 6 * (x))) )
#define SWITCH_GRAD(x) ( -30 * (x) * (x) * (x - 1) * (x - 1) )


#endif