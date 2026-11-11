#pragma once
// Compile-time selection retains inlining inside the convection/RK group.
#if FR_NS_BR1
#include "br1_kernel.hpp"
namespace fr_viscous = fr_br1;
#else
#include "ldg_kernel.hpp"
namespace fr_viscous = fr_ldg;
#endif
