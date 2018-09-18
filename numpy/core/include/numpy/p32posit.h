#ifndef __NPY_P32POSIT_H__
#define __NPY_P32POSIT_H__

#include "softposit.h"
#include <Python.h>
#include <numpy/npy_math.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * posit routines
 */

/* Conversions */
float npy_posit32_to_float(npy_posit32 p);
double npy_posit32_to_double(npy_posit32 p);
npy_posit32 npy_float_to_posit32(float f);
npy_posit32 npy_double_to_posit32(double d);
/* Comparisons */
int npy_posit32_eq(npy_posit32 p1, npy_posit32 p2);
int npy_posit32_ne(npy_posit32 p1, npy_posit32 p2);
int npy_posit32_le(npy_posit32 p1, npy_posit32 p2);
int npy_posit32_lt(npy_posit32 p1, npy_posit32 p2);
int npy_posit32_ge(npy_posit32 p1, npy_posit32 p2);
int npy_posit32_gt(npy_posit32 p1, npy_posit32 p2);
/* Miscellaneous functions */
int npy_posit32_iszero(npy_posit32 p);
int npy_posit32_isnan(npy_posit32 p);
int npy_posit32_isinf(npy_posit32 p);
int npy_posit32_isfinite(npy_posit32 p);
int npy_posit32_signbit(npy_posit32 p);
npy_posit32 npy_posit32_copysign(npy_posit32 x, npy_posit32 y);
npy_posit32 npy_posit32_spacing(npy_posit32 p);
npy_posit32 npy_posit32_nextafter(npy_posit32 x, npy_posit32 y);
npy_posit32 npy_posit32_divmod(npy_posit32 x, npy_posit32 y, npy_posit32 *modulus);

/*
 * Half-precision constants
 */

#define NPY_POSIT32_ZERO   (0x00000000u)
#define NPY_POSIT32_ONE    (0x40000000u)
#define NPY_POSIT32_NEGONE (0xC0000000u)
#define NPY_POSIT32_NAR    (0x80000000u)

#define NPY_MAX_POSIT32    (0x7FFFFFFFu)

/*
 * Bit-level conversions
 */

npy_uint32 npy_halfbits_to_posit32bits(npy_uint16 h);
npy_uint32 npy_floatbits_to_posit32bits(npy_uint32 f);
npy_uint32 npy_doublebits_to_posit32bits(npy_uint64 d);
npy_uint16 npy_posit32bits_to_halfbits(npy_uint32 p);
npy_uint32 npy_posit32bits_to_floatbits(npy_uint32 p);
npy_uint64 npy_posit32bits_to_doublebits(npy_uint32 p);
npy_uint8 npy_posit32bits_to_posit8bits(npy_uint32 p);
npy_uint16 npy_posit32bits_to_posit16bits(npy_uint32 p);

#ifdef __cplusplus
}
#endif

#endif
