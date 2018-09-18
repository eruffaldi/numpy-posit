#ifndef __NPY_P16POSIT_H__
#define __NPY_P16POSIT_H__

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
float npy_posit16_to_float(npy_posit16 p);
double npy_posit16_to_double(npy_posit16 p);
npy_posit16 npy_float_to_posit16(float f);
npy_posit16 npy_double_to_posit16(double d);
/* Comparisons */
int npy_posit16_eq(npy_posit16 p1, npy_posit16 p2);
int npy_posit16_ne(npy_posit16 p1, npy_posit16 p2);
int npy_posit16_le(npy_posit16 p1, npy_posit16 p2);
int npy_posit16_lt(npy_posit16 p1, npy_posit16 p2);
int npy_posit16_ge(npy_posit16 p1, npy_posit16 p2);
int npy_posit16_gt(npy_posit16 p1, npy_posit16 p2);
/* Miscellaneous functions */
int npy_posit16_iszero(npy_posit16 p);
int npy_posit16_isnan(npy_posit16 p);
int npy_posit16_isinf(npy_posit16 p);
int npy_posit16_isfinite(npy_posit16 p);
int npy_posit16_signbit(npy_posit16 p);
npy_posit16 npy_posit16_copysign(npy_posit16 x, npy_posit16 y);
npy_posit16 npy_posit16_spacing(npy_posit16 p);
npy_posit16 npy_posit16_nextafter(npy_posit16 x, npy_posit16 y);
npy_posit16 npy_posit16_divmod(npy_posit16 x, npy_posit16 y, npy_posit16 *modulus);

/*
 * Half-precision constants
 */

#define NPY_POSIT16_ZERO   (0x0000u)
#define NPY_POSIT16_ONE    (0x4000u)
#define NPY_POSIT16_NEGONE (0xC000u)
#define NPY_POSIT16_NAR    (0x8000u)

#define NPY_MAX_POSIT16    (0x7FFFu)

/*
 * Bit-level conversions
 */

npy_uint16 npy_halfbits_to_posit16bits(npy_uint16 h);
npy_uint16 npy_floatbits_to_posit16bits(npy_uint32 f);
npy_uint16 npy_doublebits_to_posit16bits(npy_uint64 d);
npy_uint16 npy_posit16bits_to_halfbits(npy_uint16 p);
npy_uint32 npy_posit16bits_to_floatbits(npy_uint16 p);
npy_uint64 npy_posit16bits_to_doublebits(npy_uint16 p);
npy_uint8 npy_posit16bits_to_posit8bits(npy_uint16 p);
npy_uint32 npy_posit16bits_to_posit32bits(npy_uint16 p);

#ifdef __cplusplus
}
#endif

#endif
