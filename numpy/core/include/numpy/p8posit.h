#ifndef __NPY_P8POSIT_H__
#define __NPY_P8POSIT_H__

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
float npy_posit8_to_float(npy_posit8 p);
double npy_posit8_to_double(npy_posit8 p);
npy_posit8 npy_float_to_posit8(float f);
npy_posit8 npy_double_to_posit8(double d);
/* Comparisons */
int npy_posit8_eq(npy_posit8 p1, npy_posit8 p2);
int npy_posit8_ne(npy_posit8 p1, npy_posit8 p2);
int npy_posit8_le(npy_posit8 p1, npy_posit8 p2);
int npy_posit8_lt(npy_posit8 p1, npy_posit8 p2);
int npy_posit8_ge(npy_posit8 p1, npy_posit8 p2);
int npy_posit8_gt(npy_posit8 p1, npy_posit8 p2);
/* Miscellaneous functions */
int npy_posit8_iszero(npy_posit8 p);
int npy_posit8_isnan(npy_posit8 p);
int npy_posit8_isinf(npy_posit8 p);
int npy_posit8_isfinite(npy_posit8 p);
int npy_posit8_signbit(npy_posit8 p);
npy_posit8 npy_posit8_copysign(npy_posit8 x, npy_posit8 y);
npy_posit8 npy_posit8_spacing(npy_posit8 p);
npy_posit8 npy_posit8_nextafter(npy_posit8 x, npy_posit8 y);
npy_posit8 npy_posit8_divmod(npy_posit8 x, npy_posit8 y, npy_posit8 *modulus);

/*
 * Half-precision constants
 */

#define NPY_POSIT8_ZERO   (0x00u)
#define NPY_POSIT8_ONE    (0x40u)
#define NPY_POSIT8_NEGONE (0xC0u)
#define NPY_POSIT8_NAR    (0x80u)

#define NPY_MAX_POSIT8    (0x7Fu)

/*
 * Bit-level conversions
 */

npy_uint8 npy_halfbits_to_posit8bits(npy_uint16 h);
npy_uint8 npy_floatbits_to_posit8bits(npy_uint32 f);
npy_uint8 npy_doublebits_to_posit8bits(npy_uint64 d);
npy_uint16 npy_posit8bits_to_halfbits(npy_uint8 p);
npy_uint32 npy_posit8bits_to_floatbits(npy_uint8 p);
npy_uint64 npy_posit8bits_to_doublebits(npy_uint8 p);
npy_uint16 npy_posit8bits_to_posit16bits(npy_uint8 p);
npy_uint32 npy_posit8bits_to_posit32bits(npy_uint8 p);

#ifdef __cplusplus
}
#endif

#endif
