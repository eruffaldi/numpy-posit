#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "softposit.h"
#include "numpy/p32posit.h"


float npy_posit32_to_float(npy_posit32 p)
{
    posit32_t u = { .v=p };
    float f = (float)convertP32ToDouble(u);
    return f;
}

double npy_posit32_to_double(npy_posit32 p)
{
    posit32_t u = { .v=p };
    return convertP32ToDouble(u);
}

npy_posit32 npy_float_to_posit32(float f)
{
    posit32_t u = convertDoubleToP32((double)f);
    return u.v;
}

npy_posit32 npy_double_to_posit32(double d)
{
    posit32_t u = convertDoubleToP32(d);
    return u.v;
}

int npy_posit32_iszero(npy_posit32 p)
{
    return p == 0u;
}

int npy_posit32_isnan(npy_posit32 p)
{
    return p == 0x80000000u;
}

int npy_posit32_isinf(npy_posit32 p)
{
    return npy_posit32_isnan(p);
}

int npy_posit32_isfinite(npy_posit32 p)
{
    return !npy_posit32_isnan(p);
}

int npy_posit32_signbit(npy_posit32 p)
{
    return (p&0x80000000u) != 0;
}

npy_posit32 npy_posit32_spacing(npy_posit32 p)
{
    // FIXME
    npy_posit32 ret = 1u;
    return ret;
}

npy_posit32 npy_posit32_copysign(npy_posit32 x, npy_posit32 y)
{
    return ((x&0x80000000u) == (y&0x80000000u)) ? x : -x;
}

npy_posit32 npy_posit32_nextafter(npy_posit32 x, npy_posit32 y)
{
    // FIXME
    npy_posit32 ret = 1u;
    return ret;
}

int npy_posit32_eq_nonan(npy_posit32 p1, npy_posit32 p2)
{
    return p1 == p2;
}

int npy_posit32_eq(npy_posit32 p1, npy_posit32 p2)
{
    /*
     * The equality cases are as follows:
     *   - If either value is NaN, never equal.
     *   - If the values are equal, equal.
     *   - If the values are both signed zeros, equal.
     */
    return p1 == p2;
}

int npy_posit32_ne(npy_posit32 p1, npy_posit32 p2)
{
    return !npy_posit32_eq(p1, p2);
}

int npy_posit32_lt_nonan(npy_posit32 p1, npy_posit32 p2)
{
    const int32_t s1 = *((int32_t*)&p1);
    const int32_t s2 = *((int32_t*)&p2);
    return s1 < s2;
}

int npy_posit32_lt(npy_posit32 p1, npy_posit32 p2)
{
    return npy_posit32_lt_nonan(p1, p2);
}

int npy_posit32_gt(npy_posit32 p1, npy_posit32 p2)
{
    return npy_posit32_lt(p2, p1);
}

int npy_posit32_le_nonan(npy_posit32 p1, npy_posit32 p2)
{
    const int32_t s1 = *((int32_t*)&p1);
    const int32_t s2 = *((int32_t*)&p2);
    return s1 <= s2;
}

int npy_posit32_le(npy_posit32 p1, npy_posit32 p2)
{
    return npy_posit32_le_nonan(p1, p2);
}

int npy_posit32_ge(npy_posit32 p1, npy_posit32 p2)
{
    return npy_posit32_le(p2, p1);
}

npy_posit32 npy_posit32_divmod(npy_posit32 h1, npy_posit32 h2, npy_posit32 *modulus)
{
    float fh1 = npy_posit32_to_float(h1);
    float fh2 = npy_posit32_to_float(h2);
    float div, mod;

    div = npy_divmodf(fh1, fh2, &mod);
    *modulus = npy_float_to_posit32(mod);
    return npy_float_to_posit32(div);
}

npy_uint32 npy_halfbits_to_posit32bits(npy_uint16 h)
{
    // FIXME
    return 0;
}

npy_uint32 npy_floatbits_to_posit32bits(npy_uint32 f)
{
    union { float ret; npy_uint32 retbits; } conv = { .retbits=f };
    posit32_t u = convertDoubleToP32((double)conv.ret);
    return u.v;
}

npy_uint32 npy_doublebits_to_posit32bits(npy_uint64 d)
{
    union { double ret; npy_uint64 retbits; } conv = { .retbits=d };
    posit32_t u = convertDoubleToP32(conv.ret);
    return u.v;
}

npy_uint16 npy_posit32bits_to_halfbits(npy_uint32 p)
{
    // FIXME
    return 0;
}

npy_uint32 npy_posit32bits_to_floatbits(npy_uint32 p)
{
    posit32_t u = { .v=p };
    union { float ret; npy_uint32 retbits; } conv;
    conv.ret = (float)convertP32ToDouble(u);
    return conv.retbits;
}

npy_uint64 npy_posit32bits_to_doublebits(npy_uint32 p)
{
    posit32_t u = { .v=p };
    union { double ret; npy_uint64 retbits; } conv;
    conv.ret = convertP32ToDouble(u);
    return conv.retbits;
}
