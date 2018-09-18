#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "softposit.h"
#include "numpy/halffloat.h"
#include "numpy/p8posit.h"


float npy_posit8_to_float(npy_posit8 p)
{
    posit8_t u = { .v=p };
    float f = (float)convertP8ToDouble(u);
    return f;
}

double npy_posit8_to_double(npy_posit8 p)
{
    posit8_t u = { .v=p };
    return convertP8ToDouble(u);
}

npy_posit8 npy_float_to_posit8(float f)
{
    posit8_t u = convertDoubleToP8((double)f);
    return u.v;
}

npy_posit8 npy_double_to_posit8(double d)
{
    posit8_t u = convertDoubleToP8(d);
    return u.v;
}

int npy_posit8_iszero(npy_posit8 p)
{
    return p == 0u;
}

int npy_posit8_isnan(npy_posit8 p)
{
    return p == 0x80u;
}

int npy_posit8_isinf(npy_posit8 p)
{
    return npy_posit8_isnan(p);
}

int npy_posit8_isfinite(npy_posit8 p)
{
    return !npy_posit8_isnan(p);
}

int npy_posit8_signbit(npy_posit8 p)
{
    return (p&0x80u) != 0;
}

npy_posit8 npy_posit8_spacing(npy_posit8 p)
{
    // FIXME(xman)
    npy_posit8 ret = 1u;
    return ret;
}

npy_posit8 npy_posit8_copysign(npy_posit8 x, npy_posit8 y)
{
    return ((x&0x80u) == (y&0x80u)) ? x : -x;
}

npy_posit8 npy_posit8_nextafter(npy_posit8 x, npy_posit8 y)
{
    // FIXME(xman)
    npy_posit8 ret = 1u;
    return ret;
}

int npy_posit8_eq_nonan(npy_posit8 p1, npy_posit8 p2)
{
    return p1 == p2;
}

int npy_posit8_eq(npy_posit8 p1, npy_posit8 p2)
{
    return p1 == p2;
}

int npy_posit8_ne(npy_posit8 p1, npy_posit8 p2)
{
    return !npy_posit8_eq(p1, p2);
}

int npy_posit8_lt_nonan(npy_posit8 p1, npy_posit8 p2)
{
    const int8_t s1 = *((int8_t*)&p1);
    const int8_t s2 = *((int8_t*)&p2);
    return s1 < s2;
}

int npy_posit8_lt(npy_posit8 p1, npy_posit8 p2)
{
    return npy_posit8_lt_nonan(p1, p2);
}

int npy_posit8_gt(npy_posit8 p1, npy_posit8 p2)
{
    return npy_posit8_lt(p2, p1);
}

int npy_posit8_le_nonan(npy_posit8 p1, npy_posit8 p2)
{
    const int8_t s1 = *((int8_t*)&p1);
    const int8_t s2 = *((int8_t*)&p2);
    return s1 <= s2;
}

int npy_posit8_le(npy_posit8 p1, npy_posit8 p2)
{
    return npy_posit8_le_nonan(p1, p2);
}

int npy_posit8_ge(npy_posit8 p1, npy_posit8 p2)
{
    return npy_posit8_le(p2, p1);
}

npy_posit8 npy_posit8_divmod(npy_posit8 h1, npy_posit8 h2, npy_posit8 *modulus)
{
    float fh1 = npy_posit8_to_float(h1);
    float fh2 = npy_posit8_to_float(h2);
    float div, mod;

    div = npy_divmodf(fh1, fh2, &mod);
    *modulus = npy_float_to_posit8(mod);
    return npy_float_to_posit8(div);
}

npy_uint8 npy_halfbits_to_posit8bits(npy_uint16 h)
{
    npy_uint64 d = npy_halfbits_to_doublebits(h);
    union { double ret; npy_uint64 retbits; } conv = { .retbits=d };
    posit8_t u = convertDoubleToP8(conv.ret);
    return u.v;
}

npy_uint8 npy_floatbits_to_posit8bits(npy_uint32 f)
{
    union { float ret; npy_uint32 retbits; } conv = { .retbits=f };
    posit8_t u = convertDoubleToP8((double)conv.ret);
    return u.v;
}

npy_uint8 npy_doublebits_to_posit8bits(npy_uint64 d)
{
    union { double ret; npy_uint64 retbits; } conv = { .retbits=d };
    posit8_t u = convertDoubleToP8(conv.ret);
    return u.v;
}

npy_uint16 npy_posit8bits_to_halfbits(npy_uint8 p)
{
    posit8_t u = { .v=p };
    union { double ret; npy_uint64 retbits; } conv;
    conv.ret = convertP8ToDouble(u);
    return npy_doublebits_to_halfbits(conv.retbits);
}

npy_uint32 npy_posit8bits_to_floatbits(npy_uint8 p)
{
    posit8_t u = { .v=p };
    union { float ret; npy_uint32 retbits; } conv;
    conv.ret = (float)convertP8ToDouble(u);
    return conv.retbits;
}

npy_uint64 npy_posit8bits_to_doublebits(npy_uint8 p)
{
    posit8_t u = { .v=p };
    union { double ret; npy_uint64 retbits; } conv;
    conv.ret = convertP8ToDouble(u);
    return conv.retbits;
}

npy_uint16 npy_posit8bits_to_posit16bits(npy_uint8 p)
{
    posit8_t u = { .v=p };
    posit16_t r = p8_to_p16(u);
    return r.v;
}

npy_uint32 npy_posit8bits_to_posit32bits(npy_uint8 p)
{
    posit8_t u = { .v=p };
    posit32_t r = p8_to_p32(u);
    return r.v;
}
