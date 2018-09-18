#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "softposit.h"
#include "numpy/halffloat.h"
#include "numpy/p16posit.h"


float npy_posit16_to_float(npy_posit16 p)
{
    posit16_t u = { .v=p };
    float f = (float)convertP16ToDouble(u);
    return f;
}

double npy_posit16_to_double(npy_posit16 p)
{
    posit16_t u = { .v=p };
    return convertP16ToDouble(u);
}

npy_posit16 npy_float_to_posit16(float f)
{
    posit16_t u = convertDoubleToP16((double)f);
    return u.v;
}

npy_posit16 npy_double_to_posit16(double d)
{
    posit16_t u = convertDoubleToP16(d);
    return u.v;
}

int npy_posit16_iszero(npy_posit16 p)
{
    return p == 0u;
}

int npy_posit16_isnan(npy_posit16 p)
{
    return p == 0x8000u;
}

int npy_posit16_isinf(npy_posit16 p)
{
    return npy_posit16_isnan(p);
}

int npy_posit16_isfinite(npy_posit16 p)
{
    return !npy_posit16_isnan(p);
}

int npy_posit16_signbit(npy_posit16 p)
{
    return (p&0x8000u) != 0;
}

npy_posit16 npy_posit16_spacing(npy_posit16 p)
{
    // FIXME(xman)
    npy_posit16 ret = 1u;
    return ret;
}

npy_posit16 npy_posit16_copysign(npy_posit16 x, npy_posit16 y)
{
    return ((x&0x8000u) == (y&0x8000u)) ? x : -x;
}

npy_posit16 npy_posit16_nextafter(npy_posit16 x, npy_posit16 y)
{
    // FIXME(xman)
    npy_posit16 ret = 1u;
    return ret;
}

int npy_posit16_eq(npy_posit16 p1, npy_posit16 p2)
{
    return p1 == p2;
}

int npy_posit16_ne(npy_posit16 p1, npy_posit16 p2)
{
    return !npy_posit16_eq(p1, p2);
}

int npy_posit16_lt(npy_posit16 p1, npy_posit16 p2)
{
    const int16_t s1 = *((int16_t*)&p1);
    const int16_t s2 = *((int16_t*)&p2);
    return s1 < s2;
}

int npy_posit16_gt(npy_posit16 p1, npy_posit16 p2)
{
    return npy_posit16_lt(p2, p1);
}

int npy_posit16_le(npy_posit16 p1, npy_posit16 p2)
{
    const int16_t s1 = *((int16_t*)&p1);
    const int16_t s2 = *((int16_t*)&p2);
    return s1 <= s2;
}

int npy_posit16_ge(npy_posit16 p1, npy_posit16 p2)
{
    return npy_posit16_le(p2, p1);
}

npy_posit16 npy_posit16_divmod(npy_posit16 h1, npy_posit16 h2, npy_posit16 *modulus)
{
    float fh1 = npy_posit16_to_float(h1);
    float fh2 = npy_posit16_to_float(h2);
    float div, mod;

    div = npy_divmodf(fh1, fh2, &mod);
    *modulus = npy_float_to_posit16(mod);
    return npy_float_to_posit16(div);
}

npy_uint16 npy_halfbits_to_posit16bits(npy_uint16 h)
{
    npy_uint64 d = npy_halfbits_to_doublebits(h);
    union { double ret; npy_uint64 retbits; } conv = { .retbits=d };
    posit16_t u = convertDoubleToP16(conv.ret);
    return u.v;
}

npy_uint16 npy_floatbits_to_posit16bits(npy_uint32 f)
{
    union { float ret; npy_uint32 retbits; } conv = { .retbits=f };
    posit16_t u = convertDoubleToP16((double)conv.ret);
    return u.v;
}

npy_uint16 npy_doublebits_to_posit16bits(npy_uint64 d)
{
    union { double ret; npy_uint64 retbits; } conv = { .retbits=d };
    posit16_t u = convertDoubleToP16(conv.ret);
    return u.v;
}

npy_uint16 npy_posit16bits_to_halfbits(npy_uint16 p)
{
    posit16_t u = { .v=p };
    union { double ret; npy_uint64 retbits; } conv;
    conv.ret = convertP16ToDouble(u);
    return npy_doublebits_to_halfbits(conv.retbits);
}

npy_uint32 npy_posit16bits_to_floatbits(npy_uint16 p)
{
    posit16_t u = { .v=p };
    union { float ret; npy_uint32 retbits; } conv;
    conv.ret = (float)convertP16ToDouble(u);
    return conv.retbits;
}

npy_uint64 npy_posit16bits_to_doublebits(npy_uint16 p)
{
    posit16_t u = { .v=p };
    union { double ret; npy_uint64 retbits; } conv;
    conv.ret = convertP16ToDouble(u);
    return conv.retbits;
}

npy_uint8 npy_posit16bits_to_posit8bits(npy_uint16 p)
{
    posit16_t u = { .v=p };
    posit8_t r = p16_to_p8(u);
    return r.v;
}

npy_uint32 npy_posit16bits_to_posit32bits(npy_uint16 p)
{
    posit16_t u = { .v=p };
    posit32_t r = p16_to_p32(u);
    return r.v;
}
