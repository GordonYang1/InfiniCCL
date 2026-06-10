#ifndef INFINI_CCL_OMPI_HALF_PRECISION_H_
#define INFINI_CCL_OMPI_HALF_PRECISION_H_

#include <cstdint>
#include <cstring>

namespace infini::ccl {

// Host-side bit conversions between IEEE-754 binary32 and the two 16-bit
// floating-point formats (`kFloat16` / `kBFloat16`). They allow reduction
// operations to stage half-precision payloads through `float` buffers, since
// MPI has no predefined reduction support for 16-bit floats.

inline float Fp16BitsToFloat(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1Fu;
  uint32_t mantissa = h & 0x3FFu;
  uint32_t bits;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;  // signed zero
    } else {
      // Subnormal half: renormalize into the float exponent range.
      int shift = 0;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x3FFu;
      bits = sign | ((127u - 15u - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1Fu) {
    bits = sign | 0x7F800000u | (mantissa << 13);  // inf / nan
  } else {
    bits = sign | ((exponent - 15u + 127u) << 23) | (mantissa << 13);
  }

  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline uint16_t FloatToFp16Bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));

  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  const uint32_t exponent = (bits >> 23) & 0xFFu;
  uint32_t mantissa = bits & 0x7FFFFFu;

  if (exponent == 0xFFu) {  // inf / nan
    return static_cast<uint16_t>(sign | 0x7C00u | (mantissa ? 0x200u : 0u));
  }

  const int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
  if (half_exponent >= 0x1F) {
    return static_cast<uint16_t>(sign | 0x7C00u);  // overflow to inf
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return sign;  // underflow to signed zero
    }
    // Subnormal half: shift mantissa (with the implicit bit) into place,
    // rounding to nearest-even.
    mantissa |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
    uint16_t half = static_cast<uint16_t>(mantissa >> shift);
    const uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (half & 1u))) {
      ++half;
    }
    return static_cast<uint16_t>(sign | half);
  }

  // Normal half: drop 13 mantissa bits, rounding to nearest-even. A mantissa
  // carry may bump the exponent (and saturate to inf), which is correct.
  uint16_t half =
      static_cast<uint16_t>((half_exponent << 10) | (mantissa >> 13));
  const uint32_t remainder = mantissa & 0x1FFFu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
    ++half;
  }
  return static_cast<uint16_t>(sign | half);
}

inline float Bf16BitsToFloat(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline uint16_t FloatToBf16Bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));

  if (((bits >> 23) & 0xFFu) == 0xFFu && (bits & 0x7FFFFFu)) {
    // Quiet the NaN so truncation cannot turn it into an infinity.
    return static_cast<uint16_t>((bits >> 16) | 0x40u);
  }

  // Round to nearest-even before truncating the low 16 bits.
  const uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_OMPI_HALF_PRECISION_H_
