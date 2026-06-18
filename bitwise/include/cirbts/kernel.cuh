#pragma once

#include "openfhe.h"
#include "phantom.h"

/**
 * The signed digit decomposition which takes an RLWE ciphertext input and outputs a vector of its digits.
 * This implementation performs an approximate gadget decomposition (ignoring low-order bits).
 *
 * @param output vector of decomposed digits of the input element (results are accumulated via +=)
 * @param input vector containing the input RLWE ciphertext (typically 2 polynomials)
 * @param params wrapper object containing all RingGSW scheme parameters
 * @param Q modulus for the RingGSW/RingLWE scheme (extracted from params)
 * @param baseG gadget base used in bootstrapping (extracted from params)
 * @param digitsGA number of digits used for the approximate decomposition (extracted from params)
 * @param N number of coefficients in RGSW ciphertext (extracted from params)
 */

__device__ void device_SignedDigitDecompose2(BasicInteger* output, const BasicInteger& input,
                                             const BasicInteger& Q, const uint32_t& baseG, uint32_t digits, const size_t& N);

__global__ void kernel_SignedDigitDecompose2(BasicInteger* output, const BasicInteger* input,
                                             BasicInteger Q, uint32_t baseG, uint32_t digits, size_t N);
__global__ void kernel_SignedDigitDecompose2_Int64(int64_t* output, const BasicInteger* input,
                                                   BasicInteger Q, uint32_t baseG, uint32_t digits, size_t N);
// Signed digit decomposition with residual (ignore_bits) extraction.
// - digits_out: [digits][2N] signed digits (coefficient domain)
// - residual_out: [2N] signed residual (low bits)
__global__ void kernel_SignedDigitDecompose2_Residual(int64_t* digits_out, int64_t* residual_out, const BasicInteger* input,
                                                      BasicInteger Q, uint32_t baseG, uint32_t digits, size_t N);

// Convert signed digits to [0, Q) representation for NTT.
__global__ void kernel_DigitsSignedToModQ(BasicInteger* output, const int64_t* input, BasicInteger Q, size_t total);

// Incrementally add coefficient-domain delta into digit+residual representation.
// - delta: [2N] coefficients in [0, Q)
// - overflow_flag: set to 1 if carry propagates past the most significant digit.
__global__ void kernel_AddDelta_ToDigitsResidual(int64_t* digits, int64_t* residual, const BasicInteger* delta, BasicInteger Q,
                                                 uint32_t baseG, uint32_t digits_count, size_t N, uint32_t* overflow_flag);

// Fused: SignedDigitDecompose2 (binary) + forward NTT for each digit polynomial.
// - input: 2 polynomials in coefficient domain (c0||c1), length N each
// - output: digitsG2 polynomials in evaluation domain, layout [digitsG2][N]
__global__ void kernel_SignedDigitDecompose2_FusedFNWT(BasicInteger* output, const BasicInteger* input, BasicInteger Q, uint32_t baseG,
                                                       uint32_t digits, size_t N, const BasicInteger* twiddles,
                                                       const BasicInteger* twiddles_shoup, const DModulus* modulus);
// Fused: signed digits (already computed) + forward NTT.
// - input: digitsG2 polynomials in coefficient domain, layout [digitsG2][N]
// - output: digitsG2 polynomials in evaluation domain, layout [digitsG2][N]
__global__ void kernel_DigitsSigned_FusedFNWT(BasicInteger* output, const int64_t* input, BasicInteger Q, size_t N,
                                              const BasicInteger* twiddles, const BasicInteger* twiddles_shoup,
                                              const DModulus* modulus);

// Fused: SignedDigitDecompose2 (binary) + forward NTT for each digit polynomial, batched over independent ciphertexts.
// - input:  [batch][2][N] polynomials in coefficient domain (c0||c1)
// - output: [batch][digitsG2][N] polynomials in evaluation domain
__global__ void kernel_SignedDigitDecompose2_FusedFNWT_Batch(BasicInteger* output, const BasicInteger* input, BasicInteger Q, uint32_t baseG,
                                                             uint32_t digits, size_t N, uint32_t batch, const BasicInteger* twiddles,
                                                             const BasicInteger* twiddles_shoup, const DModulus* modulus);
__global__ void kernel_INWT_Decompose2_FusedFNWT_Batch_AES2048_D1(
    BasicInteger* output, const BasicInteger* input,
    const BasicInteger* itwiddles, const BasicInteger* itwiddles_shoup,
    const BasicInteger* twiddles, const BasicInteger* twiddles_shoup,
    const DModulus* modulus, const BasicInteger* scalar,
    const BasicInteger* scalar_shoup, BasicInteger Q, uint32_t baseG,
    uint32_t batch);

// Fused: SignedDigitDecompose (approx) + forward NTT for each digit polynomial.
// - input: numLUT polynomials in coefficient domain, layout [numLUT][N]
// - output: (digits*numLUT) polynomials in evaluation domain, layout [(digit*numLUT + lut)][N]
__global__ void kernel_SignedDigitDecompose_FusedFNWT(BasicInteger* output, const BasicInteger* input, BasicInteger Q, uint32_t base,
                                                     uint32_t digits, size_t N, uint32_t numLUT, const BasicInteger* twiddles,
                                                     const BasicInteger* twiddles_shoup, const DModulus* modulus);

__global__ void kernel_SignedDigitDecompose(BasicInteger* output, const BasicInteger* input, BasicInteger Q,
                                            uint32_t baseHT, uint32_t digitsHT, size_t N, size_t numLUT);

// Exact signed digit decomposition (ignoring the first digit) used by LMKCDEY.
// - Ciphertext version: input is 2 polynomials concatenated (c0 || c1), output is digitsG2 polynomials.
// - Poly version: input is 1 polynomial, output is digitsG polynomials.
__global__ void kernel_SignedDigitDecomposeLMKCDEY_Ciphertext(BasicInteger* output, const BasicInteger* input, BasicInteger Q,
                                                              uint32_t baseG, uint32_t digitsG2, size_t N);
__global__ void kernel_SignedDigitDecomposeLMKCDEY_Poly(BasicInteger* output, const BasicInteger* input, BasicInteger Q,
                                                        uint32_t baseG, uint32_t digitsG, size_t N);
// Fused: LMKCDEY signed digit decomposition (skip first digit) + forward NTT.
// - Ciphertext version: input is 2 polynomials concatenated (c0||c1), output is digitsG2 polynomials.
// - Poly version: input is 1 polynomial, output is digitsG polynomials.
__global__ void kernel_SignedDigitDecomposeLMKCDEY_Ciphertext_FusedFNWT(BasicInteger* output, const BasicInteger* input,
                                                                        BasicInteger Q, uint32_t baseG, uint32_t digitsG2,
                                                                        size_t N, const BasicInteger* twiddles,
                                                                        const BasicInteger* twiddles_shoup,
                                                                        const DModulus* modulus);
__global__ void kernel_SignedDigitDecomposeLMKCDEY_Poly_FusedFNWT(BasicInteger* output, const BasicInteger* input,
                                                                  BasicInteger Q, uint32_t baseG, uint32_t digitsG,
                                                                  size_t N, const BasicInteger* twiddles,
                                                                  const BasicInteger* twiddles_shoup,
                                                                  const DModulus* modulus);

__device__ void device_SignedDigitDecompose(BasicInteger* output, const BasicInteger& input, const BasicInteger& Q,
                                            const uint32_t& baseHT, uint32_t digitsHT, size_t stride);

__global__ void kernel_EvalAccCore_Binary(BasicInteger* d_acc, const BasicInteger* d_dct, const BasicInteger* d_BSKey,
                                          const BasicInteger* d_BSKey_shoup, const BasicInteger* d_monic_polys, size_t N,
                                          const DModulus* mod, uint32_t digitsG2, size_t ek_dim3_index,
                                          const uint32_t* d_indexPos);
__global__ void kernel_EvalAccCore_Binary_MB2(BasicInteger* d_acc, const BasicInteger* d_dct, const BasicInteger* d_BSKey,
                                              const BasicInteger* d_BSKey_shoup, const BasicInteger* d_BSKey_pair,
                                              const BasicInteger* d_BSKey_pair_shoup, const BasicInteger* d_monic_polys,
                                              size_t N, const DModulus* mod, uint32_t digitsG2, size_t lweIndex0,
                                              size_t lweIndex1, const uint32_t* d_indexPos);
__global__ void kernel_EvalAccCore_Binary_MB2_SoA(BasicInteger* d_acc, const BasicInteger* d_dct, const BasicInteger* d_BSKey,
                                                  const BasicInteger* d_BSKey_shoup, const BasicInteger* d_BSKey_pair,
                                                  const BasicInteger* d_BSKey_pair_shoup, const BasicInteger* d_monic_polys,
                                                  size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                  size_t lweIndex0, size_t lweIndex1, const uint32_t* d_indexPos);
__global__ void kernel_EvalAccCore_Binary_MB2_SoA_Smem(BasicInteger* d_acc, const BasicInteger* d_dct, const BasicInteger* d_BSKey,
                                                       const BasicInteger* d_BSKey_shoup, const BasicInteger* d_BSKey_pair,
                                                       const BasicInteger* d_BSKey_pair_shoup, const BasicInteger* d_monic_polys,
                                                       size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                       size_t lweIndex0, size_t lweIndex1, const uint32_t* d_indexPos);
__global__ void kernel_EvalAccCore_Binary_Grouped(BasicInteger* d_acc, const BasicInteger* d_dct, const BasicInteger* d_BSKey0,
                                                  const BasicInteger* d_BSKey0_shoup, const BasicInteger* d_monic_polys,
                                                  size_t N, const DModulus* mod, uint32_t digitsG2, size_t key_group,
                                                  size_t lweIndex, const uint32_t* d_indexPos);
__global__ void kernel_EvalAccCore_Binary_Grouped_SoA(BasicInteger* d_acc, const BasicInteger* d_dct, const BasicInteger* d_BSKey0,
                                                      const BasicInteger* d_BSKey0_shoup, const BasicInteger* d_monic_polys,
                                                      size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                      size_t key_group, size_t lweIndex, const uint32_t* d_indexPos);
__global__ void kernel_EvalAccCore_Binary_Grouped_SoA_Smem(BasicInteger* d_acc, const BasicInteger* d_dct, const BasicInteger* d_BSKey0,
                                                           const BasicInteger* d_BSKey0_shoup, const BasicInteger* d_monic_polys,
                                                           size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                           size_t key_group, size_t lweIndex, const uint32_t* d_indexPos);
__global__ void kernel_EvalAccCore_Binary_MB2_Grouped(BasicInteger* d_acc, const BasicInteger* d_dct,
                                                      const BasicInteger* d_BSKey0, const BasicInteger* d_BSKey0_shoup,
                                                      const BasicInteger* d_BSKey1, const BasicInteger* d_BSKey1_shoup,
                                                      const BasicInteger* d_BSKey_pair, const BasicInteger* d_BSKey_pair_shoup,
                                                      const BasicInteger* d_monic_polys, size_t N, const DModulus* mod,
                                                      uint32_t digitsG2, size_t key_group, size_t lweIndex0, size_t lweIndex1,
                                                      const uint32_t* d_indexPos);
__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_SoA(BasicInteger* d_acc, const BasicInteger* d_dct,
                                                          const BasicInteger* d_BSKey0, const BasicInteger* d_BSKey0_shoup,
                                                          const BasicInteger* d_BSKey1, const BasicInteger* d_BSKey1_shoup,
                                                          const BasicInteger* d_BSKey_pair, const BasicInteger* d_BSKey_pair_shoup,
                                                          const BasicInteger* d_monic_polys, size_t N, uint32_t tiles,
                                                          const DModulus* mod, uint32_t digitsG2, size_t key_group,
                                                          size_t lweIndex0, size_t lweIndex1, const uint32_t* d_indexPos);
__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_SoA_Smem(BasicInteger* d_acc, const BasicInteger* d_dct,
                                                               const BasicInteger* d_BSKey0, const BasicInteger* d_BSKey0_shoup,
                                                               const BasicInteger* d_BSKey1, const BasicInteger* d_BSKey1_shoup,
                                                               const BasicInteger* d_BSKey_pair, const BasicInteger* d_BSKey_pair_shoup,
                                                               const BasicInteger* d_monic_polys, size_t N, uint32_t tiles,
                                                               const DModulus* mod, uint32_t digitsG2, size_t key_group,
                                                               size_t lweIndex0, size_t lweIndex1, const uint32_t* d_indexPos);
// EvalAcc core update that also outputs the per-step delta (evaluation domain).
__global__ void kernel_EvalAccCore_Binary_Delta(BasicInteger* acc, BasicInteger* delta, const BasicInteger* dct,
                                                const BasicInteger* ek, const BasicInteger* ek_shoup,
                                                const BasicInteger* monic_polys, size_t N, const DModulus* mod,
                                                uint32_t digitsG2, size_t lweIndex, const uint32_t* indexPos);

__global__ void kernel_SwizzleRFKey(BasicInteger* out, const BasicInteger* in, size_t N, uint32_t digitsG2,
                                    uint32_t tiles, size_t lwe_n);

template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_Swizzle(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_swizzle,
                                                  const BasicInteger* ek_shoup_swizzle, const BasicInteger* monic_polys,
                                                  size_t N, uint32_t tiles, const DModulus* mod, size_t lweIndex,
                                                  const uint32_t* indexPos);
template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_Grouped_Swizzle(BasicInteger* acc, const BasicInteger* dct,
                                                          const BasicInteger* ek_swizzle, const BasicInteger* ek_shoup_swizzle,
                                                          const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                          const DModulus* mod, size_t key_group, size_t lweIndex,
                                                          const uint32_t* indexPos);
template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Swizzle(BasicInteger* acc, const BasicInteger* dct,
                                                              const BasicInteger* ek0_swizzle,
                                                              const BasicInteger* ek0_shoup_swizzle,
                                                              const BasicInteger* ek1_swizzle,
                                                              const BasicInteger* ek1_shoup_swizzle,
                                                              const BasicInteger* ek_pair_swizzle,
                                                              const BasicInteger* ek_pair_shoup_swizzle,
                                                              const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                              const DModulus* mod, size_t key_group, size_t lweIndex0,
                                                              size_t lweIndex1, const uint32_t* indexPos);
template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_Swizzle_Delta(BasicInteger* acc, BasicInteger* delta, const BasicInteger* dct,
                                                        const BasicInteger* ek_swizzle, const BasicInteger* ek_shoup_swizzle,
                                                        const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                        const DModulus* mod, size_t lweIndex, const uint32_t* indexPos);
__global__ void kernel_EvalAccCore_Binary_SoA(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                              const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys, size_t N,
                                              uint32_t tiles, const DModulus* mod, uint32_t digitsG2, size_t lweIndex,
                                              const uint32_t* indexPos);
__global__ void kernel_EvalAccCore_Binary_SoA_Smem(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                   const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys, size_t N,
                                                   uint32_t tiles, const DModulus* mod, uint32_t digitsG2, size_t lweIndex,
                                                   const uint32_t* indexPos);
template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_SoA_T(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys, size_t N,
                                                uint32_t tiles, const DModulus* mod, size_t lweIndex,
                                                const uint32_t* indexPos);
template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_SoA_T_Smem(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                     const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys, size_t N,
                                                     uint32_t tiles, const DModulus* mod, size_t lweIndex,
                                                     const uint32_t* indexPos);

// Batched EvalAcc core update (GINX):
// - acc:      [batch][2][N] accumulator polynomials (evaluation domain)
// - dct:      [batch][digitsG2][N] decomposed digits (evaluation domain)
// - indexPos: [batch][n] monic index positions
__global__ void kernel_EvalAccCore_Binary_Batch(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                                const BasicInteger* ek_shoup, const BasicInteger* monic_polys, size_t N,
                                                const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex,
                                                const uint32_t* d_indexPos, uint32_t indexStride);
// Phantom-style index layout: d_indexPos is [n][batch], indexStride = batch.
__global__ void kernel_EvalAccCore_Binary_Batch_NMajor(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                                       const BasicInteger* ek_shoup, const BasicInteger* monic_polys, size_t N,
                                                       const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex,
                                                       const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_Batch_SoA(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                    const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys,
                                                    size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                    uint32_t lweIndex, const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_Batch_SoA_NMajor(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                           const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys,
                                                           size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                           uint32_t lweIndex, const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_Batch_SoA_Smem(BasicInteger* acc, const BasicInteger* dct,
                                                         const BasicInteger* ek_soa, const BasicInteger* ek_shoup_soa,
                                                         const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                         const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex,
                                                         const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_Batch_SoA_Smem_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                const BasicInteger* ek_soa, const BasicInteger* ek_shoup_soa,
                                                                const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex,
                                                                const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB2_Batch(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                                    const BasicInteger* ek_shoup, const BasicInteger* ek_pair,
                                                    const BasicInteger* ek_pair_shoup, const BasicInteger* monic_polys,
                                                    size_t N, const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                    uint32_t lweIndex1, const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB2_Batch_NMajor(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                                           const BasicInteger* ek_shoup, const BasicInteger* ek_pair,
                                                           const BasicInteger* ek_pair_shoup, const BasicInteger* monic_polys,
                                                           size_t N, const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                           uint32_t lweIndex1, const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB2_Batch_SoA(BasicInteger* acc, const BasicInteger* dct,
                                                        const BasicInteger* ek_soa, const BasicInteger* ek_shoup_soa,
                                                        const BasicInteger* ek_pair_soa,
                                                        const BasicInteger* ek_pair_shoup_soa,
                                                        const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                        const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                        uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                        uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB2_Batch_SoA_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                               const BasicInteger* ek_soa, const BasicInteger* ek_shoup_soa,
                                                               const BasicInteger* ek_pair_soa,
                                                               const BasicInteger* ek_pair_shoup_soa,
                                                               const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                               const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                               uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                               uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem(BasicInteger* acc, const BasicInteger* dct,
                                                             const BasicInteger* ek_soa,
                                                             const BasicInteger* ek_shoup_soa,
                                                             const BasicInteger* ek_pair_soa,
                                                             const BasicInteger* ek_pair_shoup_soa,
                                                             const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                             const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                             uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                             uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                    const BasicInteger* ek_soa,
                                                                    const BasicInteger* ek_shoup_soa,
                                                                    const BasicInteger* ek_pair_soa,
                                                                    const BasicInteger* ek_pair_shoup_soa,
                                                                    const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                    const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                                    uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                                    uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_Grouped_Batch(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek0,
                                                        const BasicInteger* ek0_shoup, const BasicInteger* monic_polys,
                                                        size_t N, const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                        uint32_t lweIndex, const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_Grouped_Batch_NMajor(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek0,
                                                               const BasicInteger* ek0_shoup, const BasicInteger* monic_polys,
                                                               size_t N, const DModulus* mod, uint32_t digitsG2,
                                                               uint32_t key_group, uint32_t lweIndex,
                                                               const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_Grouped_Batch_SoA(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek0,
                                                            const BasicInteger* ek0_shoup, const BasicInteger* monic_polys,
                                                            size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                            uint32_t key_group, uint32_t lweIndex, const uint32_t* d_indexPos,
                                                            uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_Grouped_Batch_SoA_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                   const BasicInteger* ek0,
                                                                   const BasicInteger* ek0_shoup,
                                                                   const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                   const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                                   uint32_t lweIndex, const uint32_t* d_indexPos,
                                                                   uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem(BasicInteger* acc, const BasicInteger* dct,
                                                                 const BasicInteger* ek0, const BasicInteger* ek0_shoup,
                                                                 const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                 const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                                 uint32_t lweIndex, const uint32_t* d_indexPos,
                                                                 uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                        const BasicInteger* ek0,
                                                                        const BasicInteger* ek0_shoup,
                                                                        const BasicInteger* monic_polys, size_t N,
                                                                        uint32_t tiles, const DModulus* mod,
                                                                        uint32_t digitsG2, uint32_t key_group,
                                                                        uint32_t lweIndex, const uint32_t* d_indexPos,
                                                                        uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch(BasicInteger* acc, const BasicInteger* dct,
                                                            const BasicInteger* ek0, const BasicInteger* ek0_shoup,
                                                            const BasicInteger* ek1, const BasicInteger* ek1_shoup,
                                                            const BasicInteger* ek_pair, const BasicInteger* ek_pair_shoup,
                                                            const BasicInteger* monic_polys, size_t N, const DModulus* mod,
                                                            uint32_t digitsG2, uint32_t key_group, uint32_t lweIndex0,
                                                            uint32_t lweIndex1, const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                   const BasicInteger* ek0, const BasicInteger* ek0_shoup,
                                                                   const BasicInteger* ek1, const BasicInteger* ek1_shoup,
                                                                   const BasicInteger* ek_pair, const BasicInteger* ek_pair_shoup,
                                                                   const BasicInteger* monic_polys, size_t N, const DModulus* mod,
                                                                   uint32_t digitsG2, uint32_t key_group, uint32_t lweIndex0,
                                                                   uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                                   uint32_t indexStride);

__global__ void kernel_EvalAccCore_Binary_MB3_Grouped_Batch(BasicInteger* acc, const BasicInteger* dct,
                                                            const BasicInteger* ek0, const BasicInteger* ek0_shoup,
                                                            const BasicInteger* ek1, const BasicInteger* ek1_shoup,
                                                            const BasicInteger* ek2, const BasicInteger* ek2_shoup,
                                                            const BasicInteger* ek01, const BasicInteger* ek01_shoup,
                                                            const BasicInteger* ek02, const BasicInteger* ek02_shoup,
                                                            const BasicInteger* ek12, const BasicInteger* ek12_shoup,
                                                            const BasicInteger* ek012, const BasicInteger* ek012_shoup,
                                                            const BasicInteger* monic_polys, size_t N, const DModulus* mod,
                                                            uint32_t digitsG2, uint32_t key_group, uint32_t lweIndex0,
                                                            uint32_t lweIndex1, uint32_t lweIndex2,
                                                            const uint32_t* d_indexPos, uint32_t indexStride);

__global__ void kernel_EvalAccCore_Binary_MB3_Grouped_Batch_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                   const BasicInteger* ek0, const BasicInteger* ek0_shoup,
                                                                   const BasicInteger* ek1, const BasicInteger* ek1_shoup,
                                                                   const BasicInteger* ek2, const BasicInteger* ek2_shoup,
                                                                   const BasicInteger* ek01, const BasicInteger* ek01_shoup,
                                                                   const BasicInteger* ek02, const BasicInteger* ek02_shoup,
                                                                   const BasicInteger* ek12, const BasicInteger* ek12_shoup,
                                                                   const BasicInteger* ek012, const BasicInteger* ek012_shoup,
                                                                   const BasicInteger* monic_polys, size_t N, const DModulus* mod,
                                                                   uint32_t digitsG2, uint32_t key_group, uint32_t lweIndex0,
                                                                   uint32_t lweIndex1, uint32_t lweIndex2,
                                                                   const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA(BasicInteger* acc, const BasicInteger* dct,
                                                                const BasicInteger* ek0_soa, const BasicInteger* ek0_shoup_soa,
                                                                const BasicInteger* ek1_soa, const BasicInteger* ek1_shoup_soa,
                                                                const BasicInteger* ek2_soa, const BasicInteger* ek2_shoup_soa,
                                                                const BasicInteger* ek01_soa, const BasicInteger* ek01_shoup_soa,
                                                                const BasicInteger* ek02_soa, const BasicInteger* ek02_shoup_soa,
                                                                const BasicInteger* ek12_soa, const BasicInteger* ek12_shoup_soa,
                                                                const BasicInteger* ek012_soa, const BasicInteger* ek012_shoup_soa,
                                                                const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                                uint32_t lweIndex0, uint32_t lweIndex1, uint32_t lweIndex2,
                                                                const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                       const BasicInteger* ek0_soa, const BasicInteger* ek0_shoup_soa,
                                                                       const BasicInteger* ek1_soa, const BasicInteger* ek1_shoup_soa,
                                                                       const BasicInteger* ek2_soa, const BasicInteger* ek2_shoup_soa,
                                                                       const BasicInteger* ek01_soa, const BasicInteger* ek01_shoup_soa,
                                                                       const BasicInteger* ek02_soa, const BasicInteger* ek02_shoup_soa,
                                                                       const BasicInteger* ek12_soa, const BasicInteger* ek12_shoup_soa,
                                                                       const BasicInteger* ek012_soa, const BasicInteger* ek012_shoup_soa,
                                                                       const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                       const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                                       uint32_t lweIndex0, uint32_t lweIndex1, uint32_t lweIndex2,
                                                                       const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_NMajor_AES1024_D4(
    BasicInteger* acc, const BasicInteger* dct,
    const BasicInteger* ek0_soa, const BasicInteger* ek0_shoup_soa,
    const BasicInteger* ek1_soa, const BasicInteger* ek1_shoup_soa,
    const BasicInteger* ek2_soa, const BasicInteger* ek2_shoup_soa,
    const BasicInteger* ek01_soa, const BasicInteger* ek01_shoup_soa,
    const BasicInteger* ek02_soa, const BasicInteger* ek02_shoup_soa,
    const BasicInteger* ek12_soa, const BasicInteger* ek12_shoup_soa,
    const BasicInteger* ek012_soa, const BasicInteger* ek012_shoup_soa,
    const BasicInteger* monic_polys, const DModulus* mod, uint32_t key_group,
    uint32_t lweIndex0, uint32_t lweIndex1, uint32_t lweIndex2,
    const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_NMajor_AES1024_D4_SplitAcc(
    BasicInteger* acc, const BasicInteger* dct,
    const BasicInteger* ek0_soa, const BasicInteger* ek0_shoup_soa,
    const BasicInteger* ek1_soa, const BasicInteger* ek1_shoup_soa,
    const BasicInteger* ek2_soa, const BasicInteger* ek2_shoup_soa,
    const BasicInteger* ek01_soa, const BasicInteger* ek01_shoup_soa,
    const BasicInteger* ek02_soa, const BasicInteger* ek02_shoup_soa,
    const BasicInteger* ek12_soa, const BasicInteger* ek12_shoup_soa,
    const BasicInteger* ek012_soa, const BasicInteger* ek012_shoup_soa,
    const BasicInteger* monic_polys, const DModulus* mod, uint32_t key_group,
    uint32_t lweIndex0, uint32_t lweIndex1, uint32_t lweIndex2,
    const uint32_t* d_indexPos, uint32_t indexStride, uint32_t component, bool lazyReduce);
__global__ void kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem(BasicInteger* acc, const BasicInteger* dct,
                                                                      const BasicInteger* ek0_soa,
                                                                      const BasicInteger* ek0_shoup_soa,
                                                                      const BasicInteger* ek1_soa,
                                                                      const BasicInteger* ek1_shoup_soa,
                                                                      const BasicInteger* ek2_soa,
                                                                      const BasicInteger* ek2_shoup_soa,
                                                                      const BasicInteger* ek01_soa,
                                                                      const BasicInteger* ek01_shoup_soa,
                                                                      const BasicInteger* ek02_soa,
                                                                      const BasicInteger* ek02_shoup_soa,
                                                                      const BasicInteger* ek12_soa,
                                                                      const BasicInteger* ek12_shoup_soa,
                                                                      const BasicInteger* ek012_soa,
                                                                      const BasicInteger* ek012_shoup_soa,
                                                                      const BasicInteger* monic_polys, size_t N,
                                                                      uint32_t tiles, const DModulus* mod,
                                                                      uint32_t digitsG2, uint32_t key_group,
                                                                      uint32_t lweIndex0, uint32_t lweIndex1,
                                                                      uint32_t lweIndex2, const uint32_t* d_indexPos,
                                                                      uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                             const BasicInteger* ek0_soa,
                                                                             const BasicInteger* ek0_shoup_soa,
                                                                             const BasicInteger* ek1_soa,
                                                                             const BasicInteger* ek1_shoup_soa,
                                                                             const BasicInteger* ek2_soa,
                                                                             const BasicInteger* ek2_shoup_soa,
                                                                             const BasicInteger* ek01_soa,
                                                                             const BasicInteger* ek01_shoup_soa,
                                                                             const BasicInteger* ek02_soa,
                                                                             const BasicInteger* ek02_shoup_soa,
                                                                             const BasicInteger* ek12_soa,
                                                                             const BasicInteger* ek12_shoup_soa,
                                                                             const BasicInteger* ek012_soa,
                                                                             const BasicInteger* ek012_shoup_soa,
                                                                             const BasicInteger* monic_polys, size_t N,
                                                                             uint32_t tiles, const DModulus* mod,
                                                                             uint32_t digitsG2, uint32_t key_group,
                                                                             uint32_t lweIndex0, uint32_t lweIndex1,
                                                                             uint32_t lweIndex2, const uint32_t* d_indexPos,
                                                                             uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA(BasicInteger* acc, const BasicInteger* dct,
                                                                const BasicInteger* ek0, const BasicInteger* ek0_shoup,
                                                                const BasicInteger* ek1, const BasicInteger* ek1_shoup,
                                                                const BasicInteger* ek_pair, const BasicInteger* ek_pair_shoup,
                                                                const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                                uint32_t lweIndex0, uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                                uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                       const BasicInteger* ek0,
                                                                       const BasicInteger* ek0_shoup,
                                                                       const BasicInteger* ek1,
                                                                       const BasicInteger* ek1_shoup,
                                                                       const BasicInteger* ek_pair,
                                                                       const BasicInteger* ek_pair_shoup,
                                                                       const BasicInteger* monic_polys, size_t N,
                                                                       uint32_t tiles, const DModulus* mod,
                                                                       uint32_t digitsG2, uint32_t key_group,
                                                                       uint32_t lweIndex0, uint32_t lweIndex1,
                                                                       const uint32_t* d_indexPos, uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_Smem(BasicInteger* acc, const BasicInteger* dct,
                                                                     const BasicInteger* ek0, const BasicInteger* ek0_shoup,
                                                                     const BasicInteger* ek1, const BasicInteger* ek1_shoup,
                                                                     const BasicInteger* ek_pair, const BasicInteger* ek_pair_shoup,
                                                                     const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                     const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                                     uint32_t lweIndex0, uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                                     uint32_t indexStride);
__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_Smem_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                            const BasicInteger* ek0,
                                                                            const BasicInteger* ek0_shoup,
                                                                            const BasicInteger* ek1,
                                                                            const BasicInteger* ek1_shoup,
                                                                            const BasicInteger* ek_pair,
                                                                            const BasicInteger* ek_pair_shoup,
                                                                            const BasicInteger* monic_polys, size_t N,
                                                                            uint32_t tiles, const DModulus* mod,
                                                                            uint32_t digitsG2, uint32_t key_group,
                                                                            uint32_t lweIndex0, uint32_t lweIndex1,
                                                                            const uint32_t* d_indexPos, uint32_t indexStride);

// End-to-end EvalAcc pipeline (GINX, NTT backend):
// - INWT (eval -> coeff) on the accumulator
// - signed digit decompose (coeff)
// - forward NTT (digits)
// - key multiply + monic update (acc in eval domain)
__global__ void kernel_EvalAccPipeline_Binary_GINX(BasicInteger* acc, BasicInteger* tmp,
                                                   const BasicInteger* ek, const BasicInteger* ek_shoup,
                                                   const BasicInteger* monic_polys,
                                                   const BasicInteger* itwiddles, const BasicInteger* itwiddles_shoup,
                                                   const BasicInteger* twiddles, const BasicInteger* twiddles_shoup,
                                                   const DModulus* modulus, const BasicInteger* scalar,
                                                   const BasicInteger* scalar_shoup, size_t N, uint32_t baseG,
                                                   uint32_t digitsGA, size_t lweIndex, const uint32_t* indexPos);
__global__ void kernel_EvalAccPipeline_Binary_Grouped(BasicInteger* acc, BasicInteger* tmp,
                                                      const BasicInteger* ek, const BasicInteger* ek_shoup,
                                                      const BasicInteger* monic_polys,
                                                      const BasicInteger* itwiddles, const BasicInteger* itwiddles_shoup,
                                                      const BasicInteger* twiddles, const BasicInteger* twiddles_shoup,
                                                      const DModulus* modulus, const BasicInteger* scalar,
                                                      const BasicInteger* scalar_shoup, size_t N, uint32_t baseG,
                                                      uint32_t digitsGA, size_t key_group, size_t lweIndex,
                                                      const uint32_t* indexPos);
__global__ void kernel_EvalAccPipeline_Binary_MB2_Grouped(BasicInteger* acc, BasicInteger* tmp,
                                                          const BasicInteger* ek0, const BasicInteger* ek0_shoup,
                                                          const BasicInteger* ek1, const BasicInteger* ek1_shoup,
                                                          const BasicInteger* ek_pair, const BasicInteger* ek_pair_shoup,
                                                          const BasicInteger* monic_polys,
                                                          const BasicInteger* itwiddles, const BasicInteger* itwiddles_shoup,
                                                          const BasicInteger* twiddles, const BasicInteger* twiddles_shoup,
                                                          const DModulus* modulus, const BasicInteger* scalar,
                                                          const BasicInteger* scalar_shoup, size_t N, uint32_t baseG,
                                                          uint32_t digitsGA, size_t key_group, size_t lweIndex0,
                                                          size_t lweIndex1, const uint32_t* indexPos);

template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_Batch_Swizzle(BasicInteger* acc, const BasicInteger* dct,
                                                        const BasicInteger* ek_swizzle,
                                                        const BasicInteger* ek_shoup_swizzle,
                                                        const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                        const DModulus* mod, uint32_t lweIndex,
                                                        const uint32_t* d_indexPos, uint32_t indexStride);
template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_Grouped_Batch_Swizzle(BasicInteger* acc, const BasicInteger* dct,
                                                                const BasicInteger* ek_swizzle,
                                                                const BasicInteger* ek_shoup_swizzle,
                                                                const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                const DModulus* mod, uint32_t key_group, uint32_t lweIndex,
                                                                const uint32_t* d_indexPos, uint32_t indexStride);
template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch_Swizzle(BasicInteger* acc, const BasicInteger* dct,
                                                                    const BasicInteger* ek0_swizzle,
                                                                    const BasicInteger* ek0_shoup_swizzle,
                                                                    const BasicInteger* ek1_swizzle,
                                                                    const BasicInteger* ek1_shoup_swizzle,
                                                                    const BasicInteger* ek_pair_swizzle,
                                                                    const BasicInteger* ek_pair_shoup_swizzle,
                                                                    const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                    const DModulus* mod, uint32_t key_group,
                                                                    uint32_t lweIndex0, uint32_t lweIndex1,
                                                                    const uint32_t* d_indexPos, uint32_t indexStride);

extern template __global__ void kernel_EvalAccCore_Binary_Swizzle<2>(BasicInteger* acc, const BasicInteger* dct,
                                                                     const BasicInteger* ek_swizzle,
                                                                     const BasicInteger* ek_shoup_swizzle,
                                                                     const BasicInteger* monic_polys, size_t N,
                                                                     uint32_t tiles, const DModulus* mod,
                                                                     size_t lweIndex, const uint32_t* indexPos);
extern template __global__ void kernel_EvalAccCore_Binary_Swizzle_Delta<2>(BasicInteger* acc, BasicInteger* delta,
                                                                           const BasicInteger* dct,
                                                                           const BasicInteger* ek_swizzle,
                                                                           const BasicInteger* ek_shoup_swizzle,
                                                                           const BasicInteger* monic_polys, size_t N,
                                                                           uint32_t tiles, const DModulus* mod,
                                                                           size_t lweIndex, const uint32_t* indexPos);
extern template __global__ void kernel_EvalAccCore_Binary_Batch_Swizzle<2>(BasicInteger* acc, const BasicInteger* dct,
                                                                           const BasicInteger* ek_swizzle,
                                                                           const BasicInteger* ek_shoup_swizzle,
                                                                           const BasicInteger* monic_polys, size_t N,
                                                                           uint32_t tiles, const DModulus* mod,
                                                                           uint32_t lweIndex, const uint32_t* d_indexPos,
                                                                           uint32_t indexStride);
extern template __global__ void kernel_EvalAccCore_Binary_Grouped_Swizzle<2>(BasicInteger* acc, const BasicInteger* dct,
                                                                             const BasicInteger* ek_swizzle,
                                                                             const BasicInteger* ek_shoup_swizzle,
                                                                             const BasicInteger* monic_polys, size_t N,
                                                                             uint32_t tiles, const DModulus* mod,
                                                                             size_t key_group, size_t lweIndex,
                                                                             const uint32_t* indexPos);
extern template __global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Swizzle<2>(BasicInteger* acc, const BasicInteger* dct,
                                                                                 const BasicInteger* ek0_swizzle,
                                                                                 const BasicInteger* ek0_shoup_swizzle,
                                                                                 const BasicInteger* ek1_swizzle,
                                                                                 const BasicInteger* ek1_shoup_swizzle,
                                                                                 const BasicInteger* ek_pair_swizzle,
                                                                                 const BasicInteger* ek_pair_shoup_swizzle,
                                                                                 const BasicInteger* monic_polys, size_t N,
                                                                                 uint32_t tiles, const DModulus* mod,
                                                                                 size_t key_group, size_t lweIndex0,
                                                                                 size_t lweIndex1, const uint32_t* indexPos);
extern template __global__ void kernel_EvalAccCore_Binary_Grouped_Batch_Swizzle<2>(BasicInteger* acc, const BasicInteger* dct,
                                                                                   const BasicInteger* ek_swizzle,
                                                                                   const BasicInteger* ek_shoup_swizzle,
                                                                                   const BasicInteger* monic_polys, size_t N,
                                                                                   uint32_t tiles, const DModulus* mod,
                                                                                   uint32_t key_group, uint32_t lweIndex,
                                                                                   const uint32_t* d_indexPos,
                                                                                   uint32_t indexStride);
extern template __global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch_Swizzle<2>(BasicInteger* acc, const BasicInteger* dct,
                                                                                       const BasicInteger* ek0_swizzle,
                                                                                       const BasicInteger* ek0_shoup_swizzle,
                                                                                       const BasicInteger* ek1_swizzle,
                                                                                       const BasicInteger* ek1_shoup_swizzle,
                                                                                       const BasicInteger* ek_pair_swizzle,
                                                                                       const BasicInteger* ek_pair_shoup_swizzle,
                                                                                       const BasicInteger* monic_polys, size_t N,
                                                                                       uint32_t tiles, const DModulus* mod,
                                                                                       uint32_t key_group, uint32_t lweIndex0,
                                                                                       uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                                                       uint32_t indexStride);
extern template __global__ void kernel_EvalAccCore_Binary_SoA_T<2>(BasicInteger* acc, const BasicInteger* dct,
                                                                   const BasicInteger* ek_soa,
                                                                   const BasicInteger* ek_shoup_soa,
                                                                   const BasicInteger* monic_polys, size_t N,
                                                                   uint32_t tiles, const DModulus* mod,
                                                                   size_t lweIndex, const uint32_t* indexPos);
extern template __global__ void kernel_EvalAccCore_Binary_SoA_T<4>(BasicInteger* acc, const BasicInteger* dct,
                                                                   const BasicInteger* ek_soa,
                                                                   const BasicInteger* ek_shoup_soa,
                                                                   const BasicInteger* monic_polys, size_t N,
                                                                   uint32_t tiles, const DModulus* mod,
                                                                   size_t lweIndex, const uint32_t* indexPos);
extern template __global__ void kernel_EvalAccCore_Binary_SoA_T_Smem<2>(BasicInteger* acc, const BasicInteger* dct,
                                                                        const BasicInteger* ek_soa,
                                                                        const BasicInteger* ek_shoup_soa,
                                                                        const BasicInteger* monic_polys, size_t N,
                                                                        uint32_t tiles, const DModulus* mod,
                                                                        size_t lweIndex, const uint32_t* indexPos);
extern template __global__ void kernel_EvalAccCore_Binary_SoA_T_Smem<4>(BasicInteger* acc, const BasicInteger* dct,
                                                                        const BasicInteger* ek_soa,
                                                                        const BasicInteger* ek_shoup_soa,
                                                                        const BasicInteger* monic_polys, size_t N,
                                                                        uint32_t tiles, const DModulus* mod,
                                                                        size_t lweIndex, const uint32_t* indexPos);

__global__ void kernel_ModMulConst(BasicInteger* d_acc, BasicInteger monomial_inv, const DModulus* modulus, size_t N);

__global__ void kernel_ModMulScalar(BasicInteger* d_acc, const BasicInteger* monomial_inv, const DModulus* modulus, size_t N);

// Batched bootstrap init: acc[b] = (0, LUT) in evaluation form.
__global__ void kernel_InitBootstrapAccBatch(BasicInteger* d_acc, const BasicInteger* d_lut, size_t N, uint32_t batch);

// Batched multiply: acc[b].c1 *= monomial_inv[b] (evaluation form).
__global__ void kernel_ModMulScalarBatch(BasicInteger* d_acc, const BasicInteger* monomial_inv, const DModulus* modulus, size_t N,
                                         uint32_t batch);

// Batched bootstrap init with fused b special-MS + monomial gather:
// acc[b].c0 = 0, acc[b].c1 = lut * monic_polys[b_idx] (evaluation form).
__global__ void kernel_InitBootstrapAccBatch_FusedB(BasicInteger* d_acc, const BasicInteger* d_lut, const BasicInteger* monic_polys,
                                                    const BasicInteger* d_b, uint32_t batch, uint32_t twoN, BasicInteger Q_lwe,
                                                    uint32_t bitwidth, const DModulus* modulus, uint32_t N);

// Special modulus switching (device): compute a_ms index positions for each batch LWE a.
__global__ void kernel_SpecialMS_IndexPos_Batch(const BasicInteger* d_a, uint32_t* d_indexPos, uint32_t n, uint32_t batch,
                                                uint32_t a_stride, uint32_t twoN, BasicInteger Q, uint32_t bitwidth);
// Phantom-style layout: indexPos stored as [n][batch] (n-major).
__global__ void kernel_SpecialMS_IndexPos_Batch_NMajor(const BasicInteger* d_a, uint32_t* d_indexPos, uint32_t n, uint32_t batch,
                                                       uint32_t a_stride, uint32_t twoN, BasicInteger Q, uint32_t bitwidth);

// Special modulus switching (device): compute b_ms indices per ciphertext.
__global__ void kernel_SpecialMS_B_Batch(const BasicInteger* d_b, uint32_t* d_b_idx, uint32_t batch, uint32_t twoN,
                                         BasicInteger Q, uint32_t bitwidth);

// Gather monomial_inv from a [2N][N] monic table for each ciphertext.
__global__ void kernel_GatherMonomialInv_Batch(BasicInteger* d_monomial_inv, const BasicInteger* monic_polys,
                                               const uint32_t* d_b_idx, uint32_t N, uint32_t batch);

// EvalAcc batch kernels with fused special-MS (indexPos computed from d_a on the fly).
__global__ void kernel_EvalAccCore_Binary_Batch_MS(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                                   const BasicInteger* ek_shoup, const BasicInteger* monic_polys, size_t N,
                                                   const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex,
                                                   const BasicInteger* d_a, uint32_t a_stride, uint32_t twoN,
                                                   BasicInteger Q_lwe, uint32_t bitwidth);
__global__ void kernel_EvalAccCore_Binary_Batch_SoA_MS(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                       const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys,
                                                       size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                       uint32_t lweIndex, const BasicInteger* d_a, uint32_t a_stride,
                                                       uint32_t twoN, BasicInteger Q_lwe, uint32_t bitwidth);
__global__ void kernel_EvalAccCore_Binary_Batch_SoA_Smem_MS(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                            const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys,
                                                            size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                            uint32_t lweIndex, const BasicInteger* d_a, uint32_t a_stride,
                                                            uint32_t twoN, BasicInteger Q_lwe, uint32_t bitwidth);
__global__ void kernel_EvalAccCore_Binary_MB2_Batch_MS(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                                       const BasicInteger* ek_shoup, const BasicInteger* ek_pair,
                                                       const BasicInteger* ek_pair_shoup, const BasicInteger* monic_polys,
                                                       size_t N, const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                       uint32_t lweIndex1, const BasicInteger* d_a, uint32_t a_stride,
                                                       uint32_t twoN, BasicInteger Q_lwe, uint32_t bitwidth);
__global__ void kernel_EvalAccCore_Binary_MB2_Batch_SoA_MS(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                           const BasicInteger* ek_shoup_soa, const BasicInteger* ek_pair_soa,
                                                           const BasicInteger* ek_pair_shoup_soa, const BasicInteger* monic_polys,
                                                           size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                           uint32_t lweIndex0, uint32_t lweIndex1, const BasicInteger* d_a,
                                                           uint32_t a_stride, uint32_t twoN, BasicInteger Q_lwe, uint32_t bitwidth);
__global__ void kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem_MS(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                                const BasicInteger* ek_shoup_soa, const BasicInteger* ek_pair_soa,
                                                                const BasicInteger* ek_pair_shoup_soa, const BasicInteger* monic_polys,
                                                                size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                                uint32_t lweIndex0, uint32_t lweIndex1, const BasicInteger* d_a,
                                                                uint32_t a_stride, uint32_t twoN, BasicInteger Q_lwe, uint32_t bitwidth);

__global__ void kernel_ModAddGpowScaled(BasicInteger* d_c1, const BasicInteger* d_Gpow, BasicInteger n_inv,
                                        const DModulus* modulus, size_t numLUT);

// Batched: for each ciphertext b, add Gpow[i] >> 1 scaled by n_inv into acc[b].c1[i] (coefficient domain).
__global__ void kernel_ModAddGpowScaledBatch(BasicInteger* d_acc, const BasicInteger* d_Gpow, BasicInteger n_inv, const DModulus* modulus,
                                             uint32_t N, uint32_t numLUT, uint32_t batch);

__global__ void kernel_FusedNormalizeAndAdd(BasicInteger* d_c0, BasicInteger* d_c1, BasicInteger* d_Gpow,
                                            BasicInteger n_inv, BasicInteger numLUT, const DModulus* modulus);

__global__ void kernel_GenMVRLWEs(BasicInteger* d_MV_RLWEs_c0, BasicInteger* d_MV_RLWEs_c1, const BasicInteger* d_acc,
                                  const BasicInteger* d_monomials, const DModulus* modulus, size_t N, size_t numLUT, uint32_t batch);

//GenerateAllAutoMaps
__global__ void kernel_GenerateAllAutoMaps(uint32_t* d_AllMaps, uint32_t N, uint32_t logn,
                                           uint32_t m_mask, uint32_t numAuto);

__global__ void kernel_GenerateCoefficientAutoMaps(uint32_t* d_AllMaps, uint32_t N, uint32_t logn,
                                           uint32_t m_mask, uint32_t numAuto);

// EvalHT 
__global__ void kernel_Fused_Permute_Decompose(BasicInteger* d_digits_out, const BasicInteger* d_in_c0, BasicInteger Q,
                                               uint32_t baseHT, uint32_t digitsHT, size_t N, size_t numLUT);

__global__ void kernel_Permute_Only(BasicInteger* d_out, const BasicInteger* d_in, const uint32_t* d_map,
                                    size_t N, size_t numLUT, BasicInteger Q);

// Fused: permute (evaluation) + inverse NTT + signed-digit decompose (coefficient).
// - input: numLUT polynomials in evaluation form, layout [numLUT][N]
// - output: digits polynomials in coefficient form, layout [(digit*numLUT + lut)][N]
__global__ void kernel_INWT_Permute_SignedDigitDecompose(BasicInteger* d_digits_out, const BasicInteger* d_in_eval, const uint32_t* d_map,
                                                         const BasicInteger* itwiddles, const BasicInteger* itwiddles_shoup,
                                                         const DModulus* modulus, const BasicInteger* scalar,
                                                         const BasicInteger* scalar_shoup, uint32_t base, uint32_t digits, size_t N,
                                                         uint32_t numLUT);

// Fused: inverse NTT + signed-digit decompose (coefficient).
// - input: numLUT polynomials in evaluation form, layout [numLUT][N]
// - output: digits polynomials in coefficient form, layout [(digit*numLUT + lut)][N]
__global__ void kernel_INWT_SignedDigitDecompose(BasicInteger* d_digits_out, const BasicInteger* d_in_eval, const BasicInteger* itwiddles,
                                                const BasicInteger* itwiddles_shoup, const DModulus* modulus, const BasicInteger* scalar,
                                                const BasicInteger* scalar_shoup, uint32_t base, uint32_t digits, size_t N,
                                                uint32_t numLUT);

__global__ void kernel_SaveToRGSW(BasicInteger* d_RGSW, const BasicInteger* d_c0, const BasicInteger* d_c1,
                                  int row_type, size_t N, size_t numLUT);
                                  
__global__ void kernel_MultAdd(BasicInteger* d_res_c0, BasicInteger* d_res_c1, const BasicInteger* d_digits,
                               const BasicInteger* d_HTKeys, const DModulus* modulus, uint32_t numDigits, size_t N, size_t numLUT);

__global__ void kernel_AddInPlace(BasicInteger* dst, const BasicInteger* add, const DModulus* modulus, size_t N);

// Fused: (digits in NTT domain) -> key-switch multiply-add -> HomTrace update (includes permute(c1)).
__global__ void kernel_MultAddUpdate_HT_PermuteC1(BasicInteger* d_next_c0, BasicInteger* d_next_c1, const BasicInteger* d_digits,
                                                 const BasicInteger* d_HTKeys, const BasicInteger* d_HTKeys_shoup,
                                                 const BasicInteger* d_curr_c0, const BasicInteger* d_curr_c1,
                                                 const uint32_t* d_map, const DModulus* modulus, uint32_t numDigits, size_t N,
                                                 size_t numLUT, size_t total_size);
__global__ void kernel_MultAddUpdate_HT_PermuteC1_Save(BasicInteger* d_next_c0, BasicInteger* d_next_c1,
                                                       BasicInteger* d_RGSW, const BasicInteger* d_digits,
                                                       const BasicInteger* d_HTKeys, const BasicInteger* d_HTKeys_shoup,
                                                       const BasicInteger* d_curr_c0, const BasicInteger* d_curr_c1,
                                                       const uint32_t* d_map, const DModulus* modulus, uint32_t numDigits,
                                                       size_t N, size_t numLUT, size_t total_size);
__global__ void kernel_MultAddUpdate_HT_PermuteC1_SoA(BasicInteger* d_next_c0, BasicInteger* d_next_c1, const BasicInteger* d_digits,
                                                     const BasicInteger* d_HTKeys, const BasicInteger* d_HTKeys_shoup,
                                                     const BasicInteger* d_curr_c0, const BasicInteger* d_curr_c1,
                                                     const uint32_t* d_map, const DModulus* modulus, uint32_t numDigits, size_t N,
                                                     uint32_t tiles, size_t numLUT, size_t total_size);
__global__ void kernel_MultAddUpdate_HT_PermuteC1_SoA_Smem(BasicInteger* d_next_c0, BasicInteger* d_next_c1,
                                                          const BasicInteger* d_digits, const BasicInteger* d_HTKeys,
                                                          const BasicInteger* d_HTKeys_shoup, const BasicInteger* d_curr_c0,
                                                          const BasicInteger* d_curr_c1, const uint32_t* d_map,
                                                          const DModulus* modulus, uint32_t numDigits, size_t N, uint32_t tiles,
                                                          size_t numLUT, size_t total_size);
// Fused: HT update + SS accumulate (pipeline mode).
__global__ void kernel_MultAddUpdate_HTSS_PermuteC1(BasicInteger* d_next_c0, BasicInteger* d_next_c1,
                                                   BasicInteger* d_ss_c0, BasicInteger* d_ss_c1,
                                                   const BasicInteger* d_digits, const BasicInteger* d_HTKeys,
                                                   const BasicInteger* d_HTKeys_shoup, const BasicInteger* d_SSKeys,
                                                   const BasicInteger* d_SSKeys_shoup, const BasicInteger* d_curr_c0,
                                                   const BasicInteger* d_curr_c1, const uint32_t* d_map,
                                                   const DModulus* modulus, uint32_t numDigits, size_t N, size_t numLUT,
                                                   size_t total_size);
__global__ void kernel_MultAddUpdate_HTSS_PermuteC1_SoA(BasicInteger* d_next_c0, BasicInteger* d_next_c1,
                                                       BasicInteger* d_ss_c0, BasicInteger* d_ss_c1,
                                                       const BasicInteger* d_digits, const BasicInteger* d_HTKeys,
                                                       const BasicInteger* d_HTKeys_shoup, const BasicInteger* d_SSKeys,
                                                       const BasicInteger* d_SSKeys_shoup, const BasicInteger* d_curr_c0,
                                                       const BasicInteger* d_curr_c1, const uint32_t* d_map,
                                                       const DModulus* modulus, uint32_t numDigits, size_t N, uint32_t tiles,
                                                       size_t numLUT, size_t total_size);
__global__ void kernel_MultAddUpdate_HTSS_PermuteC1_SoA_Smem(BasicInteger* d_next_c0, BasicInteger* d_next_c1,
                                                            BasicInteger* d_ss_c0, BasicInteger* d_ss_c1,
                                                            const BasicInteger* d_digits, const BasicInteger* d_HTKeys,
                                                            const BasicInteger* d_HTKeys_shoup, const BasicInteger* d_SSKeys,
                                                            const BasicInteger* d_SSKeys_shoup, const BasicInteger* d_curr_c0,
                                                            const BasicInteger* d_curr_c1, const uint32_t* d_map,
                                                            const DModulus* modulus, uint32_t numDigits, size_t N,
                                                            uint32_t tiles, size_t numLUT, size_t total_size);

__global__ void kernel_Update_HT_PermuteC1(BasicInteger* d_next_c0, BasicInteger* d_next_c1, const BasicInteger* d_ks_c0,
                                          const BasicInteger* d_ks_c1, const BasicInteger* d_curr_c0,
                                          const BasicInteger* d_curr_c1, const uint32_t* d_map, const DModulus* modulus,
                                          size_t N, size_t total_size);

// Fused: (digits in NTT domain) -> key-switch multiply-add -> SchemeSwitch update.
__global__ void kernel_MultAddUpdate_SS(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_digits,
                                       const BasicInteger* d_SSKeys, const BasicInteger* d_SSKeys_shoup,
                                       const BasicInteger* d_backup_c1, const DModulus* modulus, uint32_t numDigits, size_t N,
                                       size_t numLUT, size_t total_size);
__global__ void kernel_MultAddUpdate_SS_Save(BasicInteger* d_out_c0, BasicInteger* d_out_c1, BasicInteger* d_RGSW,
                                             const BasicInteger* d_digits, const BasicInteger* d_SSKeys,
                                             const BasicInteger* d_SSKeys_shoup, const BasicInteger* d_backup_c1,
                                             const DModulus* modulus, uint32_t numDigits, size_t N, size_t numLUT,
                                             size_t total_size);
__global__ void kernel_MultAddUpdate_SS_SoA(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_digits,
                                           const BasicInteger* d_SSKeys, const BasicInteger* d_SSKeys_shoup,
                                           const BasicInteger* d_backup_c1, const DModulus* modulus, uint32_t numDigits, size_t N,
                                           uint32_t tiles, size_t numLUT, size_t total_size);
__global__ void kernel_MultAddUpdate_SS_SoA_Smem(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_digits,
                                                const BasicInteger* d_SSKeys, const BasicInteger* d_SSKeys_shoup,
                                                const BasicInteger* d_backup_c1, const DModulus* modulus, uint32_t numDigits,
                                                size_t N, uint32_t tiles, size_t numLUT, size_t total_size);
// Accumulate-only variant: adds SS key-switch contributions into existing outputs (no backup_c1 add).
__global__ void kernel_MultAddUpdate_SS_Accumulate(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_digits,
                                                   const BasicInteger* d_SSKeys, const BasicInteger* d_SSKeys_shoup,
                                                   const DModulus* modulus, uint32_t numDigits, size_t N, size_t numLUT,
                                                   size_t total_size);
__global__ void kernel_MultAddUpdate_SS_Accumulate_SoA(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_digits,
                                                      const BasicInteger* d_SSKeys, const BasicInteger* d_SSKeys_shoup,
                                                      const DModulus* modulus, uint32_t numDigits, size_t N, uint32_t tiles,
                                                      size_t numLUT, size_t total_size);

__global__ void kernel_Update_SS(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_ks_c0,
                                 const BasicInteger* d_ks_c1, const BasicInteger* d_backup_c1, const DModulus* modulus,
                                 size_t N, size_t total_size);
