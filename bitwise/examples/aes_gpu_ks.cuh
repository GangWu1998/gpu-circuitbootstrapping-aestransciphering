/*
 * =============================================================================
 * File: aes_gpu_ks.cuh
 * Purpose: Example-only GPU offload for TLWE_N -> TLWE_n LWE key switching.
 *
 * This is used by the AES transciphering/T-table demos to remove the dominant
 * CPU bottleneck in "SampleExtractIndex + KeySwitch (N->n)" when extracting
 * 128 output bits from RLWE state.
 *
 * Constraints (by design, for simplicity/perf):
 *   - Uses a packed uint16_t representation of the key switching key (KSK)
 *     and ciphertext coefficients, therefore requires qKS <= 65535.
 *   - Requires qKS and baseKS to be powers of two, with digitCount <= 2.
 *     (Typical fast setting: qKS=16384, baseKS=128 => digitCount=2.)
 *
 * The packed layout is SoA across output coefficients so that threads with
 * consecutive (k) indices read coalesced.
 * =============================================================================
 */
#pragma once

#include "phantom.h"

#include "lwe-cryptoparameters.h"
#include "lwe-keyswitchkey.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace aes_example {

inline bool IsPowerOfTwo(uint32_t x) {
    return x && ((x & (x - 1u)) == 0u);
}

inline uint32_t Log2Pow2(uint32_t x) {
    uint32_t r = 0;
    while ((1u << r) < x) {
        ++r;
    }
    return r;
}

inline uint32_t DigitCountPow(uint32_t q, uint32_t base) {
    uint64_t v = 1;
    uint32_t d = 0;
    while (v < q) {
        v *= base;
        ++d;
        if (d > 64) {
            break;
        }
    }
    return d;
}

struct GpuLweKeySwitchKeyU16 {
    uint16_t* d_ksk{};
    size_t elems{};
    uint32_t n{};
    uint32_t N{};
    uint32_t baseKS{};
    uint32_t baseKS_log2{};
    uint32_t digitCount{};
    uint32_t qKS{};
    uint32_t qKS_mask{};

    GpuLweKeySwitchKeyU16() = default;
    ~GpuLweKeySwitchKeyU16() {
        if (d_ksk) {
            cudaFree(d_ksk);
            d_ksk = nullptr;
        }
    }

    GpuLweKeySwitchKeyU16(const GpuLweKeySwitchKeyU16&) = delete;
    GpuLweKeySwitchKeyU16& operator=(const GpuLweKeySwitchKeyU16&) = delete;

    GpuLweKeySwitchKeyU16(GpuLweKeySwitchKeyU16&& other) noexcept {
        *this = std::move(other);
    }
    GpuLweKeySwitchKeyU16& operator=(GpuLweKeySwitchKeyU16&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (d_ksk) {
            cudaFree(d_ksk);
        }
        d_ksk         = other.d_ksk;
        elems         = other.elems;
        n             = other.n;
        N             = other.N;
        baseKS        = other.baseKS;
        baseKS_log2   = other.baseKS_log2;
        digitCount    = other.digitCount;
        qKS           = other.qKS;
        qKS_mask      = other.qKS_mask;
        other.d_ksk   = nullptr;
        other.elems   = 0;
        other.n       = 0;
        other.N       = 0;
        other.baseKS  = 0;
        other.digitCount = 0;
        other.qKS     = 0;
        other.qKS_mask = 0;
        other.baseKS_log2 = 0;
        return *this;
    }

    static bool Supported(const std::shared_ptr<lbcrypto::LWECryptoParams>& params) {
        if (!params) {
            return false;
        }
        const uint64_t qks64 = params->GetqKS().ConvertToInt();
        if (qks64 == 0 || qks64 > std::numeric_limits<uint16_t>::max()) {
            return false;
        }
        const uint32_t qks = static_cast<uint32_t>(qks64);
        const uint32_t bks = params->GetBaseKS();
        if (!IsPowerOfTwo(qks) || !IsPowerOfTwo(bks)) {
            return false;
        }
        const uint32_t digits = DigitCountPow(qks, bks);
        if (digits == 0 || digits > 2) {
            return false;
        }
        return true;
    }

    static bool TryInit(GpuLweKeySwitchKeyU16* out, const std::shared_ptr<lbcrypto::LWECryptoParams>& params,
                        const lbcrypto::ConstLWESwitchingKey& ksk, std::string* why = nullptr) {
        if (!out) {
            return false;
        }
        if (!params || !ksk) {
            if (why) {
                *why = "null params/ksk";
            }
            return false;
        }
        const uint64_t qks64 = params->GetqKS().ConvertToInt();
        if (qks64 == 0 || qks64 > std::numeric_limits<uint16_t>::max()) {
            if (why) {
                *why = "qKS too large for uint16 packing";
            }
            return false;
        }

        const uint32_t qks = static_cast<uint32_t>(qks64);
        const uint32_t bks = params->GetBaseKS();
        if (!IsPowerOfTwo(qks) || !IsPowerOfTwo(bks)) {
            if (why) {
                *why = "requires power-of-two qKS and baseKS";
            }
            return false;
        }

        const uint32_t digits = DigitCountPow(qks, bks);
        if (digits == 0 || digits > 2) {
            if (why) {
                *why = "only digitCount<=2 is supported";
            }
            return false;
        }

        const uint32_t n = params->Getn();
        const uint32_t N = params->GetN();
        if (!n || !N) {
            if (why) {
                *why = "invalid n/N";
            }
            return false;
        }

        const size_t outLen = static_cast<size_t>(n) + 1;
        const size_t elems = static_cast<size_t>(digits) * static_cast<size_t>(N) * static_cast<size_t>(bks) * outLen;
        if (elems == 0 || elems > (static_cast<size_t>(1) << 31)) {
            if (why) {
                *why = "packed key size too large";
            }
            return false;
        }

        std::vector<uint16_t> h_ksk(elems);
        const auto& keyA = ksk->GetElementsA();
        const auto& keyB = ksk->GetElementsB();

        for (uint32_t j = 0; j < digits; ++j) {
            for (uint32_t i = 0; i < N; ++i) {
                const auto& refA_i = keyA[i];
                const auto& refB_i = keyB[i];
                for (uint32_t a0 = 0; a0 < bks; ++a0) {
                    const auto& refA = refA_i[a0][j];
                    const auto& refB = refB_i[a0][j];
                    const size_t base = ((((static_cast<size_t>(j) * N) + i) * bks) + a0) * outLen;
                    for (uint32_t k = 0; k < n; ++k) {
                        h_ksk[base + k] = static_cast<uint16_t>(refA[k].ConvertToInt());
                    }
                    h_ksk[base + n] = static_cast<uint16_t>(refB.ConvertToInt());
                }
            }
        }

        uint16_t* dptr = nullptr;
        PHANTOM_CHECK_CUDA(cudaMalloc(&dptr, elems * sizeof(uint16_t)));
        PHANTOM_CHECK_CUDA(cudaMemcpy(dptr, h_ksk.data(), elems * sizeof(uint16_t), cudaMemcpyHostToDevice));

        out->d_ksk       = dptr;
        out->elems       = elems;
        out->n           = n;
        out->N           = N;
        out->baseKS      = bks;
        out->baseKS_log2 = Log2Pow2(bks);
        out->digitCount  = digits;
        out->qKS         = qks;
        out->qKS_mask    = qks - 1u;
        return true;
    }
};

struct GpuLweKeySwitchWorkspaceU16 {
    uint16_t* d_in_a{};
    uint16_t* d_in_b{};
    uint16_t* d_out{};
    // Optional extraction-pack helpers (example-only): RLWE coeff buffers and per-sample extract map.
    uint64_t* d_rlwe_a{};
    uint64_t* d_rlwe_b{};
    uint32_t* d_extract_info{};
    uint16_t* d_extract_coeff_map{};
    uint32_t extractCoeffMapRows{};
    size_t maxBatch{};
    uint32_t n{};
    uint32_t N{};
    cudaStream_t stream{};
    bool owns_stream{true};

    GpuLweKeySwitchWorkspaceU16() = default;
    explicit GpuLweKeySwitchWorkspaceU16(size_t maxBatch_, uint32_t N_, uint32_t n_) : maxBatch(maxBatch_), n(n_), N(N_) {
        PHANTOM_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        owns_stream = true;
        PHANTOM_CHECK_CUDA(cudaMalloc(&d_in_a, maxBatch * static_cast<size_t>(N) * sizeof(uint16_t)));
        PHANTOM_CHECK_CUDA(cudaMalloc(&d_in_b, maxBatch * sizeof(uint16_t)));
        PHANTOM_CHECK_CUDA(cudaMalloc(&d_out, maxBatch * (static_cast<size_t>(n) + 1) * sizeof(uint16_t)));
        PHANTOM_CHECK_CUDA(cudaMalloc(&d_rlwe_a, 4ull * static_cast<size_t>(N) * sizeof(uint64_t)));
        PHANTOM_CHECK_CUDA(cudaMalloc(&d_rlwe_b, 4ull * static_cast<size_t>(N) * sizeof(uint64_t)));
        PHANTOM_CHECK_CUDA(cudaMalloc(&d_extract_info, maxBatch * sizeof(uint32_t)));
    }

    // Variant that runs all kernels on a caller-provided stream (non-owning).
    // Useful when you want extraction+keyswitch to stay on the same stream as other GPU work
    // (e.g., AES LUT/CMUX stream) to avoid extra events/synchronization.
    explicit GpuLweKeySwitchWorkspaceU16(size_t maxBatch_, uint32_t N_, uint32_t n_, cudaStream_t external_stream)
        : maxBatch(maxBatch_), n(n_), N(N_), stream(external_stream) {
        if (!stream) {
            PHANTOM_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            owns_stream = true;
        } else {
            owns_stream = false;
        }
        PHANTOM_CHECK_CUDA(cudaMalloc(&d_in_a, maxBatch * static_cast<size_t>(N) * sizeof(uint16_t)));
        PHANTOM_CHECK_CUDA(cudaMalloc(&d_in_b, maxBatch * sizeof(uint16_t)));
        PHANTOM_CHECK_CUDA(cudaMalloc(&d_out, maxBatch * (static_cast<size_t>(n) + 1) * sizeof(uint16_t)));
        PHANTOM_CHECK_CUDA(cudaMalloc(&d_rlwe_a, 4ull * static_cast<size_t>(N) * sizeof(uint64_t)));
        PHANTOM_CHECK_CUDA(cudaMalloc(&d_rlwe_b, 4ull * static_cast<size_t>(N) * sizeof(uint64_t)));
        PHANTOM_CHECK_CUDA(cudaMalloc(&d_extract_info, maxBatch * sizeof(uint32_t)));
    }

    ~GpuLweKeySwitchWorkspaceU16() {
        if (d_in_a) {
            cudaFree(d_in_a);
            d_in_a = nullptr;
        }
        if (d_in_b) {
            cudaFree(d_in_b);
            d_in_b = nullptr;
        }
        if (d_out) {
            cudaFree(d_out);
            d_out = nullptr;
        }
        if (d_rlwe_a) {
            cudaFree(d_rlwe_a);
            d_rlwe_a = nullptr;
        }
        if (d_rlwe_b) {
            cudaFree(d_rlwe_b);
            d_rlwe_b = nullptr;
        }
        if (d_extract_info) {
            cudaFree(d_extract_info);
            d_extract_info = nullptr;
        }
        if (d_extract_coeff_map) {
            cudaFree(d_extract_coeff_map);
            d_extract_coeff_map = nullptr;
        }
        if (stream && owns_stream) {
            cudaStreamDestroy(stream);
        }
        stream = nullptr;
    }

    GpuLweKeySwitchWorkspaceU16(const GpuLweKeySwitchWorkspaceU16&) = delete;
    GpuLweKeySwitchWorkspaceU16& operator=(const GpuLweKeySwitchWorkspaceU16&) = delete;

    bool EnsureExtractCoeffMap() {
        if (!d_extract_coeff_map) {
            InitExtractCoeffMap();
        }
        return d_extract_coeff_map && extractCoeffMapRows >= N;
    }

  private:
    void InitExtractCoeffMap() {
        // Encodes SampleExtract coefficient selection as (source_coeff | sign_bit).
        // The uint16 packing supports the AES parameters used here (N=2048).
        if (!N || N > 0x7FFFu || N > 8192u) {
            return;
        }
        const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(N);
        std::vector<uint16_t> h_map(elems);
        for (uint32_t idx = 0; idx < N; ++idx) {
            uint16_t* row = h_map.data() + static_cast<size_t>(idx) * N;
            for (uint32_t coeff = 0; coeff < N; ++coeff) {
                const bool neg = coeff > idx;
                const uint32_t src = neg ? (N + idx - coeff) : (idx - coeff);
                row[coeff] = static_cast<uint16_t>(src | (neg ? 0x8000u : 0u));
            }
        }
        PHANTOM_CHECK_CUDA(cudaMalloc(&d_extract_coeff_map, elems * sizeof(uint16_t)));
        PHANTOM_CHECK_CUDA(cudaMemcpy(d_extract_coeff_map, h_map.data(), elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
        extractCoeffMapRows = N;
    }
};

static __global__ void kernel_KeySwitchU16(uint16_t* __restrict__ out, const uint16_t* __restrict__ in_a,
                                           const uint16_t* __restrict__ in_b, const uint16_t* __restrict__ ksk,
                                           uint32_t batch, uint32_t N, uint32_t outLen, uint32_t stride_i, uint32_t stride_j,
                                           uint32_t baseKS_log2, uint32_t baseKS_mask, uint32_t qKS_mask) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = batch * outLen;
    if (tid >= total) {
        return;
    }

    const uint32_t t = tid / outLen;
    const uint32_t k = tid - t * outLen;
    uint32_t acc = (k + 1 == outLen) ? static_cast<uint32_t>(in_b[t]) : 0u;

    const uint16_t* a_t = in_a + static_cast<size_t>(t) * N;
    const uint16_t* ksk_j0 = ksk;
    const uint16_t* ksk_j1 = ksk + stride_j;
    uint32_t off = k;
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t ai = static_cast<uint32_t>(a_t[i]);
        const uint32_t d0 = ai & baseKS_mask;
        const uint32_t d1 = (ai >> baseKS_log2) & baseKS_mask;

        acc -= static_cast<uint32_t>(ksk_j0[off + d0 * outLen]);
        acc -= static_cast<uint32_t>(ksk_j1[off + d1 * outLen]);
        off += stride_i;
    }

    out[tid] = static_cast<uint16_t>(acc & qKS_mask);
}

static __device__ __forceinline__ uint16_t modswitch_u64_to_pow2_u16(uint64_t v, double q_over_Q, uint32_t q_mask) {
    // q_over_Q = q / Q, where q is power-of-two <= 65535. We rely on q << Q so that
    // double rounding error cannot affect the final integer rounding.
    const double x = static_cast<double>(v) * q_over_Q;
    const uint32_t r = __double2uint_rn(x);
    return static_cast<uint16_t>(r & q_mask);
}

static __device__ __forceinline__ uint16_t sample_extract_a_u16(const uint64_t* __restrict__ a_word, uint32_t idx,
                                                                uint32_t coeff, uint32_t N, uint64_t Q,
                                                                double q_over_Q, uint32_t q_mask) {
    uint64_t v{};
    if (coeff <= idx) {
        v = a_word[idx - coeff];
    } else {
        const uint32_t k = static_cast<uint32_t>(N + idx - coeff);
        const uint64_t ak = a_word[k];
        v = (ak == 0) ? 0 : (Q - ak);
    }
    return modswitch_u64_to_pow2_u16(v, q_over_Q, q_mask);
}

static __device__ __forceinline__ uint16_t sample_extract_a_mapped_u16(const uint64_t* __restrict__ a_word,
                                                                       const uint16_t* __restrict__ coeff_map,
                                                                       uint32_t idx, uint32_t coeff, uint32_t N,
                                                                       uint64_t Q, double q_over_Q, uint32_t q_mask) {
    const uint16_t encoded = coeff_map[static_cast<size_t>(idx) * N + coeff];
    const uint32_t src = static_cast<uint32_t>(encoded & 0x7FFFu);
    uint64_t v = a_word[src];
    if (encoded & 0x8000u) {
        v = (v == 0) ? 0 : (Q - v);
    }
    return modswitch_u64_to_pow2_u16(v, q_over_Q, q_mask);
}

static __device__ __forceinline__ uint64_t u128_div_u32_round_ks(uint64_t hi, uint64_t lo, uint32_t d) {
    const uint64_t add = static_cast<uint64_t>(d >> 1);
    const uint64_t lo_prev = lo;
    lo += add;
    if (lo < lo_prev) {
        hi += 1u;
    }

    uint32_t w0 = static_cast<uint32_t>(lo);
    uint32_t w1 = static_cast<uint32_t>(lo >> 32);
    uint32_t w2 = static_cast<uint32_t>(hi);
    uint32_t w3 = static_cast<uint32_t>(hi >> 32);

    uint64_t r = 0;
    uint32_t qwords[4] = {0, 0, 0, 0};
    for (int i = 3; i >= 0; --i) {
        const uint64_t cur = (r << 32) | static_cast<uint64_t>(i == 3 ? w3 : (i == 2 ? w2 : (i == 1 ? w1 : w0)));
        const uint32_t q = static_cast<uint32_t>(cur / d);
        r = cur - static_cast<uint64_t>(q) * d;
        qwords[i] = q;
    }
    return (static_cast<uint64_t>(qwords[1]) << 32) | qwords[0];
}

template <bool QDivisible>
static __device__ __forceinline__ BasicInteger ks_u16_to_lwe(uint32_t v, uint64_t scale, uint32_t qks, uint64_t q) {
    if (QDivisible) {
        return static_cast<BasicInteger>(static_cast<uint64_t>(v) * scale);
    }
    const uint64_t lo = static_cast<uint64_t>(v) * q;
    const uint64_t hi = __umul64hi(static_cast<uint64_t>(v), q);
    return static_cast<BasicInteger>(u128_div_u32_round_ks(hi, lo, qks));
}

// Pack 128 TLWE_N ciphertexts (mod qKS, packed uint16) from 4 RLWE words in COEFFICIENT form.
// sample_info[t] encodes (word_id << 16) | coeff_index.
static __global__ void kernel_SampleExtractPackU16(uint16_t* __restrict__ out_a, uint16_t* __restrict__ out_b,
                                                   const uint64_t* __restrict__ rlwe_a, const uint64_t* __restrict__ rlwe_b,
                                                   const uint32_t* __restrict__ sample_info, uint32_t batch, uint32_t N,
                                                   uint64_t Q, double q_over_Q, uint32_t q_mask) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t t = static_cast<uint32_t>(blockIdx.y);
    if (t >= batch || coeff >= N) {
        return;
    }

    const uint32_t info = sample_info[t];
    const uint32_t word = info >> 16;
    const uint32_t idx = info & 0xFFFFu;
    const uint64_t* a_word = rlwe_a + static_cast<size_t>(word) * N;
    const uint64_t* b_word = rlwe_b + static_cast<size_t>(word) * N;

    uint64_t v{};
    if (coeff <= idx) {
        v = a_word[idx - coeff];
    } else {
        const uint32_t k = static_cast<uint32_t>(N + idx - coeff);
        const uint64_t ak = a_word[k];
        v = (ak == 0) ? 0 : (Q - ak);
    }
    out_a[static_cast<size_t>(t) * N + coeff] = modswitch_u64_to_pow2_u16(v, q_over_Q, q_mask);

    if (coeff == 0) {
        out_b[t] = modswitch_u64_to_pow2_u16(b_word[idx], q_over_Q, q_mask);
    }
}

// Fused SampleExtract + KeySwitch. It skips materializing the intermediate TLWE_N
// batch in ws.d_in_a/d_in_b and consumes the extracted coefficients inside KS.
template <bool UsePreencodedMap>
static __global__ void kernel_SampleExtractKeySwitchU16(uint16_t* __restrict__ out, const uint64_t* __restrict__ rlwe_a,
                                                        const uint64_t* __restrict__ rlwe_b,
                                                        const uint32_t* __restrict__ sample_info,
                                                        const uint16_t* __restrict__ extract_coeff_map,
                                                        const uint16_t* __restrict__ ksk, uint32_t batch, uint32_t N,
                                                        uint32_t outLen, uint32_t stride_i, uint32_t stride_j,
                                                        uint32_t baseKS_log2, uint32_t baseKS_mask, uint32_t qKS_mask,
                                                        uint64_t Q, double q_over_Q) {
    const uint32_t t = static_cast<uint32_t>(blockIdx.y);
    const uint32_t block_k0 = static_cast<uint32_t>(blockIdx.x) * blockDim.x;
    if (t >= batch || block_k0 >= outLen) {
        return;
    }
    const uint32_t k = block_k0 + threadIdx.x;

    const uint32_t info = sample_info[t];
    const uint32_t word = info >> 16;
    const uint32_t idx = info & 0xFFFFu;
    const uint64_t* a_word = rlwe_a + static_cast<size_t>(word) * N;
    const uint64_t* b_word = rlwe_b + static_cast<size_t>(word) * N;

    const bool active = (k < outLen);
    uint32_t acc = 0u;
    if (active && k + 1u == outLen) {
        acc = static_cast<uint32_t>(modswitch_u64_to_pow2_u16(b_word[idx], q_over_Q, qKS_mask));
    }

    const uint16_t* ksk_j0 = ksk;
    const uint16_t* ksk_j1 = ksk + stride_j;
    __shared__ uint16_t s_a[256];
    for (uint32_t tile = 0; tile < N; tile += blockDim.x) {
        const uint32_t coeff = tile + threadIdx.x;
        if (coeff < N) {
            if (UsePreencodedMap) {
                s_a[threadIdx.x] = sample_extract_a_mapped_u16(a_word, extract_coeff_map, idx, coeff, N, Q, q_over_Q, qKS_mask);
            } else {
                s_a[threadIdx.x] = sample_extract_a_u16(a_word, idx, coeff, N, Q, q_over_Q, qKS_mask);
            }
        }
        __syncthreads();

        if (active) {
            const uint32_t tile_count = min(blockDim.x, N - tile);
            for (uint32_t j = 0; j < tile_count; ++j) {
                const uint32_t ai = static_cast<uint32_t>(s_a[j]);
                const uint32_t d0 = ai & baseKS_mask;
                const uint32_t d1 = (ai >> baseKS_log2) & baseKS_mask;
                const uint32_t off = (tile + j) * stride_i + k;
                acc -= static_cast<uint32_t>(ksk_j0[off + d0 * outLen]);
                acc -= static_cast<uint32_t>(ksk_j1[off + d1 * outLen]);
            }
        }
        __syncthreads();
    }

    if (active) {
        out[static_cast<size_t>(t) * outLen + k] = static_cast<uint16_t>(acc & qKS_mask);
    }
}

template <bool UsePreencodedMap, bool QDivisible>
static __global__ void kernel_SampleExtractKeySwitchToLWE(BasicInteger* __restrict__ out_a,
                                                          BasicInteger* __restrict__ out_b,
                                                          const uint64_t* __restrict__ rlwe_a,
                                                          const uint64_t* __restrict__ rlwe_b,
                                                          const uint32_t* __restrict__ sample_info,
                                                          const uint16_t* __restrict__ extract_coeff_map,
                                                          const uint16_t* __restrict__ ksk, uint32_t batch,
                                                          uint32_t N, uint32_t n, uint32_t outLen,
                                                          uint32_t stride_i, uint32_t stride_j,
                                                          uint32_t baseKS_log2, uint32_t baseKS_mask,
                                                          uint32_t qKS_mask, uint64_t Q, double q_over_Q,
                                                          uint64_t scale, uint32_t qks, uint64_t q,
                                                          uint32_t out_stride) {
    const uint32_t t = static_cast<uint32_t>(blockIdx.y);
    const uint32_t block_k0 = static_cast<uint32_t>(blockIdx.x) * blockDim.x;
    if (t >= batch || block_k0 >= outLen) {
        return;
    }
    const uint32_t k = block_k0 + threadIdx.x;

    const uint32_t info = sample_info[t];
    const uint32_t word = info >> 16;
    const uint32_t idx = info & 0xFFFFu;
    const uint64_t* a_word = rlwe_a + static_cast<size_t>(word) * N;
    const uint64_t* b_word = rlwe_b + static_cast<size_t>(word) * N;

    const bool active = (k < outLen);
    uint32_t acc = 0u;
    if (active && k == n) {
        acc = static_cast<uint32_t>(modswitch_u64_to_pow2_u16(b_word[idx], q_over_Q, qKS_mask));
    }

    const uint16_t* ksk_j0 = ksk;
    const uint16_t* ksk_j1 = ksk + stride_j;
    __shared__ uint16_t s_a[256];
    for (uint32_t tile = 0; tile < N; tile += blockDim.x) {
        const uint32_t coeff = tile + threadIdx.x;
        if (coeff < N) {
            if (UsePreencodedMap) {
                s_a[threadIdx.x] = sample_extract_a_mapped_u16(a_word, extract_coeff_map, idx, coeff, N, Q, q_over_Q, qKS_mask);
            } else {
                s_a[threadIdx.x] = sample_extract_a_u16(a_word, idx, coeff, N, Q, q_over_Q, qKS_mask);
            }
        }
        __syncthreads();

        if (active) {
            const uint32_t tile_count = min(blockDim.x, N - tile);
            for (uint32_t j = 0; j < tile_count; ++j) {
                const uint32_t ai = static_cast<uint32_t>(s_a[j]);
                const uint32_t d0 = ai & baseKS_mask;
                const uint32_t d1 = (ai >> baseKS_log2) & baseKS_mask;
                const uint32_t off = (tile + j) * stride_i + k;
                acc -= static_cast<uint32_t>(ksk_j0[off + d0 * outLen]);
                acc -= static_cast<uint32_t>(ksk_j1[off + d1 * outLen]);
            }
        }
        __syncthreads();
    }

    if (active) {
        const uint32_t u16 = acc & qKS_mask;
        const BasicInteger val = ks_u16_to_lwe<QDivisible>(u16, scale, qks, q);
        if (k == n) {
            out_b[t] = val;
        } else {
            out_a[static_cast<size_t>(t) * out_stride + k] = val;
        }
    }
}

inline bool KeySwitchBatchDevice(const GpuLweKeySwitchKeyU16& key, GpuLweKeySwitchWorkspaceU16& ws, size_t batch) {
    if (!key.d_ksk || !ws.d_in_a || !ws.d_in_b || !ws.d_out) {
        return false;
    }
    if (batch == 0 || batch > ws.maxBatch) {
        return false;
    }
    if (key.n != ws.n || key.N != ws.N) {
        return false;
    }
    if (key.digitCount != 2) {
        return false;
    }

    const uint32_t outLen = key.n + 1u;
    const uint32_t stride_i = key.baseKS * outLen;
    const uint32_t stride_j = key.N * stride_i;
    const uint32_t total = static_cast<uint32_t>(batch) * outLen;
    const dim3 block(256);
    const dim3 grid((total + block.x - 1) / block.x);
    kernel_KeySwitchU16<<<grid, block, 0, ws.stream>>>(ws.d_out, ws.d_in_a, ws.d_in_b, key.d_ksk, static_cast<uint32_t>(batch),
                                                       key.N, outLen, stride_i, stride_j, key.baseKS_log2, key.baseKS - 1u,
                                                       key.qKS_mask);
    PHANTOM_CHECK_CUDA_LAST();
    return true;
}

inline bool KeySwitchBatchDeviceFromExtractedRLWE(const GpuLweKeySwitchKeyU16& key, GpuLweKeySwitchWorkspaceU16& ws,
                                                  const uint64_t* d_rlwe_a, const uint64_t* d_rlwe_b,
                                                  const uint32_t* d_extract_info, size_t batch, uint64_t Q,
                                                  double q_over_Q) {
    if (!key.d_ksk || !ws.d_out || !d_rlwe_a || !d_rlwe_b || !d_extract_info) {
        return false;
    }
    if (batch == 0 || batch > ws.maxBatch || batch > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (key.n != ws.n || key.N != ws.N) {
        return false;
    }
    if (key.digitCount != 2) {
        return false;
    }

    const uint32_t outLen = key.n + 1u;
    const uint32_t stride_i = key.baseKS * outLen;
    const uint32_t stride_j = key.N * stride_i;
    const dim3 block(256);
    const dim3 grid((outLen + block.x - 1) / block.x, static_cast<uint32_t>(batch));
    kernel_SampleExtractKeySwitchU16<false><<<grid, block, 0, ws.stream>>>(
        ws.d_out, d_rlwe_a, d_rlwe_b, d_extract_info, nullptr, key.d_ksk, static_cast<uint32_t>(batch), key.N, outLen, stride_i,
        stride_j, key.baseKS_log2, key.baseKS - 1u, key.qKS_mask, Q, q_over_Q);
    PHANTOM_CHECK_CUDA_LAST();
    return true;
}

inline bool KeySwitchBatchDeviceFromMappedExtractedRLWE(const GpuLweKeySwitchKeyU16& key, GpuLweKeySwitchWorkspaceU16& ws,
                                                        const uint64_t* d_rlwe_a, const uint64_t* d_rlwe_b,
                                                        const uint32_t* d_extract_info, size_t batch, uint64_t Q,
                                                        double q_over_Q) {
    if (!ws.EnsureExtractCoeffMap() || ws.extractCoeffMapRows < key.N) {
        return false;
    }
    if (!key.d_ksk || !ws.d_out || !d_rlwe_a || !d_rlwe_b || !d_extract_info) {
        return false;
    }
    if (batch == 0 || batch > ws.maxBatch || batch > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (key.n != ws.n || key.N != ws.N) {
        return false;
    }
    if (key.digitCount != 2) {
        return false;
    }

    const uint32_t outLen = key.n + 1u;
    const uint32_t stride_i = key.baseKS * outLen;
    const uint32_t stride_j = key.N * stride_i;
    const dim3 block(256);
    const dim3 grid((outLen + block.x - 1) / block.x, static_cast<uint32_t>(batch));
    kernel_SampleExtractKeySwitchU16<true><<<grid, block, 0, ws.stream>>>(
        ws.d_out, d_rlwe_a, d_rlwe_b, d_extract_info, ws.d_extract_coeff_map, key.d_ksk, static_cast<uint32_t>(batch), key.N,
        outLen, stride_i, stride_j, key.baseKS_log2, key.baseKS - 1u, key.qKS_mask, Q, q_over_Q);
    PHANTOM_CHECK_CUDA_LAST();
    return true;
}

template <bool UsePreencodedMap, bool QDivisible>
inline bool KeySwitchBatchDeviceFromExtractedRLWEToLWEImpl(const GpuLweKeySwitchKeyU16& key, GpuLweKeySwitchWorkspaceU16& ws,
                                                          const uint64_t* d_rlwe_a, const uint64_t* d_rlwe_b,
                                                          const uint32_t* d_extract_info, BasicInteger* d_out_a,
                                                          BasicInteger* d_out_b, size_t batch, uint64_t Q,
                                                          double q_over_Q, uint64_t scale, uint64_t q_lwe,
                                                          uint32_t out_stride) {
    if (!key.d_ksk || !d_rlwe_a || !d_rlwe_b || !d_extract_info || !d_out_a || !d_out_b) {
        return false;
    }
    if (UsePreencodedMap && (!ws.EnsureExtractCoeffMap() || ws.extractCoeffMapRows < key.N)) {
        return false;
    }
    if (batch == 0 || batch > ws.maxBatch || batch > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (key.n != ws.n || key.N != ws.N || out_stride < key.n) {
        return false;
    }
    if (key.digitCount != 2) {
        return false;
    }

    const uint32_t outLen = key.n + 1u;
    const uint32_t stride_i = key.baseKS * outLen;
    const uint32_t stride_j = key.N * stride_i;
    const dim3 block(256);
    const dim3 grid((outLen + block.x - 1) / block.x, static_cast<uint32_t>(batch));
    kernel_SampleExtractKeySwitchToLWE<UsePreencodedMap, QDivisible><<<grid, block, 0, ws.stream>>>(
        d_out_a, d_out_b, d_rlwe_a, d_rlwe_b, d_extract_info, UsePreencodedMap ? ws.d_extract_coeff_map : nullptr,
        key.d_ksk, static_cast<uint32_t>(batch), key.N, key.n, outLen, stride_i, stride_j, key.baseKS_log2, key.baseKS - 1u,
        key.qKS_mask, Q, q_over_Q, scale, key.qKS, q_lwe, out_stride);
    PHANTOM_CHECK_CUDA_LAST();
    return true;
}

inline bool KeySwitchBatchDeviceFromExtractedRLWEToLWE(const GpuLweKeySwitchKeyU16& key, GpuLweKeySwitchWorkspaceU16& ws,
                                                       const uint64_t* d_rlwe_a, const uint64_t* d_rlwe_b,
                                                       const uint32_t* d_extract_info, BasicInteger* d_out_a,
                                                       BasicInteger* d_out_b, size_t batch, uint64_t Q,
                                                       double q_over_Q, bool q_divisible, uint64_t scale,
                                                       uint64_t q_lwe, uint32_t out_stride) {
    if (q_divisible) {
        return KeySwitchBatchDeviceFromExtractedRLWEToLWEImpl<false, true>(
            key, ws, d_rlwe_a, d_rlwe_b, d_extract_info, d_out_a, d_out_b, batch, Q, q_over_Q, scale, q_lwe, out_stride);
    }
    return KeySwitchBatchDeviceFromExtractedRLWEToLWEImpl<false, false>(
        key, ws, d_rlwe_a, d_rlwe_b, d_extract_info, d_out_a, d_out_b, batch, Q, q_over_Q, scale, q_lwe, out_stride);
}

inline bool KeySwitchBatchDeviceFromMappedExtractedRLWEToLWE(const GpuLweKeySwitchKeyU16& key,
                                                             GpuLweKeySwitchWorkspaceU16& ws,
                                                             const uint64_t* d_rlwe_a, const uint64_t* d_rlwe_b,
                                                             const uint32_t* d_extract_info, BasicInteger* d_out_a,
                                                             BasicInteger* d_out_b, size_t batch, uint64_t Q,
                                                             double q_over_Q, bool q_divisible, uint64_t scale,
                                                             uint64_t q_lwe, uint32_t out_stride) {
    if (q_divisible) {
        return KeySwitchBatchDeviceFromExtractedRLWEToLWEImpl<true, true>(
            key, ws, d_rlwe_a, d_rlwe_b, d_extract_info, d_out_a, d_out_b, batch, Q, q_over_Q, scale, q_lwe, out_stride);
    }
    return KeySwitchBatchDeviceFromExtractedRLWEToLWEImpl<true, false>(
        key, ws, d_rlwe_a, d_rlwe_b, d_extract_info, d_out_a, d_out_b, batch, Q, q_over_Q, scale, q_lwe, out_stride);
}

inline bool KeySwitchBatch(const GpuLweKeySwitchKeyU16& key, GpuLweKeySwitchWorkspaceU16& ws, size_t batch,
                           const uint16_t* h_in_a, const uint16_t* h_in_b, uint16_t* h_out) {
    if (!h_in_a || !h_in_b || !h_out) {
        return false;
    }
    if (!key.d_ksk || !ws.d_in_a || !ws.d_in_b || !ws.d_out) {
        return false;
    }
    if (batch == 0 || batch > ws.maxBatch) {
        return false;
    }
    if (key.n != ws.n || key.N != ws.N) {
        return false;
    }
    if (key.digitCount != 2) {
        return false;
    }

    const uint32_t outLen = key.n + 1u;
    const uint32_t stride_i = key.baseKS * outLen;
    const uint32_t stride_j = key.N * stride_i;
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(ws.d_in_a, h_in_a, batch * static_cast<size_t>(key.N) * sizeof(uint16_t), cudaMemcpyHostToDevice, ws.stream));
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(ws.d_in_b, h_in_b, batch * sizeof(uint16_t), cudaMemcpyHostToDevice, ws.stream));

    const uint32_t total = static_cast<uint32_t>(batch) * outLen;
    const dim3 block(256);
    const dim3 grid((total + block.x - 1) / block.x);
    kernel_KeySwitchU16<<<grid, block, 0, ws.stream>>>(ws.d_out, ws.d_in_a, ws.d_in_b, key.d_ksk, static_cast<uint32_t>(batch),
                                                       key.N, outLen, stride_i, stride_j, key.baseKS_log2, key.baseKS - 1u,
                                                       key.qKS_mask);
    PHANTOM_CHECK_CUDA_LAST();

    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(h_out, ws.d_out, total * sizeof(uint16_t), cudaMemcpyDeviceToHost, ws.stream));
    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(ws.stream));
    return true;
}

}  // namespace aes_example
