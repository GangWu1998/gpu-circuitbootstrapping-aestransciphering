#include "kernel.cuh"
#include "compact_ntt.cuh"
#include "modarith.cuh"

using namespace lbcrypto;

__device__ void
device_SignedDigitDecompose(BasicInteger* out, const BasicInteger* in,
                            const uint32_t QHalf, const NativeInteger::SignedNativeInt Q_int,
                            const int gBits, const int gBitsMaxBits,
                            const uint32_t digitsG2, const size_t N, size_t coeff_idx) {
    auto t = in[coeff_idx];
    auto d = static_cast<NativeInteger::SignedNativeInt>(t < QHalf ? t : t - Q_int);
    auto r = (d << gBitsMaxBits) >> gBitsMaxBits;
    d = (d - r) >> gBits;

    for (size_t i = 0; i < digitsG2; i += 2) {
        r = (d << gBitsMaxBits) >> gBitsMaxBits;
        d = (d - r) >> gBits;
        if (r < 0) r += Q_int;
        out[i * N + coeff_idx] = r;
    }
}

__global__ void kernel_SignedDigitDecompose(BasicInteger* output, const BasicInteger* input,
                                            BasicInteger Q, uint32_t baseG, size_t N) {
    const size_t poly_idx = blockIdx.x;
    const size_t batch_idx = blockIdx.y;
    const size_t digitsG2 = gridDim.x;

    for (size_t N_idx = threadIdx.x; N_idx < N; N_idx += blockDim.x) {
        uint32_t QHalf = Q >> 1;
        auto Q_int = static_cast<NativeInteger::SignedNativeInt>(Q);
        int gBits = __clz(static_cast<int32_t>(__brev(baseG)));
        int maxBits = sizeof(NativeInteger) * 8;
        int gBitsMaxBits = maxBits - gBits;

        auto t = input[batch_idx * 2 * N + (poly_idx & 1) * N + N_idx];
        auto d = static_cast<NativeInteger::SignedNativeInt>(t < QHalf ? t : t - Q_int);
        auto r = (d << gBitsMaxBits) >> gBitsMaxBits;
        d = (d - r) >> gBits;

        for (size_t i = 0; i < (poly_idx >> 1) + 1; i++) {
            r = (d << gBitsMaxBits) >> gBitsMaxBits;
            d = (d - r) >> gBits;
            r += static_cast<NativeInteger::SignedNativeInt>(
                (static_cast<BasicInteger>(r) >> (sizeof(BasicInteger) * 8 - 1)) * Q_int);
        }

        output[batch_idx * digitsG2 * N + poly_idx * N + N_idx] = r;
    }
}

// the main rounding operation used in ModSwitch (as described in Section 3 of
// https://eprint.iacr.org/2014/816) The idea is that Round(x) = 0.5 + Floor(x)
__device__ BasicInteger device_RoundqQ(const BasicInteger& v, const BasicInteger& q, const BasicInteger& Q) {
    const BasicInteger tmp = std::floor(0.5 + static_cast<double>(v) * static_cast<double>(q) / static_cast<double>(Q));
    return modFast(tmp, q);
}

__global__ void kernel_EvalAccCoreDM(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* RingGSWACCKey,
                                     const size_t N, const BasicInteger mod, const BasicInteger mont,
                                     const size_t digitsG2) {
    const BasicInteger Q = mod;
    const BasicInteger Q2 = Q << 1;
    const size_t poly_idx = blockIdx.x;
    for (size_t N_idx = threadIdx.x; N_idx < N; N_idx += blockDim.x) {
        BasicInteger tmp = modMulMontLazy(dct[N_idx], RingGSWACCKey[poly_idx * N + N_idx], Q, mont);

        for (size_t digitsG2_index = 1; digitsG2_index < digitsG2; ++digitsG2_index) {
            BasicInteger dct_coeff = dct[digitsG2_index * N + N_idx];
            tmp += modMulMontLazy(dct_coeff, RingGSWACCKey[digitsG2_index * 2 * N + poly_idx * N + N_idx], Q, mont);
            tmp = modFast(tmp, Q2);
        }

        acc[poly_idx * N + N_idx] = tmp;
    }
}

__global__ void kernel_EvalAccCoreDM_batch(BasicInteger* acc, const BasicInteger* dct, BasicInteger** acc_keys,
                                           const size_t N, const BasicInteger mod, const BasicInteger mont,
                                           const size_t digitsG2) {
    const BasicInteger Q = mod;
    const BasicInteger Q2 = Q << 1;
    const size_t poly_idx = blockIdx.x;
    const size_t batch_idx = blockIdx.y;
    const BasicInteger* acc_key = acc_keys[batch_idx];

    for (size_t N_idx = threadIdx.x; N_idx < N; N_idx += blockDim.x) {
        BasicInteger dct_first = dct[batch_idx * digitsG2 * N + N_idx];
        BasicInteger tmp = modMulMontLazy(dct_first, acc_key[poly_idx * N + N_idx], Q, mont);

        for (size_t digitsG2_idx = 1; digitsG2_idx < digitsG2; ++digitsG2_idx) {
            BasicInteger dct_coeff = dct[batch_idx * digitsG2 * N + digitsG2_idx * N + N_idx];
            tmp += modMulMontLazy(dct_coeff, acc_key[digitsG2_idx * 2 * N + poly_idx * N + N_idx], Q, mont);
            tmp = modFast(tmp, Q2);
        }

        acc[batch_idx * 2 * N + poly_idx * N + N_idx] = tmp;
    }
}

__global__ void kernel_EvalAccCoreCGGI(BasicInteger* acc, const BasicInteger* dct,
                                       const BasicInteger* d_ACCKey0, const BasicInteger* d_ACCKey1,
                                       const BasicInteger* monic_polys, size_t N,
                                       BasicInteger mod, BasicInteger mont,
                                       size_t digitsG2, size_t indexPos, size_t indexNeg) {
    const BasicInteger Q = mod;
    const BasicInteger Q2 = Q << 1;
    const size_t poly_idx = blockIdx.x;
    for (size_t N_idx = threadIdx.x; N_idx < N; N_idx += blockDim.x) {
        // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial
        const auto pos_monic_coeff = monic_polys[indexPos * N + N_idx];
        const auto neg_monic_coeff = monic_polys[indexNeg * N + N_idx];

        BasicInteger dct_first = dct[N_idx];
        BasicInteger tmp0 = modMulMontLazy(dct_first, d_ACCKey0[poly_idx * N + N_idx], Q, mont);
        BasicInteger tmp1 = modMulMontLazy(dct_first, d_ACCKey1[poly_idx * N + N_idx], Q, mont);

        for (uint32_t digitsG2_idx = 1; digitsG2_idx < digitsG2; ++digitsG2_idx) {
            BasicInteger dct_coeff = dct[digitsG2_idx * N + N_idx];

            tmp0 += modMulMontLazy(dct_coeff, d_ACCKey0[digitsG2_idx * 2 * N + poly_idx * N + N_idx], Q, mont);
            tmp0 = modFast(tmp0, Q2);

            tmp1 += modMulMontLazy(dct_coeff, d_ACCKey1[digitsG2_idx * 2 * N + poly_idx * N + N_idx], Q, mont);
            tmp1 = modFast(tmp1, Q2);
        }

        wide_type<BasicInteger> acc_wide = addWide(mulWide(tmp0, pos_monic_coeff),
                                                   mulWide(tmp1, neg_monic_coeff));

        acc[poly_idx * N + N_idx] = modWideMontLazy(acc_wide, Q, mont);
    }
}

__global__ void kernel_EvalAccCoreCGGI_binary(BasicInteger* acc, const BasicInteger* dct,
                                              const BasicInteger* d_ACCKey, const BasicInteger* monic_polys, size_t N,
                                              BasicInteger mod, BasicInteger mont,
                                              size_t digitsG2, size_t indexPos) {
    const BasicInteger Q = mod;
    const BasicInteger Q2 = Q << 1;
    const size_t poly_idx = blockIdx.x;
    for (size_t N_idx = threadIdx.x; N_idx < N; N_idx += blockDim.x) {
        // acc = acc + dct * ek * monomial
        const auto pos_monic_coeff = monic_polys[indexPos * N + N_idx];

        BasicInteger tmp = modMulMontLazy(dct[N_idx], d_ACCKey[poly_idx * N + N_idx], Q, mont);

        for (uint32_t digitsG2_idx = 1; digitsG2_idx < digitsG2; ++digitsG2_idx) {
            BasicInteger dct_coeff = dct[digitsG2_idx * N + N_idx];

            tmp += modMulMontLazy(dct_coeff, d_ACCKey[digitsG2_idx * 2 * N + poly_idx * N + N_idx], Q, mont);
            tmp = modFast(tmp, Q2);
        }

        acc[poly_idx * N + N_idx] = modMulMontLazy(tmp, pos_monic_coeff, Q, mont);
    }
}

__global__ void kernel_EvalAccCoreCGGI_batch(BasicInteger* acc, const BasicInteger* dct,
                                             const BasicInteger* acc_key0, const BasicInteger* acc_key1,
                                             const BasicInteger* monic_polys, size_t N,
                                             BasicInteger mod, BasicInteger mont, size_t digitsG2,
                                             const uint32_t* indexPos_batch, const uint32_t* indexNeg_batch) {
    const size_t poly_idx = blockIdx.x;
    const size_t batch_idx = blockIdx.y;

    uint32_t indexPos = indexPos_batch[batch_idx];
    uint32_t indexNeg = indexNeg_batch[batch_idx];

    const BasicInteger Q = mod;
    const BasicInteger Q2 = Q << 1;

    for (size_t N_idx = threadIdx.x; N_idx < N; N_idx += blockDim.x) {
        // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial
        const auto pos_monic_coeff = monic_polys[indexPos * N + N_idx];
        const auto neg_monic_coeff = monic_polys[indexNeg * N + N_idx];

        BasicInteger dct_first = dct[batch_idx * digitsG2 * N + N_idx];

        BasicInteger tmp0 = modMulMontLazy(dct_first, acc_key0[poly_idx * N + N_idx], Q, mont);
        BasicInteger tmp1 = modMulMontLazy(dct_first, acc_key1[poly_idx * N + N_idx], Q, mont);

        for (uint32_t digitsG2_idx = 1; digitsG2_idx < digitsG2; ++digitsG2_idx) {
            BasicInteger dct_coeff = dct[batch_idx * digitsG2 * N + digitsG2_idx * N + N_idx];

            tmp0 += modMulMontLazy(dct_coeff, acc_key0[digitsG2_idx * 2 * N + poly_idx * N + N_idx], Q, mont);
            tmp0 = modFast(tmp0, Q2);

            tmp1 += modMulMontLazy(dct_coeff, acc_key1[digitsG2_idx * 2 * N + poly_idx * N + N_idx], Q, mont);
            tmp1 = modFast(tmp1, Q2);
        }

        wide_type<BasicInteger> acc_wide = addWide(mulWide(tmp0, pos_monic_coeff),
                                                   mulWide(tmp1, neg_monic_coeff));

        acc[batch_idx * 2 * N + poly_idx * N + N_idx] = modWideMontLazy(acc_wide, Q, mont);
    }
}

__global__ void kernel_EvalAccCoreCGGI_binary_batch(BasicInteger* acc, const BasicInteger* dct,
                                                    const BasicInteger* acc_key,
                                                    const BasicInteger* monic_polys, size_t N,
                                                    BasicInteger mod, BasicInteger mont,
                                                    size_t digitsG2,
                                                    const uint32_t* indexPos_batch) {
    const size_t poly_idx = blockIdx.x;
    const size_t batch_idx = blockIdx.y;

    uint32_t indexPos = indexPos_batch[batch_idx];

    const BasicInteger Q = mod;
    const BasicInteger Q2 = Q << 1;

    for (size_t N_idx = threadIdx.x; N_idx < N; N_idx += blockDim.x) {
        // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial
        const auto pos_monic_coeff = monic_polys[indexPos * N + N_idx];

        BasicInteger tmp = modMulMontLazy(dct[batch_idx * digitsG2 * N + N_idx], acc_key[poly_idx * N + N_idx],
                                          Q, mont);

        for (uint32_t digitsG2_idx = 1; digitsG2_idx < digitsG2; ++digitsG2_idx) {
            BasicInteger dct_coeff = dct[batch_idx * digitsG2 * N + digitsG2_idx * N + N_idx];

            tmp += modMulMontLazy(dct_coeff, acc_key[digitsG2_idx * 2 * N + poly_idx * N + N_idx], Q, mont);
            tmp = modFast(tmp, Q2);
        }

        acc[batch_idx * 2 * N + poly_idx * N + N_idx] = modMulMontLazy(tmp, pos_monic_coeff, Q, mont);
    }
}

__global__ void kernel_element_add(BasicInteger* output, const BasicInteger* input1, const BasicInteger* input2,
                                   size_t dim, BasicInteger mod) {
    const size_t global_block_idx = blockIdx.y * gridDim.x + blockIdx.x;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x)
        output[global_block_idx * dim + i] = modFast(input1[global_block_idx * dim + i] + input2[global_block_idx * dim + i], mod);
}

__global__ void kernel_automorphism_modSwitch(BasicInteger* output, const BasicInteger* input,
                                              BasicInteger qKS, BasicInteger Q, size_t logN) {
    size_t N = 1 << logN;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t k = 2 * N - 1;
    size_t jk = j * k;
    size_t mask = N - 1;
    BasicInteger tmp = ((jk >> logN) & 0x1) ? Q - input[blockIdx.y * 2 * N + j] : input[blockIdx.y * 2 * N + j];
    output[blockIdx.y * 2 * N + (jk & mask)] = device_RoundqQ(tmp, qKS, Q);
}

__global__ void kernel_scale_by_p(BasicInteger* ct_Q, const BasicInteger* ct_PQ, size_t dim,
                                  BasicInteger Q, BasicInteger PQ) {
    const size_t global_block_idx = blockIdx.y * gridDim.x + blockIdx.x;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x)
        ct_Q[global_block_idx * dim + i] = device_RoundqQ(ct_PQ[global_block_idx * dim + i], Q, PQ);
}

__global__ void kernel_LWEKeySwitch_modSwitch(BasicInteger* res_ct_A,
                                              const BasicInteger* ct_A, const BasicInteger* ksk_A,
                                              size_t n, size_t N,
                                              BasicInteger qAfter, BasicInteger qKS, BasicInteger log_baseKS,
                                              size_t digitCount) {
    BasicInteger baseKS = 1 << log_baseKS;

    BasicInteger sum_A = 0;

    for (size_t n_idx = blockIdx.x * blockDim.x + threadIdx.x; n_idx < n; n_idx += gridDim.x * blockDim.x) {
        for (size_t N_idx = 0; N_idx < N; ++N_idx) {
            BasicInteger ct_Ai = ct_A[blockIdx.y * 2 * N + N_idx];
            for (size_t j = 0; j < digitCount; ++j) {
                const auto a0 = (ct_Ai >> (j * log_baseKS)) & (baseKS - 1);
                sum_A += ksk_A[N_idx * baseKS * digitCount * n + a0 * digitCount * n + j * n + n_idx];
                sum_A = modFast(sum_A, qKS);
            }
        }
        res_ct_A[blockIdx.y * n + n_idx] = device_RoundqQ(qKS - sum_A, qAfter, qKS);
    }
}
