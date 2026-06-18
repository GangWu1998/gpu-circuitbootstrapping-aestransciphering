#include "ntt.cuh"
#include "butterfly.cuh"
#include "uintmodmath.cuh"

using namespace phantom::arith;

/** forward NTT transformation, with N (num of operands) up to 2048,
 * to ensure all operation completed in one block.
 * @param[inout] inout The value to operate and the returned result
 * @param[in] twiddles The pre-computated forward NTT table
 * @param[in] mod The coeff modulus value
 * @param[in] n The poly degreee
 * @param[in] logn The logarithm of n
 * @param[in] numOfGroups
 * @param[in] iter The current iteration in forward NTT transformation
 */
__global__ void inplace_fnwt_radix2(uint64_t *inout,
                                    const uint64_t *twiddles,
                                    const uint64_t *twiddles_shoup,
                                    const DModulus *modulus,
                                    size_t coeff_mod_size,
                                    size_t start_mod_idx,
                                    size_t n) {
    extern __shared__ uint64_t buffer[];

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n / 2 * coeff_mod_size; // deal with 2 data per thread
         i += blockDim.x * gridDim.x) {
        size_t mod_idx = i / (n / 2) + start_mod_idx;
        size_t tid = i % (n / 2);

        // modulus
        const DModulus *modulus_table = modulus;
        uint64_t mod = modulus_table[mod_idx].value();
        uint64_t mod2 = mod << 1;

        size_t pairsInGroup;
        size_t k, j, glbIdx, bufIdx; // k = psi_step
        uint64_t samples[2];

        for (size_t numOfGroups = 1; numOfGroups < n; numOfGroups <<= 1) {
            pairsInGroup = n / numOfGroups / 2;

            k = tid / pairsInGroup;
            j = tid % pairsInGroup;
            glbIdx = 2 * k * pairsInGroup + j;
            bufIdx = glbIdx % n;
            glbIdx += mod_idx * n;

            uint64_t psi = twiddles[numOfGroups + k + n * mod_idx];
            uint64_t psi_shoup = twiddles_shoup[numOfGroups + k + n * mod_idx];

            if (numOfGroups == 1) {
                samples[0] = inout[glbIdx];
                samples[1] = inout[glbIdx + pairsInGroup];
            } else {
                samples[0] = buffer[bufIdx];
                samples[1] = buffer[bufIdx + pairsInGroup];
            }
            ct_butterfly(samples[0], samples[1], psi, psi_shoup, mod);

            if (numOfGroups == n >> 1) {
                csub_q(samples[0], mod2);
                csub_q(samples[0], mod);
                csub_q(samples[1], mod2);
                csub_q(samples[1], mod);
                inout[glbIdx] = samples[0];
                inout[glbIdx + pairsInGroup] = samples[1];
            } else {
                buffer[bufIdx] = samples[0];
                buffer[bufIdx + pairsInGroup] = samples[1];
                __syncthreads();
            }
        }
    }
}

__global__ void inplace_fnwt_radix2_opt(uint64_t *inout,
                                        const uint64_t *twiddles,
                                        const uint64_t *twiddles_shoup,
                                        const DModulus *modulus,
                                        const size_t n) {
    extern __shared__ uint64_t buffer[];

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t mod_idx = i / (n / 2);
    const size_t tid = i % (n / 2);

    // modulus
    const DModulus *modulus_table = modulus;
    const uint64_t mod = modulus_table[mod_idx].value();
    const uint64_t mod2 = mod << 1;

    constexpr size_t numOfGroups = 1;
    const size_t pairsInGroup = n / numOfGroups / 2;
    const size_t k = tid / pairsInGroup; // k = psi_step
    const uint64_t psi = twiddles[numOfGroups + k + n * mod_idx];
    const uint64_t psi_shoup = twiddles_shoup[numOfGroups + k + n * mod_idx];
    const size_t j = tid % pairsInGroup;
    size_t glbIdx = 2 * k * pairsInGroup + j;
    const size_t bufIdx = glbIdx % n;
    glbIdx += mod_idx * n;
    uint64_t samples0 = inout[glbIdx];
    uint64_t samples1 = inout[glbIdx + pairsInGroup];
    ct_butterfly(samples0, samples1, psi, psi_shoup, mod);
    buffer[bufIdx] = samples0;
    buffer[bufIdx + pairsInGroup] = samples1;
    __syncthreads();

    for (size_t numOfGroups = 2; numOfGroups < n / 2; numOfGroups <<= 1) {
        const size_t pairsInGroup = n / numOfGroups / 2;
        const size_t k = tid / pairsInGroup; // k = psi_step
        const uint64_t psi = twiddles[numOfGroups + k + n * mod_idx];
        const uint64_t psi_shoup = twiddles_shoup[numOfGroups + k + n * mod_idx];
        const size_t j = tid % pairsInGroup;
        const size_t glbIdx = 2 * k * pairsInGroup + j;
        const size_t bufIdx = glbIdx % n;
        uint64_t samples0 = buffer[bufIdx];
        uint64_t samples1 = buffer[bufIdx + pairsInGroup];
        ct_butterfly(samples0, samples1, psi, psi_shoup, mod);
        buffer[bufIdx] = samples0;
        buffer[bufIdx + pairsInGroup] = samples1;
        __syncthreads();
    }

    const size_t numOfGroups_last = n / 2;
    const size_t pairsInGroup_last = n / numOfGroups_last / 2;
    const size_t k_last = tid / pairsInGroup_last; // k = psi_step
    const uint64_t psi_last = twiddles[numOfGroups_last + k_last + n * mod_idx];
    const uint64_t psi_last_shoup = twiddles_shoup[numOfGroups_last + k_last + n * mod_idx];
    const size_t j_last = tid % pairsInGroup_last;
    size_t glbIdx_last = 2 * k_last * pairsInGroup_last + j_last;
    const size_t bufIdx_last = glbIdx_last % n;
    uint64_t samples0_last = buffer[bufIdx_last];
    uint64_t samples1_last = buffer[bufIdx_last + pairsInGroup_last];
    ct_butterfly(samples0_last, samples1_last, psi_last, psi_last_shoup, mod);
    csub_q(samples0_last, mod2);
    csub_q(samples0_last, mod);
    csub_q(samples1_last, mod2);
    csub_q(samples1_last, mod);
    glbIdx_last += mod_idx * n;
    inout[glbIdx_last] = samples0_last;
    inout[glbIdx_last + pairsInGroup_last] = samples1_last;
}

void fnwt_1d(uint64_t *inout,
             const uint64_t *twiddles,
             const uint64_t *twiddles_shoup,
             const DModulus *modulus,
             size_t dim,
             size_t coeff_modulus_size,
             size_t start_modulus_idx,
             const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);

    inplace_fnwt_radix2<<<coeff_modulus_size, dim / 2, per_block_memory, stream>>>(
            inout,
            twiddles,
            twiddles_shoup,
            modulus,
            coeff_modulus_size,
            start_modulus_idx,
            dim);
}

void fnwt_1d_opt(uint64_t *inout,
                 const uint64_t *twiddles,
                 const uint64_t *twiddles_shoup,
                 const DModulus *modulus,
                 size_t dim,
                 size_t coeff_modulus_size,
                 size_t start_modulus_idx,
                 const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);

    inplace_fnwt_radix2_opt<<<coeff_modulus_size, dim / 2, per_block_memory, stream>>>(
            inout,
            twiddles,
            twiddles_shoup,
            modulus,
            dim);
}

/** backward NTT transformation, with N (num of operands) up to 2048,
 * to ensure all operation completed in one block.
 * @param[inout] inout The value to operate and the returned result
 * @param[in] inverse_twiddles The pre-computated backward NTT table
 * @param[in] mod The coeff modulus value
 * @param[in] n The poly degreee
 * @param[in] logn The logarithm of n
 * @param[in] numOfGroups
 * @param[in] iter The current iteration in backward NTT transformation
 */
__global__ void inplace_inwt_radix2(uint64_t *inout,
                                    const uint64_t *itwiddles,
                                    const uint64_t *itwiddles_shoup,
                                    const DModulus *modulus,
                                    const uint64_t *scalar, const uint64_t *scalar_shoup,
                                    size_t coeff_mod_size,
                                    size_t start_mod_idx,
                                    size_t n) {
    extern __shared__ uint64_t buffer[];

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n / 2 * coeff_mod_size;
         i += blockDim.x * gridDim.x) {
        size_t mod_idx = i / (n / 2) + start_mod_idx;
        size_t tid = i % (n / 2);

        size_t pairsInGroup;
        size_t k, j, glbIdx, bufIdx;
        uint64_t samples[2];

        const DModulus *modulus_table = modulus;
        uint64_t mod = modulus_table[mod_idx].value();

        const uint64_t scalar_ = scalar[mod_idx];
        const uint64_t scalar_shoup_ = scalar_shoup[mod_idx];

        for (size_t _numOfGroups = n / 2; _numOfGroups >= 1; _numOfGroups >>= 1) {
            pairsInGroup = n / _numOfGroups / 2;
            k = tid / pairsInGroup;
            j = tid % pairsInGroup;
            glbIdx = 2 * k * pairsInGroup + j;
            bufIdx = glbIdx % n;
            glbIdx += mod_idx * n;
            uint64_t psi = itwiddles[_numOfGroups + k + mod_idx * n];
            uint64_t psi_shoup = itwiddles_shoup[_numOfGroups + k + mod_idx * n];
            if (_numOfGroups == n >> 1) {
                samples[0] = inout[glbIdx];
                samples[1] = inout[glbIdx + pairsInGroup];
            } else {
                samples[0] = buffer[bufIdx];
                samples[1] = buffer[bufIdx + pairsInGroup];
            }

            gs_butterfly(samples[0], samples[1], psi, psi_shoup, mod);

            if (_numOfGroups == 1) {
                // final reduction
                csub_q(samples[0], mod);
                csub_q(samples[1], mod);
            }

            if (_numOfGroups == 1) {
                samples[0] = multiply_and_reduce_shoup(samples[0], scalar_, scalar_shoup_, mod);
                inout[glbIdx] = samples[0];
                inout[glbIdx + pairsInGroup] = samples[1];
            } else {
                buffer[bufIdx] = samples[0];
                buffer[bufIdx + pairsInGroup] = samples[1];
                __syncthreads();
            }
        }
    }
}

__global__ void inplace_inwt_radix2_outofplace(uint64_t *out, const uint64_t *in,
                                               const uint64_t *itwiddles, const uint64_t *itwiddles_shoup,
                                               const DModulus *modulus, const uint64_t *scalar,
                                               const uint64_t *scalar_shoup, size_t coeff_mod_size, size_t start_mod_idx,
                                               size_t n) {
    extern __shared__ uint64_t buffer[];

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 2 * coeff_mod_size; i += blockDim.x * gridDim.x) {
        size_t mod_idx = i / (n / 2) + start_mod_idx;
        size_t tid     = i % (n / 2);

        size_t pairsInGroup;
        size_t k, j, glbIdx, bufIdx;
        uint64_t samples[2];

        const DModulus *modulus_table = modulus;
        uint64_t mod                 = modulus_table[mod_idx].value();

        const uint64_t scalar_       = scalar[mod_idx];
        const uint64_t scalar_shoup_ = scalar_shoup[mod_idx];

        const uint64_t *in_poly = in + mod_idx * n;

        for (size_t _numOfGroups = n / 2; _numOfGroups >= 1; _numOfGroups >>= 1) {
            pairsInGroup = n / _numOfGroups / 2;
            k            = tid / pairsInGroup;
            j            = tid % pairsInGroup;
            glbIdx       = 2 * k * pairsInGroup + j;
            bufIdx       = glbIdx % n;

            const size_t local_glbIdx = glbIdx;
            glbIdx += mod_idx * n;

            uint64_t psi       = itwiddles[_numOfGroups + k + mod_idx * n];
            uint64_t psi_shoup = itwiddles_shoup[_numOfGroups + k + mod_idx * n];
            if (_numOfGroups == n >> 1) {
                samples[0] = in_poly[local_glbIdx];
                samples[1] = in_poly[local_glbIdx + pairsInGroup];
            }
            else {
                samples[0] = buffer[bufIdx];
                samples[1] = buffer[bufIdx + pairsInGroup];
            }

            gs_butterfly(samples[0], samples[1], psi, psi_shoup, mod);

            if (_numOfGroups == 1) {
                // final reduction
                csub_q(samples[0], mod);
                csub_q(samples[1], mod);
            }

            if (_numOfGroups == 1) {
                samples[0] = multiply_and_reduce_shoup(samples[0], scalar_, scalar_shoup_, mod);
                out[glbIdx] = samples[0];
                out[glbIdx + pairsInGroup] = samples[1];
            }
            else {
                buffer[bufIdx] = samples[0];
                buffer[bufIdx + pairsInGroup] = samples[1];
                __syncthreads();
            }
        }
    }
}

__global__ void inplace_inwt_radix2_permute(uint64_t *out, const uint64_t *in, const uint32_t *d_map,
                                            const uint64_t *itwiddles, const uint64_t *itwiddles_shoup,
                                            const DModulus *modulus, const uint64_t *scalar,
                                            const uint64_t *scalar_shoup, size_t coeff_mod_size, size_t start_mod_idx,
                                            size_t n) {
    extern __shared__ uint64_t buffer[];

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 2 * coeff_mod_size; i += blockDim.x * gridDim.x) {
        size_t mod_idx = i / (n / 2) + start_mod_idx;
        size_t tid     = i % (n / 2);

        size_t pairsInGroup;
        size_t k, j, glbIdx, bufIdx;
        uint64_t samples[2];

        const DModulus *modulus_table = modulus;
        uint64_t mod                 = modulus_table[mod_idx].value();

        const uint64_t scalar_       = scalar[mod_idx];
        const uint64_t scalar_shoup_ = scalar_shoup[mod_idx];

        const uint64_t *in_poly = in + mod_idx * n;

        for (size_t _numOfGroups = n / 2; _numOfGroups >= 1; _numOfGroups >>= 1) {
            pairsInGroup = n / _numOfGroups / 2;
            k            = tid / pairsInGroup;
            j            = tid % pairsInGroup;
            glbIdx       = 2 * k * pairsInGroup + j;
            bufIdx       = glbIdx % n;

            const size_t local_glbIdx = glbIdx;
            glbIdx += mod_idx * n;

            uint64_t psi       = itwiddles[_numOfGroups + k + mod_idx * n];
            uint64_t psi_shoup = itwiddles_shoup[_numOfGroups + k + mod_idx * n];

            if (_numOfGroups == n >> 1) {
                const size_t local0 = local_glbIdx;
                const size_t local1 = local_glbIdx + pairsInGroup;

                uint32_t src0 = d_map[local0];
                bool negate0  = false;
                if (src0 >= n) {
                    src0 -= static_cast<uint32_t>(n);
                    negate0 = true;
                }
                uint64_t val0 = in_poly[src0];
                if (negate0 && val0) {
                    val0 = mod - val0;
                }

                uint32_t src1 = d_map[local1];
                bool negate1  = false;
                if (src1 >= n) {
                    src1 -= static_cast<uint32_t>(n);
                    negate1 = true;
                }
                uint64_t val1 = in_poly[src1];
                if (negate1 && val1) {
                    val1 = mod - val1;
                }

                samples[0] = val0;
                samples[1] = val1;
            }
            else {
                samples[0] = buffer[bufIdx];
                samples[1] = buffer[bufIdx + pairsInGroup];
            }

            gs_butterfly(samples[0], samples[1], psi, psi_shoup, mod);

            if (_numOfGroups == 1) {
                // final reduction
                csub_q(samples[0], mod);
                csub_q(samples[1], mod);
            }

            if (_numOfGroups == 1) {
                samples[0] = multiply_and_reduce_shoup(samples[0], scalar_, scalar_shoup_, mod);
                out[glbIdx] = samples[0];
                out[glbIdx + pairsInGroup] = samples[1];
            }
            else {
                buffer[bufIdx] = samples[0];
                buffer[bufIdx + pairsInGroup] = samples[1];
                __syncthreads();
            }
        }
    }
}

void inwt_1d(uint64_t *inout,
             const uint64_t *itwiddles, const uint64_t *itwiddles_shoup, const DModulus *modulus,
             const uint64_t *scalar, const uint64_t *scalar_shoup,
             size_t dim, size_t coeff_modulus_size, size_t start_modulus_idx,
             const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);

    inplace_inwt_radix2<<<coeff_modulus_size, dim / 2, per_block_memory, stream>>>(
            inout,
            itwiddles,
            itwiddles_shoup,
            modulus,
            scalar, scalar_shoup,
            coeff_modulus_size,
            start_modulus_idx,
            dim);
}

void inwt_1d_opt(uint64_t *inout,
                 const uint64_t *itwiddles, const uint64_t *itwiddles_shoup, const DModulus *modulus,
                 const uint64_t *scalar, const uint64_t *scalar_shoup,
                 size_t dim, size_t coeff_modulus_size, size_t start_modulus_idx,
                 const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);

    inplace_inwt_radix2<<<coeff_modulus_size, dim / 2, per_block_memory, stream>>>(
            inout,
            itwiddles,
            itwiddles_shoup,
            modulus,
            scalar, scalar_shoup,
            coeff_modulus_size,
            start_modulus_idx,
            dim);
}

void inwt_1d_opt_outofplace(uint64_t *out, const uint64_t *in, const uint64_t *itwiddles, const uint64_t *itwiddles_shoup,
                            const DModulus *modulus, const uint64_t *scalar, const uint64_t *scalar_shoup, size_t dim,
                            size_t coeff_modulus_size, size_t start_modulus_idx, const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);

    inplace_inwt_radix2_outofplace<<<coeff_modulus_size, dim / 2, per_block_memory, stream>>>(
            out,
            in,
            itwiddles,
            itwiddles_shoup,
            modulus,
            scalar, scalar_shoup,
            coeff_modulus_size,
            start_modulus_idx,
            dim);
}

void inwt_1d_opt_permute(uint64_t *out, const uint64_t *in, const uint32_t *d_map, const uint64_t *itwiddles,
                         const uint64_t *itwiddles_shoup, const DModulus *modulus, const uint64_t *scalar,
                         const uint64_t *scalar_shoup, size_t dim, size_t coeff_modulus_size, size_t start_modulus_idx,
                         const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);

    inplace_inwt_radix2_permute<<<coeff_modulus_size, dim / 2, per_block_memory, stream>>>(
            out,
            in,
            d_map,
            itwiddles,
            itwiddles_shoup,
            modulus,
            scalar, scalar_shoup,
            coeff_modulus_size,
            start_modulus_idx,
            dim);
}

__global__ void inplace_fnwt_radix2_opt_batched(uint64_t *inout,
                                                const uint64_t *twiddles,
                                                const uint64_t *twiddles_shoup,
                                                const DModulus *modulus,
                                                size_t n,
                                                size_t batch) {
    extern __shared__ uint64_t buffer[];

    const size_t mod_idx = blockIdx.x;
    const size_t tid     = threadIdx.x;
    if (mod_idx >= batch || tid >= n / 2) {
        return;
    }

    const uint64_t mod  = modulus[0].value();
    const uint64_t mod2 = mod << 1;

    constexpr size_t numOfGroups0 = 1;
    const size_t pairsInGroup0    = n / 2;
    const size_t k0               = tid / pairsInGroup0;
    const uint64_t psi0           = twiddles[numOfGroups0 + k0];
    const uint64_t psi0_shoup     = twiddles_shoup[numOfGroups0 + k0];
    const size_t j0               = tid % pairsInGroup0;
    size_t glbIdx0                = 2 * k0 * pairsInGroup0 + j0;
    const size_t bufIdx0          = glbIdx0;
    glbIdx0 += mod_idx * n;
    uint64_t samples0 = inout[glbIdx0];
    uint64_t samples1 = inout[glbIdx0 + pairsInGroup0];
    ct_butterfly(samples0, samples1, psi0, psi0_shoup, mod);
    buffer[bufIdx0]                 = samples0;
    buffer[bufIdx0 + pairsInGroup0] = samples1;
    __syncthreads();

    for (size_t numOfGroups = 2; numOfGroups < n / 2; numOfGroups <<= 1) {
        const size_t pairsInGroup = n / numOfGroups / 2;
        const size_t k            = tid / pairsInGroup;
        const uint64_t psi        = twiddles[numOfGroups + k];
        const uint64_t psi_shoup  = twiddles_shoup[numOfGroups + k];
        const size_t j            = tid % pairsInGroup;
        const size_t glbIdx       = 2 * k * pairsInGroup + j;
        const size_t bufIdx       = glbIdx % n;
        uint64_t a0               = buffer[bufIdx];
        uint64_t a1               = buffer[bufIdx + pairsInGroup];
        ct_butterfly(a0, a1, psi, psi_shoup, mod);
        buffer[bufIdx]                 = a0;
        buffer[bufIdx + pairsInGroup]  = a1;
        __syncthreads();
    }

    const size_t numOfGroups_last     = n / 2;
    const size_t pairsInGroup_last    = n / numOfGroups_last / 2;
    const size_t k_last               = tid / pairsInGroup_last;
    const uint64_t psi_last           = twiddles[numOfGroups_last + k_last];
    const uint64_t psi_last_shoup     = twiddles_shoup[numOfGroups_last + k_last];
    const size_t j_last               = tid % pairsInGroup_last;
    size_t glbIdx_last                = 2 * k_last * pairsInGroup_last + j_last;
    const size_t bufIdx_last          = glbIdx_last % n;
    uint64_t samples0_last            = buffer[bufIdx_last];
    uint64_t samples1_last            = buffer[bufIdx_last + pairsInGroup_last];
    ct_butterfly(samples0_last, samples1_last, psi_last, psi_last_shoup, mod);
    csub_q(samples0_last, mod2);
    csub_q(samples0_last, mod);
    csub_q(samples1_last, mod2);
    csub_q(samples1_last, mod);
    glbIdx_last += mod_idx * n;
    inout[glbIdx_last]                      = samples0_last;
    inout[glbIdx_last + pairsInGroup_last]  = samples1_last;
}

__global__ void inplace_inwt_radix2_batched(uint64_t *inout,
                                            const uint64_t *itwiddles,
                                            const uint64_t *itwiddles_shoup,
                                            const DModulus *modulus,
                                            const uint64_t *scalar,
                                            const uint64_t *scalar_shoup,
                                            size_t n,
                                            size_t batch) {
    extern __shared__ uint64_t buffer[];

    const size_t mod_idx = blockIdx.x;
    const size_t tid     = threadIdx.x;
    if (mod_idx >= batch || tid >= n / 2) {
        return;
    }

    const uint64_t mod         = modulus[0].value();
    const uint64_t scalar_     = scalar[0];
    const uint64_t scalar_sh   = scalar_shoup[0];

    size_t pairsInGroup;
    size_t k, j, glbIdx, bufIdx;
    uint64_t samples[2];

    for (size_t numOfGroups = n / 2; numOfGroups >= 1; numOfGroups >>= 1) {
        pairsInGroup = n / numOfGroups / 2;
        k            = tid / pairsInGroup;
        j            = tid % pairsInGroup;
        glbIdx       = 2 * k * pairsInGroup + j;
        bufIdx       = glbIdx % n;

        const uint64_t psi       = itwiddles[numOfGroups + k];
        const uint64_t psi_shoup = itwiddles_shoup[numOfGroups + k];

        if (numOfGroups == n >> 1) {
            const size_t base = mod_idx * n;
            samples[0]        = inout[base + glbIdx];
            samples[1]        = inout[base + glbIdx + pairsInGroup];
        }
        else {
            samples[0] = buffer[bufIdx];
            samples[1] = buffer[bufIdx + pairsInGroup];
        }

        gs_butterfly(samples[0], samples[1], psi, psi_shoup, mod);

        if (numOfGroups == 1) {
            csub_q(samples[0], mod);
            csub_q(samples[1], mod);
            samples[0] = multiply_and_reduce_shoup(samples[0], scalar_, scalar_sh, mod);
            const size_t base = mod_idx * n;
            inout[base + glbIdx]              = samples[0];
            inout[base + glbIdx + pairsInGroup] = samples[1];
        }
        else {
            buffer[bufIdx]                = samples[0];
            buffer[bufIdx + pairsInGroup] = samples[1];
            __syncthreads();
        }
    }
}

__global__ void inplace_inwt_radix2_outofplace_batched(uint64_t *out, const uint64_t *in,
                                                       const uint64_t *itwiddles, const uint64_t *itwiddles_shoup,
                                                       const DModulus *modulus, const uint64_t *scalar,
                                                       const uint64_t *scalar_shoup, size_t n, size_t batch) {
    extern __shared__ uint64_t buffer[];

    const size_t mod_idx = blockIdx.x;
    const size_t tid     = threadIdx.x;
    if (mod_idx >= batch || tid >= n / 2) {
        return;
    }

    const uint64_t mod        = modulus[0].value();
    const uint64_t scalar_    = scalar[0];
    const uint64_t scalar_sh  = scalar_shoup[0];
    const uint64_t *in_poly   = in + mod_idx * n;

    size_t pairsInGroup;
    size_t k, j, glbIdx, bufIdx;
    uint64_t samples[2];

    for (size_t numOfGroups = n / 2; numOfGroups >= 1; numOfGroups >>= 1) {
        pairsInGroup = n / numOfGroups / 2;
        k            = tid / pairsInGroup;
        j            = tid % pairsInGroup;
        glbIdx       = 2 * k * pairsInGroup + j;
        bufIdx       = glbIdx % n;

        const uint64_t psi       = itwiddles[numOfGroups + k];
        const uint64_t psi_shoup = itwiddles_shoup[numOfGroups + k];

        if (numOfGroups == n >> 1) {
            samples[0] = in_poly[glbIdx];
            samples[1] = in_poly[glbIdx + pairsInGroup];
        }
        else {
            samples[0] = buffer[bufIdx];
            samples[1] = buffer[bufIdx + pairsInGroup];
        }

        gs_butterfly(samples[0], samples[1], psi, psi_shoup, mod);

        if (numOfGroups == 1) {
            csub_q(samples[0], mod);
            csub_q(samples[1], mod);
            samples[0] = multiply_and_reduce_shoup(samples[0], scalar_, scalar_sh, mod);
            const size_t base = mod_idx * n;
            out[base + glbIdx]              = samples[0];
            out[base + glbIdx + pairsInGroup] = samples[1];
        }
        else {
            buffer[bufIdx]                = samples[0];
            buffer[bufIdx + pairsInGroup] = samples[1];
            __syncthreads();
        }
    }
}

__global__ void inplace_inwt_radix2_permute_batched(uint64_t *out, const uint64_t *in, const uint32_t *d_map,
                                                    const uint64_t *itwiddles, const uint64_t *itwiddles_shoup,
                                                    const DModulus *modulus, const uint64_t *scalar,
                                                    const uint64_t *scalar_shoup, size_t n, size_t batch) {
    extern __shared__ uint64_t buffer[];

    const size_t mod_idx = blockIdx.x;
    const size_t tid     = threadIdx.x;
    if (mod_idx >= batch || tid >= n / 2) {
        return;
    }

    const uint64_t mod        = modulus[0].value();
    const uint64_t scalar_    = scalar[0];
    const uint64_t scalar_sh  = scalar_shoup[0];
    const uint64_t *in_poly   = in + mod_idx * n;

    size_t pairsInGroup;
    size_t k, j, glbIdx, bufIdx;
    uint64_t samples[2];

    for (size_t numOfGroups = n / 2; numOfGroups >= 1; numOfGroups >>= 1) {
        pairsInGroup = n / numOfGroups / 2;
        k            = tid / pairsInGroup;
        j            = tid % pairsInGroup;
        glbIdx       = 2 * k * pairsInGroup + j;
        bufIdx       = glbIdx % n;

        const uint64_t psi       = itwiddles[numOfGroups + k];
        const uint64_t psi_shoup = itwiddles_shoup[numOfGroups + k];

        if (numOfGroups == n >> 1) {
            const size_t local0 = glbIdx;
            const size_t local1 = glbIdx + pairsInGroup;

            uint32_t src0 = d_map[local0];
            bool negate0  = false;
            if (src0 >= n) {
                src0 -= static_cast<uint32_t>(n);
                negate0 = true;
            }
            uint64_t val0 = in_poly[src0];
            if (negate0 && val0) {
                val0 = mod - val0;
            }

            uint32_t src1 = d_map[local1];
            bool negate1  = false;
            if (src1 >= n) {
                src1 -= static_cast<uint32_t>(n);
                negate1 = true;
            }
            uint64_t val1 = in_poly[src1];
            if (negate1 && val1) {
                val1 = mod - val1;
            }

            samples[0] = val0;
            samples[1] = val1;
        }
        else {
            samples[0] = buffer[bufIdx];
            samples[1] = buffer[bufIdx + pairsInGroup];
        }

        gs_butterfly(samples[0], samples[1], psi, psi_shoup, mod);

        if (numOfGroups == 1) {
            csub_q(samples[0], mod);
            csub_q(samples[1], mod);
            samples[0] = multiply_and_reduce_shoup(samples[0], scalar_, scalar_sh, mod);
            const size_t base = mod_idx * n;
            out[base + glbIdx]              = samples[0];
            out[base + glbIdx + pairsInGroup] = samples[1];
        }
        else {
            buffer[bufIdx]                = samples[0];
            buffer[bufIdx + pairsInGroup] = samples[1];
            __syncthreads();
        }
    }
}

void fnwt_1d_opt_batched(uint64_t *inout, const uint64_t *twiddles, const uint64_t *twiddles_shoup,
                         const DModulus *modulus, size_t dim, size_t batch, const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);
    inplace_fnwt_radix2_opt_batched<<<batch, dim / 2, per_block_memory, stream>>>(
            inout, twiddles, twiddles_shoup, modulus, dim, batch);
}

void inwt_1d_opt_batched(uint64_t *inout, const uint64_t *itwiddles, const uint64_t *itwiddles_shoup,
                         const DModulus *modulus, const uint64_t *scalar, const uint64_t *scalar_shoup,
                         size_t dim, size_t batch, const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);
    inplace_inwt_radix2_batched<<<batch, dim / 2, per_block_memory, stream>>>(
            inout, itwiddles, itwiddles_shoup, modulus, scalar, scalar_shoup, dim, batch);
}

void inwt_1d_opt_outofplace_batched(uint64_t *out, const uint64_t *in, const uint64_t *itwiddles,
                                    const uint64_t *itwiddles_shoup, const DModulus *modulus, const uint64_t *scalar,
                                    const uint64_t *scalar_shoup, size_t dim, size_t batch, const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);
    inplace_inwt_radix2_outofplace_batched<<<batch, dim / 2, per_block_memory, stream>>>(
            out, in, itwiddles, itwiddles_shoup, modulus, scalar, scalar_shoup, dim, batch);
}

void inwt_1d_opt_permute_batched(uint64_t *out, const uint64_t *in, const uint32_t *d_map, const uint64_t *itwiddles,
                                 const uint64_t *itwiddles_shoup, const DModulus *modulus, const uint64_t *scalar,
                                 const uint64_t *scalar_shoup, size_t dim, size_t batch, const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);
    inplace_inwt_radix2_permute_batched<<<batch, dim / 2, per_block_memory, stream>>>(
            out, in, d_map, itwiddles, itwiddles_shoup, modulus, scalar, scalar_shoup, dim, batch);
}
