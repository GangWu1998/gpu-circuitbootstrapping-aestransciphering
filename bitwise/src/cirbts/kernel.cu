#include "cirbts/kernel.cuh"
#include "ntt.cuh"
#include "butterfly.cuh"

using namespace lbcrypto;

namespace {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ unsigned int cirbts_smem_ptr(void* ptr) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cirbts_cp_async_8(void* smem_ptr, const void* gmem_ptr) {
    const unsigned int smem = cirbts_smem_ptr(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(smem), "l"(gmem_ptr));
}

__device__ __forceinline__ void cirbts_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cirbts_cp_async_wait() {
    asm volatile("cp.async.wait_group 0;");
}
#else
__device__ __forceinline__ void cirbts_cp_async_8(void* smem_ptr, const void* gmem_ptr) {
    *reinterpret_cast<BasicInteger*>(smem_ptr) = *reinterpret_cast<const BasicInteger*>(gmem_ptr);
}

__device__ __forceinline__ void cirbts_cp_async_commit() {}
__device__ __forceinline__ void cirbts_cp_async_wait() {}
#endif
}  // namespace

__device__ __forceinline__ uint32_t device_SpecialMS_IndexPos(BasicInteger v, uint32_t twoN, BasicInteger Q, uint32_t bitwidth);

__device__ void device_SignedDigitDecompose2(BasicInteger *output, const BasicInteger &input, const BasicInteger &Q, 
                                             const uint32_t &baseG, uint32_t digits,const size_t &N){

    // baseG is a power of two
    const int gBits        = __ffs(baseG) - 1;
    const int maxBits      = static_cast<int>(sizeof(BasicInteger) * 8);
    const int logQ         = maxBits - __clzll(static_cast<unsigned long long>(Q));
    int ignore_bits        = logQ - static_cast<int>(digits) * gBits;
    if (ignore_bits < 0) {
        ignore_bits = 0;
    }
    uint64_t QHalf         = Q >> 1;
    const auto gBitsMaxBits      = maxBits - ignore_bits;
    const auto gBitsMaxBits0     = maxBits - gBits;
    auto Q_int             = static_cast<NativeInteger::SignedNativeInt>(Q);

    auto t = input;
    auto d = static_cast<NativeInteger::SignedNativeInt>(t < QHalf ? t : t - Q_int);

    auto r = static_cast<NativeInteger::SignedNativeInt>(0);
    if (ignore_bits != 0) {
        r = (d << gBitsMaxBits) >> gBitsMaxBits;
        d = (d - r) >> ignore_bits;
    }

    for(size_t i = 0; i < digits; i++){
        r = (d << gBitsMaxBits0) >>gBitsMaxBits0;
        d = (d - r) >> gBits;
        if(r < 0) r += Q_int;
        output[i * 2 * N] = r;
    }
}

__device__ void device_SignedDigitDecompose2_Int64(int64_t* output, const BasicInteger& input, const BasicInteger& Q,
                                                   const uint32_t& baseG, uint32_t digits, const size_t& N) {
    const int gBits        = __ffs(baseG) - 1;
    const int maxBits      = static_cast<int>(sizeof(BasicInteger) * 8);
    const int logQ         = maxBits - __clzll(static_cast<unsigned long long>(Q));
    int ignore_bits        = logQ - static_cast<int>(digits) * gBits;
    if (ignore_bits < 0) {
        ignore_bits = 0;
    }
    const uint64_t QHalf   = Q >> 1;
    const auto gBitsMaxBits  = maxBits - ignore_bits;
    const auto gBitsMaxBits0 = maxBits - gBits;
    const auto Q_int       = static_cast<NativeInteger::SignedNativeInt>(Q);

    auto t = input;
    auto d = static_cast<NativeInteger::SignedNativeInt>(t < QHalf ? t : t - Q_int);
    auto r = static_cast<NativeInteger::SignedNativeInt>(0);

    if (ignore_bits != 0) {
        r = (d << gBitsMaxBits) >> gBitsMaxBits;
        d = (d - r) >> ignore_bits;
    }

    for (size_t i = 0; i < digits; ++i) {
        r = (d << gBitsMaxBits0) >> gBitsMaxBits0;
        d = (d - r) >> gBits;
        if (r < 0) {
            r += Q_int;
        }
        const uint64_t r_u = static_cast<uint64_t>(r);
        const int64_t r_s = (r_u > QHalf) ? (static_cast<int64_t>(r_u) - static_cast<int64_t>(Q))
                                          : static_cast<int64_t>(r_u);
        output[i * 2 * N] = r_s;
    }
}

__device__ void device_SignedDigitDecompose2_WithResidual(int64_t* digits_out, int64_t* residual_out, const BasicInteger& input,
                                                          const BasicInteger& Q, const uint32_t& baseG, uint32_t digits,
                                                          const size_t& N) {
    const int gBits        = __ffs(baseG) - 1;
    const int maxBits      = static_cast<int>(sizeof(BasicInteger) * 8);
    const int logQ         = maxBits - __clzll(static_cast<unsigned long long>(Q));
    int ignore_bits        = logQ - static_cast<int>(digits) * gBits;
    if (ignore_bits < 0) {
        ignore_bits = 0;
    }
    const int gBitsMaxBits  = maxBits - ignore_bits;
    const int gBitsMaxBits0 = maxBits - gBits;
    const auto Q_int       = static_cast<NativeInteger::SignedNativeInt>(Q);

    const uint64_t QHalf   = Q >> 1;
    auto d                = static_cast<NativeInteger::SignedNativeInt>(input < QHalf ? input : input - Q_int);
    auto r                = static_cast<NativeInteger::SignedNativeInt>(0);

    if (ignore_bits != 0) {
        r = (d << gBitsMaxBits) >> gBitsMaxBits;
        d = (d - r) >> ignore_bits;
    }
    residual_out[0] = static_cast<int64_t>(r);

    for (uint32_t i = 0; i < digits; ++i) {
        r = (d << gBitsMaxBits0) >> gBitsMaxBits0;
        d = (d - r) >> gBits;
        digits_out[i * 2 * N] = static_cast<int64_t>(r);
    }
}

__global__ void kernel_SignedDigitDecompose2(BasicInteger *output, const BasicInteger *input,
                                            BasicInteger Q, uint32_t baseG, uint32_t digits, size_t N){
    size_t poly_id  = blockIdx.y;
    size_t coeff_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(coeff_id >= N) return;
    size_t output_offset = poly_id * N + coeff_id;
    device_SignedDigitDecompose2(output + output_offset, input[poly_id * N + coeff_id], Q, baseG, digits, N);                                           
}

__global__ void kernel_SignedDigitDecompose2_Int64(int64_t* output, const BasicInteger* input, BasicInteger Q,
                                                   uint32_t baseG, uint32_t digits, size_t N) {
    const size_t poly_id  = blockIdx.y;
    const size_t coeff_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_id >= N) {
        return;
    }
    const size_t output_offset = poly_id * N + coeff_id;
    device_SignedDigitDecompose2_Int64(output + output_offset, input[poly_id * N + coeff_id], Q, baseG, digits, N);
}

__global__ void kernel_SignedDigitDecompose2_Residual(int64_t* digits_out, int64_t* residual_out, const BasicInteger* input,
                                                      BasicInteger Q, uint32_t baseG, uint32_t digits, size_t N) {
    const size_t poly_id  = blockIdx.y;
    const size_t coeff_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_id >= N) {
        return;
    }
    const size_t offset = poly_id * N + coeff_id;
    device_SignedDigitDecompose2_WithResidual(digits_out + offset, residual_out + offset, input[offset], Q, baseG, digits, N);
}

__global__ void kernel_DigitsSignedToModQ(BasicInteger* output, const int64_t* input, BasicInteger Q, size_t total) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t v = input[idx];
    output[idx] = (v < 0) ? static_cast<BasicInteger>(v + static_cast<int64_t>(Q))
                          : static_cast<BasicInteger>(v);
}

__global__ void kernel_AddDelta_ToDigitsResidual(int64_t* digits, int64_t* residual, const BasicInteger* delta, BasicInteger Q,
                                                 uint32_t baseG, uint32_t digits_count, size_t N, uint32_t* overflow_flag) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(2) * N;
    if (idx >= total) {
        return;
    }

    const uint64_t QHalf = Q >> 1;
    const int64_t delta_s =
        (delta[idx] > QHalf) ? (static_cast<int64_t>(delta[idx]) - static_cast<int64_t>(Q))
                             : static_cast<int64_t>(delta[idx]);

    const int gBits   = __ffs(baseG) - 1;
    const int maxBits = static_cast<int>(sizeof(BasicInteger) * 8);
    const int logQ    = maxBits - __clzll(static_cast<unsigned long long>(Q));
    int ignore_bits   = logQ - static_cast<int>(digits_count) * gBits;
    if (ignore_bits < 0) {
        ignore_bits = 0;
    }

    int64_t carry = delta_s;
    if (ignore_bits > 0) {
        const int64_t half  = static_cast<int64_t>(1) << (ignore_bits - 1);
        int64_t total_res   = residual[idx] + carry;
        int64_t carry0      = total_res >> ignore_bits;
        int64_t res         = total_res - (carry0 << ignore_bits);
        if (res >= half) {
            res -= (static_cast<int64_t>(1) << ignore_bits);
            carry0 += 1;
        }
        if (res < -half) {
            res += (static_cast<int64_t>(1) << ignore_bits);
            carry0 -= 1;
        }
        residual[idx] = res;
        carry = carry0;
    }
    else {
        residual[idx] = 0;
    }

    const int64_t half_digit = static_cast<int64_t>(baseG) >> 1;
    const size_t stride = static_cast<size_t>(2) * N;
    for (uint32_t d = 0; d < digits_count; ++d) {
        const size_t offset = static_cast<size_t>(d) * stride + idx;
        int64_t val = digits[offset] + carry;
        int64_t carry_out = val >> gBits;
        int64_t rem = val - (carry_out << gBits);
        if (rem >= half_digit) {
            rem -= static_cast<int64_t>(baseG);
            carry_out += 1;
        }
        if (rem < -half_digit) {
            rem += static_cast<int64_t>(baseG);
            carry_out -= 1;
        }
        digits[offset] = rem;
        carry = carry_out;
    }

    if (carry != 0) {
        atomicExch(overflow_flag, 1u);
    }
}

__device__ __forceinline__ BasicInteger device_SignedDigitDecompose2_SelectDigit(const BasicInteger& input, const BasicInteger& Q,
                                                                                 const uint32_t baseG, const uint32_t digits,
                                                                                 const uint32_t digit_idx) {
    // baseG is a power of two
    const int gBits   = __ffs(baseG) - 1;
    const int maxBits = static_cast<int>(sizeof(BasicInteger) * 8);
    const int logQ    = maxBits - __clzll(static_cast<unsigned long long>(Q));
    int ignore_bits   = logQ - static_cast<int>(digits) * gBits;
    if (ignore_bits < 0) {
        ignore_bits = 0;
    }

    const BasicInteger QHalf     = Q >> 1;
    const int gBitsMaxBits       = maxBits - ignore_bits;
    const int gBitsMaxBits0      = maxBits - gBits;
    const auto Q_int             = static_cast<NativeInteger::SignedNativeInt>(Q);
    auto d                        = static_cast<NativeInteger::SignedNativeInt>(input < QHalf ? input : input - Q_int);
    auto r                        = static_cast<NativeInteger::SignedNativeInt>(0);

    if (ignore_bits != 0) {
        r = (d << gBitsMaxBits) >> gBitsMaxBits;
        d = (d - r) >> ignore_bits;
    }

    for (uint32_t i = 0; i <= digit_idx; ++i) {
        r = (d << gBitsMaxBits0) >> gBitsMaxBits0;
        d = (d - r) >> gBits;
        if (r < 0) {
            r += Q_int;
        }
    }

    return static_cast<BasicInteger>(r);
}

__device__ __forceinline__ BasicInteger device_SignedDigitDecompose_SelectDigit(const BasicInteger& input, const BasicInteger& Q,
                                                                                const uint32_t base, const uint32_t digits,
                                                                                const uint32_t digit_idx) {
    // base is a power of two
    const int gBits   = __ffs(base) - 1;
    const int maxBits = static_cast<int>(sizeof(BasicInteger) * 8);
    const int logQ    = maxBits - __clzll(static_cast<unsigned long long>(Q));
    int ignore_bits   = logQ - static_cast<int>(digits) * gBits;
    if (ignore_bits < 0) {
        ignore_bits = 0;
    }

    const BasicInteger QHalf     = Q >> 1;
    const int gBitsMaxBits       = maxBits - ignore_bits;
    const int gBitsMaxBits0      = maxBits - gBits;
    const auto Q_int             = static_cast<NativeInteger::SignedNativeInt>(Q);
    auto d                        = static_cast<NativeInteger::SignedNativeInt>(input < QHalf ? input : input - Q_int);
    auto r                        = static_cast<NativeInteger::SignedNativeInt>(0);

    if (ignore_bits != 0) {
        r = (d << gBitsMaxBits) >> gBitsMaxBits;
        d = (d - r) >> ignore_bits;
    }

    for (uint32_t i = 0; i <= digit_idx; ++i) {
        r = (d << gBitsMaxBits0) >> gBitsMaxBits0;
        d = (d - r) >> gBits;
        if (r < 0) {
            r += Q_int;
        }
    }

    return static_cast<BasicInteger>(r);
}

__global__ void kernel_SignedDigitDecompose2_FusedFNWT(BasicInteger* output, const BasicInteger* input, BasicInteger Q, uint32_t baseG,
                                                       uint32_t digits, size_t N, const BasicInteger* twiddles,
                                                       const BasicInteger* twiddles_shoup, const DModulus* modulus) {
    extern __shared__ BasicInteger buffer[];

    const size_t poly_idx = blockIdx.x;
    const size_t tid      = threadIdx.x;
    const size_t tpb      = blockDim.x;

    const size_t mod_idx = poly_idx;
    const BasicInteger mod  = modulus[0].value();
    const BasicInteger mod2 = mod << 1;

    const uint32_t ct_poly   = static_cast<uint32_t>(poly_idx & 1u);
    const uint32_t digit_idx = static_cast<uint32_t>(poly_idx >> 1);

    // stage 1: compute digit values on the fly and start the radix-2 NTT.
    {
        constexpr size_t numOfGroups0 = 1;
        const size_t pairsInGroup0    = N / 2;
        const BasicInteger psi0       = twiddles[numOfGroups0];
        const BasicInteger psi0_shoup = twiddles_shoup[numOfGroups0];

        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t glbIdx0 = butterflyId;
            BasicInteger samples0 = device_SignedDigitDecompose2_SelectDigit(
                input[static_cast<size_t>(ct_poly) * N + glbIdx0], Q, baseG, digits, digit_idx);
            BasicInteger samples1 = device_SignedDigitDecompose2_SelectDigit(
                input[static_cast<size_t>(ct_poly) * N + glbIdx0 + pairsInGroup0], Q, baseG, digits, digit_idx);

            phantom::arith::ct_butterfly(samples0, samples1, psi0, psi0_shoup, mod);
            buffer[glbIdx0]                 = samples0;
            buffer[glbIdx0 + pairsInGroup0] = samples1;
        }
        __syncthreads();
    }

    for (size_t numOfGroups = 2; numOfGroups < N / 2; numOfGroups <<= 1) {
        const size_t pairsInGroup = N / numOfGroups / 2;
        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t k        = butterflyId / pairsInGroup;
            const size_t j        = butterflyId - k * pairsInGroup;
            const size_t glbIdx   = 2 * k * pairsInGroup + j;
            const BasicInteger psi    = twiddles[numOfGroups + k];
            const BasicInteger psi_sh = twiddles_shoup[numOfGroups + k];

            BasicInteger a0 = buffer[glbIdx];
            BasicInteger a1 = buffer[glbIdx + pairsInGroup];
            phantom::arith::ct_butterfly(a0, a1, psi, psi_sh, mod);
            buffer[glbIdx]                 = a0;
            buffer[glbIdx + pairsInGroup]  = a1;
        }
        __syncthreads();
    }

    // last stage: write directly to output (no need to store back to shared memory).
    {
        const size_t numOfGroups_last     = N / 2;
        const size_t pairsInGroup_last    = 1;
        const size_t out_base             = mod_idx * N;
        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t k_last               = butterflyId;
            const BasicInteger psi_last       = twiddles[numOfGroups_last + k_last];
            const BasicInteger psi_last_shoup = twiddles_shoup[numOfGroups_last + k_last];
            const size_t glbIdx_last          = 2 * butterflyId;

            BasicInteger a0_last = buffer[glbIdx_last];
            BasicInteger a1_last = buffer[glbIdx_last + pairsInGroup_last];
            phantom::arith::ct_butterfly(a0_last, a1_last, psi_last, psi_last_shoup, mod);
            phantom::arith::csub_q(a0_last, mod2);
            phantom::arith::csub_q(a0_last, mod);
            phantom::arith::csub_q(a1_last, mod2);
            phantom::arith::csub_q(a1_last, mod);

            output[out_base + glbIdx_last]                    = a0_last;
            output[out_base + glbIdx_last + pairsInGroup_last] = a1_last;
        }
    }
}

__global__ void kernel_DigitsSigned_FusedFNWT(BasicInteger* output, const int64_t* input, BasicInteger Q, size_t N,
                                              const BasicInteger* twiddles, const BasicInteger* twiddles_shoup,
                                              const DModulus* modulus) {
    extern __shared__ BasicInteger buffer[];

    const size_t poly_idx = blockIdx.x;
    const size_t tid      = threadIdx.x;
    const size_t tpb      = blockDim.x;

    const BasicInteger mod  = modulus[0].value();
    const BasicInteger mod2 = mod << 1;
    const size_t in_base    = poly_idx * N;

    // stage 1: load signed digits and start the radix-2 NTT.
    {
        constexpr size_t numOfGroups0 = 1;
        const size_t pairsInGroup0    = N / 2;
        const BasicInteger psi0       = twiddles[numOfGroups0];
        const BasicInteger psi0_shoup = twiddles_shoup[numOfGroups0];

        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t glbIdx0 = butterflyId;
            const int64_t s0 = input[in_base + glbIdx0];
            const int64_t s1 = input[in_base + glbIdx0 + pairsInGroup0];
            BasicInteger samples0 = (s0 < 0) ? static_cast<BasicInteger>(s0 + static_cast<int64_t>(Q))
                                             : static_cast<BasicInteger>(s0);
            BasicInteger samples1 = (s1 < 0) ? static_cast<BasicInteger>(s1 + static_cast<int64_t>(Q))
                                             : static_cast<BasicInteger>(s1);

            phantom::arith::ct_butterfly(samples0, samples1, psi0, psi0_shoup, mod);
            buffer[glbIdx0]                 = samples0;
            buffer[glbIdx0 + pairsInGroup0] = samples1;
        }
        __syncthreads();
    }

    for (size_t numOfGroups = 2; numOfGroups < N / 2; numOfGroups <<= 1) {
        const size_t pairsInGroup = N / numOfGroups / 2;
        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t k        = butterflyId / pairsInGroup;
            const size_t j        = butterflyId - k * pairsInGroup;
            const size_t glbIdx   = 2 * k * pairsInGroup + j;
            const BasicInteger psi    = twiddles[numOfGroups + k];
            const BasicInteger psi_sh = twiddles_shoup[numOfGroups + k];

            BasicInteger a0 = buffer[glbIdx];
            BasicInteger a1 = buffer[glbIdx + pairsInGroup];
            phantom::arith::ct_butterfly(a0, a1, psi, psi_sh, mod);
            buffer[glbIdx]                = a0;
            buffer[glbIdx + pairsInGroup] = a1;
        }
        __syncthreads();
    }

    // last stage: write directly to output.
    {
        const size_t numOfGroups_last     = N / 2;
        const size_t pairsInGroup_last    = 1;
        const size_t out_base             = poly_idx * N;
        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t k_last               = butterflyId;
            const BasicInteger psi_last       = twiddles[numOfGroups_last + k_last];
            const BasicInteger psi_last_shoup = twiddles_shoup[numOfGroups_last + k_last];
            const size_t glbIdx_last          = 2 * butterflyId;

            BasicInteger a0_last = buffer[glbIdx_last];
            BasicInteger a1_last = buffer[glbIdx_last + pairsInGroup_last];
            phantom::arith::ct_butterfly(a0_last, a1_last, psi_last, psi_last_shoup, mod);
            phantom::arith::csub_q(a0_last, mod2);
            phantom::arith::csub_q(a0_last, mod);
            phantom::arith::csub_q(a1_last, mod2);
            phantom::arith::csub_q(a1_last, mod);

            output[out_base + glbIdx_last]                    = a0_last;
            output[out_base + glbIdx_last + pairsInGroup_last] = a1_last;
        }
    }
}

__global__ void kernel_SignedDigitDecompose2_FusedFNWT_Batch(BasicInteger* output, const BasicInteger* input, BasicInteger Q, uint32_t baseG,
                                                             uint32_t digits, size_t N, uint32_t batch, const BasicInteger* twiddles,
                                                             const BasicInteger* twiddles_shoup, const DModulus* modulus) {
    extern __shared__ BasicInteger buffer[];

    const size_t poly_idx = blockIdx.x;
    const size_t tid      = threadIdx.x;
    const size_t tpb      = blockDim.x;

    const uint32_t digitsG2 = digits << 1;
    const size_t batch_idx  = poly_idx / digitsG2;
    if (batch_idx >= batch) {
        return;
    }
    const uint32_t inner    = static_cast<uint32_t>(poly_idx - batch_idx * digitsG2);

    const size_t mod_idx    = poly_idx;
    const BasicInteger mod  = modulus[0].value();
    const BasicInteger mod2 = mod << 1;

    const uint32_t ct_poly   = inner & 1u;
    const uint32_t digit_idx = inner >> 1;

    const BasicInteger* in_poly = input + (batch_idx * 2 + ct_poly) * N;

    // stage 1: compute digit values on the fly and start the radix-2 NTT.
    {
        constexpr size_t numOfGroups0 = 1;
        const size_t pairsInGroup0    = N / 2;
        const BasicInteger psi0       = twiddles[numOfGroups0];
        const BasicInteger psi0_shoup = twiddles_shoup[numOfGroups0];

        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t glbIdx0 = butterflyId;
            BasicInteger samples0 = device_SignedDigitDecompose2_SelectDigit(in_poly[glbIdx0], Q, baseG, digits, digit_idx);
            BasicInteger samples1 =
                device_SignedDigitDecompose2_SelectDigit(in_poly[glbIdx0 + pairsInGroup0], Q, baseG, digits, digit_idx);

            phantom::arith::ct_butterfly(samples0, samples1, psi0, psi0_shoup, mod);
            buffer[glbIdx0]                 = samples0;
            buffer[glbIdx0 + pairsInGroup0] = samples1;
        }
        __syncthreads();
    }

    for (size_t numOfGroups = 2; numOfGroups < N / 2; numOfGroups <<= 1) {
        const size_t pairsInGroup = N / numOfGroups / 2;
        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t k        = butterflyId / pairsInGroup;
            const size_t j        = butterflyId - k * pairsInGroup;
            const size_t glbIdx   = 2 * k * pairsInGroup + j;
            const BasicInteger psi    = twiddles[numOfGroups + k];
            const BasicInteger psi_sh = twiddles_shoup[numOfGroups + k];

            BasicInteger a0 = buffer[glbIdx];
            BasicInteger a1 = buffer[glbIdx + pairsInGroup];
            phantom::arith::ct_butterfly(a0, a1, psi, psi_sh, mod);
            buffer[glbIdx]                 = a0;
            buffer[glbIdx + pairsInGroup]  = a1;
        }
        __syncthreads();
    }

    // last stage: write directly to output.
    {
        const size_t numOfGroups_last     = N / 2;
        const size_t pairsInGroup_last    = 1;
        const size_t out_base             = mod_idx * N;
        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t k_last               = butterflyId;
            const BasicInteger psi_last       = twiddles[numOfGroups_last + k_last];
            const BasicInteger psi_last_shoup = twiddles_shoup[numOfGroups_last + k_last];
            const size_t glbIdx_last          = 2 * butterflyId;

            BasicInteger a0_last = buffer[glbIdx_last];
            BasicInteger a1_last = buffer[glbIdx_last + pairsInGroup_last];
            phantom::arith::ct_butterfly(a0_last, a1_last, psi_last, psi_last_shoup, mod);
            phantom::arith::csub_q(a0_last, mod2);
            phantom::arith::csub_q(a0_last, mod);
            phantom::arith::csub_q(a1_last, mod2);
            phantom::arith::csub_q(a1_last, mod);

            output[out_base + glbIdx_last]                     = a0_last;
            output[out_base + glbIdx_last + pairsInGroup_last]  = a1_last;
        }
    }
}

__global__ void kernel_INWT_Decompose2_FusedFNWT_Batch_AES2048_D1(
    BasicInteger* output, const BasicInteger* input,
    const BasicInteger* itwiddles, const BasicInteger* itwiddles_shoup,
    const BasicInteger* twiddles, const BasicInteger* twiddles_shoup,
    const DModulus* modulus, const BasicInteger* scalar,
    const BasicInteger* scalar_shoup, BasicInteger Q, uint32_t baseG,
    uint32_t batch) {
    constexpr size_t kN = 2048;
    constexpr uint32_t kDigits = 1;
    constexpr uint32_t kDigitsG2 = kDigits * 2;

    extern __shared__ BasicInteger shared[];
    BasicInteger* coeff = shared;
    BasicInteger* work = shared + kN;

    const size_t poly_idx = blockIdx.x;
    const size_t tid = threadIdx.x;
    const size_t tpb = blockDim.x;
    if (poly_idx >= static_cast<size_t>(batch) * 2u) {
        return;
    }

    const BasicInteger mod = modulus[0].value();
    const BasicInteger mod2 = mod << 1;
    const BasicInteger scalar_ = scalar[0];
    const BasicInteger scalar_shoup_ = scalar_shoup[0];
    const BasicInteger* in_poly = input + poly_idx * kN;

#pragma unroll
    for (size_t numOfGroups = kN / 2; numOfGroups >= 1; numOfGroups >>= 1) {
        const size_t pairsInGroup = kN / numOfGroups / 2;
        for (size_t butterflyId = tid; butterflyId < kN / 2; butterflyId += tpb) {
            const size_t k = butterflyId / pairsInGroup;
            const size_t j = butterflyId - k * pairsInGroup;
            const size_t glbIdx = 2 * k * pairsInGroup + j;
            const BasicInteger psi = itwiddles[numOfGroups + k];
            const BasicInteger psi_shoup = itwiddles_shoup[numOfGroups + k];

            BasicInteger samples0;
            BasicInteger samples1;
            if (numOfGroups == kN / 2) {
                samples0 = in_poly[glbIdx];
                samples1 = in_poly[glbIdx + pairsInGroup];
            } else {
                samples0 = coeff[glbIdx];
                samples1 = coeff[glbIdx + pairsInGroup];
            }

            phantom::arith::gs_butterfly(samples0, samples1, psi, psi_shoup, mod);
            if (numOfGroups == 1) {
                phantom::arith::csub_q(samples0, mod);
                phantom::arith::csub_q(samples1, mod);
                samples0 = phantom::arith::multiply_and_reduce_shoup(samples0, scalar_, scalar_shoup_, mod);
            }
            coeff[glbIdx] = samples0;
            coeff[glbIdx + pairsInGroup] = samples1;
        }
        __syncthreads();
    }

    const size_t batch_idx = poly_idx >> 1;
    const uint32_t ct_poly = static_cast<uint32_t>(poly_idx & 1u);
#pragma unroll
    for (uint32_t digit_idx = 0; digit_idx < kDigits; ++digit_idx) {
        {
            constexpr size_t numOfGroups0 = 1;
            constexpr size_t pairsInGroup0 = kN / 2;
            const BasicInteger psi0 = twiddles[numOfGroups0];
            const BasicInteger psi0_shoup = twiddles_shoup[numOfGroups0];
            for (size_t butterflyId = tid; butterflyId < kN / 2; butterflyId += tpb) {
                const size_t glbIdx0 = butterflyId;
                BasicInteger samples0 =
                    device_SignedDigitDecompose2_SelectDigit(coeff[glbIdx0], Q, baseG, kDigits, digit_idx);
                BasicInteger samples1 =
                    device_SignedDigitDecompose2_SelectDigit(coeff[glbIdx0 + pairsInGroup0], Q, baseG, kDigits, digit_idx);

                phantom::arith::ct_butterfly(samples0, samples1, psi0, psi0_shoup, mod);
                work[glbIdx0] = samples0;
                work[glbIdx0 + pairsInGroup0] = samples1;
            }
            __syncthreads();
        }

#pragma unroll
        for (size_t numOfGroups = 2; numOfGroups < kN / 2; numOfGroups <<= 1) {
            const size_t pairsInGroup = kN / numOfGroups / 2;
            for (size_t butterflyId = tid; butterflyId < kN / 2; butterflyId += tpb) {
                const size_t k = butterflyId / pairsInGroup;
                const size_t j = butterflyId - k * pairsInGroup;
                const size_t glbIdx = 2 * k * pairsInGroup + j;
                const BasicInteger psi = twiddles[numOfGroups + k];
                const BasicInteger psi_shoup = twiddles_shoup[numOfGroups + k];

                BasicInteger a0 = work[glbIdx];
                BasicInteger a1 = work[glbIdx + pairsInGroup];
                phantom::arith::ct_butterfly(a0, a1, psi, psi_shoup, mod);
                work[glbIdx] = a0;
                work[glbIdx + pairsInGroup] = a1;
            }
            __syncthreads();
        }

        {
            constexpr size_t numOfGroupsLast = kN / 2;
            constexpr size_t pairsInGroupLast = 1;
            const size_t out_inner = static_cast<size_t>(digit_idx) * 2u + ct_poly;
            const size_t out_base = (batch_idx * kDigitsG2 + out_inner) * kN;
            for (size_t butterflyId = tid; butterflyId < kN / 2; butterflyId += tpb) {
                const size_t glbIdxLast = 2 * butterflyId;
                const BasicInteger psi = twiddles[numOfGroupsLast + butterflyId];
                const BasicInteger psi_shoup = twiddles_shoup[numOfGroupsLast + butterflyId];

                BasicInteger a0 = work[glbIdxLast];
                BasicInteger a1 = work[glbIdxLast + pairsInGroupLast];
                phantom::arith::ct_butterfly(a0, a1, psi, psi_shoup, mod);
                phantom::arith::csub_q(a0, mod2);
                phantom::arith::csub_q(a0, mod);
                phantom::arith::csub_q(a1, mod2);
                phantom::arith::csub_q(a1, mod);
                output[out_base + glbIdxLast] = a0;
                output[out_base + glbIdxLast + pairsInGroupLast] = a1;
            }
        }
    }
}

__global__ void kernel_SignedDigitDecompose_FusedFNWT(BasicInteger* output, const BasicInteger* input, BasicInteger Q, uint32_t base,
                                                      uint32_t digits, size_t N, uint32_t numLUT, const BasicInteger* twiddles,
                                                      const BasicInteger* twiddles_shoup, const DModulus* modulus) {
    extern __shared__ BasicInteger buffer[];

    const size_t poly_idx = blockIdx.x;
    const size_t tid      = threadIdx.x;
    const size_t tpb      = blockDim.x;

    const uint32_t lut      = static_cast<uint32_t>(poly_idx % numLUT);
    const uint32_t digit_idx = static_cast<uint32_t>(poly_idx / numLUT);

    const size_t mod_idx      = poly_idx;
    const BasicInteger mod    = modulus[0].value();
    const BasicInteger mod2   = mod << 1;

    const BasicInteger* in_poly = input + static_cast<size_t>(lut) * N;

    // stage 1: compute digit values on the fly and start the radix-2 NTT.
    {
        constexpr size_t numOfGroups0 = 1;
        const size_t pairsInGroup0    = N / 2;
        const BasicInteger psi0       = twiddles[numOfGroups0];
        const BasicInteger psi0_shoup = twiddles_shoup[numOfGroups0];

        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t glbIdx0 = butterflyId;
            BasicInteger samples0 = device_SignedDigitDecompose_SelectDigit(in_poly[glbIdx0], Q, base, digits, digit_idx);
            BasicInteger samples1 =
                device_SignedDigitDecompose_SelectDigit(in_poly[glbIdx0 + pairsInGroup0], Q, base, digits, digit_idx);

            phantom::arith::ct_butterfly(samples0, samples1, psi0, psi0_shoup, mod);
            buffer[glbIdx0]                 = samples0;
            buffer[glbIdx0 + pairsInGroup0] = samples1;
        }
        __syncthreads();
    }

    for (size_t numOfGroups = 2; numOfGroups < N / 2; numOfGroups <<= 1) {
        const size_t pairsInGroup = N / numOfGroups / 2;
        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t k        = butterflyId / pairsInGroup;
            const size_t j        = butterflyId - k * pairsInGroup;
            const size_t glbIdx   = 2 * k * pairsInGroup + j;
            const BasicInteger psi    = twiddles[numOfGroups + k];
            const BasicInteger psi_sh = twiddles_shoup[numOfGroups + k];

            BasicInteger a0 = buffer[glbIdx];
            BasicInteger a1 = buffer[glbIdx + pairsInGroup];
            phantom::arith::ct_butterfly(a0, a1, psi, psi_sh, mod);
            buffer[glbIdx]                 = a0;
            buffer[glbIdx + pairsInGroup]  = a1;
        }
        __syncthreads();
    }

    // last stage: write directly to output.
    {
        const size_t numOfGroups_last     = N / 2;
        const size_t pairsInGroup_last    = 1;
        const size_t out_base             = mod_idx * N;
        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t k_last               = butterflyId;
            const BasicInteger psi_last       = twiddles[numOfGroups_last + k_last];
            const BasicInteger psi_last_shoup = twiddles_shoup[numOfGroups_last + k_last];
            const size_t glbIdx_last          = 2 * butterflyId;

            BasicInteger a0_last = buffer[glbIdx_last];
            BasicInteger a1_last = buffer[glbIdx_last + pairsInGroup_last];
            phantom::arith::ct_butterfly(a0_last, a1_last, psi_last, psi_last_shoup, mod);
            phantom::arith::csub_q(a0_last, mod2);
            phantom::arith::csub_q(a0_last, mod);
            phantom::arith::csub_q(a1_last, mod2);
            phantom::arith::csub_q(a1_last, mod);

            output[out_base + glbIdx_last]                    = a0_last;
            output[out_base + glbIdx_last + pairsInGroup_last] = a1_last;
        }
    }
}

__device__ void device_SignedDigitDecompose(BasicInteger *output, const BasicInteger &input, const BasicInteger &Q, 
                                            const uint32_t &baseHT, uint32_t digitsHT, size_t stride){
    // __ffs returns 1-based index of least significant 1-bit; subtract 1 to get ctz
    const int gBits       = __ffs(baseHT) - 1; 
    const int maxBits     = static_cast<int>(sizeof(BasicInteger) * 8);
    const int logQ        = maxBits - __clzll(static_cast<unsigned long long>(Q));
    int ignore_bits       = logQ - static_cast<int>(digitsHT) * gBits;
    if (ignore_bits < 0) {
        ignore_bits = 0;
    }

    const int gBitsMaxBits  = maxBits - ignore_bits;
    const int gBitsMaxBits0 = maxBits - gBits; 
    auto Q_int        = static_cast<NativeInteger::SignedNativeInt>(Q);
    uint64_t QHalf    = Q >> 1; 
    
    auto t = input;
    auto d = static_cast<NativeInteger::SignedNativeInt>(t < QHalf ? t : t - Q_int);

    auto r = static_cast<NativeInteger::SignedNativeInt>(0);
    if (ignore_bits != 0) {
        r = (d << gBitsMaxBits) >> gBitsMaxBits;
        d = (d - r) >> ignore_bits;
    }
    for(size_t i = 0; i < digitsHT; i++){
        r = (d << gBitsMaxBits0) >> gBitsMaxBits0;
        d = (d - r) >> gBits;
        if(r < 0) r += Q_int;
        output[i * stride] = static_cast<BasicInteger>(r);
    }
}


__global__ void kernel_SignedDigitDecompose(BasicInteger *output, const BasicInteger *input, BasicInteger Q,
                                            uint32_t baseHT, uint32_t digitsHT, size_t N, size_t numLUT){
    size_t lut_id   = blockIdx.y;
    size_t coeff_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx      = lut_id * N + coeff_id;   
    size_t stride   = numLUT * N;

    if (coeff_id >= N || lut_id >= numLUT) return;

    device_SignedDigitDecompose(output + idx, input[idx], Q, baseHT, digitsHT, stride);
} 

__global__ void kernel_SignedDigitDecomposeLMKCDEY_Ciphertext(BasicInteger* output, const BasicInteger* input, BasicInteger Q,
                                                              uint32_t baseG, uint32_t digitsG2, size_t N) {
    size_t coeff = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    BasicInteger QHalf = Q >> 1;
    auto Q_int         = static_cast<NativeInteger::SignedNativeInt>(Q);
    auto t0            = input[coeff];
    auto d0            = static_cast<NativeInteger::SignedNativeInt>(t0 < QHalf ? t0 : t0 - Q_int);
    auto t1            = input[N + coeff];
    auto d1            = static_cast<NativeInteger::SignedNativeInt>(t1 < QHalf ? t1 : t1 - Q_int);

    // baseG is a power of two
    int gBits = __ffs(baseG) - 1;
    int maxBits = static_cast<int>(sizeof(BasicInteger) * 8);
    int gBitsMaxBits = maxBits - gBits;

    // OpenFHE LMKCDEY uses an approximate gadget decomposition that ignores the first digit.
    {
        auto r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
        d0 = (d0 - r0) >> gBits;
        auto r1 = (d1 << gBitsMaxBits) >> gBitsMaxBits;
        d1 = (d1 - r1) >> gBits;
    }

    for (uint32_t d = 0; d < digitsG2; d += 2) {
        auto r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
        d0 = (d0 - r0) >> gBits;
        if (r0 < 0)
            r0 += Q_int;
        output[(static_cast<size_t>(d) + 0) * N + coeff] = static_cast<BasicInteger>(r0);

        auto r1 = (d1 << gBitsMaxBits) >> gBitsMaxBits;
        d1 = (d1 - r1) >> gBits;
        if (r1 < 0)
            r1 += Q_int;
        output[(static_cast<size_t>(d) + 1) * N + coeff] = static_cast<BasicInteger>(r1);
    }
}

__global__ void kernel_SignedDigitDecomposeLMKCDEY_Poly(BasicInteger* output, const BasicInteger* input, BasicInteger Q,
                                                        uint32_t baseG, uint32_t digitsG, size_t N) {
    size_t coeff = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    BasicInteger QHalf = Q >> 1;
    auto Q_int         = static_cast<NativeInteger::SignedNativeInt>(Q);
    auto t0            = input[coeff];
    auto d0            = static_cast<NativeInteger::SignedNativeInt>(t0 < QHalf ? t0 : t0 - Q_int);

    // baseG is a power of two
    int gBits = __ffs(baseG) - 1;
    int maxBits = static_cast<int>(sizeof(BasicInteger) * 8);
    int gBitsMaxBits = maxBits - gBits;

    // OpenFHE LMKCDEY uses an approximate gadget decomposition that ignores the first digit.
    {
        auto r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
        d0 = (d0 - r0) >> gBits;
    }

    for (uint32_t d = 0; d < digitsG; ++d) {
        auto r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
        d0 = (d0 - r0) >> gBits;
        if (r0 < 0)
            r0 += Q_int;
        output[static_cast<size_t>(d) * N + coeff] = static_cast<BasicInteger>(r0);
    }
}

__device__ BasicInteger device_SignedDigitDecomposeLMKCDEY_SelectDigit(const BasicInteger& input, BasicInteger Q,
                                                                       uint32_t baseG, uint32_t digit_idx) {
    BasicInteger QHalf = Q >> 1;
    auto Q_int         = static_cast<NativeInteger::SignedNativeInt>(Q);
    auto t0            = input;
    auto d0            = static_cast<NativeInteger::SignedNativeInt>(t0 < QHalf ? t0 : t0 - Q_int);

    // baseG is a power of two
    int gBits = __ffs(baseG) - 1;
    int maxBits = static_cast<int>(sizeof(BasicInteger) * 8);
    int gBitsMaxBits = maxBits - gBits;

    // Skip the first digit (LMKCDEY approximate decomposition).
    {
        auto r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
        d0 = (d0 - r0) >> gBits;
    }

    for (uint32_t d = 0; d <= digit_idx; ++d) {
        auto r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
        d0 = (d0 - r0) >> gBits;
        if (d == digit_idx) {
            if (r0 < 0)
                r0 += Q_int;
            return static_cast<BasicInteger>(r0);
        }
    }
    return 0;
}

__global__ void kernel_SignedDigitDecomposeLMKCDEY_Ciphertext_FusedFNWT(BasicInteger* output, const BasicInteger* input,
                                                                        BasicInteger Q, uint32_t baseG, uint32_t digitsG2,
                                                                        size_t N, const BasicInteger* twiddles,
                                                                        const BasicInteger* twiddles_shoup,
                                                                        const DModulus* modulus) {
    extern __shared__ BasicInteger buffer[];

    const size_t poly_idx = blockIdx.x;
    const size_t tid      = threadIdx.x;
    const size_t tpb      = blockDim.x;

    const uint32_t digit_idx = static_cast<uint32_t>(poly_idx);
    const uint32_t digit_pos = digit_idx >> 1;
    const bool use_c1        = (digit_idx & 1u) != 0;

    const size_t mod_idx    = poly_idx;
    const BasicInteger mod  = modulus[0].value();
    const BasicInteger mod2 = mod << 1;

    const BasicInteger* in_poly = input + (use_c1 ? N : 0);

    // stage 1: compute digit values on the fly and start the radix-2 NTT.
    {
        constexpr size_t numOfGroups0 = 1;
        const size_t pairsInGroup0    = N / 2;
        const BasicInteger psi0       = twiddles[numOfGroups0];
        const BasicInteger psi0_shoup = twiddles_shoup[numOfGroups0];

        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t glbIdx0 = butterflyId;
            BasicInteger samples0 =
                device_SignedDigitDecomposeLMKCDEY_SelectDigit(in_poly[glbIdx0], Q, baseG, digit_pos);
            BasicInteger samples1 =
                device_SignedDigitDecomposeLMKCDEY_SelectDigit(in_poly[glbIdx0 + pairsInGroup0], Q, baseG, digit_pos);

            phantom::arith::ct_butterfly(samples0, samples1, psi0, psi0_shoup, mod);
            buffer[glbIdx0]                 = samples0;
            buffer[glbIdx0 + pairsInGroup0] = samples1;
        }
        __syncthreads();
    }

    for (size_t numOfGroups = 2; numOfGroups < N / 2; numOfGroups <<= 1) {
        const size_t pairsInGroup = N / numOfGroups / 2;
        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t k        = butterflyId / pairsInGroup;
            const size_t j        = butterflyId - k * pairsInGroup;
            const size_t glbIdx   = 2 * k * pairsInGroup + j;
            const BasicInteger psi    = twiddles[numOfGroups + k];
            const BasicInteger psi_sh = twiddles_shoup[numOfGroups + k];

            BasicInteger a0 = buffer[glbIdx];
            BasicInteger a1 = buffer[glbIdx + pairsInGroup];
            phantom::arith::ct_butterfly(a0, a1, psi, psi_sh, mod);
            buffer[glbIdx]                = a0;
            buffer[glbIdx + pairsInGroup] = a1;
        }
        __syncthreads();
    }

    // last stage: write directly to output.
    {
        const size_t numOfGroups_last     = N / 2;
        const size_t pairsInGroup_last    = 1;
        const size_t out_base             = mod_idx * N;
        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t k_last               = butterflyId;
            const BasicInteger psi_last       = twiddles[numOfGroups_last + k_last];
            const BasicInteger psi_last_shoup = twiddles_shoup[numOfGroups_last + k_last];
            const size_t glbIdx_last          = 2 * butterflyId;

            BasicInteger a0_last = buffer[glbIdx_last];
            BasicInteger a1_last = buffer[glbIdx_last + pairsInGroup_last];
            phantom::arith::ct_butterfly(a0_last, a1_last, psi_last, psi_last_shoup, mod);
            phantom::arith::csub_q(a0_last, mod2);
            phantom::arith::csub_q(a0_last, mod);
            phantom::arith::csub_q(a1_last, mod2);
            phantom::arith::csub_q(a1_last, mod);

            output[out_base + glbIdx_last]                    = a0_last;
            output[out_base + glbIdx_last + pairsInGroup_last] = a1_last;
        }
    }
}

__global__ void kernel_SignedDigitDecomposeLMKCDEY_Poly_FusedFNWT(BasicInteger* output, const BasicInteger* input,
                                                                  BasicInteger Q, uint32_t baseG, uint32_t digitsG,
                                                                  size_t N, const BasicInteger* twiddles,
                                                                  const BasicInteger* twiddles_shoup,
                                                                  const DModulus* modulus) {
    extern __shared__ BasicInteger buffer[];

    const size_t poly_idx = blockIdx.x;
    const size_t tid      = threadIdx.x;
    const size_t tpb      = blockDim.x;

    const uint32_t digit_idx = static_cast<uint32_t>(poly_idx);

    const size_t mod_idx    = poly_idx;
    const BasicInteger mod  = modulus[0].value();
    const BasicInteger mod2 = mod << 1;

    const BasicInteger* in_poly = input;

    // stage 1: compute digit values on the fly and start the radix-2 NTT.
    {
        constexpr size_t numOfGroups0 = 1;
        const size_t pairsInGroup0    = N / 2;
        const BasicInteger psi0       = twiddles[numOfGroups0];
        const BasicInteger psi0_shoup = twiddles_shoup[numOfGroups0];

        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t glbIdx0 = butterflyId;
            BasicInteger samples0 =
                device_SignedDigitDecomposeLMKCDEY_SelectDigit(in_poly[glbIdx0], Q, baseG, digit_idx);
            BasicInteger samples1 =
                device_SignedDigitDecomposeLMKCDEY_SelectDigit(in_poly[glbIdx0 + pairsInGroup0], Q, baseG, digit_idx);

            phantom::arith::ct_butterfly(samples0, samples1, psi0, psi0_shoup, mod);
            buffer[glbIdx0]                 = samples0;
            buffer[glbIdx0 + pairsInGroup0] = samples1;
        }
        __syncthreads();
    }

    for (size_t numOfGroups = 2; numOfGroups < N / 2; numOfGroups <<= 1) {
        const size_t pairsInGroup = N / numOfGroups / 2;
        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t k        = butterflyId / pairsInGroup;
            const size_t j        = butterflyId - k * pairsInGroup;
            const size_t glbIdx   = 2 * k * pairsInGroup + j;
            const BasicInteger psi    = twiddles[numOfGroups + k];
            const BasicInteger psi_sh = twiddles_shoup[numOfGroups + k];

            BasicInteger a0 = buffer[glbIdx];
            BasicInteger a1 = buffer[glbIdx + pairsInGroup];
            phantom::arith::ct_butterfly(a0, a1, psi, psi_sh, mod);
            buffer[glbIdx]                = a0;
            buffer[glbIdx + pairsInGroup] = a1;
        }
        __syncthreads();
    }

    // last stage: write directly to output.
    {
        const size_t numOfGroups_last     = N / 2;
        const size_t pairsInGroup_last    = 1;
        const size_t out_base             = mod_idx * N;
        for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
            const size_t k_last               = butterflyId;
            const BasicInteger psi_last       = twiddles[numOfGroups_last + k_last];
            const BasicInteger psi_last_shoup = twiddles_shoup[numOfGroups_last + k_last];
            const size_t glbIdx_last          = 2 * butterflyId;

            BasicInteger a0_last = buffer[glbIdx_last];
            BasicInteger a1_last = buffer[glbIdx_last + pairsInGroup_last];
            phantom::arith::ct_butterfly(a0_last, a1_last, psi_last, psi_last_shoup, mod);
            phantom::arith::csub_q(a0_last, mod2);
            phantom::arith::csub_q(a0_last, mod);
            phantom::arith::csub_q(a1_last, mod2);
            phantom::arith::csub_q(a1_last, mod);

            output[out_base + glbIdx_last]                    = a0_last;
            output[out_base + glbIdx_last + pairsInGroup_last] = a1_last;
        }
    }
}
   
__global__ void kernel_EvalAccCore_Binary(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                         const BasicInteger* ek_shoup, const BasicInteger* monic_polys, size_t N,
                                         const DModulus* mod, uint32_t digitsG2, size_t ek_dim3_index,
                                         const uint32_t* d_indexPos) {
                                    
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};
    size_t rgsw_dim3_index  = blockIdx.x * blockDim.x + threadIdx.x;
    if (rgsw_dim3_index >= N) {
        return;
    }

    const uint32_t indexPos = d_indexPos[ek_dim3_index];
    const BasicInteger pos_monic_coeff = monic_polys[static_cast<size_t>(indexPos) * N + rgsw_dim3_index];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    for (uint32_t rgsw_dim1_index = 0; rgsw_dim1_index < digitsG2; ++rgsw_dim1_index) {
        BasicInteger dct_coeff = dct[rgsw_dim1_index * N + rgsw_dim3_index];
        const size_t key_base =
            ek_dim3_index * static_cast<size_t>(digitsG2) * 2 * N + static_cast<size_t>(rgsw_dim1_index) * 2 * N + rgsw_dim3_index;

        const BasicInteger k0 = ek[key_base];
        const BasicInteger k0_shoup = ek_shoup[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + N;
        const BasicInteger k1 = ek[key_base1];
        const BasicInteger k1_shoup = ek_shoup[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, pos_monic_coeff);

    BasicInteger acc_t0 = acc[0 * N + rgsw_dim3_index] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);

    phantom::arith::csub_q(acc_t0, Q);
    acc[0 * N + rgsw_dim3_index] = acc_t0;


    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, pos_monic_coeff);
    BasicInteger acc_t1 = acc[1 * N + rgsw_dim3_index] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[1 * N + rgsw_dim3_index] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Grouped(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek0,
                                                  const BasicInteger* ek0_shoup, const BasicInteger* monic_polys, size_t N,
                                                  const DModulus* mod, uint32_t digitsG2, size_t key_group,
                                                  size_t lweIndex, const uint32_t* d_indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};
    const size_t coeff      = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = d_indexPos[lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    const size_t key_stride = static_cast<size_t>(digitsG2) * 2u * N;
    const size_t key_base0 = key_group * key_stride;
    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
        const size_t key_base = key_base0 + static_cast<size_t>(d) * 2u * N + coeff;

        const BasicInteger k0 = ek0[key_base];
        const BasicInteger k0_shoup = ek0_shoup[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + N;
        const BasicInteger k1 = ek0[key_base1];
        const BasicInteger k1_shoup = ek0_shoup[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 = acc[N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_MB2(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                              const BasicInteger* ek_shoup, const BasicInteger* ek_pair,
                                              const BasicInteger* ek_pair_shoup, const BasicInteger* monic_polys, size_t N,
                                              const DModulus* mod, uint32_t digitsG2, size_t lweIndex0, size_t lweIndex1,
                                              const uint32_t* d_indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};
    const size_t coeff      = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[lweIndex0];
    const uint32_t pos1 = d_indexPos[lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t key_stride = static_cast<size_t>(digitsG2) * 2u * N;
    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];

        const size_t base0 = lweIndex0 * key_stride + static_cast<size_t>(d) * 2u * N + coeff;
        const size_t base1 = lweIndex1 * key_stride + static_cast<size_t>(d) * 2u * N + coeff;

        const BasicInteger k0 = ek[base0];
        const BasicInteger k0_shoup = ek_shoup[base0];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const size_t base0_1 = base0 + N;
        const BasicInteger k1 = ek[base0_1];
        const BasicInteger k1_shoup = ek_shoup[base0_1];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const BasicInteger k0b = ek[base1];
        const BasicInteger k0b_shoup = ek_shoup[base1];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const size_t base1_1 = base1 + N;
        const BasicInteger k1b = ek[base1_1];
        const BasicInteger k1b_shoup = ek_shoup[base1_1];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair[base0];
        const BasicInteger kp0_shoup = ek_pair_shoup[base0];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair[base0_1];
        const BasicInteger kp1_shoup = ek_pair_shoup[base0_1];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    BasicInteger acc0 = acc[coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[coeff] = acc0;

    BasicInteger acc1 = acc[N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Grouped(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek0,
                                                      const BasicInteger* ek0_shoup, const BasicInteger* ek1,
                                                      const BasicInteger* ek1_shoup, const BasicInteger* ek_pair,
                                                      const BasicInteger* ek_pair_shoup, const BasicInteger* monic_polys,
                                                      size_t N, const DModulus* mod, uint32_t digitsG2, size_t key_group,
                                                      size_t lweIndex0, size_t lweIndex1, const uint32_t* d_indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};
    const size_t coeff      = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[lweIndex0];
    const uint32_t pos1 = d_indexPos[lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t key_stride = static_cast<size_t>(digitsG2) * 2u * N;
    const size_t base = key_group * key_stride;
    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * N + coeff;

        const BasicInteger k0 = ek0[key_base];
        const BasicInteger k0_shoup = ek0_shoup[key_base];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const size_t key_base1 = key_base + N;
        const BasicInteger k1 = ek0[key_base1];
        const BasicInteger k1_shoup = ek0_shoup[key_base1];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const BasicInteger k0b = ek1[key_base];
        const BasicInteger k0b_shoup = ek1_shoup[key_base];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const BasicInteger k1b = ek1[key_base1];
        const BasicInteger k1b_shoup = ek1_shoup[key_base1];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair[key_base];
        const BasicInteger kp0_shoup = ek_pair_shoup[key_base];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair[key_base1];
        const BasicInteger kp1_shoup = ek_pair_shoup[key_base1];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    BasicInteger acc0 = acc[coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[coeff] = acc0;

    BasicInteger acc1 = acc[N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_SoA(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                  const BasicInteger* ek_shoup_soa, const BasicInteger* ek_pair_soa,
                                                  const BasicInteger* ek_pair_shoup_soa, const BasicInteger* monic_polys,
                                                  size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                  size_t lweIndex0, size_t lweIndex1, const uint32_t* d_indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};
    const size_t coeff      = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[lweIndex0];
    const uint32_t pos1 = d_indexPos[lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base0 = ((lweIndex0 * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t base1 = ((lweIndex1 * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t basep = base0;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
        const size_t key_base0 = base0 + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_base1 = base1 + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_basep = basep + static_cast<size_t>(d) * 2u * 32u;

        const BasicInteger k0 = ek_soa[key_base0];
        const BasicInteger k0_shoup = ek_shoup_soa[key_base0];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const size_t key_base0_1 = key_base0 + 32u;
        const BasicInteger k1 = ek_soa[key_base0_1];
        const BasicInteger k1_shoup = ek_shoup_soa[key_base0_1];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const BasicInteger k0b = ek_soa[key_base1];
        const BasicInteger k0b_shoup = ek_shoup_soa[key_base1];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const size_t key_base1_1 = key_base1 + 32u;
        const BasicInteger k1b = ek_soa[key_base1_1];
        const BasicInteger k1b_shoup = ek_shoup_soa[key_base1_1];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair_soa[key_basep];
        const BasicInteger kp0_shoup = ek_pair_shoup_soa[key_basep];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const size_t key_basep_1 = key_basep + 32u;
        const BasicInteger kp1 = ek_pair_soa[key_basep_1];
        const BasicInteger kp1_shoup = ek_pair_shoup_soa[key_basep_1];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    BasicInteger acc0 = acc[coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[coeff] = acc0;

    BasicInteger acc1 = acc[N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_SoA(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek0_soa,
                                                          const BasicInteger* ek0_shoup_soa, const BasicInteger* ek1_soa,
                                                          const BasicInteger* ek1_shoup_soa, const BasicInteger* ek_pair_soa,
                                                          const BasicInteger* ek_pair_shoup_soa, const BasicInteger* monic_polys,
                                                          size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                          size_t key_group, size_t lweIndex0, size_t lweIndex1,
                                                          const uint32_t* d_indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};
    const size_t coeff      = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[lweIndex0];
    const uint32_t pos1 = d_indexPos[lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((key_group * tiles + tile) * digitsG2) * 2u * 32u + lane;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;

        const BasicInteger k0 = ek0_soa[key_base];
        const BasicInteger k0_shoup = ek0_shoup_soa[key_base];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const BasicInteger k1 = ek0_soa[key_base + 32u];
        const BasicInteger k1_shoup = ek0_shoup_soa[key_base + 32u];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const BasicInteger k0b = ek1_soa[key_base];
        const BasicInteger k0b_shoup = ek1_shoup_soa[key_base];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const BasicInteger k1b = ek1_soa[key_base + 32u];
        const BasicInteger k1b_shoup = ek1_shoup_soa[key_base + 32u];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair_soa[key_base];
        const BasicInteger kp0_shoup = ek_pair_shoup_soa[key_base];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair_soa[key_base + 32u];
        const BasicInteger kp1_shoup = ek_pair_shoup_soa[key_base + 32u];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    BasicInteger acc0 = acc[coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[coeff] = acc0;

    BasicInteger acc1 = acc[N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_SoA_Smem(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                       const BasicInteger* ek_shoup_soa, const BasicInteger* ek_pair_soa,
                                                       const BasicInteger* ek_pair_shoup_soa, const BasicInteger* monic_polys,
                                                       size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                       size_t lweIndex0, size_t lweIndex1, const uint32_t* d_indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};
    const size_t coeff      = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[lweIndex0];
    const uint32_t pos1 = d_indexPos[lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base0 = ((lweIndex0 * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t base1 = ((lweIndex1 * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t basep = base0;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 8u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base0 = base0 + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_base1 = base1 + static_cast<size_t>(d) * 2u * 32u;

        cirbts_cp_async_8(buf + lane + 0u * 32u, ek_soa + key_base0);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek_shoup_soa + key_base0);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek_soa + key_base0 + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek_shoup_soa + key_base0 + 32u);

        cirbts_cp_async_8(buf + lane + 4u * 32u, ek_soa + key_base1);
        cirbts_cp_async_8(buf + lane + 5u * 32u, ek_shoup_soa + key_base1);
        cirbts_cp_async_8(buf + lane + 6u * 32u, ek_soa + key_base1 + 32u);
        cirbts_cp_async_8(buf + lane + 7u * 32u, ek_shoup_soa + key_base1 + 32u);

        cirbts_cp_async_commit();
    };

    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];
            tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            const BasicInteger k0b = cur[lane + 4u * 32u];
            const BasicInteger k0b_shoup = cur[lane + 5u * 32u];
            tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
            const BasicInteger k1b = cur[lane + 6u * 32u];
            const BasicInteger k1b_shoup = cur[lane + 7u * 32u];
            tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

            const size_t key_basep = basep + static_cast<size_t>(d) * 2u * 32u;
            const BasicInteger kp0 = ek_pair_soa[key_basep];
            const BasicInteger kp0_shoup = ek_pair_shoup_soa[key_basep];
            tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
            const BasicInteger kp1 = ek_pair_soa[key_basep + 32u];
            const BasicInteger kp1_shoup = ek_pair_shoup_soa[key_basep + 32u];
            tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);

            buf ^= 1u;
        }
    }

    BasicInteger acc0 = acc[coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[coeff] = acc0;

    BasicInteger acc1 = acc[N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_SoA_Smem(BasicInteger* acc, const BasicInteger* dct,
                                                               const BasicInteger* ek0_soa, const BasicInteger* ek0_shoup_soa,
                                                               const BasicInteger* ek1_soa, const BasicInteger* ek1_shoup_soa,
                                                               const BasicInteger* ek_pair_soa, const BasicInteger* ek_pair_shoup_soa,
                                                               const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                               const DModulus* mod, uint32_t digitsG2, size_t key_group,
                                                               size_t lweIndex0, size_t lweIndex1, const uint32_t* d_indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};
    const size_t coeff      = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[lweIndex0];
    const uint32_t pos1 = d_indexPos[lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((key_group * tiles + tile) * digitsG2) * 2u * 32u + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 8u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;

        cirbts_cp_async_8(buf + lane + 0u * 32u, ek0_soa + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek0_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek0_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek0_shoup_soa + key_base + 32u);

        cirbts_cp_async_8(buf + lane + 4u * 32u, ek1_soa + key_base);
        cirbts_cp_async_8(buf + lane + 5u * 32u, ek1_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 6u * 32u, ek1_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 7u * 32u, ek1_shoup_soa + key_base + 32u);

        cirbts_cp_async_commit();
    };

    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];
            tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            const BasicInteger k0b = cur[lane + 4u * 32u];
            const BasicInteger k0b_shoup = cur[lane + 5u * 32u];
            tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
            const BasicInteger k1b = cur[lane + 6u * 32u];
            const BasicInteger k1b_shoup = cur[lane + 7u * 32u];
            tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

            const size_t key_basep = base + static_cast<size_t>(d) * 2u * 32u;
            const BasicInteger kp0 = ek_pair_soa[key_basep];
            const BasicInteger kp0_shoup = ek_pair_shoup_soa[key_basep];
            tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
            const BasicInteger kp1 = ek_pair_soa[key_basep + 32u];
            const BasicInteger kp1_shoup = ek_pair_shoup_soa[key_basep + 32u];
            tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);

            buf ^= 1u;
        }
    }

    BasicInteger acc0 = acc[coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[coeff] = acc0;

    BasicInteger acc1 = acc[N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[N + coeff] = acc1;
}

__global__ void kernel_EvalAccPipeline_Binary_GINX(BasicInteger* acc, BasicInteger* tmp,
                                                   const BasicInteger* ek, const BasicInteger* ek_shoup,
                                                   const BasicInteger* monic_polys,
                                                   const BasicInteger* itwiddles, const BasicInteger* itwiddles_shoup,
                                                   const BasicInteger* twiddles, const BasicInteger* twiddles_shoup,
                                                   const DModulus* modulus, const BasicInteger* scalar,
                                                   const BasicInteger* scalar_shoup, size_t N, uint32_t baseG,
                                                   uint32_t digitsGA, size_t lweIndex, const uint32_t* indexPos) {
    extern __shared__ BasicInteger buffer[];
    BasicInteger* coeff = buffer;
    BasicInteger* nttbuf = buffer + N;

    const BasicInteger Q          = modulus[0].value();
    BasicInteger Q_ratio[2]       = {modulus[0].const_ratio()[0], modulus[0].const_ratio()[1]};
    const BasicInteger scalar_    = scalar[0];
    const BasicInteger scalar_sh  = scalar_shoup[0];
    const uint32_t digitsG2       = digitsGA << 1;
    const size_t tid              = threadIdx.x;
    const size_t tpb              = blockDim.x;

    const uint32_t indexPosLocal = indexPos[lweIndex];

    // Zero per-coefficient accumulators in global scratch (tmp = [2][N]).
    for (size_t coeff_idx = tid; coeff_idx < N; coeff_idx += tpb) {
        tmp[coeff_idx]       = 0;
        tmp[N + coeff_idx]   = 0;
    }
    __syncthreads();

    // Process c0 and c1 sequentially to fit in shared memory.
    for (uint32_t ct_poly = 0; ct_poly < 2; ++ct_poly) {
        // INWT: eval -> coeff for the current polynomial (acc is in eval domain).
        for (size_t numOfGroups = N / 2; numOfGroups >= 1; numOfGroups >>= 1) {
            const size_t pairsInGroup = N / numOfGroups / 2;
            for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
                const size_t k        = butterflyId / pairsInGroup;
                const size_t j        = butterflyId % pairsInGroup;
                const size_t glbIdx   = 2 * k * pairsInGroup + j;

                const BasicInteger psi       = itwiddles[numOfGroups + k];
                const BasicInteger psi_shoup = itwiddles_shoup[numOfGroups + k];

                BasicInteger a0{};
                BasicInteger a1{};
                if (numOfGroups == N / 2) {
                    const size_t base = static_cast<size_t>(ct_poly) * N;
                    a0 = acc[base + glbIdx];
                    a1 = acc[base + glbIdx + pairsInGroup];
                } else {
                    a0 = nttbuf[glbIdx];
                    a1 = nttbuf[glbIdx + pairsInGroup];
                }

                phantom::arith::gs_butterfly(a0, a1, psi, psi_shoup, Q);

                if (numOfGroups == 1) {
                    phantom::arith::csub_q(a0, Q);
                    phantom::arith::csub_q(a1, Q);
                    a0 = phantom::arith::multiply_and_reduce_shoup(a0, scalar_, scalar_sh, Q);
                    coeff[glbIdx]                 = a0;
                    coeff[glbIdx + pairsInGroup]  = a1;
                } else {
                    nttbuf[glbIdx]                = a0;
                    nttbuf[glbIdx + pairsInGroup] = a1;
                }
            }
            __syncthreads();
        }

        // For each digit, compute digit polynomial (NTT domain) and update accumulator.
        for (uint32_t digit_idx = 0; digit_idx < digitsGA; ++digit_idx) {
            // Forward NTT with on-the-fly digit extraction.
            {
                const size_t pairsInGroup0    = N / 2;
                const BasicInteger psi0       = twiddles[1];
                const BasicInteger psi0_shoup = twiddles_shoup[1];
                for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
                    const size_t glbIdx0 = butterflyId;
                    BasicInteger samples0 = device_SignedDigitDecompose2_SelectDigit(
                        coeff[glbIdx0], Q, baseG, digitsGA, digit_idx);
                    BasicInteger samples1 = device_SignedDigitDecompose2_SelectDigit(
                        coeff[glbIdx0 + pairsInGroup0], Q, baseG, digitsGA, digit_idx);
                    phantom::arith::ct_butterfly(samples0, samples1, psi0, psi0_shoup, Q);
                    nttbuf[glbIdx0]                 = samples0;
                    nttbuf[glbIdx0 + pairsInGroup0] = samples1;
                }
                __syncthreads();
            }

            for (size_t numOfGroups = 2; numOfGroups < N / 2; numOfGroups <<= 1) {
                const size_t pairsInGroup = N / numOfGroups / 2;
                for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
                    const size_t k      = butterflyId / pairsInGroup;
                    const size_t j      = butterflyId % pairsInGroup;
                    const size_t glbIdx = 2 * k * pairsInGroup + j;
                    const BasicInteger psi       = twiddles[numOfGroups + k];
                    const BasicInteger psi_shoup = twiddles_shoup[numOfGroups + k];

                    BasicInteger a0 = nttbuf[glbIdx];
                    BasicInteger a1 = nttbuf[glbIdx + pairsInGroup];
                    phantom::arith::ct_butterfly(a0, a1, psi, psi_shoup, Q);
                    nttbuf[glbIdx]                = a0;
                    nttbuf[glbIdx + pairsInGroup] = a1;
                }
                __syncthreads();
            }

            {
                const size_t numOfGroups_last     = N / 2;
                const size_t pairsInGroup_last    = 1;
                for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
                    const size_t k_last               = butterflyId;
                    const BasicInteger psi_last       = twiddles[numOfGroups_last + k_last];
                    const BasicInteger psi_last_shoup = twiddles_shoup[numOfGroups_last + k_last];
                    const size_t glbIdx_last          = 2 * butterflyId;

                    BasicInteger a0_last = nttbuf[glbIdx_last];
                    BasicInteger a1_last = nttbuf[glbIdx_last + pairsInGroup_last];
                    phantom::arith::ct_butterfly(a0_last, a1_last, psi_last, psi_last_shoup, Q);
                    phantom::arith::csub_q(a0_last, Q << 1);
                    phantom::arith::csub_q(a0_last, Q);
                    phantom::arith::csub_q(a1_last, Q << 1);
                    phantom::arith::csub_q(a1_last, Q);

                    nttbuf[glbIdx_last]                    = a0_last;
                    nttbuf[glbIdx_last + pairsInGroup_last] = a1_last;
                }
                __syncthreads();
            }

            const uint32_t rgsw_dim1_index = (digit_idx << 1) | ct_poly;
            for (size_t coeff_idx = tid; coeff_idx < N; coeff_idx += tpb) {
                const BasicInteger dct_coeff = nttbuf[coeff_idx];
                const size_t key_base = lweIndex * static_cast<size_t>(digitsG2) * 2 * N +
                                        static_cast<size_t>(rgsw_dim1_index) * 2 * N + coeff_idx;
                const BasicInteger k0 = ek[key_base];
                const BasicInteger k0_shoup = ek_shoup[key_base];
                const BasicInteger k1 = ek[key_base + N];
                const BasicInteger k1_shoup = ek_shoup[key_base + N];

                const BasicInteger tmp0 = phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
                const BasicInteger tmp1 = phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

                BasicInteger acc_tmp0 = tmp[coeff_idx] + tmp0;
                BasicInteger acc_tmp1 = tmp[N + coeff_idx] + tmp1;
                phantom::arith::csub_q(acc_tmp0, Q);
                phantom::arith::csub_q(acc_tmp1, Q);
                tmp[coeff_idx]     = acc_tmp0;
                tmp[N + coeff_idx] = acc_tmp1;
            }
            __syncthreads();
        }
    }

    // Apply monic and update accumulator once (after all digits from both polys).
    for (size_t coeff_idx = tid; coeff_idx < N; coeff_idx += tpb) {
        const BasicInteger monic_coeff = monic_polys[static_cast<size_t>(indexPosLocal) * N + coeff_idx];
        const BasicInteger sum0 = tmp[coeff_idx];
        const BasicInteger sum1 = tmp[N + coeff_idx];

        phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(sum0, monic_coeff);
        BasicInteger acc_t0 = acc[0 * N + coeff_idx] +
                              phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
        phantom::arith::csub_q(acc_t0, Q);
        acc[0 * N + coeff_idx] = acc_t0;

        phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(sum1, monic_coeff);
        BasicInteger acc_t1 = acc[1 * N + coeff_idx] +
                              phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
        phantom::arith::csub_q(acc_t1, Q);
        acc[1 * N + coeff_idx] = acc_t1;
    }
}

__global__ void kernel_EvalAccPipeline_Binary_Grouped(BasicInteger* acc, BasicInteger* tmp,
                                                      const BasicInteger* ek, const BasicInteger* ek_shoup,
                                                      const BasicInteger* monic_polys,
                                                      const BasicInteger* itwiddles, const BasicInteger* itwiddles_shoup,
                                                      const BasicInteger* twiddles, const BasicInteger* twiddles_shoup,
                                                      const DModulus* modulus, const BasicInteger* scalar,
                                                      const BasicInteger* scalar_shoup, size_t N, uint32_t baseG,
                                                      uint32_t digitsGA, size_t key_group, size_t lweIndex,
                                                      const uint32_t* indexPos) {
    extern __shared__ BasicInteger buffer[];
    BasicInteger* coeff = buffer;
    BasicInteger* nttbuf = buffer + N;

    const BasicInteger Q          = modulus[0].value();
    BasicInteger Q_ratio[2]       = {modulus[0].const_ratio()[0], modulus[0].const_ratio()[1]};
    const BasicInteger scalar_    = scalar[0];
    const BasicInteger scalar_sh  = scalar_shoup[0];
    const uint32_t digitsG2       = digitsGA << 1;
    const size_t tid              = threadIdx.x;
    const size_t tpb              = blockDim.x;

    const uint32_t indexPosLocal = indexPos[lweIndex];

    // Zero per-coefficient accumulators in global scratch (tmp = [2][N]).
    for (size_t coeff_idx = tid; coeff_idx < N; coeff_idx += tpb) {
        tmp[coeff_idx]       = 0;
        tmp[N + coeff_idx]   = 0;
    }
    __syncthreads();

    // Process c0 and c1 sequentially to fit in shared memory.
    for (uint32_t ct_poly = 0; ct_poly < 2; ++ct_poly) {
        // INWT: eval -> coeff for the current polynomial (acc is in eval domain).
        for (size_t numOfGroups = N / 2; numOfGroups >= 1; numOfGroups >>= 1) {
            const size_t pairsInGroup = N / numOfGroups / 2;
            for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
                const size_t k        = butterflyId / pairsInGroup;
                const size_t j        = butterflyId % pairsInGroup;
                const size_t glbIdx   = 2 * k * pairsInGroup + j;

                const BasicInteger psi       = itwiddles[numOfGroups + k];
                const BasicInteger psi_shoup = itwiddles_shoup[numOfGroups + k];

                BasicInteger a0{};
                BasicInteger a1{};
                if (numOfGroups == N / 2) {
                    const size_t base = static_cast<size_t>(ct_poly) * N;
                    a0 = acc[base + glbIdx];
                    a1 = acc[base + glbIdx + pairsInGroup];
                } else {
                    a0 = nttbuf[glbIdx];
                    a1 = nttbuf[glbIdx + pairsInGroup];
                }

                phantom::arith::gs_butterfly(a0, a1, psi, psi_shoup, Q);

                if (numOfGroups == 1) {
                    phantom::arith::csub_q(a0, Q);
                    phantom::arith::csub_q(a1, Q);
                    a0 = phantom::arith::multiply_and_reduce_shoup(a0, scalar_, scalar_sh, Q);
                    coeff[glbIdx]                 = a0;
                    coeff[glbIdx + pairsInGroup]  = a1;
                } else {
                    nttbuf[glbIdx]                = a0;
                    nttbuf[glbIdx + pairsInGroup] = a1;
                }
            }
            __syncthreads();
        }

        // For each digit, compute digit polynomial (NTT domain) and update accumulator.
        for (uint32_t digit_idx = 0; digit_idx < digitsGA; ++digit_idx) {
            // Forward NTT with on-the-fly digit extraction.
            {
                const size_t pairsInGroup0    = N / 2;
                const BasicInteger psi0       = twiddles[1];
                const BasicInteger psi0_shoup = twiddles_shoup[1];
                for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
                    const size_t glbIdx0 = butterflyId;
                    BasicInteger samples0 = device_SignedDigitDecompose2_SelectDigit(
                        coeff[glbIdx0], Q, baseG, digitsGA, digit_idx);
                    BasicInteger samples1 = device_SignedDigitDecompose2_SelectDigit(
                        coeff[glbIdx0 + pairsInGroup0], Q, baseG, digitsGA, digit_idx);
                    phantom::arith::ct_butterfly(samples0, samples1, psi0, psi0_shoup, Q);
                    nttbuf[glbIdx0]                 = samples0;
                    nttbuf[glbIdx0 + pairsInGroup0] = samples1;
                }
                __syncthreads();
            }

            for (size_t numOfGroups = 2; numOfGroups < N / 2; numOfGroups <<= 1) {
                const size_t pairsInGroup = N / numOfGroups / 2;
                for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
                    const size_t k      = butterflyId / pairsInGroup;
                    const size_t j      = butterflyId % pairsInGroup;
                    const size_t glbIdx = 2 * k * pairsInGroup + j;
                    const BasicInteger psi       = twiddles[numOfGroups + k];
                    const BasicInteger psi_shoup = twiddles_shoup[numOfGroups + k];

                    BasicInteger a0 = nttbuf[glbIdx];
                    BasicInteger a1 = nttbuf[glbIdx + pairsInGroup];
                    phantom::arith::ct_butterfly(a0, a1, psi, psi_shoup, Q);
                    nttbuf[glbIdx]                = a0;
                    nttbuf[glbIdx + pairsInGroup] = a1;
                }
                __syncthreads();
            }

            {
                const size_t numOfGroups_last     = N / 2;
                const size_t pairsInGroup_last    = 1;
                for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
                    const size_t k_last               = butterflyId;
                    const BasicInteger psi_last       = twiddles[numOfGroups_last + k_last];
                    const BasicInteger psi_last_shoup = twiddles_shoup[numOfGroups_last + k_last];
                    const size_t glbIdx_last          = 2 * butterflyId;

                    BasicInteger a0_last = nttbuf[glbIdx_last];
                    BasicInteger a1_last = nttbuf[glbIdx_last + pairsInGroup_last];
                    phantom::arith::ct_butterfly(a0_last, a1_last, psi_last, psi_last_shoup, Q);
                    phantom::arith::csub_q(a0_last, Q << 1);
                    phantom::arith::csub_q(a0_last, Q);
                    phantom::arith::csub_q(a1_last, Q << 1);
                    phantom::arith::csub_q(a1_last, Q);

                    nttbuf[glbIdx_last]                    = a0_last;
                    nttbuf[glbIdx_last + pairsInGroup_last] = a1_last;
                }
                __syncthreads();
            }

            const uint32_t rgsw_dim1_index = (digit_idx << 1) | ct_poly;
            for (size_t coeff_idx = tid; coeff_idx < N; coeff_idx += tpb) {
                const BasicInteger dct_coeff = nttbuf[coeff_idx];
                const size_t key_base = key_group * static_cast<size_t>(digitsG2) * 2 * N +
                                        static_cast<size_t>(rgsw_dim1_index) * 2 * N + coeff_idx;
                const BasicInteger k0 = ek[key_base];
                const BasicInteger k0_shoup = ek_shoup[key_base];
                const BasicInteger k1 = ek[key_base + N];
                const BasicInteger k1_shoup = ek_shoup[key_base + N];

                const BasicInteger tmp0 = phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
                const BasicInteger tmp1 = phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

                BasicInteger acc_tmp0 = tmp[coeff_idx] + tmp0;
                BasicInteger acc_tmp1 = tmp[N + coeff_idx] + tmp1;
                phantom::arith::csub_q(acc_tmp0, Q);
                phantom::arith::csub_q(acc_tmp1, Q);
                tmp[coeff_idx]     = acc_tmp0;
                tmp[N + coeff_idx] = acc_tmp1;
            }
            __syncthreads();
        }
    }

    // Apply monic and update accumulator once (after all digits from both polys).
    for (size_t coeff_idx = tid; coeff_idx < N; coeff_idx += tpb) {
        const BasicInteger monic_coeff = monic_polys[static_cast<size_t>(indexPosLocal) * N + coeff_idx];
        const BasicInteger sum0 = tmp[coeff_idx];
        const BasicInteger sum1 = tmp[N + coeff_idx];

        phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(sum0, monic_coeff);
        BasicInteger acc_t0 = acc[0 * N + coeff_idx] +
                              phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
        phantom::arith::csub_q(acc_t0, Q);
        acc[0 * N + coeff_idx] = acc_t0;

        phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(sum1, monic_coeff);
        BasicInteger acc_t1 = acc[1 * N + coeff_idx] +
                              phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
        phantom::arith::csub_q(acc_t1, Q);
        acc[1 * N + coeff_idx] = acc_t1;
    }
}

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
                                                          size_t lweIndex1, const uint32_t* indexPos) {
    extern __shared__ BasicInteger buffer[];
    BasicInteger* coeff = buffer;
    BasicInteger* nttbuf = buffer + N;

    const BasicInteger Q          = modulus[0].value();
    BasicInteger Q_ratio[2]       = {modulus[0].const_ratio()[0], modulus[0].const_ratio()[1]};
    const BasicInteger scalar_    = scalar[0];
    const BasicInteger scalar_sh  = scalar_shoup[0];
    const uint32_t digitsG2       = digitsGA << 1;
    const size_t tid              = threadIdx.x;
    const size_t tpb              = blockDim.x;

    // Zero per-coefficient accumulators in global scratch (tmp = [2][N]).
    for (size_t coeff_idx = tid; coeff_idx < N; coeff_idx += tpb) {
        tmp[coeff_idx]       = 0;
        tmp[N + coeff_idx]   = 0;
    }
    __syncthreads();

    const uint32_t pos0 = indexPos[lweIndex0];
    const uint32_t pos1 = indexPos[lweIndex1];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);

    // Process c0 and c1 sequentially to fit in shared memory.
    for (uint32_t ct_poly = 0; ct_poly < 2; ++ct_poly) {
        // INWT: eval -> coeff for the current polynomial (acc is in eval domain).
        for (size_t numOfGroups = N / 2; numOfGroups >= 1; numOfGroups >>= 1) {
            const size_t pairsInGroup = N / numOfGroups / 2;
            for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
                const size_t k        = butterflyId / pairsInGroup;
                const size_t j        = butterflyId % pairsInGroup;
                const size_t glbIdx   = 2 * k * pairsInGroup + j;

                const BasicInteger psi       = itwiddles[numOfGroups + k];
                const BasicInteger psi_shoup = itwiddles_shoup[numOfGroups + k];

                BasicInteger a0{};
                BasicInteger a1{};
                if (numOfGroups == N / 2) {
                    const size_t base = static_cast<size_t>(ct_poly) * N;
                    a0 = acc[base + glbIdx];
                    a1 = acc[base + glbIdx + pairsInGroup];
                } else {
                    a0 = nttbuf[glbIdx];
                    a1 = nttbuf[glbIdx + pairsInGroup];
                }

                phantom::arith::gs_butterfly(a0, a1, psi, psi_shoup, Q);

                if (numOfGroups == 1) {
                    phantom::arith::csub_q(a0, Q);
                    phantom::arith::csub_q(a1, Q);
                    a0 = phantom::arith::multiply_and_reduce_shoup(a0, scalar_, scalar_sh, Q);
                    coeff[glbIdx]                 = a0;
                    coeff[glbIdx + pairsInGroup]  = a1;
                } else {
                    nttbuf[glbIdx]                = a0;
                    nttbuf[glbIdx + pairsInGroup] = a1;
                }
            }
            __syncthreads();
        }

        // For each digit, compute digit polynomial (NTT domain) and update accumulator directly.
        for (uint32_t digit_idx = 0; digit_idx < digitsGA; ++digit_idx) {
            // Forward NTT with on-the-fly digit extraction.
            {
                const size_t pairsInGroup0    = N / 2;
                const BasicInteger psi0       = twiddles[1];
                const BasicInteger psi0_shoup = twiddles_shoup[1];
                for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
                    const size_t glbIdx0 = butterflyId;
                    BasicInteger samples0 = device_SignedDigitDecompose2_SelectDigit(
                        coeff[glbIdx0], Q, baseG, digitsGA, digit_idx);
                    BasicInteger samples1 = device_SignedDigitDecompose2_SelectDigit(
                        coeff[glbIdx0 + pairsInGroup0], Q, baseG, digitsGA, digit_idx);
                    phantom::arith::ct_butterfly(samples0, samples1, psi0, psi0_shoup, Q);
                    nttbuf[glbIdx0]                 = samples0;
                    nttbuf[glbIdx0 + pairsInGroup0] = samples1;
                }
                __syncthreads();
            }

            for (size_t numOfGroups = 2; numOfGroups < N / 2; numOfGroups <<= 1) {
                const size_t pairsInGroup = N / numOfGroups / 2;
                for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
                    const size_t k      = butterflyId / pairsInGroup;
                    const size_t j      = butterflyId % pairsInGroup;
                    const size_t glbIdx = 2 * k * pairsInGroup + j;
                    const BasicInteger psi       = twiddles[numOfGroups + k];
                    const BasicInteger psi_shoup = twiddles_shoup[numOfGroups + k];

                    BasicInteger a0 = nttbuf[glbIdx];
                    BasicInteger a1 = nttbuf[glbIdx + pairsInGroup];
                    phantom::arith::ct_butterfly(a0, a1, psi, psi_shoup, Q);
                    nttbuf[glbIdx]                = a0;
                    nttbuf[glbIdx + pairsInGroup] = a1;
                }
                __syncthreads();
            }

            {
                const size_t numOfGroups_last     = N / 2;
                const size_t pairsInGroup_last    = 1;
                for (size_t butterflyId = tid; butterflyId < N / 2; butterflyId += tpb) {
                    const size_t k_last               = butterflyId;
                    const BasicInteger psi_last       = twiddles[numOfGroups_last + k_last];
                    const BasicInteger psi_last_shoup = twiddles_shoup[numOfGroups_last + k_last];
                    const size_t glbIdx_last          = 2 * butterflyId;

                    BasicInteger a0_last = nttbuf[glbIdx_last];
                    BasicInteger a1_last = nttbuf[glbIdx_last + pairsInGroup_last];
                    phantom::arith::ct_butterfly(a0_last, a1_last, psi_last, psi_last_shoup, Q);
                    phantom::arith::csub_q(a0_last, Q << 1);
                    phantom::arith::csub_q(a0_last, Q);
                    phantom::arith::csub_q(a1_last, Q << 1);
                    phantom::arith::csub_q(a1_last, Q);

                    nttbuf[glbIdx_last]                    = a0_last;
                    nttbuf[glbIdx_last + pairsInGroup_last] = a1_last;
                }
                __syncthreads();
            }

            const uint32_t rgsw_dim1_index = (digit_idx << 1) | ct_poly;
            for (size_t coeff_idx = tid; coeff_idx < N; coeff_idx += tpb) {
                const BasicInteger dct_coeff = nttbuf[coeff_idx];
                const size_t key_base = key_group * static_cast<size_t>(digitsG2) * 2 * N +
                                        static_cast<size_t>(rgsw_dim1_index) * 2 * N + coeff_idx;

                const BasicInteger k0 = ek0[key_base];
                const BasicInteger k0_shoup = ek0_shoup[key_base];
                const BasicInteger k1v = ek0[key_base + N];
                const BasicInteger k1_shoup = ek0_shoup[key_base + N];
                const BasicInteger k0b = ek1[key_base];
                const BasicInteger k0b_shoup = ek1_shoup[key_base];
                const BasicInteger k1b = ek1[key_base + N];
                const BasicInteger k1b_shoup = ek1_shoup[key_base + N];
                const BasicInteger kp0 = ek_pair[key_base];
                const BasicInteger kp0_shoup = ek_pair_shoup[key_base];
                const BasicInteger kp1 = ek_pair[key_base + N];
                const BasicInteger kp1_shoup = ek_pair_shoup[key_base + N];

                const BasicInteger tmp0_0 = phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
                const BasicInteger tmp1_0 = phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1v, k1_shoup, Q);
                const BasicInteger tmp0_1 = phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
                const BasicInteger tmp1_1 = phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);
                const BasicInteger tmp0_p = phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
                const BasicInteger tmp1_p = phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);

                const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff_idx];
                const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff_idx];
                const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff_idx];
                BasicInteger monic01 = monic_sum;
                monic01 += (Q - monic0);
                phantom::arith::csub_q(monic01, Q);
                monic01 += (Q - monic1);
                phantom::arith::csub_q(monic01, Q);

                BasicInteger delta0 = tmp[coeff_idx];
                phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
                delta0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
                phantom::arith::csub_q(delta0, Q);
                prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
                delta0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
                phantom::arith::csub_q(delta0, Q);
                prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
                delta0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
                phantom::arith::csub_q(delta0, Q);
                tmp[coeff_idx] = delta0;

                BasicInteger delta1 = tmp[N + coeff_idx];
                phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
                delta1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
                phantom::arith::csub_q(delta1, Q);
                prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
                delta1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
                phantom::arith::csub_q(delta1, Q);
                prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
                delta1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
                phantom::arith::csub_q(delta1, Q);
                tmp[N + coeff_idx] = delta1;
            }
            __syncthreads();
        }
    }

    // Apply accumulated deltas once (after all digits from both polys).
    for (size_t coeff_idx = tid; coeff_idx < N; coeff_idx += tpb) {
        BasicInteger acc0 = acc[coeff_idx] + tmp[coeff_idx];
        phantom::arith::csub_q(acc0, Q);
        acc[coeff_idx] = acc0;

        BasicInteger acc1 = acc[N + coeff_idx] + tmp[N + coeff_idx];
        phantom::arith::csub_q(acc1, Q);
        acc[N + coeff_idx] = acc1;
    }
}

__global__ void kernel_EvalAccCore_Binary_Delta(BasicInteger* acc, BasicInteger* delta, const BasicInteger* dct,
                                                const BasicInteger* ek, const BasicInteger* ek_shoup,
                                                const BasicInteger* monic_polys, size_t N, const DModulus* mod,
                                                uint32_t digitsG2, size_t ek_dim3_index, const uint32_t* d_indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};
    size_t rgsw_dim3_index  = blockIdx.x * blockDim.x + threadIdx.x;
    if (rgsw_dim3_index >= N) {
        return;
    }

    const uint32_t indexPos = d_indexPos[ek_dim3_index];
    const BasicInteger pos_monic_coeff = monic_polys[static_cast<size_t>(indexPos) * N + rgsw_dim3_index];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    for (uint32_t rgsw_dim1_index = 0; rgsw_dim1_index < digitsG2; ++rgsw_dim1_index) {
        BasicInteger dct_coeff = dct[rgsw_dim1_index * N + rgsw_dim3_index];
        const size_t key_base =
            ek_dim3_index * static_cast<size_t>(digitsG2) * 2 * N + static_cast<size_t>(rgsw_dim1_index) * 2 * N + rgsw_dim3_index;

        const BasicInteger k0 = ek[key_base];
        const BasicInteger k0_shoup = ek_shoup[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + N;
        const BasicInteger k1 = ek[key_base1];
        const BasicInteger k1_shoup = ek_shoup[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, pos_monic_coeff);
    BasicInteger delta0 = phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    BasicInteger acc_t0 = acc[0 * N + rgsw_dim3_index] + delta0;
    phantom::arith::csub_q(acc_t0, Q);
    acc[0 * N + rgsw_dim3_index] = acc_t0;
    delta[0 * N + rgsw_dim3_index] = delta0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, pos_monic_coeff);
    BasicInteger delta1 = phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    BasicInteger acc_t1 = acc[1 * N + rgsw_dim3_index] + delta1;
    phantom::arith::csub_q(acc_t1, Q);
    acc[1 * N + rgsw_dim3_index] = acc_t1;
    delta[1 * N + rgsw_dim3_index] = delta1;
}

__global__ void kernel_SwizzleRFKey(BasicInteger* out, const BasicInteger* in, size_t N, uint32_t digitsG2,
                                    uint32_t tiles, size_t lwe_n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = lwe_n * static_cast<size_t>(digitsG2) * 2 * N;
    if (idx >= total) {
        return;
    }

    size_t tmp = idx;
    const size_t coeff = tmp % N;
    tmp /= N;
    const size_t poly = tmp % 2;
    tmp /= 2;
    const size_t digit = tmp % digitsG2;
    tmp /= digitsG2;
    const size_t lweIndex = tmp;

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t out_idx = (((lweIndex * tiles + tile) * digitsG2 + digit) * 2 + poly) * 32 + lane;
    out[out_idx] = in[idx];
}

__global__ void kernel_EvalAccCore_Binary_Grouped_SoA(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek0_soa,
                                                      const BasicInteger* ek0_shoup_soa, const BasicInteger* monic_polys,
                                                      size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                      size_t key_group, size_t lweIndex, const uint32_t* indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = indexPos[lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((key_group * tiles + tile) * digitsG2) * 2u * 32u + lane;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        const BasicInteger k0 = ek0_soa[key_base];
        const BasicInteger k0_shoup = ek0_shoup_soa[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + 32u;
        const BasicInteger k1 = ek0_soa[key_base1];
        const BasicInteger k1_shoup = ek0_shoup_soa[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 = acc[N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Grouped_SoA_Smem(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek0_soa,
                                                           const BasicInteger* ek0_shoup_soa, const BasicInteger* monic_polys,
                                                           size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                           size_t key_group, size_t lweIndex, const uint32_t* indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = indexPos[lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((key_group * tiles + tile) * digitsG2) * 2u * 32u + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 4u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, ek0_soa + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek0_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek0_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek0_shoup_soa + key_base + 32u);
        cirbts_cp_async_commit();
    };

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];

            tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            buf ^= 1u;
        }
    }

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 = acc[N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[N + coeff] = acc_t1;
}

template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_Swizzle(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_swizzle,
                                                  const BasicInteger* ek_shoup_swizzle, const BasicInteger* monic_polys,
                                                  size_t N, uint32_t tiles, const DModulus* mod, size_t lweIndex,
                                                  const uint32_t* indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = indexPos[lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((lweIndex * tiles + tile) * DigitsG2) * 2 * 32 + lane;

#pragma unroll
    for (uint32_t d = 0; d < DigitsG2; ++d) {
        const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2 * 32;
        const BasicInteger k0 = ek_swizzle[key_base];
        const BasicInteger k0_shoup = ek_shoup_swizzle[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + 32;
        const BasicInteger k1 = ek_swizzle[key_base1];
        const BasicInteger k1_shoup = ek_shoup_swizzle[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 = acc[N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[N + coeff] = acc_t1;
}

template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_Grouped_Swizzle(BasicInteger* acc, const BasicInteger* dct,
                                                          const BasicInteger* ek_swizzle,
                                                          const BasicInteger* ek_shoup_swizzle,
                                                          const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                          const DModulus* mod, size_t key_group, size_t lweIndex,
                                                          const uint32_t* indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = indexPos[lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((key_group * tiles + tile) * DigitsG2) * 2u * 32u + lane;

#pragma unroll
    for (uint32_t d = 0; d < DigitsG2; ++d) {
        const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        const BasicInteger k0 = ek_swizzle[key_base];
        const BasicInteger k0_shoup = ek_shoup_swizzle[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + 32u;
        const BasicInteger k1 = ek_swizzle[key_base1];
        const BasicInteger k1_shoup = ek_shoup_swizzle[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 = acc[N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[N + coeff] = acc_t1;
}

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
                                                              size_t lweIndex1, const uint32_t* indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = indexPos[lweIndex0];
    const uint32_t pos1 = indexPos[lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((key_group * tiles + tile) * DigitsG2) * 2u * 32u + lane;

#pragma unroll
    for (uint32_t d = 0; d < DigitsG2; ++d) {
        const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;

        const BasicInteger k0 = ek0_swizzle[key_base];
        const BasicInteger k0_shoup = ek0_shoup_swizzle[key_base];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const BasicInteger k1 = ek0_swizzle[key_base + 32u];
        const BasicInteger k1_shoup = ek0_shoup_swizzle[key_base + 32u];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const BasicInteger k0b = ek1_swizzle[key_base];
        const BasicInteger k0b_shoup = ek1_shoup_swizzle[key_base];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const BasicInteger k1b = ek1_swizzle[key_base + 32u];
        const BasicInteger k1b_shoup = ek1_shoup_swizzle[key_base + 32u];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair_swizzle[key_base];
        const BasicInteger kp0_shoup = ek_pair_shoup_swizzle[key_base];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair_swizzle[key_base + 32u];
        const BasicInteger kp1_shoup = ek_pair_shoup_swizzle[key_base + 32u];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    BasicInteger acc0 = acc[coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[coeff] = acc0;

    BasicInteger acc1 = acc[N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_SoA(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                              const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys, size_t N,
                                              uint32_t tiles, const DModulus* mod, uint32_t digitsG2, size_t lweIndex,
                                              const uint32_t* indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = indexPos[lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((lweIndex * tiles + tile) * digitsG2) * 2 * 32 + lane;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2 * 32;
        const BasicInteger k0 = ek_soa[key_base];
        const BasicInteger k0_shoup = ek_shoup_soa[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + 32;
        const BasicInteger k1 = ek_soa[key_base1];
        const BasicInteger k1_shoup = ek_shoup_soa[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 = acc[N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_SoA_Smem(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                   const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys, size_t N,
                                                   uint32_t tiles, const DModulus* mod, uint32_t digitsG2, size_t lweIndex,
                                                   const uint32_t* indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = indexPos[lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((lweIndex * tiles + tile) * digitsG2) * 2 * 32 + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 4u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, ek_soa + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek_shoup_soa + key_base + 32u);
        cirbts_cp_async_commit();
    };

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];

            tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            buf ^= 1u;
        }
    }

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 = acc[N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[N + coeff] = acc_t1;
}

template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_SoA_T(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys, size_t N,
                                                uint32_t tiles, const DModulus* mod, size_t lweIndex,
                                                const uint32_t* indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = indexPos[lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((lweIndex * tiles + tile) * DigitsG2) * 2 * 32 + lane;

#pragma unroll
    for (uint32_t d = 0; d < DigitsG2; ++d) {
        const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2 * 32;
        const BasicInteger k0 = ek_soa[key_base];
        const BasicInteger k0_shoup = ek_shoup_soa[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + 32;
        const BasicInteger k1 = ek_soa[key_base1];
        const BasicInteger k1_shoup = ek_shoup_soa[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 = acc[N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[N + coeff] = acc_t1;
}

template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_SoA_T_Smem(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                     const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys, size_t N,
                                                     uint32_t tiles, const DModulus* mod, size_t lweIndex,
                                                     const uint32_t* indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = indexPos[lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((lweIndex * tiles + tile) * DigitsG2) * 2 * 32 + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 4u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, ek_soa + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek_shoup_soa + key_base + 32u);
        cirbts_cp_async_commit();
    };

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    if (DigitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
#pragma unroll
        for (uint32_t d = 0; d < DigitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < DigitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];

            tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            buf ^= 1u;
        }
    }

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 = acc[N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[N + coeff] = acc_t1;
}

template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_Swizzle_Delta(BasicInteger* acc, BasicInteger* delta, const BasicInteger* dct,
                                                        const BasicInteger* ek_swizzle, const BasicInteger* ek_shoup_swizzle,
                                                        const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                        const DModulus* mod, size_t lweIndex, const uint32_t* indexPos) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = indexPos[lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((lweIndex * tiles + tile) * DigitsG2) * 2 * 32 + lane;

#pragma unroll
    for (uint32_t d = 0; d < DigitsG2; ++d) {
        const BasicInteger dct_coeff = dct[static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2 * 32;
        const BasicInteger k0 = ek_swizzle[key_base];
        const BasicInteger k0_shoup = ek_shoup_swizzle[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + 32;
        const BasicInteger k1 = ek_swizzle[key_base1];
        const BasicInteger k1_shoup = ek_shoup_swizzle[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger delta0 = phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    BasicInteger acc_t0 = acc[coeff] + delta0;
    phantom::arith::csub_q(acc_t0, Q);
    acc[coeff] = acc_t0;
    delta[coeff] = delta0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger delta1 = phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    BasicInteger acc_t1 = acc[N + coeff] + delta1;
    phantom::arith::csub_q(acc_t1, Q);
    acc[N + coeff] = acc_t1;
    delta[N + coeff] = delta1;
}

__global__ void kernel_EvalAccCore_Binary_Batch(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                                const BasicInteger* ek_shoup, const BasicInteger* monic_polys, size_t N,
                                                const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex,
                                                const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t indexPos = d_indexPos[static_cast<size_t>(batch_idx) * indexStride + lweIndex];
    const BasicInteger pos_monic_coeff = monic_polys[static_cast<size_t>(indexPos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t ek_base  = static_cast<size_t>(lweIndex) * digitsG2 * 2 * N;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];

        const size_t key_base = ek_base + static_cast<size_t>(d) * 2 * N + coeff;
        const BasicInteger k0 = ek[key_base];
        const BasicInteger k0_shoup = ek_shoup[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + N;
        const BasicInteger k1 = ek[key_base1];
        const BasicInteger k1_shoup = ek_shoup[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, pos_monic_coeff);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, pos_monic_coeff);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Batch_NMajor(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                                       const BasicInteger* ek_shoup, const BasicInteger* monic_polys, size_t N,
                                                       const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex,
                                                       const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t indexPos = d_indexPos[static_cast<size_t>(lweIndex) * indexStride + batch_idx];
    const BasicInteger pos_monic_coeff = monic_polys[static_cast<size_t>(indexPos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t ek_base  = static_cast<size_t>(lweIndex) * digitsG2 * 2 * N;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];

        const size_t key_base = ek_base + static_cast<size_t>(d) * 2 * N + coeff;
        const BasicInteger k0 = ek[key_base];
        const BasicInteger k0_shoup = ek_shoup[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + N;
        const BasicInteger k1 = ek[key_base1];
        const BasicInteger k1_shoup = ek_shoup[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, pos_monic_coeff);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, pos_monic_coeff);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Grouped_Batch(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek0,
                                                        const BasicInteger* ek0_shoup, const BasicInteger* monic_polys,
                                                        size_t N, const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                        uint32_t lweIndex, const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t indexPos = d_indexPos[static_cast<size_t>(batch_idx) * indexStride + lweIndex];
    const BasicInteger pos_monic_coeff = monic_polys[static_cast<size_t>(indexPos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t ek_base  = static_cast<size_t>(key_group) * digitsG2 * 2 * N;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];

        const size_t key_base = ek_base + static_cast<size_t>(d) * 2 * N + coeff;
        const BasicInteger k0 = ek0[key_base];
        const BasicInteger k0_shoup = ek0_shoup[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + N;
        const BasicInteger k1 = ek0[key_base1];
        const BasicInteger k1_shoup = ek0_shoup[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, pos_monic_coeff);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, pos_monic_coeff);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Grouped_Batch_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                               const BasicInteger* ek0, const BasicInteger* ek0_shoup,
                                                               const BasicInteger* monic_polys, size_t N,
                                                               const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                               uint32_t lweIndex, const uint32_t* d_indexPos,
                                                               uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = d_indexPos[static_cast<size_t>(lweIndex) * indexStride + batch_idx];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t ek_base  = static_cast<size_t>(key_group) * digitsG2 * 2 * N;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];

        const size_t key_base = ek_base + static_cast<size_t>(d) * 2 * N + coeff;
        const BasicInteger k0 = ek0[key_base];
        const BasicInteger k0_shoup = ek0_shoup[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + N;
        const BasicInteger k1 = ek0[key_base1];
        const BasicInteger k1_shoup = ek0_shoup[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;

    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Batch(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                                    const BasicInteger* ek_shoup, const BasicInteger* ek_pair,
                                                    const BasicInteger* ek_pair_shoup, const BasicInteger* monic_polys,
                                                    size_t N, const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                    uint32_t lweIndex1, const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const size_t index_base = static_cast<size_t>(batch_idx) * indexStride;
    const uint32_t pos0 = d_indexPos[index_base + lweIndex0];
    const uint32_t pos1 = d_indexPos[index_base + lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t key_stride = static_cast<size_t>(digitsG2) * 2 * N;
    const size_t base0 = static_cast<size_t>(lweIndex0) * key_stride;
    const size_t base1 = static_cast<size_t>(lweIndex1) * key_stride;
    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];

        const size_t key_base0 = base0 + static_cast<size_t>(d) * 2 * N + coeff;
        const BasicInteger k0 = ek[key_base0];
        const BasicInteger k0_shoup = ek_shoup[key_base0];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const size_t key_base0_1 = key_base0 + N;
        const BasicInteger k1 = ek[key_base0_1];
        const BasicInteger k1_shoup = ek_shoup[key_base0_1];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const size_t key_base1 = base1 + static_cast<size_t>(d) * 2 * N + coeff;
        const BasicInteger k0b = ek[key_base1];
        const BasicInteger k0b_shoup = ek_shoup[key_base1];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const size_t key_base1_1 = key_base1 + N;
        const BasicInteger k1b = ek[key_base1_1];
        const BasicInteger k1b_shoup = ek_shoup[key_base1_1];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair[key_base0];
        const BasicInteger kp0_shoup = ek_pair_shoup[key_base0];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair[key_base0_1];
        const BasicInteger kp1_shoup = ek_pair_shoup[key_base0_1];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                       const BasicInteger* ek0_soa,
                                                                       const BasicInteger* ek0_shoup_soa,
                                                                       const BasicInteger* ek1_soa,
                                                                       const BasicInteger* ek1_shoup_soa,
                                                                       const BasicInteger* ek_pair_soa,
                                                                       const BasicInteger* ek_pair_shoup_soa,
                                                                       const BasicInteger* monic_polys, size_t N,
                                                                       uint32_t tiles, const DModulus* mod,
                                                                       uint32_t digitsG2, uint32_t key_group,
                                                                       uint32_t lweIndex0, uint32_t lweIndex1,
                                                                       const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[static_cast<size_t>(lweIndex0) * indexStride + batch_idx];
    const uint32_t pos1 = d_indexPos[static_cast<size_t>(lweIndex1) * indexStride + batch_idx];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;

        const BasicInteger k0 = ek0_soa[key_base];
        const BasicInteger k0_shoup = ek0_shoup_soa[key_base];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const BasicInteger k1 = ek0_soa[key_base + 32u];
        const BasicInteger k1_shoup = ek0_shoup_soa[key_base + 32u];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const BasicInteger k0b = ek1_soa[key_base];
        const BasicInteger k0b_shoup = ek1_shoup_soa[key_base];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const BasicInteger k1b = ek1_soa[key_base + 32u];
        const BasicInteger k1b_shoup = ek1_shoup_soa[key_base + 32u];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair_soa[key_base];
        const BasicInteger kp0_shoup = ek_pair_shoup_soa[key_base];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair_soa[key_base + 32u];
        const BasicInteger kp1_shoup = ek_pair_shoup_soa[key_base + 32u];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Batch_NMajor(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                                           const BasicInteger* ek_shoup, const BasicInteger* ek_pair,
                                                           const BasicInteger* ek_pair_shoup, const BasicInteger* monic_polys,
                                                           size_t N, const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                           uint32_t lweIndex1, const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[static_cast<size_t>(lweIndex0) * indexStride + batch_idx];
    const uint32_t pos1 = d_indexPos[static_cast<size_t>(lweIndex1) * indexStride + batch_idx];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t key_stride = static_cast<size_t>(digitsG2) * 2 * N;
    const size_t base0 = static_cast<size_t>(lweIndex0) * key_stride;
    const size_t base1 = static_cast<size_t>(lweIndex1) * key_stride;
    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];

        const size_t key_base0 = base0 + static_cast<size_t>(d) * 2 * N + coeff;
        const BasicInteger k0 = ek[key_base0];
        const BasicInteger k0_shoup = ek_shoup[key_base0];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const size_t key_base0_1 = key_base0 + N;
        const BasicInteger k1 = ek[key_base0_1];
        const BasicInteger k1_shoup = ek_shoup[key_base0_1];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const size_t key_base1 = base1 + static_cast<size_t>(d) * 2 * N + coeff;
        const BasicInteger k0b = ek[key_base1];
        const BasicInteger k0b_shoup = ek_shoup[key_base1];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const size_t key_base1_1 = key_base1 + N;
        const BasicInteger k1b = ek[key_base1_1];
        const BasicInteger k1b_shoup = ek_shoup[key_base1_1];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair[key_base0];
        const BasicInteger kp0_shoup = ek_pair_shoup[key_base0];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair[key_base0_1];
        const BasicInteger kp1_shoup = ek_pair_shoup[key_base0_1];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Batch_SoA_MS(BasicInteger* acc, const BasicInteger* dct,
                                                           const BasicInteger* ek_soa, const BasicInteger* ek_shoup_soa,
                                                           const BasicInteger* ek_pair_soa,
                                                           const BasicInteger* ek_pair_shoup_soa,
                                                           const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                           const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                           uint32_t lweIndex1, const BasicInteger* d_a, uint32_t a_stride,
                                                           uint32_t twoN, BasicInteger Q_lwe, uint32_t bitwidth) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    __shared__ uint32_t pos0_shared;
    __shared__ uint32_t pos1_shared;
    if (threadIdx.x == 0) {
        const size_t base = static_cast<size_t>(batch_idx) * a_stride;
        const BasicInteger v0 = d_a[base + lweIndex0];
        const BasicInteger v1 = d_a[base + lweIndex1];
        pos0_shared = device_SpecialMS_IndexPos(v0, twoN, Q_lwe, bitwidth);
        pos1_shared = device_SpecialMS_IndexPos(v1, twoN, Q_lwe, bitwidth);
    }
    __syncthreads();

    const uint32_t pos0 = pos0_shared;
    const uint32_t pos1 = pos1_shared;
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base0 = ((static_cast<size_t>(lweIndex0) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t base1 = ((static_cast<size_t>(lweIndex1) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t basep = base0;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base0 = base0 + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_base1 = base1 + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_basep = basep + static_cast<size_t>(d) * 2u * 32u;

        const BasicInteger k0 = ek_soa[key_base0];
        const BasicInteger k0_shoup = ek_shoup_soa[key_base0];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const BasicInteger k1 = ek_soa[key_base0 + 32u];
        const BasicInteger k1_shoup = ek_shoup_soa[key_base0 + 32u];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const BasicInteger k0b = ek_soa[key_base1];
        const BasicInteger k0b_shoup = ek_shoup_soa[key_base1];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const BasicInteger k1b = ek_soa[key_base1 + 32u];
        const BasicInteger k1b_shoup = ek_shoup_soa[key_base1 + 32u];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair_soa[key_basep];
        const BasicInteger kp0_shoup = ek_pair_shoup_soa[key_basep];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair_soa[key_basep + 32u];
        const BasicInteger kp1_shoup = ek_pair_shoup_soa[key_basep + 32u];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem_MS(BasicInteger* acc, const BasicInteger* dct,
                                                                const BasicInteger* ek_soa, const BasicInteger* ek_shoup_soa,
                                                                const BasicInteger* ek_pair_soa, const BasicInteger* ek_pair_shoup_soa,
                                                                const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                                uint32_t lweIndex1, const BasicInteger* d_a, uint32_t a_stride,
                                                                uint32_t twoN, BasicInteger Q_lwe, uint32_t bitwidth) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    __shared__ uint32_t pos0_shared;
    __shared__ uint32_t pos1_shared;
    if (threadIdx.x == 0) {
        const size_t base = static_cast<size_t>(batch_idx) * a_stride;
        const BasicInteger v0 = d_a[base + lweIndex0];
        const BasicInteger v1 = d_a[base + lweIndex1];
        pos0_shared = device_SpecialMS_IndexPos(v0, twoN, Q_lwe, bitwidth);
        pos1_shared = device_SpecialMS_IndexPos(v1, twoN, Q_lwe, bitwidth);
    }
    __syncthreads();

    const uint32_t pos0 = pos0_shared;
    const uint32_t pos1 = pos1_shared;
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base0 = ((static_cast<size_t>(lweIndex0) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t base1 = ((static_cast<size_t>(lweIndex1) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t basep = base0;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 6u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base0 = base0 + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_base1 = base1 + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_basep = basep + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, ek_soa + key_base0);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek_shoup_soa + key_base0);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek_soa + key_base0 + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek_shoup_soa + key_base0 + 32u);
        cirbts_cp_async_8(buf + lane + 4u * 32u, ek_soa + key_base1);
        cirbts_cp_async_8(buf + lane + 5u * 32u, ek_shoup_soa + key_base1);
        cirbts_cp_async_commit();
        (void)key_basep;
    };

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;
    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];
            const BasicInteger k0b = cur[lane + 4u * 32u];
            const BasicInteger k0b_shoup = cur[lane + 5u * 32u];

            tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
            tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);

            const BasicInteger k1b = ek_soa[(base1 + static_cast<size_t>(d) * 2u * 32u) + 32u];
            const BasicInteger k1b_shoup = ek_shoup_soa[(base1 + static_cast<size_t>(d) * 2u * 32u) + 32u];
            tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

            const BasicInteger kp0 = ek_pair_soa[(basep + static_cast<size_t>(d) * 2u * 32u)];
            const BasicInteger kp0_shoup = ek_pair_shoup_soa[(basep + static_cast<size_t>(d) * 2u * 32u)];
            const BasicInteger kp1 = ek_pair_soa[(basep + static_cast<size_t>(d) * 2u * 32u) + 32u];
            const BasicInteger kp1_shoup = ek_pair_shoup_soa[(basep + static_cast<size_t>(d) * 2u * 32u) + 32u];
            tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
            tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);

            buf ^= 1u;
        }
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Batch_MS(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                                       const BasicInteger* ek_shoup, const BasicInteger* ek_pair,
                                                       const BasicInteger* ek_pair_shoup, const BasicInteger* monic_polys,
                                                       size_t N, const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                       uint32_t lweIndex1, const BasicInteger* d_a, uint32_t a_stride,
                                                       uint32_t twoN, BasicInteger Q_lwe, uint32_t bitwidth) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    __shared__ uint32_t pos0_shared;
    __shared__ uint32_t pos1_shared;
    if (threadIdx.x == 0) {
        const size_t base = static_cast<size_t>(batch_idx) * a_stride;
        const BasicInteger v0 = d_a[base + lweIndex0];
        const BasicInteger v1 = d_a[base + lweIndex1];
        pos0_shared = device_SpecialMS_IndexPos(v0, twoN, Q_lwe, bitwidth);
        pos1_shared = device_SpecialMS_IndexPos(v1, twoN, Q_lwe, bitwidth);
    }
    __syncthreads();

    const uint32_t pos0 = pos0_shared;
    const uint32_t pos1 = pos1_shared;
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t key_stride = static_cast<size_t>(digitsG2) * 2 * N;
    const size_t base0 = static_cast<size_t>(lweIndex0) * key_stride;
    const size_t base1 = static_cast<size_t>(lweIndex1) * key_stride;
    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];

        const size_t key_base0 = base0 + static_cast<size_t>(d) * 2 * N + coeff;
        const BasicInteger k0 = ek[key_base0];
        const BasicInteger k0_shoup = ek_shoup[key_base0];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const size_t key_base0_1 = key_base0 + N;
        const BasicInteger k1 = ek[key_base0_1];
        const BasicInteger k1_shoup = ek_shoup[key_base0_1];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const size_t key_base1 = base1 + static_cast<size_t>(d) * 2 * N + coeff;
        const BasicInteger k0b = ek[key_base1];
        const BasicInteger k0b_shoup = ek_shoup[key_base1];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const size_t key_base1_1 = key_base1 + N;
        const BasicInteger k1b = ek[key_base1_1];
        const BasicInteger k1b_shoup = ek_shoup[key_base1_1];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair[key_base0];
        const BasicInteger kp0_shoup = ek_pair_shoup[key_base0];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair[key_base0_1];
        const BasicInteger kp1_shoup = ek_pair_shoup[key_base0_1];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek0,
                                                            const BasicInteger* ek0_shoup, const BasicInteger* ek1,
                                                            const BasicInteger* ek1_shoup, const BasicInteger* ek_pair,
                                                            const BasicInteger* ek_pair_shoup, const BasicInteger* monic_polys,
                                                            size_t N, const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                            uint32_t lweIndex0, uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                            uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const size_t index_base = static_cast<size_t>(batch_idx) * indexStride;
    const uint32_t pos0 = d_indexPos[index_base + lweIndex0];
    const uint32_t pos1 = d_indexPos[index_base + lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t key_stride = static_cast<size_t>(digitsG2) * 2 * N;
    const size_t base = static_cast<size_t>(key_group) * key_stride;
    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];

        const size_t key_base = base + static_cast<size_t>(d) * 2 * N + coeff;
        const BasicInteger k0 = ek0[key_base];
        const BasicInteger k0_shoup = ek0_shoup[key_base];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const size_t key_base1 = key_base + N;
        const BasicInteger k1 = ek0[key_base1];
        const BasicInteger k1_shoup = ek0_shoup[key_base1];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const BasicInteger k0b = ek1[key_base];
        const BasicInteger k0b_shoup = ek1_shoup[key_base];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const BasicInteger k1b = ek1[key_base1];
        const BasicInteger k1b_shoup = ek1_shoup[key_base1];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair[key_base];
        const BasicInteger kp0_shoup = ek_pair_shoup[key_base];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair[key_base1];
        const BasicInteger kp1_shoup = ek_pair_shoup[key_base1];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                   const BasicInteger* ek0, const BasicInteger* ek0_shoup,
                                                                   const BasicInteger* ek1, const BasicInteger* ek1_shoup,
                                                                   const BasicInteger* ek_pair, const BasicInteger* ek_pair_shoup,
                                                                   const BasicInteger* monic_polys, size_t N, const DModulus* mod,
                                                                   uint32_t digitsG2, uint32_t key_group, uint32_t lweIndex0,
                                                                   uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                                   uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[static_cast<size_t>(lweIndex0) * indexStride + batch_idx];
    const uint32_t pos1 = d_indexPos[static_cast<size_t>(lweIndex1) * indexStride + batch_idx];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t key_stride = static_cast<size_t>(digitsG2) * 2 * N;
    const size_t base = static_cast<size_t>(key_group) * key_stride;
    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];

        const size_t key_base = base + static_cast<size_t>(d) * 2 * N + coeff;
        const BasicInteger k0 = ek0[key_base];
        const BasicInteger k0_shoup = ek0_shoup[key_base];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const size_t key_base1 = key_base + N;
        const BasicInteger k1 = ek0[key_base1];
        const BasicInteger k1_shoup = ek0_shoup[key_base1];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const BasicInteger k0b = ek1[key_base];
        const BasicInteger k0b_shoup = ek1_shoup[key_base];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const BasicInteger k1b = ek1[key_base1];
        const BasicInteger k1b_shoup = ek1_shoup[key_base1];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair[key_base];
        const BasicInteger kp0_shoup = ek_pair_shoup[key_base];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair[key_base1];
        const BasicInteger kp1_shoup = ek_pair_shoup[key_base1];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

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
                                                            const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const size_t index_base = static_cast<size_t>(batch_idx) * indexStride;
    const uint32_t pos0 = d_indexPos[index_base + lweIndex0];
    const uint32_t pos1 = d_indexPos[index_base + lweIndex1];
    const uint32_t pos2 = d_indexPos[index_base + lweIndex2];

    const BasicInteger m0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger m1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const BasicInteger m2 = monic_polys[static_cast<size_t>(pos2) * N + coeff];

    const uint32_t pos01 = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos02 = (pos0 + pos2) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos12 = (pos1 + pos2) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos012 = (pos01 + pos2) & static_cast<uint32_t>((N << 1) - 1);

    const BasicInteger ms01 = monic_polys[static_cast<size_t>(pos01) * N + coeff];
    const BasicInteger ms02 = monic_polys[static_cast<size_t>(pos02) * N + coeff];
    const BasicInteger ms12 = monic_polys[static_cast<size_t>(pos12) * N + coeff];
    const BasicInteger ms012 = monic_polys[static_cast<size_t>(pos012) * N + coeff];

    BasicInteger m01 = ms01;
    m01 += (Q - m0);
    phantom::arith::csub_q(m01, Q);
    m01 += (Q - m1);
    phantom::arith::csub_q(m01, Q);

    BasicInteger m02 = ms02;
    m02 += (Q - m0);
    phantom::arith::csub_q(m02, Q);
    m02 += (Q - m2);
    phantom::arith::csub_q(m02, Q);

    BasicInteger m12 = ms12;
    m12 += (Q - m1);
    phantom::arith::csub_q(m12, Q);
    m12 += (Q - m2);
    phantom::arith::csub_q(m12, Q);

    BasicInteger m012 = ms012;
    m012 += (Q - ms01);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms02);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms12);
    phantom::arith::csub_q(m012, Q);
    m012 += m0;
    phantom::arith::csub_q(m012, Q);
    m012 += m1;
    phantom::arith::csub_q(m012, Q);
    m012 += m2;
    phantom::arith::csub_q(m012, Q);

    BasicInteger t00 = 0, t10 = 0;
    BasicInteger t01 = 0, t11 = 0;
    BasicInteger t02 = 0, t12 = 0;
    BasicInteger t001 = 0, t101 = 0;
    BasicInteger t002 = 0, t102 = 0;
    BasicInteger t012 = 0, t112 = 0;
    BasicInteger t0123 = 0, t1123 = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t key_stride = static_cast<size_t>(digitsG2) * 2 * N;
    const size_t base = static_cast<size_t>(key_group) * key_stride;
    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2 * N + coeff;
        const size_t key_base1 = key_base + N;

        const BasicInteger k00 = ek0[key_base];
        const BasicInteger k00s = ek0_shoup[key_base];
        t00 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k00, k00s, Q);
        const BasicInteger k10 = ek0[key_base1];
        const BasicInteger k10s = ek0_shoup[key_base1];
        t10 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k10, k10s, Q);

        const BasicInteger k01 = ek1[key_base];
        const BasicInteger k01s = ek1_shoup[key_base];
        t01 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k01, k01s, Q);
        const BasicInteger k11 = ek1[key_base1];
        const BasicInteger k11s = ek1_shoup[key_base1];
        t11 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k11, k11s, Q);

        const BasicInteger k02 = ek2[key_base];
        const BasicInteger k02s = ek2_shoup[key_base];
        t02 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k02, k02s, Q);
        const BasicInteger k12 = ek2[key_base1];
        const BasicInteger k12s = ek2_shoup[key_base1];
        t12 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k12, k12s, Q);

        const BasicInteger k001 = ek01[key_base];
        const BasicInteger k001s = ek01_shoup[key_base];
        t001 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k001, k001s, Q);
        const BasicInteger k101 = ek01[key_base1];
        const BasicInteger k101s = ek01_shoup[key_base1];
        t101 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k101, k101s, Q);

        const BasicInteger k002 = ek02[key_base];
        const BasicInteger k002s = ek02_shoup[key_base];
        t002 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k002, k002s, Q);
        const BasicInteger k102 = ek02[key_base1];
        const BasicInteger k102s = ek02_shoup[key_base1];
        t102 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k102, k102s, Q);

        const BasicInteger k012 = ek12[key_base];
        const BasicInteger k012s = ek12_shoup[key_base];
        t012 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k012, k012s, Q);
        const BasicInteger k112 = ek12[key_base1];
        const BasicInteger k112s = ek12_shoup[key_base1];
        t112 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k112, k112s, Q);

        const BasicInteger k0123 = ek012[key_base];
        const BasicInteger k0123s = ek012_shoup[key_base];
        t0123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0123, k0123s, Q);
        const BasicInteger k1123 = ek012[key_base1];
        const BasicInteger k1123s = ek012_shoup[key_base1];
        t1123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1123, k1123s, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t p128 = phantom::arith::multiply_uint64_uint64(t00, m0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t01, m1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t02, m2);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t001, m01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t002, m02);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t012, m12);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t0123, m012);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    p128 = phantom::arith::multiply_uint64_uint64(t10, m0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t11, m1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t12, m2);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t101, m01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t102, m02);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t112, m12);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t1123, m012);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

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
                                                                   const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[static_cast<size_t>(lweIndex0) * indexStride + batch_idx];
    const uint32_t pos1 = d_indexPos[static_cast<size_t>(lweIndex1) * indexStride + batch_idx];
    const uint32_t pos2 = d_indexPos[static_cast<size_t>(lweIndex2) * indexStride + batch_idx];

    const BasicInteger m0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger m1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const BasicInteger m2 = monic_polys[static_cast<size_t>(pos2) * N + coeff];

    const uint32_t pos01 = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos02 = (pos0 + pos2) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos12 = (pos1 + pos2) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos012 = (pos01 + pos2) & static_cast<uint32_t>((N << 1) - 1);

    const BasicInteger ms01 = monic_polys[static_cast<size_t>(pos01) * N + coeff];
    const BasicInteger ms02 = monic_polys[static_cast<size_t>(pos02) * N + coeff];
    const BasicInteger ms12 = monic_polys[static_cast<size_t>(pos12) * N + coeff];
    const BasicInteger ms012 = monic_polys[static_cast<size_t>(pos012) * N + coeff];

    BasicInteger m01 = ms01;
    m01 += (Q - m0);
    phantom::arith::csub_q(m01, Q);
    m01 += (Q - m1);
    phantom::arith::csub_q(m01, Q);

    BasicInteger m02 = ms02;
    m02 += (Q - m0);
    phantom::arith::csub_q(m02, Q);
    m02 += (Q - m2);
    phantom::arith::csub_q(m02, Q);

    BasicInteger m12 = ms12;
    m12 += (Q - m1);
    phantom::arith::csub_q(m12, Q);
    m12 += (Q - m2);
    phantom::arith::csub_q(m12, Q);

    BasicInteger m012 = ms012;
    m012 += (Q - ms01);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms02);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms12);
    phantom::arith::csub_q(m012, Q);
    m012 += m0;
    phantom::arith::csub_q(m012, Q);
    m012 += m1;
    phantom::arith::csub_q(m012, Q);
    m012 += m2;
    phantom::arith::csub_q(m012, Q);

    BasicInteger t00 = 0, t10 = 0;
    BasicInteger t01 = 0, t11 = 0;
    BasicInteger t02 = 0, t12 = 0;
    BasicInteger t001 = 0, t101 = 0;
    BasicInteger t002 = 0, t102 = 0;
    BasicInteger t012 = 0, t112 = 0;
    BasicInteger t0123 = 0, t1123 = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t key_stride = static_cast<size_t>(digitsG2) * 2 * N;
    const size_t base = static_cast<size_t>(key_group) * key_stride;
    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2 * N + coeff;
        const size_t key_base1 = key_base + N;

        const BasicInteger k00 = ek0[key_base];
        const BasicInteger k00s = ek0_shoup[key_base];
        t00 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k00, k00s, Q);
        const BasicInteger k10 = ek0[key_base1];
        const BasicInteger k10s = ek0_shoup[key_base1];
        t10 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k10, k10s, Q);

        const BasicInteger k01 = ek1[key_base];
        const BasicInteger k01s = ek1_shoup[key_base];
        t01 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k01, k01s, Q);
        const BasicInteger k11 = ek1[key_base1];
        const BasicInteger k11s = ek1_shoup[key_base1];
        t11 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k11, k11s, Q);

        const BasicInteger k02 = ek2[key_base];
        const BasicInteger k02s = ek2_shoup[key_base];
        t02 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k02, k02s, Q);
        const BasicInteger k12 = ek2[key_base1];
        const BasicInteger k12s = ek2_shoup[key_base1];
        t12 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k12, k12s, Q);

        const BasicInteger k001 = ek01[key_base];
        const BasicInteger k001s = ek01_shoup[key_base];
        t001 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k001, k001s, Q);
        const BasicInteger k101 = ek01[key_base1];
        const BasicInteger k101s = ek01_shoup[key_base1];
        t101 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k101, k101s, Q);

        const BasicInteger k002 = ek02[key_base];
        const BasicInteger k002s = ek02_shoup[key_base];
        t002 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k002, k002s, Q);
        const BasicInteger k102 = ek02[key_base1];
        const BasicInteger k102s = ek02_shoup[key_base1];
        t102 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k102, k102s, Q);

        const BasicInteger k012 = ek12[key_base];
        const BasicInteger k012s = ek12_shoup[key_base];
        t012 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k012, k012s, Q);
        const BasicInteger k112 = ek12[key_base1];
        const BasicInteger k112s = ek12_shoup[key_base1];
        t112 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k112, k112s, Q);

        const BasicInteger k0123 = ek012[key_base];
        const BasicInteger k0123s = ek012_shoup[key_base];
        t0123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0123, k0123s, Q);
        const BasicInteger k1123 = ek012[key_base1];
        const BasicInteger k1123s = ek012_shoup[key_base1];
        t1123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1123, k1123s, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t p128 = phantom::arith::multiply_uint64_uint64(t00, m0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t01, m1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t02, m2);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t001, m01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t002, m02);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t012, m12);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t0123, m012);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    p128 = phantom::arith::multiply_uint64_uint64(t10, m0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t11, m1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t12, m2);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t101, m01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t102, m02);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t112, m12);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t1123, m012);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

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
                                                                const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const size_t index_base = static_cast<size_t>(batch_idx) * indexStride;
    const uint32_t pos0 = d_indexPos[index_base + lweIndex0];
    const uint32_t pos1 = d_indexPos[index_base + lweIndex1];
    const uint32_t pos2 = d_indexPos[index_base + lweIndex2];

    const BasicInteger m0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger m1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const BasicInteger m2 = monic_polys[static_cast<size_t>(pos2) * N + coeff];

    const uint32_t pos01 = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos02 = (pos0 + pos2) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos12 = (pos1 + pos2) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos012 = (pos01 + pos2) & static_cast<uint32_t>((N << 1) - 1);

    const BasicInteger ms01 = monic_polys[static_cast<size_t>(pos01) * N + coeff];
    const BasicInteger ms02 = monic_polys[static_cast<size_t>(pos02) * N + coeff];
    const BasicInteger ms12 = monic_polys[static_cast<size_t>(pos12) * N + coeff];
    const BasicInteger ms012 = monic_polys[static_cast<size_t>(pos012) * N + coeff];

    BasicInteger m01 = ms01;
    m01 += (Q - m0);
    phantom::arith::csub_q(m01, Q);
    m01 += (Q - m1);
    phantom::arith::csub_q(m01, Q);

    BasicInteger m02 = ms02;
    m02 += (Q - m0);
    phantom::arith::csub_q(m02, Q);
    m02 += (Q - m2);
    phantom::arith::csub_q(m02, Q);

    BasicInteger m12 = ms12;
    m12 += (Q - m1);
    phantom::arith::csub_q(m12, Q);
    m12 += (Q - m2);
    phantom::arith::csub_q(m12, Q);

    BasicInteger m012 = ms012;
    m012 += (Q - ms01);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms02);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms12);
    phantom::arith::csub_q(m012, Q);
    m012 += m0;
    phantom::arith::csub_q(m012, Q);
    m012 += m1;
    phantom::arith::csub_q(m012, Q);
    m012 += m2;
    phantom::arith::csub_q(m012, Q);

    BasicInteger t00 = 0, t10 = 0;
    BasicInteger t01 = 0, t11 = 0;
    BasicInteger t02 = 0, t12 = 0;
    BasicInteger t001 = 0, t101 = 0;
    BasicInteger t002 = 0, t102 = 0;
    BasicInteger t012 = 0, t112 = 0;
    BasicInteger t0123 = 0, t1123 = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_base1 = key_base + 32u;

        const BasicInteger k00 = ek0_soa[key_base];
        const BasicInteger k00s = ek0_shoup_soa[key_base];
        t00 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k00, k00s, Q);
        const BasicInteger k10 = ek0_soa[key_base1];
        const BasicInteger k10s = ek0_shoup_soa[key_base1];
        t10 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k10, k10s, Q);

        const BasicInteger k01 = ek1_soa[key_base];
        const BasicInteger k01s = ek1_shoup_soa[key_base];
        t01 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k01, k01s, Q);
        const BasicInteger k11 = ek1_soa[key_base1];
        const BasicInteger k11s = ek1_shoup_soa[key_base1];
        t11 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k11, k11s, Q);

        const BasicInteger k02 = ek2_soa[key_base];
        const BasicInteger k02s = ek2_shoup_soa[key_base];
        t02 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k02, k02s, Q);
        const BasicInteger k12 = ek2_soa[key_base1];
        const BasicInteger k12s = ek2_shoup_soa[key_base1];
        t12 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k12, k12s, Q);

        const BasicInteger k001 = ek01_soa[key_base];
        const BasicInteger k001s = ek01_shoup_soa[key_base];
        t001 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k001, k001s, Q);
        const BasicInteger k101 = ek01_soa[key_base1];
        const BasicInteger k101s = ek01_shoup_soa[key_base1];
        t101 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k101, k101s, Q);

        const BasicInteger k002 = ek02_soa[key_base];
        const BasicInteger k002s = ek02_shoup_soa[key_base];
        t002 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k002, k002s, Q);
        const BasicInteger k102 = ek02_soa[key_base1];
        const BasicInteger k102s = ek02_shoup_soa[key_base1];
        t102 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k102, k102s, Q);

        const BasicInteger k012 = ek12_soa[key_base];
        const BasicInteger k012s = ek12_shoup_soa[key_base];
        t012 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k012, k012s, Q);
        const BasicInteger k112 = ek12_soa[key_base1];
        const BasicInteger k112s = ek12_shoup_soa[key_base1];
        t112 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k112, k112s, Q);

        const BasicInteger k0123 = ek012_soa[key_base];
        const BasicInteger k0123s = ek012_shoup_soa[key_base];
        t0123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0123, k0123s, Q);
        const BasicInteger k1123 = ek012_soa[key_base1];
        const BasicInteger k1123s = ek012_shoup_soa[key_base1];
        t1123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1123, k1123s, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t p128 = phantom::arith::multiply_uint64_uint64(t00, m0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t01, m1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t02, m2);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t001, m01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t002, m02);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t012, m12);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t0123, m012);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    p128 = phantom::arith::multiply_uint64_uint64(t10, m0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t11, m1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t12, m2);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t101, m01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t102, m02);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t112, m12);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t1123, m012);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ __launch_bounds__(256, 3) void kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_NMajor(BasicInteger* acc, const BasicInteger* dct,
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
                                                                       const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[static_cast<size_t>(lweIndex0) * indexStride + batch_idx];
    const uint32_t pos1 = d_indexPos[static_cast<size_t>(lweIndex1) * indexStride + batch_idx];
    const uint32_t pos2 = d_indexPos[static_cast<size_t>(lweIndex2) * indexStride + batch_idx];

    const BasicInteger m0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger m1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const BasicInteger m2 = monic_polys[static_cast<size_t>(pos2) * N + coeff];

    const uint32_t pos01 = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos02 = (pos0 + pos2) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos12 = (pos1 + pos2) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos012 = (pos01 + pos2) & static_cast<uint32_t>((N << 1) - 1);

    const BasicInteger ms01 = monic_polys[static_cast<size_t>(pos01) * N + coeff];
    const BasicInteger ms02 = monic_polys[static_cast<size_t>(pos02) * N + coeff];
    const BasicInteger ms12 = monic_polys[static_cast<size_t>(pos12) * N + coeff];
    const BasicInteger ms012 = monic_polys[static_cast<size_t>(pos012) * N + coeff];

    BasicInteger m01 = ms01;
    m01 += (Q - m0);
    phantom::arith::csub_q(m01, Q);
    m01 += (Q - m1);
    phantom::arith::csub_q(m01, Q);

    BasicInteger m02 = ms02;
    m02 += (Q - m0);
    phantom::arith::csub_q(m02, Q);
    m02 += (Q - m2);
    phantom::arith::csub_q(m02, Q);

    BasicInteger m12 = ms12;
    m12 += (Q - m1);
    phantom::arith::csub_q(m12, Q);
    m12 += (Q - m2);
    phantom::arith::csub_q(m12, Q);

    BasicInteger m012 = ms012;
    m012 += (Q - ms01);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms02);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms12);
    phantom::arith::csub_q(m012, Q);
    m012 += m0;
    phantom::arith::csub_q(m012, Q);
    m012 += m1;
    phantom::arith::csub_q(m012, Q);
    m012 += m2;
    phantom::arith::csub_q(m012, Q);

    BasicInteger t00 = 0, t10 = 0;
    BasicInteger t01 = 0, t11 = 0;
    BasicInteger t02 = 0, t12 = 0;
    BasicInteger t001 = 0, t101 = 0;
    BasicInteger t002 = 0, t102 = 0;
    BasicInteger t012 = 0, t112 = 0;
    BasicInteger t0123 = 0, t1123 = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_base1 = key_base + 32u;

        const BasicInteger k00 = ek0_soa[key_base];
        const BasicInteger k00s = ek0_shoup_soa[key_base];
        t00 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k00, k00s, Q);
        const BasicInteger k10 = ek0_soa[key_base1];
        const BasicInteger k10s = ek0_shoup_soa[key_base1];
        t10 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k10, k10s, Q);

        const BasicInteger k01 = ek1_soa[key_base];
        const BasicInteger k01s = ek1_shoup_soa[key_base];
        t01 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k01, k01s, Q);
        const BasicInteger k11 = ek1_soa[key_base1];
        const BasicInteger k11s = ek1_shoup_soa[key_base1];
        t11 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k11, k11s, Q);

        const BasicInteger k02 = ek2_soa[key_base];
        const BasicInteger k02s = ek2_shoup_soa[key_base];
        t02 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k02, k02s, Q);
        const BasicInteger k12 = ek2_soa[key_base1];
        const BasicInteger k12s = ek2_shoup_soa[key_base1];
        t12 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k12, k12s, Q);

        const BasicInteger k001 = ek01_soa[key_base];
        const BasicInteger k001s = ek01_shoup_soa[key_base];
        t001 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k001, k001s, Q);
        const BasicInteger k101 = ek01_soa[key_base1];
        const BasicInteger k101s = ek01_shoup_soa[key_base1];
        t101 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k101, k101s, Q);

        const BasicInteger k002 = ek02_soa[key_base];
        const BasicInteger k002s = ek02_shoup_soa[key_base];
        t002 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k002, k002s, Q);
        const BasicInteger k102 = ek02_soa[key_base1];
        const BasicInteger k102s = ek02_shoup_soa[key_base1];
        t102 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k102, k102s, Q);

        const BasicInteger k012 = ek12_soa[key_base];
        const BasicInteger k012s = ek12_shoup_soa[key_base];
        t012 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k012, k012s, Q);
        const BasicInteger k112 = ek12_soa[key_base1];
        const BasicInteger k112s = ek12_shoup_soa[key_base1];
        t112 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k112, k112s, Q);

        const BasicInteger k0123 = ek012_soa[key_base];
        const BasicInteger k0123s = ek012_shoup_soa[key_base];
        t0123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0123, k0123s, Q);
        const BasicInteger k1123 = ek012_soa[key_base1];
        const BasicInteger k1123s = ek012_shoup_soa[key_base1];
        t1123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1123, k1123s, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t p128 = phantom::arith::multiply_uint64_uint64(t00, m0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t01, m1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t02, m2);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t001, m01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t002, m02);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t012, m12);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t0123, m012);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    p128 = phantom::arith::multiply_uint64_uint64(t10, m0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t11, m1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t12, m2);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t101, m01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t102, m02);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t112, m12);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t1123, m012);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ __launch_bounds__(256, 3) void kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_NMajor_AES1024_D4(
    BasicInteger* __restrict__ acc, const BasicInteger* __restrict__ dct,
    const BasicInteger* __restrict__ ek0_soa, const BasicInteger* __restrict__ ek0_shoup_soa,
    const BasicInteger* __restrict__ ek1_soa, const BasicInteger* __restrict__ ek1_shoup_soa,
    const BasicInteger* __restrict__ ek2_soa, const BasicInteger* __restrict__ ek2_shoup_soa,
    const BasicInteger* __restrict__ ek01_soa, const BasicInteger* __restrict__ ek01_shoup_soa,
    const BasicInteger* __restrict__ ek02_soa, const BasicInteger* __restrict__ ek02_shoup_soa,
    const BasicInteger* __restrict__ ek12_soa, const BasicInteger* __restrict__ ek12_shoup_soa,
    const BasicInteger* __restrict__ ek012_soa, const BasicInteger* __restrict__ ek012_shoup_soa,
    const BasicInteger* __restrict__ monic_polys, const DModulus* __restrict__ mod, uint32_t key_group,
    uint32_t lweIndex0, uint32_t lweIndex1, uint32_t lweIndex2,
    const uint32_t* __restrict__ d_indexPos, uint32_t indexStride) {
    constexpr uint32_t kN = 1024;
    constexpr uint32_t kDigits = 4;
    constexpr uint32_t kTiles = kN / 32;
    constexpr uint32_t kMask = (kN << 1) - 1;

    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const uint32_t coeff     = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= kN) {
        return;
    }

    const uint32_t pos0 = d_indexPos[static_cast<size_t>(lweIndex0) * indexStride + batch_idx];
    const uint32_t pos1 = d_indexPos[static_cast<size_t>(lweIndex1) * indexStride + batch_idx];
    const uint32_t pos2 = d_indexPos[static_cast<size_t>(lweIndex2) * indexStride + batch_idx];

    const BasicInteger m0 = monic_polys[static_cast<size_t>(pos0) * kN + coeff];
    const BasicInteger m1 = monic_polys[static_cast<size_t>(pos1) * kN + coeff];
    const BasicInteger m2 = monic_polys[static_cast<size_t>(pos2) * kN + coeff];

    const uint32_t pos01  = (pos0 + pos1) & kMask;
    const uint32_t pos02  = (pos0 + pos2) & kMask;
    const uint32_t pos12  = (pos1 + pos2) & kMask;
    const uint32_t pos012 = (pos01 + pos2) & kMask;

    const BasicInteger ms01  = monic_polys[static_cast<size_t>(pos01) * kN + coeff];
    const BasicInteger ms02  = monic_polys[static_cast<size_t>(pos02) * kN + coeff];
    const BasicInteger ms12  = monic_polys[static_cast<size_t>(pos12) * kN + coeff];
    const BasicInteger ms012 = monic_polys[static_cast<size_t>(pos012) * kN + coeff];

    BasicInteger m01 = ms01;
    m01 += (Q - m0);
    phantom::arith::csub_q(m01, Q);
    m01 += (Q - m1);
    phantom::arith::csub_q(m01, Q);

    BasicInteger m02 = ms02;
    m02 += (Q - m0);
    phantom::arith::csub_q(m02, Q);
    m02 += (Q - m2);
    phantom::arith::csub_q(m02, Q);

    BasicInteger m12 = ms12;
    m12 += (Q - m1);
    phantom::arith::csub_q(m12, Q);
    m12 += (Q - m2);
    phantom::arith::csub_q(m12, Q);

    BasicInteger m012 = ms012;
    m012 += (Q - ms01);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms02);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms12);
    phantom::arith::csub_q(m012, Q);
    m012 += m0;
    phantom::arith::csub_q(m012, Q);
    m012 += m1;
    phantom::arith::csub_q(m012, Q);
    m012 += m2;
    phantom::arith::csub_q(m012, Q);

    BasicInteger t00 = 0, t10 = 0;
    BasicInteger t01 = 0, t11 = 0;
    BasicInteger t02 = 0, t12 = 0;
    BasicInteger t001 = 0, t101 = 0;
    BasicInteger t002 = 0, t102 = 0;
    BasicInteger t012 = 0, t112 = 0;
    BasicInteger t0123 = 0, t1123 = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * kDigits * kN;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * kTiles + tile) * kDigits) * 2u * 32u + lane;
#pragma unroll 1
    for (uint32_t d = 0; d < kDigits; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * kN + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_base1 = key_base + 32u;

        const BasicInteger k00 = ek0_soa[key_base];
        const BasicInteger k00s = ek0_shoup_soa[key_base];
        t00 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k00, k00s, Q);
        const BasicInteger k10 = ek0_soa[key_base1];
        const BasicInteger k10s = ek0_shoup_soa[key_base1];
        t10 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k10, k10s, Q);

        const BasicInteger k01 = ek1_soa[key_base];
        const BasicInteger k01s = ek1_shoup_soa[key_base];
        t01 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k01, k01s, Q);
        const BasicInteger k11 = ek1_soa[key_base1];
        const BasicInteger k11s = ek1_shoup_soa[key_base1];
        t11 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k11, k11s, Q);

        const BasicInteger k02 = ek2_soa[key_base];
        const BasicInteger k02s = ek2_shoup_soa[key_base];
        t02 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k02, k02s, Q);
        const BasicInteger k12 = ek2_soa[key_base1];
        const BasicInteger k12s = ek2_shoup_soa[key_base1];
        t12 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k12, k12s, Q);

        const BasicInteger k001 = ek01_soa[key_base];
        const BasicInteger k001s = ek01_shoup_soa[key_base];
        t001 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k001, k001s, Q);
        const BasicInteger k101 = ek01_soa[key_base1];
        const BasicInteger k101s = ek01_shoup_soa[key_base1];
        t101 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k101, k101s, Q);

        const BasicInteger k002 = ek02_soa[key_base];
        const BasicInteger k002s = ek02_shoup_soa[key_base];
        t002 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k002, k002s, Q);
        const BasicInteger k102 = ek02_soa[key_base1];
        const BasicInteger k102s = ek02_shoup_soa[key_base1];
        t102 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k102, k102s, Q);

        const BasicInteger k012 = ek12_soa[key_base];
        const BasicInteger k012s = ek12_shoup_soa[key_base];
        t012 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k012, k012s, Q);
        const BasicInteger k112 = ek12_soa[key_base1];
        const BasicInteger k112s = ek12_shoup_soa[key_base1];
        t112 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k112, k112s, Q);

        const BasicInteger k0123 = ek012_soa[key_base];
        const BasicInteger k0123s = ek012_shoup_soa[key_base];
        t0123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0123, k0123s, Q);
        const BasicInteger k1123 = ek012_soa[key_base1];
        const BasicInteger k1123s = ek012_shoup_soa[key_base1];
        t1123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1123, k1123s, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * kN;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t p128 = phantom::arith::multiply_uint64_uint64(t00, m0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t01, m1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t02, m2);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t001, m01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t002, m02);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t012, m12);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t0123, m012);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + kN + coeff];
    p128 = phantom::arith::multiply_uint64_uint64(t10, m0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t11, m1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t12, m2);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t101, m01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t102, m02);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t112, m12);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t1123, m012);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + kN + coeff] = acc1;
}

__global__ __launch_bounds__(256, 3) void kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_NMajor_AES1024_D4_SplitAcc(
    BasicInteger* __restrict__ acc, const BasicInteger* __restrict__ dct,
    const BasicInteger* __restrict__ ek0_soa, const BasicInteger* __restrict__ ek0_shoup_soa,
    const BasicInteger* __restrict__ ek1_soa, const BasicInteger* __restrict__ ek1_shoup_soa,
    const BasicInteger* __restrict__ ek2_soa, const BasicInteger* __restrict__ ek2_shoup_soa,
    const BasicInteger* __restrict__ ek01_soa, const BasicInteger* __restrict__ ek01_shoup_soa,
    const BasicInteger* __restrict__ ek02_soa, const BasicInteger* __restrict__ ek02_shoup_soa,
    const BasicInteger* __restrict__ ek12_soa, const BasicInteger* __restrict__ ek12_shoup_soa,
    const BasicInteger* __restrict__ ek012_soa, const BasicInteger* __restrict__ ek012_shoup_soa,
    const BasicInteger* __restrict__ monic_polys, const DModulus* __restrict__ mod, uint32_t key_group,
    uint32_t lweIndex0, uint32_t lweIndex1, uint32_t lweIndex2,
    const uint32_t* __restrict__ d_indexPos, uint32_t indexStride, uint32_t component, bool lazyReduce) {
    constexpr uint32_t kN = 1024;
    constexpr uint32_t kDigits = 4;
    constexpr uint32_t kTiles = kN / 32;
    constexpr uint32_t kMask = (kN << 1) - 1;

    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const uint32_t coeff     = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= kN) {
        return;
    }

    const uint32_t pos0 = d_indexPos[static_cast<size_t>(lweIndex0) * indexStride + batch_idx];
    const uint32_t pos1 = d_indexPos[static_cast<size_t>(lweIndex1) * indexStride + batch_idx];
    const uint32_t pos2 = d_indexPos[static_cast<size_t>(lweIndex2) * indexStride + batch_idx];

    const BasicInteger m0 = monic_polys[static_cast<size_t>(pos0) * kN + coeff];
    const BasicInteger m1 = monic_polys[static_cast<size_t>(pos1) * kN + coeff];
    const BasicInteger m2 = monic_polys[static_cast<size_t>(pos2) * kN + coeff];

    const uint32_t pos01  = (pos0 + pos1) & kMask;
    const uint32_t pos02  = (pos0 + pos2) & kMask;
    const uint32_t pos12  = (pos1 + pos2) & kMask;
    const uint32_t pos012 = (pos01 + pos2) & kMask;

    const BasicInteger ms01  = monic_polys[static_cast<size_t>(pos01) * kN + coeff];
    const BasicInteger ms02  = monic_polys[static_cast<size_t>(pos02) * kN + coeff];
    const BasicInteger ms12  = monic_polys[static_cast<size_t>(pos12) * kN + coeff];
    const BasicInteger ms012 = monic_polys[static_cast<size_t>(pos012) * kN + coeff];

    BasicInteger m01 = ms01;
    m01 += (Q - m0);
    phantom::arith::csub_q(m01, Q);
    m01 += (Q - m1);
    phantom::arith::csub_q(m01, Q);

    BasicInteger m02 = ms02;
    m02 += (Q - m0);
    phantom::arith::csub_q(m02, Q);
    m02 += (Q - m2);
    phantom::arith::csub_q(m02, Q);

    BasicInteger m12 = ms12;
    m12 += (Q - m1);
    phantom::arith::csub_q(m12, Q);
    m12 += (Q - m2);
    phantom::arith::csub_q(m12, Q);

    BasicInteger m012 = ms012;
    m012 += (Q - ms01);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms02);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms12);
    phantom::arith::csub_q(m012, Q);
    m012 += m0;
    phantom::arith::csub_q(m012, Q);
    m012 += m1;
    phantom::arith::csub_q(m012, Q);
    m012 += m2;
    phantom::arith::csub_q(m012, Q);

    BasicInteger t0 = 0, t1 = 0, t2 = 0, t01 = 0, t02 = 0, t12 = 0, t012 = 0;

    const uint32_t component_bit = component & 1u;
    const size_t dct_base = static_cast<size_t>(batch_idx) * kDigits * kN;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t component_offset = static_cast<size_t>(component_bit) * 32u;
    const size_t base = ((static_cast<size_t>(key_group) * kTiles + tile) * kDigits) * 2u * 32u + lane + component_offset;
#pragma unroll
    for (uint32_t d = 0; d < kDigits; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * kN + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;

        const BasicInteger k0 = ek0_soa[key_base];
        const BasicInteger k0s = ek0_shoup_soa[key_base];
        t0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0s, Q);

        const BasicInteger k1 = ek1_soa[key_base];
        const BasicInteger k1s = ek1_shoup_soa[key_base];
        t1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1s, Q);

        const BasicInteger k2 = ek2_soa[key_base];
        const BasicInteger k2s = ek2_shoup_soa[key_base];
        t2 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k2, k2s, Q);

        const BasicInteger k01 = ek01_soa[key_base];
        const BasicInteger k01s = ek01_shoup_soa[key_base];
        t01 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k01, k01s, Q);

        const BasicInteger k02 = ek02_soa[key_base];
        const BasicInteger k02s = ek02_shoup_soa[key_base];
        t02 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k02, k02s, Q);

        const BasicInteger k12 = ek12_soa[key_base];
        const BasicInteger k12s = ek12_shoup_soa[key_base];
        t12 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k12, k12s, Q);

        const BasicInteger k012 = ek012_soa[key_base];
        const BasicInteger k012s = ek012_shoup_soa[key_base];
        t012 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k012, k012s, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * kN + static_cast<size_t>(component_bit) * kN;
    if (lazyReduce) {
        BasicInteger accv = acc[acc_base + coeff];
        phantom::arith::uint128_t p128 = phantom::arith::multiply_uint64_uint64(t0, m0);
        accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
        p128 = phantom::arith::multiply_uint64_uint64(t1, m1);
        accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
        p128 = phantom::arith::multiply_uint64_uint64(t2, m2);
        accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
        p128 = phantom::arith::multiply_uint64_uint64(t01, m01);
        accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
        p128 = phantom::arith::multiply_uint64_uint64(t02, m02);
        accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
        p128 = phantom::arith::multiply_uint64_uint64(t12, m12);
        accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
        p128 = phantom::arith::multiply_uint64_uint64(t012, m012);
        accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
        const phantom::arith::uint128_t acc_wide{0, accv};
        acc[acc_base + coeff] = phantom::arith::barrett_reduce_uint128_uint64(acc_wide, Q, Q_ratio);
        return;
    }

    BasicInteger accv = acc[acc_base + coeff];
    phantom::arith::uint128_t p128 = phantom::arith::multiply_uint64_uint64(t0, m0);
    accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(accv, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t1, m1);
    accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(accv, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t2, m2);
    accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(accv, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t01, m01);
    accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(accv, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t02, m02);
    accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(accv, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t12, m12);
    accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(accv, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t012, m012);
    accv += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(accv, Q);
    acc[acc_base + coeff] = accv;
}

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
                                                                      uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const size_t index_base = static_cast<size_t>(batch_idx) * indexStride;
    const uint32_t pos0 = d_indexPos[index_base + lweIndex0];
    const uint32_t pos1 = d_indexPos[index_base + lweIndex1];
    const uint32_t pos2 = d_indexPos[index_base + lweIndex2];

    const BasicInteger m0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger m1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const BasicInteger m2 = monic_polys[static_cast<size_t>(pos2) * N + coeff];

    const uint32_t pos01 = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos02 = (pos0 + pos2) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos12 = (pos1 + pos2) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos012 = (pos01 + pos2) & static_cast<uint32_t>((N << 1) - 1);

    const BasicInteger ms01 = monic_polys[static_cast<size_t>(pos01) * N + coeff];
    const BasicInteger ms02 = monic_polys[static_cast<size_t>(pos02) * N + coeff];
    const BasicInteger ms12 = monic_polys[static_cast<size_t>(pos12) * N + coeff];
    const BasicInteger ms012 = monic_polys[static_cast<size_t>(pos012) * N + coeff];

    BasicInteger m01 = ms01;
    m01 += (Q - m0);
    phantom::arith::csub_q(m01, Q);
    m01 += (Q - m1);
    phantom::arith::csub_q(m01, Q);

    BasicInteger m02 = ms02;
    m02 += (Q - m0);
    phantom::arith::csub_q(m02, Q);
    m02 += (Q - m2);
    phantom::arith::csub_q(m02, Q);

    BasicInteger m12 = ms12;
    m12 += (Q - m1);
    phantom::arith::csub_q(m12, Q);
    m12 += (Q - m2);
    phantom::arith::csub_q(m12, Q);

    BasicInteger m012 = ms012;
    m012 += (Q - ms01);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms02);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms12);
    phantom::arith::csub_q(m012, Q);
    m012 += m0;
    phantom::arith::csub_q(m012, Q);
    m012 += m1;
    phantom::arith::csub_q(m012, Q);
    m012 += m2;
    phantom::arith::csub_q(m012, Q);

    BasicInteger t00 = 0, t10 = 0;
    BasicInteger t01 = 0, t11 = 0;
    BasicInteger t02 = 0, t12 = 0;
    BasicInteger t001 = 0, t101 = 0;
    BasicInteger t002 = 0, t102 = 0;
    BasicInteger t012 = 0, t112 = 0;
    BasicInteger t0123 = 0, t1123 = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 12u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, ek0_soa + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek0_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek0_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek0_shoup_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 4u * 32u, ek1_soa + key_base);
        cirbts_cp_async_8(buf + lane + 5u * 32u, ek1_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 6u * 32u, ek1_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 7u * 32u, ek1_shoup_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 8u * 32u, ek2_soa + key_base);
        cirbts_cp_async_8(buf + lane + 9u * 32u, ek2_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 10u * 32u, ek2_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 11u * 32u, ek2_shoup_soa + key_base + 32u);
        cirbts_cp_async_commit();
    };

    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
            t00 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, cur[lane + 0u * 32u], cur[lane + 1u * 32u], Q);
            t10 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, cur[lane + 2u * 32u], cur[lane + 3u * 32u], Q);
            t01 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, cur[lane + 4u * 32u], cur[lane + 5u * 32u], Q);
            t11 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, cur[lane + 6u * 32u], cur[lane + 7u * 32u], Q);
            t02 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, cur[lane + 8u * 32u], cur[lane + 9u * 32u], Q);
            t12 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, cur[lane + 10u * 32u], cur[lane + 11u * 32u], Q);

            const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
            const size_t key_base1 = key_base + 32u;

            t001 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek01_soa[key_base], ek01_shoup_soa[key_base], Q);
            t101 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek01_soa[key_base1], ek01_shoup_soa[key_base1], Q);
            t002 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek02_soa[key_base], ek02_shoup_soa[key_base], Q);
            t102 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek02_soa[key_base1], ek02_shoup_soa[key_base1], Q);
            t012 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek12_soa[key_base], ek12_shoup_soa[key_base], Q);
            t112 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek12_soa[key_base1], ek12_shoup_soa[key_base1], Q);
            t0123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek012_soa[key_base], ek012_shoup_soa[key_base], Q);
            t1123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek012_soa[key_base1], ek012_shoup_soa[key_base1], Q);

            buf ^= 1u;
        }
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t p128 = phantom::arith::multiply_uint64_uint64(t00, m0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t01, m1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t02, m2);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t001, m01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t002, m02);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t012, m12);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t0123, m012);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    p128 = phantom::arith::multiply_uint64_uint64(t10, m0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t11, m1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t12, m2);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t101, m01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t102, m02);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t112, m12);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t1123, m012);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

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
                                                                             uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[static_cast<size_t>(lweIndex0) * indexStride + batch_idx];
    const uint32_t pos1 = d_indexPos[static_cast<size_t>(lweIndex1) * indexStride + batch_idx];
    const uint32_t pos2 = d_indexPos[static_cast<size_t>(lweIndex2) * indexStride + batch_idx];

    const BasicInteger m0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger m1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const BasicInteger m2 = monic_polys[static_cast<size_t>(pos2) * N + coeff];

    const uint32_t pos01 = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos02 = (pos0 + pos2) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos12 = (pos1 + pos2) & static_cast<uint32_t>((N << 1) - 1);
    const uint32_t pos012 = (pos01 + pos2) & static_cast<uint32_t>((N << 1) - 1);

    const BasicInteger ms01 = monic_polys[static_cast<size_t>(pos01) * N + coeff];
    const BasicInteger ms02 = monic_polys[static_cast<size_t>(pos02) * N + coeff];
    const BasicInteger ms12 = monic_polys[static_cast<size_t>(pos12) * N + coeff];
    const BasicInteger ms012 = monic_polys[static_cast<size_t>(pos012) * N + coeff];

    BasicInteger m01 = ms01;
    m01 += (Q - m0);
    phantom::arith::csub_q(m01, Q);
    m01 += (Q - m1);
    phantom::arith::csub_q(m01, Q);

    BasicInteger m02 = ms02;
    m02 += (Q - m0);
    phantom::arith::csub_q(m02, Q);
    m02 += (Q - m2);
    phantom::arith::csub_q(m02, Q);

    BasicInteger m12 = ms12;
    m12 += (Q - m1);
    phantom::arith::csub_q(m12, Q);
    m12 += (Q - m2);
    phantom::arith::csub_q(m12, Q);

    BasicInteger m012 = ms012;
    m012 += (Q - ms01);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms02);
    phantom::arith::csub_q(m012, Q);
    m012 += (Q - ms12);
    phantom::arith::csub_q(m012, Q);
    m012 += m0;
    phantom::arith::csub_q(m012, Q);
    m012 += m1;
    phantom::arith::csub_q(m012, Q);
    m012 += m2;
    phantom::arith::csub_q(m012, Q);

    BasicInteger t00 = 0, t10 = 0;
    BasicInteger t01 = 0, t11 = 0;
    BasicInteger t02 = 0, t12 = 0;
    BasicInteger t001 = 0, t101 = 0;
    BasicInteger t002 = 0, t102 = 0;
    BasicInteger t012 = 0, t112 = 0;
    BasicInteger t0123 = 0, t1123 = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 12u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, ek0_soa + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek0_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek0_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek0_shoup_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 4u * 32u, ek1_soa + key_base);
        cirbts_cp_async_8(buf + lane + 5u * 32u, ek1_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 6u * 32u, ek1_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 7u * 32u, ek1_shoup_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 8u * 32u, ek2_soa + key_base);
        cirbts_cp_async_8(buf + lane + 9u * 32u, ek2_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 10u * 32u, ek2_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 11u * 32u, ek2_shoup_soa + key_base + 32u);
        cirbts_cp_async_commit();
    };

    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
            t00 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, cur[lane + 0u * 32u], cur[lane + 1u * 32u], Q);
            t10 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, cur[lane + 2u * 32u], cur[lane + 3u * 32u], Q);
            t01 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, cur[lane + 4u * 32u], cur[lane + 5u * 32u], Q);
            t11 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, cur[lane + 6u * 32u], cur[lane + 7u * 32u], Q);
            t02 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, cur[lane + 8u * 32u], cur[lane + 9u * 32u], Q);
            t12 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, cur[lane + 10u * 32u], cur[lane + 11u * 32u], Q);

            const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
            const size_t key_base1 = key_base + 32u;

            t001 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek01_soa[key_base], ek01_shoup_soa[key_base], Q);
            t101 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek01_soa[key_base1], ek01_shoup_soa[key_base1], Q);
            t002 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek02_soa[key_base], ek02_shoup_soa[key_base], Q);
            t102 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek02_soa[key_base1], ek02_shoup_soa[key_base1], Q);
            t012 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek12_soa[key_base], ek12_shoup_soa[key_base], Q);
            t112 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek12_soa[key_base1], ek12_shoup_soa[key_base1], Q);
            t0123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek012_soa[key_base], ek012_shoup_soa[key_base], Q);
            t1123 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, ek012_soa[key_base1], ek012_shoup_soa[key_base1], Q);

            buf ^= 1u;
        }
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t p128 = phantom::arith::multiply_uint64_uint64(t00, m0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t01, m1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t02, m2);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t001, m01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t002, m02);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t012, m12);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t0123, m012);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    p128 = phantom::arith::multiply_uint64_uint64(t10, m0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t11, m1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t12, m2);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t101, m01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t102, m02);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t112, m12);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    p128 = phantom::arith::multiply_uint64_uint64(t1123, m012);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(p128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_Batch_SoA(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                    const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys,
                                                    size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                    uint32_t lweIndex, const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = d_indexPos[static_cast<size_t>(batch_idx) * indexStride + lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(lweIndex) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        const BasicInteger k0 = ek_soa[key_base];
        const BasicInteger k0_shoup = ek_shoup_soa[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + 32u;
        const BasicInteger k1 = ek_soa[key_base1];
        const BasicInteger k1_shoup = ek_shoup_soa[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Batch_SoA_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                           const BasicInteger* ek_soa, const BasicInteger* ek_shoup_soa,
                                                           const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                           const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex,
                                                           const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = d_indexPos[static_cast<size_t>(lweIndex) * indexStride + batch_idx];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(lweIndex) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        const BasicInteger k0 = ek_soa[key_base];
        const BasicInteger k0_shoup = ek_shoup_soa[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + 32u;
        const BasicInteger k1 = ek_soa[key_base1];
        const BasicInteger k1_shoup = ek_shoup_soa[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Batch_SoA_Smem(BasicInteger* acc, const BasicInteger* dct,
                                                         const BasicInteger* ek_soa, const BasicInteger* ek_shoup_soa,
                                                         const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                         const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex,
                                                         const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = d_indexPos[static_cast<size_t>(batch_idx) * indexStride + lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(lweIndex) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 4u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, ek_soa + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek_shoup_soa + key_base + 32u);
        cirbts_cp_async_commit();
    };

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];

            tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            buf ^= 1u;
        }
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Batch_SoA_Smem_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                const BasicInteger* ek_soa,
                                                                const BasicInteger* ek_shoup_soa,
                                                                const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex,
                                                                const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = d_indexPos[static_cast<size_t>(lweIndex) * indexStride + batch_idx];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(lweIndex) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 4u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, ek_soa + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek_shoup_soa + key_base + 32u);
        cirbts_cp_async_commit();
    };

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];

            tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            buf ^= 1u;
        }
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Batch_MS(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek,
                                                   const BasicInteger* ek_shoup, const BasicInteger* monic_polys, size_t N,
                                                   const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex,
                                                   const BasicInteger* d_a, uint32_t a_stride, uint32_t twoN,
                                                   BasicInteger Q_lwe, uint32_t bitwidth) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    __shared__ uint32_t pos_shared;
    if (threadIdx.x == 0) {
        const BasicInteger v = d_a[static_cast<size_t>(batch_idx) * a_stride + lweIndex];
        pos_shared = device_SpecialMS_IndexPos(v, twoN, Q_lwe, bitwidth);
    }
    __syncthreads();

    const uint32_t pos = pos_shared;
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t ek_base  = static_cast<size_t>(lweIndex) * digitsG2 * 2 * N;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = ek_base + static_cast<size_t>(d) * 2 * N + coeff;
        const BasicInteger k0 = ek[key_base];
        const BasicInteger k0_shoup = ek_shoup[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + N;
        const BasicInteger k1 = ek[key_base1];
        const BasicInteger k1_shoup = ek_shoup[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Batch_SoA_MS(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek_soa,
                                                       const BasicInteger* ek_shoup_soa, const BasicInteger* monic_polys,
                                                       size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                       uint32_t lweIndex, const BasicInteger* d_a, uint32_t a_stride,
                                                       uint32_t twoN, BasicInteger Q_lwe, uint32_t bitwidth) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    __shared__ uint32_t pos_shared;
    if (threadIdx.x == 0) {
        const BasicInteger v = d_a[static_cast<size_t>(batch_idx) * a_stride + lweIndex];
        pos_shared = device_SpecialMS_IndexPos(v, twoN, Q_lwe, bitwidth);
    }
    __syncthreads();

    const uint32_t pos = pos_shared;
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(lweIndex) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        const BasicInteger k0 = ek_soa[key_base];
        const BasicInteger k0_shoup = ek_shoup_soa[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + 32u;
        const BasicInteger k1 = ek_soa[key_base1];
        const BasicInteger k1_shoup = ek_shoup_soa[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Batch_SoA_Smem_MS(BasicInteger* acc, const BasicInteger* dct,
                                                            const BasicInteger* ek_soa, const BasicInteger* ek_shoup_soa,
                                                            const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                            const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex,
                                                            const BasicInteger* d_a, uint32_t a_stride, uint32_t twoN,
                                                            BasicInteger Q_lwe, uint32_t bitwidth) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    __shared__ uint32_t pos_shared;
    if (threadIdx.x == 0) {
        const BasicInteger v = d_a[static_cast<size_t>(batch_idx) * a_stride + lweIndex];
        pos_shared = device_SpecialMS_IndexPos(v, twoN, Q_lwe, bitwidth);
    }
    __syncthreads();

    const uint32_t pos = pos_shared;
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(lweIndex) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 4u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, ek_soa + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek_shoup_soa + key_base + 32u);
        cirbts_cp_async_commit();
    };

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];

            tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            buf ^= 1u;
        }
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Grouped_Batch_SoA(BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek0_soa,
                                                            const BasicInteger* ek0_shoup_soa, const BasicInteger* monic_polys,
                                                            size_t N, uint32_t tiles, const DModulus* mod, uint32_t digitsG2,
                                                            uint32_t key_group, uint32_t lweIndex, const uint32_t* d_indexPos,
                                                            uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = d_indexPos[static_cast<size_t>(batch_idx) * indexStride + lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        const BasicInteger k0 = ek0_soa[key_base];
        const BasicInteger k0_shoup = ek0_shoup_soa[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + 32u;
        const BasicInteger k1 = ek0_soa[key_base1];
        const BasicInteger k1_shoup = ek0_shoup_soa[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Grouped_Batch_SoA_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                   const BasicInteger* ek0_soa,
                                                                   const BasicInteger* ek0_shoup_soa,
                                                                   const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                   const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                                   uint32_t lweIndex, const uint32_t* d_indexPos,
                                                                   uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = d_indexPos[static_cast<size_t>(lweIndex) * indexStride + batch_idx];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        const BasicInteger k0 = ek0_soa[key_base];
        const BasicInteger k0_shoup = ek0_shoup_soa[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + 32u;
        const BasicInteger k1 = ek0_soa[key_base1];
        const BasicInteger k1_shoup = ek0_shoup_soa[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem(BasicInteger* acc, const BasicInteger* dct,
                                                                 const BasicInteger* ek0_soa, const BasicInteger* ek0_shoup_soa,
                                                                 const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                 const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                                 uint32_t lweIndex, const uint32_t* d_indexPos,
                                                                 uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = d_indexPos[static_cast<size_t>(batch_idx) * indexStride + lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 4u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, ek0_soa + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek0_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek0_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek0_shoup_soa + key_base + 32u);
        cirbts_cp_async_commit();
    };

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];

            tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            buf ^= 1u;
        }
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                        const BasicInteger* ek0_soa,
                                                                        const BasicInteger* ek0_shoup_soa,
                                                                        const BasicInteger* monic_polys, size_t N,
                                                                        uint32_t tiles, const DModulus* mod,
                                                                        uint32_t digitsG2, uint32_t key_group,
                                                                        uint32_t lweIndex, const uint32_t* d_indexPos,
                                                                        uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = d_indexPos[static_cast<size_t>(lweIndex) * indexStride + batch_idx];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 4u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, ek0_soa + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek0_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek0_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek0_shoup_soa + key_base + 32u);
        cirbts_cp_async_commit();
    };

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;
    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];

            tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            buf ^= 1u;
        }
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Batch_SoA(BasicInteger* acc, const BasicInteger* dct,
                                                        const BasicInteger* ek_soa, const BasicInteger* ek_shoup_soa,
                                                        const BasicInteger* ek_pair_soa,
                                                        const BasicInteger* ek_pair_shoup_soa,
                                                        const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                        const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                        uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                        uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const size_t index_base = static_cast<size_t>(batch_idx) * indexStride;
    const uint32_t pos0 = d_indexPos[index_base + lweIndex0];
    const uint32_t pos1 = d_indexPos[index_base + lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base0 = ((static_cast<size_t>(lweIndex0) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t base1 = ((static_cast<size_t>(lweIndex1) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t basep = base0;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base0 = base0 + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_base1 = base1 + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_basep = basep + static_cast<size_t>(d) * 2u * 32u;

        const BasicInteger k0 = ek_soa[key_base0];
        const BasicInteger k0_shoup = ek_shoup_soa[key_base0];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const BasicInteger k1 = ek_soa[key_base0 + 32u];
        const BasicInteger k1_shoup = ek_shoup_soa[key_base0 + 32u];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const BasicInteger k0b = ek_soa[key_base1];
        const BasicInteger k0b_shoup = ek_shoup_soa[key_base1];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const BasicInteger k1b = ek_soa[key_base1 + 32u];
        const BasicInteger k1b_shoup = ek_shoup_soa[key_base1 + 32u];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair_soa[key_basep];
        const BasicInteger kp0_shoup = ek_pair_shoup_soa[key_basep];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair_soa[key_basep + 32u];
        const BasicInteger kp1_shoup = ek_pair_shoup_soa[key_basep + 32u];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA(BasicInteger* acc, const BasicInteger* dct,
                                                                const BasicInteger* ek0_soa, const BasicInteger* ek0_shoup_soa,
                                                                const BasicInteger* ek1_soa, const BasicInteger* ek1_shoup_soa,
                                                                const BasicInteger* ek_pair_soa, const BasicInteger* ek_pair_shoup_soa,
                                                                const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                                uint32_t lweIndex0, uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                                uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const size_t index_base = static_cast<size_t>(batch_idx) * indexStride;
    const uint32_t pos0 = d_indexPos[index_base + lweIndex0];
    const uint32_t pos1 = d_indexPos[index_base + lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;

        const BasicInteger k0 = ek0_soa[key_base];
        const BasicInteger k0_shoup = ek0_shoup_soa[key_base];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const BasicInteger k1 = ek0_soa[key_base + 32u];
        const BasicInteger k1_shoup = ek0_shoup_soa[key_base + 32u];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const BasicInteger k0b = ek1_soa[key_base];
        const BasicInteger k0b_shoup = ek1_shoup_soa[key_base];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const BasicInteger k1b = ek1_soa[key_base + 32u];
        const BasicInteger k1b_shoup = ek1_shoup_soa[key_base + 32u];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair_soa[key_base];
        const BasicInteger kp0_shoup = ek_pair_shoup_soa[key_base];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair_soa[key_base + 32u];
        const BasicInteger kp1_shoup = ek_pair_shoup_soa[key_base + 32u];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Batch_SoA_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                               const BasicInteger* ek_soa,
                                                               const BasicInteger* ek_shoup_soa,
                                                               const BasicInteger* ek_pair_soa,
                                                               const BasicInteger* ek_pair_shoup_soa,
                                                               const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                               const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                               uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                               uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[static_cast<size_t>(lweIndex0) * indexStride + batch_idx];
    const uint32_t pos1 = d_indexPos[static_cast<size_t>(lweIndex1) * indexStride + batch_idx];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base0 = ((static_cast<size_t>(lweIndex0) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t base1 = ((static_cast<size_t>(lweIndex1) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t basep = base0;

    for (uint32_t d = 0; d < digitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base0 = base0 + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_base1 = base1 + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_basep = basep + static_cast<size_t>(d) * 2u * 32u;

        const BasicInteger k0 = ek_soa[key_base0];
        const BasicInteger k0_shoup = ek_shoup_soa[key_base0];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const BasicInteger k1 = ek_soa[key_base0 + 32u];
        const BasicInteger k1_shoup = ek_shoup_soa[key_base0 + 32u];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const BasicInteger k0b = ek_soa[key_base1];
        const BasicInteger k0b_shoup = ek_shoup_soa[key_base1];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const BasicInteger k1b = ek_soa[key_base1 + 32u];
        const BasicInteger k1b_shoup = ek_shoup_soa[key_base1 + 32u];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair_soa[key_basep];
        const BasicInteger kp0_shoup = ek_pair_shoup_soa[key_basep];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair_soa[key_basep + 32u];
        const BasicInteger kp1_shoup = ek_pair_shoup_soa[key_basep + 32u];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem(BasicInteger* acc, const BasicInteger* dct,
                                                             const BasicInteger* ek_soa,
                                                             const BasicInteger* ek_shoup_soa,
                                                             const BasicInteger* ek_pair_soa,
                                                             const BasicInteger* ek_pair_shoup_soa,
                                                             const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                             const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                             uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                             uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const size_t index_base = static_cast<size_t>(batch_idx) * indexStride;
    const uint32_t pos0 = d_indexPos[index_base + lweIndex0];
    const uint32_t pos1 = d_indexPos[index_base + lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base0 = ((static_cast<size_t>(lweIndex0) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t base1 = ((static_cast<size_t>(lweIndex1) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t basep = base0;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 8u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base0 = base0 + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_base1 = base1 + static_cast<size_t>(d) * 2u * 32u;

        cirbts_cp_async_8(buf + lane + 0u * 32u, ek_soa + key_base0);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek_shoup_soa + key_base0);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek_soa + key_base0 + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek_shoup_soa + key_base0 + 32u);

        cirbts_cp_async_8(buf + lane + 4u * 32u, ek_soa + key_base1);
        cirbts_cp_async_8(buf + lane + 5u * 32u, ek_shoup_soa + key_base1);
        cirbts_cp_async_8(buf + lane + 6u * 32u, ek_soa + key_base1 + 32u);
        cirbts_cp_async_8(buf + lane + 7u * 32u, ek_shoup_soa + key_base1 + 32u);

        cirbts_cp_async_commit();
    };

    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];
            tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            const BasicInteger k0b = cur[lane + 4u * 32u];
            const BasicInteger k0b_shoup = cur[lane + 5u * 32u];
            tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
            const BasicInteger k1b = cur[lane + 6u * 32u];
            const BasicInteger k1b_shoup = cur[lane + 7u * 32u];
            tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

            const size_t key_basep = basep + static_cast<size_t>(d) * 2u * 32u;
            const BasicInteger kp0 = ek_pair_soa[key_basep];
            const BasicInteger kp0_shoup = ek_pair_shoup_soa[key_basep];
            tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
            const BasicInteger kp1 = ek_pair_soa[key_basep + 32u];
            const BasicInteger kp1_shoup = ek_pair_shoup_soa[key_basep + 32u];
            tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);

            buf ^= 1u;
        }
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem_NMajor(BasicInteger* acc, const BasicInteger* dct,
                                                                    const BasicInteger* ek_soa,
                                                                    const BasicInteger* ek_shoup_soa,
                                                                    const BasicInteger* ek_pair_soa,
                                                                    const BasicInteger* ek_pair_shoup_soa,
                                                                    const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                    const DModulus* mod, uint32_t digitsG2, uint32_t lweIndex0,
                                                                    uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                                    uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[static_cast<size_t>(lweIndex0) * indexStride + batch_idx];
    const uint32_t pos1 = d_indexPos[static_cast<size_t>(lweIndex1) * indexStride + batch_idx];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base0 = ((static_cast<size_t>(lweIndex0) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t base1 = ((static_cast<size_t>(lweIndex1) * tiles + tile) * digitsG2) * 2u * 32u + lane;
    const size_t basep = base0;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 8u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base0 = base0 + static_cast<size_t>(d) * 2u * 32u;
        const size_t key_base1 = base1 + static_cast<size_t>(d) * 2u * 32u;

        cirbts_cp_async_8(buf + lane + 0u * 32u, ek_soa + key_base0);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek_shoup_soa + key_base0);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek_soa + key_base0 + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek_shoup_soa + key_base0 + 32u);

        cirbts_cp_async_8(buf + lane + 4u * 32u, ek_soa + key_base1);
        cirbts_cp_async_8(buf + lane + 5u * 32u, ek_shoup_soa + key_base1);
        cirbts_cp_async_8(buf + lane + 6u * 32u, ek_soa + key_base1 + 32u);
        cirbts_cp_async_8(buf + lane + 7u * 32u, ek_shoup_soa + key_base1 + 32u);

        cirbts_cp_async_commit();
    };

    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];
            tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            const BasicInteger k0b = cur[lane + 4u * 32u];
            const BasicInteger k0b_shoup = cur[lane + 5u * 32u];
            tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
            const BasicInteger k1b = cur[lane + 6u * 32u];
            const BasicInteger k1b_shoup = cur[lane + 7u * 32u];
            tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

            const size_t key_basep = basep + static_cast<size_t>(d) * 2u * 32u;
            const BasicInteger kp0 = ek_pair_soa[key_basep];
            const BasicInteger kp0_shoup = ek_pair_shoup_soa[key_basep];
            tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
            const BasicInteger kp1 = ek_pair_soa[key_basep + 32u];
            const BasicInteger kp1_shoup = ek_pair_shoup_soa[key_basep + 32u];
            tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);

            buf ^= 1u;
        }
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_Smem(BasicInteger* acc, const BasicInteger* dct,
                                                                     const BasicInteger* ek0_soa, const BasicInteger* ek0_shoup_soa,
                                                                     const BasicInteger* ek1_soa, const BasicInteger* ek1_shoup_soa,
                                                                     const BasicInteger* ek_pair_soa, const BasicInteger* ek_pair_shoup_soa,
                                                                     const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                     const DModulus* mod, uint32_t digitsG2, uint32_t key_group,
                                                                     uint32_t lweIndex0, uint32_t lweIndex1, const uint32_t* d_indexPos,
                                                                     uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const size_t index_base = static_cast<size_t>(batch_idx) * indexStride;
    const uint32_t pos0 = d_indexPos[index_base + lweIndex0];
    const uint32_t pos1 = d_indexPos[index_base + lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 8u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;

        cirbts_cp_async_8(buf + lane + 0u * 32u, ek0_soa + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek0_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek0_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek0_shoup_soa + key_base + 32u);

        cirbts_cp_async_8(buf + lane + 4u * 32u, ek1_soa + key_base);
        cirbts_cp_async_8(buf + lane + 5u * 32u, ek1_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 6u * 32u, ek1_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 7u * 32u, ek1_shoup_soa + key_base + 32u);

        cirbts_cp_async_commit();
    };

    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];
            tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            const BasicInteger k0b = cur[lane + 4u * 32u];
            const BasicInteger k0b_shoup = cur[lane + 5u * 32u];
            tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
            const BasicInteger k1b = cur[lane + 6u * 32u];
            const BasicInteger k1b_shoup = cur[lane + 7u * 32u];
            tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

            const size_t key_basep = base + static_cast<size_t>(d) * 2u * 32u;
            const BasicInteger kp0 = ek_pair_soa[key_basep];
            const BasicInteger kp0_shoup = ek_pair_shoup_soa[key_basep];
            tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
            const BasicInteger kp1 = ek_pair_soa[key_basep + 32u];
            const BasicInteger kp1_shoup = ek_pair_shoup_soa[key_basep + 32u];
            tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);

            buf ^= 1u;
        }
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_Smem_NMajor(
    BasicInteger* acc, const BasicInteger* dct, const BasicInteger* ek0_soa, const BasicInteger* ek0_shoup_soa,
    const BasicInteger* ek1_soa, const BasicInteger* ek1_shoup_soa, const BasicInteger* ek_pair_soa,
    const BasicInteger* ek_pair_shoup_soa, const BasicInteger* monic_polys, size_t N, uint32_t tiles,
    const DModulus* mod, uint32_t digitsG2, uint32_t key_group, uint32_t lweIndex0, uint32_t lweIndex1,
    const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[static_cast<size_t>(lweIndex0) * indexStride + batch_idx];
    const uint32_t pos1 = d_indexPos[static_cast<size_t>(lweIndex1) * indexStride + batch_idx];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * digitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * digitsG2) * 2u * 32u + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 8u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, ek0_soa + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, ek0_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, ek0_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, ek0_shoup_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 4u * 32u, ek1_soa + key_base);
        cirbts_cp_async_8(buf + lane + 5u * 32u, ek1_shoup_soa + key_base);
        cirbts_cp_async_8(buf + lane + 6u * 32u, ek1_soa + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 7u * 32u, ek1_shoup_soa + key_base + 32u);
        cirbts_cp_async_commit();
    };

    if (digitsG2 > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < digitsG2; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < digitsG2) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];
            tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

            const BasicInteger k0b = cur[lane + 4u * 32u];
            const BasicInteger k0b_shoup = cur[lane + 5u * 32u];
            tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
            const BasicInteger k1b = cur[lane + 6u * 32u];
            const BasicInteger k1b_shoup = cur[lane + 7u * 32u];
            tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

            const size_t key_basep = base + static_cast<size_t>(d) * 2u * 32u;
            const BasicInteger kp0 = ek_pair_soa[key_basep];
            const BasicInteger kp0_shoup = ek_pair_shoup_soa[key_basep];
            tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
            const BasicInteger kp1 = ek_pair_soa[key_basep + 32u];
            const BasicInteger kp1_shoup = ek_pair_shoup_soa[key_basep + 32u];
            tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);

            buf ^= 1u;
        }
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_Batch_Swizzle(BasicInteger* acc, const BasicInteger* dct,
                                                        const BasicInteger* ek_swizzle,
                                                        const BasicInteger* ek_shoup_swizzle,
                                                        const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                        const DModulus* mod, uint32_t lweIndex,
                                                        const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = d_indexPos[static_cast<size_t>(batch_idx) * indexStride + lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * DigitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(lweIndex) * tiles + tile) * DigitsG2) * 2 * 32 + lane;

#pragma unroll
    for (uint32_t d = 0; d < DigitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2 * 32;
        const BasicInteger k0 = ek_swizzle[key_base];
        const BasicInteger k0_shoup = ek_shoup_swizzle[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + 32;
        const BasicInteger k1 = ek_swizzle[key_base1];
        const BasicInteger k1_shoup = ek_shoup_swizzle[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

template <uint32_t DigitsG2>
__global__ void kernel_EvalAccCore_Binary_Grouped_Batch_Swizzle(BasicInteger* acc, const BasicInteger* dct,
                                                                const BasicInteger* ek_swizzle,
                                                                const BasicInteger* ek_shoup_swizzle,
                                                                const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                const DModulus* mod, uint32_t key_group, uint32_t lweIndex,
                                                                const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos = d_indexPos[static_cast<size_t>(batch_idx) * indexStride + lweIndex];
    const BasicInteger monic = monic_polys[static_cast<size_t>(pos) * N + coeff];

    BasicInteger tmp0 = 0;
    BasicInteger tmp1 = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * DigitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * DigitsG2) * 2u * 32u + lane;

#pragma unroll
    for (uint32_t d = 0; d < DigitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        const BasicInteger k0 = ek_swizzle[key_base];
        const BasicInteger k0_shoup = ek_shoup_swizzle[key_base];
        tmp0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);

        const size_t key_base1 = key_base + 32u;
        const BasicInteger k1 = ek_swizzle[key_base1];
        const BasicInteger k1_shoup = ek_shoup_swizzle[key_base1];
        tmp1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0, monic);
    BasicInteger acc_t0 = acc[acc_base + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t0, Q);
    acc[acc_base + coeff] = acc_t0;

    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1, monic);
    BasicInteger acc_t1 =
        acc[acc_base + N + coeff] + phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc_t1, Q);
    acc[acc_base + N + coeff] = acc_t1;
}

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
                                                                    const uint32_t* d_indexPos, uint32_t indexStride) {
    BasicInteger Q          = mod[0].value();
    BasicInteger Q_ratio[2] = {mod[0].const_ratio()[0], mod[0].const_ratio()[1]};

    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const uint32_t pos0 = d_indexPos[static_cast<size_t>(batch_idx) * indexStride + lweIndex0];
    const uint32_t pos1 = d_indexPos[static_cast<size_t>(batch_idx) * indexStride + lweIndex1];
    const BasicInteger monic0 = monic_polys[static_cast<size_t>(pos0) * N + coeff];
    const BasicInteger monic1 = monic_polys[static_cast<size_t>(pos1) * N + coeff];
    const uint32_t pos_sum = (pos0 + pos1) & static_cast<uint32_t>((N << 1) - 1);
    const BasicInteger monic_sum = monic_polys[static_cast<size_t>(pos_sum) * N + coeff];
    BasicInteger monic01 = monic_sum;
    monic01 += (Q - monic0);
    phantom::arith::csub_q(monic01, Q);
    monic01 += (Q - monic1);
    phantom::arith::csub_q(monic01, Q);

    BasicInteger tmp0_0 = 0;
    BasicInteger tmp1_0 = 0;
    BasicInteger tmp0_1 = 0;
    BasicInteger tmp1_1 = 0;
    BasicInteger tmp0_p = 0;
    BasicInteger tmp1_p = 0;

    const size_t dct_base = static_cast<size_t>(batch_idx) * DigitsG2 * N;
    const size_t tile = coeff >> 5;
    const size_t lane = coeff & 31u;
    const size_t base = ((static_cast<size_t>(key_group) * tiles + tile) * DigitsG2) * 2u * 32u + lane;

#pragma unroll
    for (uint32_t d = 0; d < DigitsG2; ++d) {
        const BasicInteger dct_coeff = dct[dct_base + static_cast<size_t>(d) * N + coeff];
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;

        const BasicInteger k0 = ek0_swizzle[key_base];
        const BasicInteger k0_shoup = ek0_shoup_swizzle[key_base];
        tmp0_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0, k0_shoup, Q);
        const BasicInteger k1 = ek0_swizzle[key_base + 32u];
        const BasicInteger k1_shoup = ek0_shoup_swizzle[key_base + 32u];
        tmp1_0 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1, k1_shoup, Q);

        const BasicInteger k0b = ek1_swizzle[key_base];
        const BasicInteger k0b_shoup = ek1_shoup_swizzle[key_base];
        tmp0_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k0b, k0b_shoup, Q);
        const BasicInteger k1b = ek1_swizzle[key_base + 32u];
        const BasicInteger k1b_shoup = ek1_shoup_swizzle[key_base + 32u];
        tmp1_1 += phantom::arith::multiply_and_reduce_shoup(dct_coeff, k1b, k1b_shoup, Q);

        const BasicInteger kp0 = ek_pair_swizzle[key_base];
        const BasicInteger kp0_shoup = ek_pair_shoup_swizzle[key_base];
        tmp0_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp0, kp0_shoup, Q);
        const BasicInteger kp1 = ek_pair_swizzle[key_base + 32u];
        const BasicInteger kp1_shoup = ek_pair_shoup_swizzle[key_base + 32u];
        tmp1_p += phantom::arith::multiply_and_reduce_shoup(dct_coeff, kp1, kp1_shoup, Q);
    }

    const size_t acc_base = static_cast<size_t>(batch_idx) * 2 * N;
    BasicInteger acc0 = acc[acc_base + coeff];
    phantom::arith::uint128_t prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_0, monic0);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_1, monic1);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    prod0_128 = phantom::arith::multiply_uint64_uint64(tmp0_p, monic01);
    acc0 += phantom::arith::barrett_reduce_uint128_uint64(prod0_128, Q, Q_ratio);
    phantom::arith::csub_q(acc0, Q);
    acc[acc_base + coeff] = acc0;

    BasicInteger acc1 = acc[acc_base + N + coeff];
    phantom::arith::uint128_t prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_0, monic0);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_1, monic1);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    prod1_128 = phantom::arith::multiply_uint64_uint64(tmp1_p, monic01);
    acc1 += phantom::arith::barrett_reduce_uint128_uint64(prod1_128, Q, Q_ratio);
    phantom::arith::csub_q(acc1, Q);
    acc[acc_base + N + coeff] = acc1;
}

__global__ void kernel_ModMulScalar(BasicInteger* d_acc, const BasicInteger* monomial_inv, const DModulus* modulus, size_t N) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    BasicInteger Q          = modulus[0].value();
    BasicInteger Q_ratio[2] = {modulus[0].const_ratio()[0], modulus[0].const_ratio()[1]};

    phantom::arith::uint128_t prod_struct = phantom::arith::multiply_uint64_uint64(d_acc[idx], monomial_inv[idx]);
    d_acc[idx] = barrett_reduce_uint128_uint64(prod_struct, Q, Q_ratio);
}

__global__ void kernel_InitBootstrapAccBatch(BasicInteger* d_acc, const BasicInteger* d_lut, size_t N, uint32_t batch) {
    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    if (batch_idx >= batch) {
        return;
    }

    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    const size_t base = static_cast<size_t>(batch_idx) * 2 * N;
    d_acc[base + coeff]       = 0;
    d_acc[base + N + coeff]   = d_lut[coeff];
}

__global__ void kernel_ModMulScalarBatch(BasicInteger* d_acc, const BasicInteger* monomial_inv, const DModulus* modulus, size_t N,
                                         uint32_t batch) {
    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    if (batch_idx >= batch) {
        return;
    }

    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    BasicInteger Q          = modulus[0].value();
    BasicInteger Q_ratio[2] = {modulus[0].const_ratio()[0], modulus[0].const_ratio()[1]};

    const size_t mono_base = static_cast<size_t>(batch_idx) * N;
    const size_t acc_base  = static_cast<size_t>(batch_idx) * 2 * N + N;

    phantom::arith::uint128_t prod_struct =
        phantom::arith::multiply_uint64_uint64(d_acc[acc_base + coeff], monomial_inv[mono_base + coeff]);
    d_acc[acc_base + coeff] = barrett_reduce_uint128_uint64(prod_struct, Q, Q_ratio);
}

__device__ __forceinline__ uint32_t device_SpecialMS_IndexPos(BasicInteger v, uint32_t twoN, BasicInteger Q, uint32_t bitwidth) {
    const uint64_t shift = (bitwidth < 63u) ? (1ULL << bitwidth) : 0ULL;
    if (shift == 0ULL || twoN == 0u || Q == 0) {
        return 0u;
    }
    const double scale = static_cast<double>(twoN) / (static_cast<double>(Q) * static_cast<double>(shift));
    const double x = static_cast<double>(v) * scale;
    const uint64_t r = static_cast<uint64_t>(__double2ull_rd(x + 0.5));
    const uint64_t v_ms = (r * shift) & (static_cast<uint64_t>(twoN) - 1ULL);
    return static_cast<uint32_t>(v_ms);
}

__global__ void kernel_InitBootstrapAccBatch_FusedB(BasicInteger* d_acc, const BasicInteger* d_lut, const BasicInteger* monic_polys,
                                                    const BasicInteger* d_b, uint32_t batch, uint32_t twoN, BasicInteger Q_lwe,
                                                    uint32_t bitwidth, const DModulus* modulus, uint32_t N) {
    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    if (batch_idx >= batch) {
        return;
    }
    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }

    __shared__ uint32_t b_idx_shared;
    if (threadIdx.x == 0) {
        const BasicInteger v = d_b[batch_idx];
        b_idx_shared = device_SpecialMS_IndexPos(v, twoN, Q_lwe, bitwidth);
    }
    __syncthreads();

    const uint32_t b_idx = b_idx_shared;
    const BasicInteger monic = monic_polys[static_cast<size_t>(b_idx) * N + coeff];

    BasicInteger Q          = modulus[0].value();
    BasicInteger Q_ratio[2] = {modulus[0].const_ratio()[0], modulus[0].const_ratio()[1]};

    const size_t base = static_cast<size_t>(batch_idx) * 2 * N;
    d_acc[base + coeff] = 0;
    const BasicInteger lut_coeff = d_lut[coeff];
    phantom::arith::uint128_t prod_struct = phantom::arith::multiply_uint64_uint64(lut_coeff, monic);
    d_acc[base + N + coeff] = barrett_reduce_uint128_uint64(prod_struct, Q, Q_ratio);
}

__global__ void kernel_SpecialMS_IndexPos_Batch(const BasicInteger* d_a, uint32_t* d_indexPos, uint32_t n, uint32_t batch,
                                                uint32_t a_stride, uint32_t twoN, BasicInteger Q, uint32_t bitwidth) {
    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    if (batch_idx >= batch) {
        return;
    }
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= n) {
        return;
    }

    const uint64_t shift = (bitwidth < 63u) ? (1ULL << bitwidth) : 0ULL;
    if (shift == 0ULL || twoN == 0u || Q == 0) {
        d_indexPos[static_cast<size_t>(batch_idx) * n + coeff] = 0u;
        return;
    }

    const BasicInteger v = d_a[static_cast<size_t>(batch_idx) * a_stride + coeff];
    const double scale = static_cast<double>(twoN) / (static_cast<double>(Q) * static_cast<double>(shift));
    const double x = static_cast<double>(v) * scale;
    const uint64_t r = static_cast<uint64_t>(__double2ull_rd(x + 0.5));
    const uint64_t v_ms = (r * shift) & (static_cast<uint64_t>(twoN) - 1ULL);
    d_indexPos[static_cast<size_t>(batch_idx) * n + coeff] = static_cast<uint32_t>(v_ms);
}

__global__ void kernel_SpecialMS_IndexPos_Batch_NMajor(const BasicInteger* d_a, uint32_t* d_indexPos, uint32_t n, uint32_t batch,
                                                       uint32_t a_stride, uint32_t twoN, BasicInteger Q, uint32_t bitwidth) {
    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    if (batch_idx >= batch) {
        return;
    }
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= n) {
        return;
    }

    const uint64_t shift = (bitwidth < 63u) ? (1ULL << bitwidth) : 0ULL;
    if (shift == 0ULL || twoN == 0u || Q == 0) {
        d_indexPos[static_cast<size_t>(coeff) * batch + batch_idx] = 0u;
        return;
    }

    const BasicInteger v = d_a[static_cast<size_t>(batch_idx) * a_stride + coeff];
    const double scale = static_cast<double>(twoN) / (static_cast<double>(Q) * static_cast<double>(shift));
    const double x = static_cast<double>(v) * scale;
    const uint64_t r = static_cast<uint64_t>(__double2ull_rd(x + 0.5));
    const uint64_t v_ms = (r * shift) & (static_cast<uint64_t>(twoN) - 1ULL);
    d_indexPos[static_cast<size_t>(coeff) * batch + batch_idx] = static_cast<uint32_t>(v_ms);
}

__global__ void kernel_SpecialMS_B_Batch(const BasicInteger* d_b, uint32_t* d_b_idx, uint32_t batch, uint32_t twoN,
                                         BasicInteger Q, uint32_t bitwidth) {
    const uint32_t idx = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= batch) {
        return;
    }

    const uint64_t shift = (bitwidth < 63u) ? (1ULL << bitwidth) : 0ULL;
    if (shift == 0ULL || twoN == 0u || Q == 0) {
        d_b_idx[idx] = 0u;
        return;
    }

    const BasicInteger v = d_b[idx];
    const double scale = static_cast<double>(twoN) / (static_cast<double>(Q) * static_cast<double>(shift));
    const double x = static_cast<double>(v) * scale;
    const uint64_t r = static_cast<uint64_t>(__double2ull_rd(x + 0.5));
    const uint64_t v_ms = (r * shift) & (static_cast<uint64_t>(twoN) - 1ULL);
    d_b_idx[idx] = static_cast<uint32_t>(v_ms);
}

__global__ void kernel_GatherMonomialInv_Batch(BasicInteger* d_monomial_inv, const BasicInteger* monic_polys,
                                               const uint32_t* d_b_idx, uint32_t N, uint32_t batch) {
    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    if (batch_idx >= batch) {
        return;
    }
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }
    const uint32_t mono_row = d_b_idx[batch_idx];
    const size_t src_base = static_cast<size_t>(mono_row) * N;
    const size_t dst_base = static_cast<size_t>(batch_idx) * N;
    d_monomial_inv[dst_base + coeff] = monic_polys[src_base + coeff];
}

__global__ void kernel_ModMulConst(BasicInteger* d_acc, BasicInteger monomial_inv, const DModulus* modulus, size_t N) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    BasicInteger Q          = modulus[0].value();
    BasicInteger Q_ratio[2] = {modulus[0].const_ratio()[0], modulus[0].const_ratio()[1]};

    phantom::arith::uint128_t prod_struct = phantom::arith::multiply_uint64_uint64(d_acc[idx], monomial_inv);
    d_acc[idx] = barrett_reduce_uint128_uint64(prod_struct, Q, Q_ratio);
}

__global__ void kernel_ModAddGpowScaled(BasicInteger *d_c1, const BasicInteger *d_Gpow, BasicInteger n_inv,
                                        const DModulus *modulus, size_t numLUT){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (static_cast<size_t>(idx) >= numLUT) {
        return;
    }

    BasicInteger Q          = modulus[0].value();
    BasicInteger Q_ratio[2] = {modulus[0].const_ratio()[0], modulus[0].const_ratio()[1]};
    uint64_t val_Gpow       = d_Gpow[idx] >> 1;
    phantom::arith::uint128_t prod_gpow = phantom::arith::multiply_uint64_uint64(val_Gpow, n_inv);
    uint64_t temp = phantom::arith::barrett_reduce_uint128_uint64(prod_gpow, Q, Q_ratio);
    
    d_c1[idx] = phantom::arith::add_uint64_uint64_mod(d_c1[idx], temp, Q);
}

__global__ void kernel_ModAddGpowScaledBatch(BasicInteger* d_acc, const BasicInteger* d_Gpow, BasicInteger n_inv, const DModulus* modulus,
                                             uint32_t N, uint32_t numLUT, uint32_t batch) {
    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    if (batch_idx >= batch) {
        return;
    }
    const uint32_t idx = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numLUT) {
        return;
    }

    BasicInteger Q          = modulus[0].value();
    BasicInteger Q_ratio[2] = {modulus[0].const_ratio()[0], modulus[0].const_ratio()[1]};

    const uint64_t val_Gpow = d_Gpow[idx] >> 1;
    phantom::arith::uint128_t prod_gpow = phantom::arith::multiply_uint64_uint64(val_Gpow, n_inv);
    const uint64_t temp = phantom::arith::barrett_reduce_uint128_uint64(prod_gpow, Q, Q_ratio);

    const size_t acc_c1_base = static_cast<size_t>(batch_idx) * 2 * N + N;
    const size_t pos         = acc_c1_base + idx;
    d_acc[pos]               = phantom::arith::add_uint64_uint64_mod(d_acc[pos], temp, Q);
}

__global__ void kernel_GenMVRLWEs(BasicInteger* d_MV_RLWEs_c0, BasicInteger* d_MV_RLWEs_c1, const BasicInteger* d_acc,
                                  const BasicInteger* d_monomials, const DModulus* modulus, size_t N, size_t numLUT,
                                  uint32_t batch) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_luts = static_cast<size_t>(batch) * numLUT;
    const size_t total = total_luts * 2 * N;
    if (idx >= total) {
        return;
    }

    const size_t lut_global = idx / (2 * N);
    const size_t acc_offset = idx - lut_global * (2 * N);
    const size_t poly_idx   = acc_offset / N;
    const size_t coeff_idx  = acc_offset - poly_idx * N;

    const uint32_t batch_idx = static_cast<uint32_t>(lut_global / numLUT);
    const uint32_t lut_local = static_cast<uint32_t>(lut_global - static_cast<size_t>(batch_idx) * numLUT);

    BasicInteger Q          = modulus[0].value();
    BasicInteger Q_ratio[2] = {modulus[0].const_ratio()[0], modulus[0].const_ratio()[1]};

    const size_t in_base = static_cast<size_t>(batch_idx) * 2 * N + poly_idx * N + coeff_idx;
    BasicInteger result  = d_acc[in_base];

    if (lut_local != 0) {
        const uint64_t mono_val = d_monomials[(static_cast<size_t>(lut_local) - 1) * N + coeff_idx];
        phantom::arith::uint128_t temp = phantom::arith::multiply_uint64_uint64(result, mono_val);
        result = phantom::arith::barrett_reduce_uint128_uint64(temp, Q, Q_ratio);
    }

    const size_t out_idx = lut_global * N + coeff_idx;
    if (poly_idx == 0) {
        d_MV_RLWEs_c0[out_idx] = result;
    }
    else {
        d_MV_RLWEs_c1[out_idx] = result;
    }
}



// NEW ADD
__global__ void kernel_GenerateAllAutoMaps(uint32_t* d_AllMaps, uint32_t N, uint32_t /*logn*/,
                                           uint32_t /*m_mask*/, uint32_t numAuto){
    uint32_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level = blockIdx.y;
    if (tid >= N || level >= numAuto) return;

    uint32_t k    = (N >> level) + 1;
    uint32_t jTmp = (tid << 1) + 1;
    uint32_t idx  = (((uint64_t)jTmp * k) & (2 * N - 1)) >> 1;

    d_AllMaps[(size_t)level * N + tid] = idx;
}


__global__ void kernel_GenerateCoefficientAutoMaps(uint32_t* d_AllMaps, uint32_t N, uint32_t logn,
                                           uint32_t m_mask, uint32_t numAuto){
    uint32_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level = blockIdx.y;
 
    if (tid >= N || level >= numAuto) return;                                        
    uint32_t k = (N >> level) + 1;
    uint32_t j = phantom::arith::reverse_bits_uint32(tid, logn);

    uint32_t jTmp    = (j << 1) + 1;   
    uint32_t idx     = (((uint64_t)jTmp * k) & (2 * N - 1)) >> 1;
    uint32_t idxrev  = phantom::arith::reverse_bits_uint32(idx, logn);

    size_t out_offset = (size_t)level * N + tid;
    d_AllMaps[out_offset] = idxrev;
}


// HT
__global__ void kernel_Fused_Permute_Decompose(BasicInteger* d_digits_out, const BasicInteger* d_in_c0, BasicInteger Q,
                                               uint32_t baseHT, uint32_t digitsHT, size_t N, size_t numLUT){
    size_t lut_id       = blockIdx.y;
    size_t coeff_id     = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_id >= N || lut_id >= numLUT) {
        return;
    }
    size_t global_idx   = lut_id * N + coeff_id;
    BasicInteger val    = d_in_c0[global_idx]; // 已经 permute + INTT
    size_t out_base_idx = global_idx;
    size_t stride       = numLUT * N;

    device_SignedDigitDecompose(d_digits_out + out_base_idx, val, Q, baseHT, digitsHT, stride);
}

__global__ void kernel_Permute_Only(BasicInteger *d_out, const BasicInteger *d_in, const uint32_t *d_map, 
                                    size_t N, size_t numLUT, BasicInteger Q){
    size_t lut_id   = blockIdx.y;
    size_t coeff_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (coeff_id >= N) return;

    uint32_t raw_map_idx = d_map[coeff_id];

    bool negate      = false;
    uint32_t src_idx = raw_map_idx;

    if (src_idx >= N) {
        src_idx -= N;
        negate = true;
    }

    size_t global_src_idx  = lut_id * N + src_idx; 
    size_t global_dest_idx = lut_id * N + coeff_id;  
    BasicInteger val       = d_in[global_src_idx];

    if (negate) {
        if (val != 0) {
            val = Q - val;
        }
    }

    d_out[global_dest_idx] = val;
}

__global__ void kernel_INWT_Permute_SignedDigitDecompose(BasicInteger* d_digits_out, const BasicInteger* d_in_eval, const uint32_t* d_map,
                                                         const BasicInteger* itwiddles, const BasicInteger* itwiddles_shoup,
                                                         const DModulus* modulus, const BasicInteger* scalar,
                                                         const BasicInteger* scalar_shoup, uint32_t base, uint32_t digits, size_t N,
                                                         uint32_t numLUT) {
    extern __shared__ BasicInteger buffer[];

    const size_t lut = blockIdx.x;
    const size_t tid = threadIdx.x;
    if (lut >= numLUT || tid >= N / 2) {
        return;
    }

    const size_t mod_idx        = lut;
    const BasicInteger mod      = modulus[0].value();
    const BasicInteger scalar_  = scalar[0];
    const BasicInteger scalar_shoup_ = scalar_shoup[0];

    const BasicInteger* in_poly = d_in_eval + mod_idx * N;

    size_t pairsInGroup;
    size_t k, j, glbIdx;
    BasicInteger samples0{};
    BasicInteger samples1{};

    for (size_t numOfGroups = N / 2; numOfGroups > 0; numOfGroups >>= 1) {
        pairsInGroup = N / numOfGroups / 2;
        k            = tid / pairsInGroup;
        j            = tid % pairsInGroup;
        glbIdx       = 2 * k * pairsInGroup + j;
        const size_t bufIdx = glbIdx;

        const BasicInteger psi       = itwiddles[numOfGroups + k];
        const BasicInteger psi_shoup = itwiddles_shoup[numOfGroups + k];

        if (numOfGroups == N / 2) {
            const size_t local0 = glbIdx;
            const size_t local1 = glbIdx + pairsInGroup;

            uint32_t src0 = d_map[local0];
            bool negate0  = false;
            if (src0 >= N) {
                src0 -= static_cast<uint32_t>(N);
                negate0 = true;
            }
            BasicInteger val0 = in_poly[src0];
            if (negate0 && val0 != 0) {
                val0 = mod - val0;
            }

            uint32_t src1 = d_map[local1];
            bool negate1  = false;
            if (src1 >= N) {
                src1 -= static_cast<uint32_t>(N);
                negate1 = true;
            }
            BasicInteger val1 = in_poly[src1];
            if (negate1 && val1 != 0) {
                val1 = mod - val1;
            }

            samples0 = val0;
            samples1 = val1;
        }
        else {
            samples0 = buffer[bufIdx];
            samples1 = buffer[bufIdx + pairsInGroup];
        }

        phantom::arith::gs_butterfly(samples0, samples1, psi, psi_shoup, mod);

        if (numOfGroups == 1) {
            // final reduction + scale (match inwt_1d_opt_permute semantics)
            phantom::arith::csub_q(samples0, mod);
            phantom::arith::csub_q(samples1, mod);
            samples0 = phantom::arith::multiply_and_reduce_shoup(samples0, scalar_, scalar_shoup_, mod);

            const size_t stride     = static_cast<size_t>(numLUT) * N;
            const size_t coeff0_idx = lut * N + glbIdx;
            const size_t coeff1_idx = lut * N + glbIdx + pairsInGroup;
            device_SignedDigitDecompose(d_digits_out + coeff0_idx, samples0, mod, base, digits, stride);
            device_SignedDigitDecompose(d_digits_out + coeff1_idx, samples1, mod, base, digits, stride);
        }
        else {
            buffer[bufIdx]                = samples0;
            buffer[bufIdx + pairsInGroup] = samples1;
            __syncthreads();
        }
    }
}

__global__ void kernel_INWT_SignedDigitDecompose(BasicInteger* d_digits_out, const BasicInteger* d_in_eval, const BasicInteger* itwiddles,
                                                 const BasicInteger* itwiddles_shoup, const DModulus* modulus,
                                                 const BasicInteger* scalar, const BasicInteger* scalar_shoup, uint32_t base,
                                                 uint32_t digits, size_t N, uint32_t numLUT) {
    extern __shared__ BasicInteger buffer[];

    const size_t lut = blockIdx.x;
    const size_t tid = threadIdx.x;
    if (lut >= numLUT || tid >= N / 2) {
        return;
    }

    const size_t mod_idx        = lut;
    const BasicInteger mod      = modulus[0].value();
    const BasicInteger scalar_  = scalar[0];
    const BasicInteger scalar_shoup_ = scalar_shoup[0];

    const BasicInteger* in_poly = d_in_eval + mod_idx * N;

    size_t pairsInGroup;
    size_t k, j, glbIdx;
    BasicInteger samples0{};
    BasicInteger samples1{};

    for (size_t numOfGroups = N / 2; numOfGroups > 0; numOfGroups >>= 1) {
        pairsInGroup = N / numOfGroups / 2;
        k            = tid / pairsInGroup;
        j            = tid % pairsInGroup;
        glbIdx       = 2 * k * pairsInGroup + j;
        const size_t bufIdx = glbIdx;

        const BasicInteger psi       = itwiddles[numOfGroups + k];
        const BasicInteger psi_shoup = itwiddles_shoup[numOfGroups + k];

        if (numOfGroups == N / 2) {
            const size_t local0 = glbIdx;
            const size_t local1 = glbIdx + pairsInGroup;
            samples0            = in_poly[local0];
            samples1            = in_poly[local1];
        }
        else {
            samples0 = buffer[bufIdx];
            samples1 = buffer[bufIdx + pairsInGroup];
        }

        phantom::arith::gs_butterfly(samples0, samples1, psi, psi_shoup, mod);

        if (numOfGroups == 1) {
            // final reduction + scale (match inwt_1d_opt semantics)
            phantom::arith::csub_q(samples0, mod);
            phantom::arith::csub_q(samples1, mod);
            samples0 = phantom::arith::multiply_and_reduce_shoup(samples0, scalar_, scalar_shoup_, mod);

            const size_t stride     = static_cast<size_t>(numLUT) * N;
            const size_t coeff0_idx = lut * N + glbIdx;
            const size_t coeff1_idx = lut * N + glbIdx + pairsInGroup;
            device_SignedDigitDecompose(d_digits_out + coeff0_idx, samples0, mod, base, digits, stride);
            device_SignedDigitDecompose(d_digits_out + coeff1_idx, samples1, mod, base, digits, stride);
        }
        else {
            buffer[bufIdx]                = samples0;
            buffer[bufIdx + pairsInGroup] = samples1;
            __syncthreads();
        }
    }
}

__global__ void kernel_SaveToRGSW(BasicInteger *d_RGSW, const BasicInteger *d_c0, const BasicInteger *d_c1,  
                                  int row_type, size_t N, size_t numLUT){
    size_t lut_id   = blockIdx.y;
    size_t coeff_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (coeff_id >= N) return;

    size_t global_idx      = lut_id * N + coeff_id;
    size_t lut_base_offset = lut_id * (4 * N);
    size_t row_offset      = (row_type == 0) ? 0 : (2 * N);

    d_RGSW[lut_base_offset + row_offset + coeff_id]     = d_c0[global_idx];
    d_RGSW[lut_base_offset + row_offset + N + coeff_id] = d_c1[global_idx];
}

__global__ void kernel_MultAdd(BasicInteger *d_res_c0, BasicInteger *d_res_c1, const BasicInteger *d_digits,
                            const BasicInteger *d_HTKeys, const DModulus *modulus, uint32_t numDigits, size_t N, size_t numLUT){
    BasicInteger Q          = modulus[0].value();
    BasicInteger Q_ratio[2] = {modulus[0].const_ratio()[0], modulus[0].const_ratio()[1]};
    size_t coeff_idx        = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) return;

    // flatten lut + coeff so each blockIdx.y works on its own LUT slice
    size_t idx = static_cast<size_t>(blockIdx.y) * N + coeff_idx;
    if (idx >= numLUT * N) return;

    BasicInteger sum0     = 0;          // ct->GetElements()[0] += (dcta[d] * ev[d][0]); ?
    BasicInteger sum1     = 0;          // ct->GetElements()[1] += (dcta[d] * ev[d][1]); ?
    size_t digit_stride   = numLUT * N;

    for (uint32_t d = 0; d < numDigits; ++d) {
        BasicInteger digit_val = d_digits[d * digit_stride + idx];
        size_t key_base        = d * (2 * N);
        BasicInteger k0        = d_HTKeys[key_base + coeff_idx];
        BasicInteger k1        = d_HTKeys[key_base + N + coeff_idx];


        BasicInteger prod0 = phantom::arith::multiply_and_barrett_reduce_uint64(digit_val, k0, Q, Q_ratio);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, prod0, Q);

        BasicInteger prod1 = phantom::arith::multiply_and_barrett_reduce_uint64(digit_val, k1, Q, Q_ratio);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, prod1, Q);
    }
    d_res_c0[idx] = sum0;
    d_res_c1[idx] = sum1;
}

__global__ void kernel_AddInPlace(BasicInteger* dst, const BasicInteger* add, const DModulus* modulus, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    BasicInteger Q = modulus[0].value();
    dst[idx] = phantom::arith::add_uint64_uint64_mod(dst[idx], add[idx], Q);
}

__global__ void kernel_MultAddUpdate_HT_PermuteC1(BasicInteger* d_next_c0, BasicInteger* d_next_c1, const BasicInteger* d_digits,
                                                 const BasicInteger* d_HTKeys, const BasicInteger* d_HTKeys_shoup,
                                                 const BasicInteger* d_curr_c0, const BasicInteger* d_curr_c1,
                                                 const uint32_t* d_map, const DModulus* modulus, uint32_t numDigits, size_t N,
                                                 size_t numLUT, size_t total_size) {
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q          = modulus[0].value();

    const size_t digit_stride = numLUT * N;

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;
    for (uint32_t d = 0; d < numDigits; ++d) {
        const BasicInteger digit_val = d_digits[static_cast<size_t>(d) * digit_stride + idx];
        const size_t key_base        = static_cast<size_t>(d) * (2 * N);
        const BasicInteger k0        = d_HTKeys[key_base + coeff_idx];
        const BasicInteger k0_shoup  = d_HTKeys_shoup[key_base + coeff_idx];
        const BasicInteger k1        = d_HTKeys[key_base + N + coeff_idx];
        const BasicInteger k1_shoup  = d_HTKeys_shoup[key_base + N + coeff_idx];

        const BasicInteger prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, k0, k0_shoup, Q);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, prod0, Q);

        const BasicInteger prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, k1, k1_shoup, Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, prod1, Q);
    }

    // next_c0 = curr_c0 + ks_c0
    d_next_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_curr_c0[idx], sum0, Q);

    // next_c1 = curr_c1 + ks_c1 + permute(curr_c1)
    uint32_t raw_map_idx = d_map[coeff_idx];
    bool negate          = false;
    uint32_t src_idx     = raw_map_idx;
    if (src_idx >= N) {
        src_idx -= static_cast<uint32_t>(N);
        negate = true;
    }

    BasicInteger perm_val = d_curr_c1[lut * N + src_idx];
    if (negate && perm_val != 0) {
        perm_val = Q - perm_val;
    }

    const BasicInteger temp = phantom::arith::add_uint64_uint64_mod(d_curr_c1[idx], sum1, Q);
    d_next_c1[idx]          = phantom::arith::add_uint64_uint64_mod(temp, perm_val, Q);
}

__global__ void kernel_MultAddUpdate_HT_PermuteC1_Save(BasicInteger* d_next_c0, BasicInteger* d_next_c1,
                                                       BasicInteger* d_RGSW, const BasicInteger* d_digits,
                                                       const BasicInteger* d_HTKeys, const BasicInteger* d_HTKeys_shoup,
                                                       const BasicInteger* d_curr_c0, const BasicInteger* d_curr_c1,
                                                       const uint32_t* d_map, const DModulus* modulus, uint32_t numDigits,
                                                       size_t N, size_t numLUT, size_t total_size) {
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const size_t digit_stride = numLUT * N;

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;
    for (uint32_t d = 0; d < numDigits; ++d) {
        const BasicInteger digit_val = d_digits[static_cast<size_t>(d) * digit_stride + idx];
        const size_t key_base        = static_cast<size_t>(d) * (2 * N);
        const BasicInteger k0        = d_HTKeys[key_base + coeff_idx];
        const BasicInteger k0_shoup  = d_HTKeys_shoup[key_base + coeff_idx];
        const BasicInteger k1        = d_HTKeys[key_base + N + coeff_idx];
        const BasicInteger k1_shoup  = d_HTKeys_shoup[key_base + N + coeff_idx];

        const BasicInteger prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, k0, k0_shoup, Q);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, prod0, Q);

        const BasicInteger prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, k1, k1_shoup, Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, prod1, Q);
    }

    const BasicInteger out0 = phantom::arith::add_uint64_uint64_mod(d_curr_c0[idx], sum0, Q);

    uint32_t raw_map_idx = d_map[coeff_idx];
    bool negate          = false;
    uint32_t src_idx     = raw_map_idx;
    if (src_idx >= N) {
        src_idx -= static_cast<uint32_t>(N);
        negate = true;
    }

    BasicInteger perm_val = d_curr_c1[lut * N + src_idx];
    if (negate && perm_val != 0) {
        perm_val = Q - perm_val;
    }

    const BasicInteger temp = phantom::arith::add_uint64_uint64_mod(d_curr_c1[idx], sum1, Q);
    const BasicInteger out1 = phantom::arith::add_uint64_uint64_mod(temp, perm_val, Q);

    d_next_c0[idx] = out0;
    d_next_c1[idx] = out1;

    const size_t rgsw_base = lut * (4 * N) + 2 * N;
    d_RGSW[rgsw_base + coeff_idx] = out0;
    d_RGSW[rgsw_base + N + coeff_idx] = out1;
}

__global__ void kernel_MultAddUpdate_HT_PermuteC1_SoA(BasicInteger* d_next_c0, BasicInteger* d_next_c1, const BasicInteger* d_digits,
                                                     const BasicInteger* d_HTKeys, const BasicInteger* d_HTKeys_shoup,
                                                     const BasicInteger* d_curr_c0, const BasicInteger* d_curr_c1,
                                                     const uint32_t* d_map, const DModulus* modulus, uint32_t numDigits, size_t N,
                                                     uint32_t tiles, size_t numLUT, size_t total_size) {
    (void)tiles;
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const size_t digit_stride = numLUT * N;

    const size_t tile = coeff_idx >> 5;
    const size_t lane = coeff_idx & 31u;
    const size_t base = (tile * numDigits) * 2 * 32 + lane;

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;
    for (uint32_t d = 0; d < numDigits; ++d) {
        const BasicInteger digit_val = d_digits[static_cast<size_t>(d) * digit_stride + idx];
        const size_t key_base = base + static_cast<size_t>(d) * 2 * 32;
        const BasicInteger k0 = d_HTKeys[key_base];
        const BasicInteger k0_shoup = d_HTKeys_shoup[key_base];
        const BasicInteger k1 = d_HTKeys[key_base + 32];
        const BasicInteger k1_shoup = d_HTKeys_shoup[key_base + 32];

        const BasicInteger prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, k0, k0_shoup, Q);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, prod0, Q);

        const BasicInteger prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, k1, k1_shoup, Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, prod1, Q);
    }

    d_next_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_curr_c0[idx], sum0, Q);

    uint32_t raw_map_idx = d_map[coeff_idx];
    bool negate          = false;
    uint32_t src_idx     = raw_map_idx;
    if (src_idx >= N) {
        src_idx -= static_cast<uint32_t>(N);
        negate = true;
    }

    BasicInteger perm_val = d_curr_c1[lut * N + src_idx];
    if (negate && perm_val != 0) {
        perm_val = Q - perm_val;
    }

    const BasicInteger temp = phantom::arith::add_uint64_uint64_mod(d_curr_c1[idx], sum1, Q);
    d_next_c1[idx]          = phantom::arith::add_uint64_uint64_mod(temp, perm_val, Q);
}

__global__ void kernel_MultAddUpdate_HT_PermuteC1_SoA_Smem(BasicInteger* d_next_c0, BasicInteger* d_next_c1,
                                                          const BasicInteger* d_digits, const BasicInteger* d_HTKeys,
                                                          const BasicInteger* d_HTKeys_shoup, const BasicInteger* d_curr_c0,
                                                          const BasicInteger* d_curr_c1, const uint32_t* d_map,
                                                          const DModulus* modulus, uint32_t numDigits, size_t N, uint32_t tiles,
                                                          size_t numLUT, size_t total_size) {
    (void)tiles;
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const size_t digit_stride = numLUT * N;

    const size_t tile = coeff_idx >> 5;
    const size_t lane = coeff_idx & 31u;
    const size_t base = (tile * numDigits) * 2 * 32 + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 4u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, d_HTKeys + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, d_HTKeys_shoup + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, d_HTKeys + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, d_HTKeys_shoup + key_base + 32u);
        cirbts_cp_async_commit();
    };

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;
    if (numDigits > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < numDigits; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < numDigits) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger digit_val = d_digits[static_cast<size_t>(d) * digit_stride + idx];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];

            const BasicInteger prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, k0, k0_shoup, Q);
            sum0 = phantom::arith::add_uint64_uint64_mod(sum0, prod0, Q);

            const BasicInteger prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, k1, k1_shoup, Q);
            sum1 = phantom::arith::add_uint64_uint64_mod(sum1, prod1, Q);

            buf ^= 1u;
        }
    }

    d_next_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_curr_c0[idx], sum0, Q);

    uint32_t raw_map_idx = d_map[coeff_idx];
    bool negate          = false;
    uint32_t src_idx     = raw_map_idx;
    if (src_idx >= N) {
        src_idx -= static_cast<uint32_t>(N);
        negate = true;
    }

    BasicInteger perm_val = d_curr_c1[lut * N + src_idx];
    if (negate && perm_val != 0) {
        perm_val = Q - perm_val;
    }

    const BasicInteger temp = phantom::arith::add_uint64_uint64_mod(d_curr_c1[idx], sum1, Q);
    d_next_c1[idx]          = phantom::arith::add_uint64_uint64_mod(temp, perm_val, Q);
}

__global__ void kernel_MultAddUpdate_HTSS_PermuteC1(BasicInteger* d_next_c0, BasicInteger* d_next_c1,
                                                   BasicInteger* d_ss_c0, BasicInteger* d_ss_c1,
                                                   const BasicInteger* d_digits, const BasicInteger* d_HTKeys,
                                                   const BasicInteger* d_HTKeys_shoup, const BasicInteger* d_SSKeys,
                                                   const BasicInteger* d_SSKeys_shoup, const BasicInteger* d_curr_c0,
                                                   const BasicInteger* d_curr_c1, const uint32_t* d_map,
                                                   const DModulus* modulus, uint32_t numDigits, size_t N, size_t numLUT,
                                                   size_t total_size) {
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q      = modulus[0].value();
    const size_t digit_stride = numLUT * N;

    BasicInteger sum0_ht = 0;
    BasicInteger sum1_ht = 0;
    BasicInteger sum0_ss = 0;
    BasicInteger sum1_ss = 0;
    for (uint32_t d = 0; d < numDigits; ++d) {
        const BasicInteger digit_val = d_digits[static_cast<size_t>(d) * digit_stride + idx];
        const size_t key_base        = static_cast<size_t>(d) * (2 * N);

        const BasicInteger ht_k0       = d_HTKeys[key_base + coeff_idx];
        const BasicInteger ht_k0_shoup = d_HTKeys_shoup[key_base + coeff_idx];
        const BasicInteger ht_k1       = d_HTKeys[key_base + N + coeff_idx];
        const BasicInteger ht_k1_shoup = d_HTKeys_shoup[key_base + N + coeff_idx];

        const BasicInteger ss_k0       = d_SSKeys[key_base + coeff_idx];
        const BasicInteger ss_k0_shoup = d_SSKeys_shoup[key_base + coeff_idx];
        const BasicInteger ss_k1       = d_SSKeys[key_base + N + coeff_idx];
        const BasicInteger ss_k1_shoup = d_SSKeys_shoup[key_base + N + coeff_idx];

        const BasicInteger ht_prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, ht_k0, ht_k0_shoup, Q);
        sum0_ht = phantom::arith::add_uint64_uint64_mod(sum0_ht, ht_prod0, Q);

        const BasicInteger ht_prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, ht_k1, ht_k1_shoup, Q);
        sum1_ht = phantom::arith::add_uint64_uint64_mod(sum1_ht, ht_prod1, Q);

        const BasicInteger ss_prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, ss_k0, ss_k0_shoup, Q);
        sum0_ss = phantom::arith::add_uint64_uint64_mod(sum0_ss, ss_prod0, Q);

        const BasicInteger ss_prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, ss_k1, ss_k1_shoup, Q);
        sum1_ss = phantom::arith::add_uint64_uint64_mod(sum1_ss, ss_prod1, Q);
    }

    d_next_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_curr_c0[idx], sum0_ht, Q);

    uint32_t raw_map_idx = d_map[coeff_idx];
    bool negate          = false;
    uint32_t src_idx     = raw_map_idx;
    if (src_idx >= N) {
        src_idx -= static_cast<uint32_t>(N);
        negate = true;
    }

    BasicInteger perm_val = d_curr_c1[lut * N + src_idx];
    if (negate && perm_val != 0) {
        perm_val = Q - perm_val;
    }

    const BasicInteger temp = phantom::arith::add_uint64_uint64_mod(d_curr_c1[idx], sum1_ht, Q);
    d_next_c1[idx]          = phantom::arith::add_uint64_uint64_mod(temp, perm_val, Q);

    d_ss_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_ss_c0[idx], sum0_ss, Q);
    d_ss_c1[idx] = phantom::arith::add_uint64_uint64_mod(d_ss_c1[idx], sum1_ss, Q);
}

__global__ void kernel_MultAddUpdate_HTSS_PermuteC1_SoA(BasicInteger* d_next_c0, BasicInteger* d_next_c1,
                                                       BasicInteger* d_ss_c0, BasicInteger* d_ss_c1,
                                                       const BasicInteger* d_digits, const BasicInteger* d_HTKeys,
                                                       const BasicInteger* d_HTKeys_shoup, const BasicInteger* d_SSKeys,
                                                       const BasicInteger* d_SSKeys_shoup, const BasicInteger* d_curr_c0,
                                                       const BasicInteger* d_curr_c1, const uint32_t* d_map,
                                                       const DModulus* modulus, uint32_t numDigits, size_t N, uint32_t tiles,
                                                       size_t numLUT, size_t total_size) {
    (void)tiles;
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const size_t digit_stride = numLUT * N;

    const size_t tile = coeff_idx >> 5;
    const size_t lane = coeff_idx & 31u;
    const size_t base = (tile * numDigits) * 2 * 32 + lane;

    BasicInteger sum0_ht = 0;
    BasicInteger sum1_ht = 0;
    BasicInteger sum0_ss = 0;
    BasicInteger sum1_ss = 0;
    for (uint32_t d = 0; d < numDigits; ++d) {
        const BasicInteger digit_val = d_digits[static_cast<size_t>(d) * digit_stride + idx];
        const size_t key_base = base + static_cast<size_t>(d) * 2 * 32;

        const BasicInteger ht_k0       = d_HTKeys[key_base];
        const BasicInteger ht_k0_shoup = d_HTKeys_shoup[key_base];
        const BasicInteger ht_k1       = d_HTKeys[key_base + 32];
        const BasicInteger ht_k1_shoup = d_HTKeys_shoup[key_base + 32];

        const BasicInteger ss_k0       = d_SSKeys[key_base];
        const BasicInteger ss_k0_shoup = d_SSKeys_shoup[key_base];
        const BasicInteger ss_k1       = d_SSKeys[key_base + 32];
        const BasicInteger ss_k1_shoup = d_SSKeys_shoup[key_base + 32];

        const BasicInteger ht_prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, ht_k0, ht_k0_shoup, Q);
        sum0_ht = phantom::arith::add_uint64_uint64_mod(sum0_ht, ht_prod0, Q);

        const BasicInteger ht_prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, ht_k1, ht_k1_shoup, Q);
        sum1_ht = phantom::arith::add_uint64_uint64_mod(sum1_ht, ht_prod1, Q);

        const BasicInteger ss_prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, ss_k0, ss_k0_shoup, Q);
        sum0_ss = phantom::arith::add_uint64_uint64_mod(sum0_ss, ss_prod0, Q);

        const BasicInteger ss_prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, ss_k1, ss_k1_shoup, Q);
        sum1_ss = phantom::arith::add_uint64_uint64_mod(sum1_ss, ss_prod1, Q);
    }

    d_next_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_curr_c0[idx], sum0_ht, Q);

    uint32_t raw_map_idx = d_map[coeff_idx];
    bool negate          = false;
    uint32_t src_idx     = raw_map_idx;
    if (src_idx >= N) {
        src_idx -= static_cast<uint32_t>(N);
        negate = true;
    }

    BasicInteger perm_val = d_curr_c1[lut * N + src_idx];
    if (negate && perm_val != 0) {
        perm_val = Q - perm_val;
    }

    const BasicInteger temp = phantom::arith::add_uint64_uint64_mod(d_curr_c1[idx], sum1_ht, Q);
    d_next_c1[idx]          = phantom::arith::add_uint64_uint64_mod(temp, perm_val, Q);

    d_ss_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_ss_c0[idx], sum0_ss, Q);
    d_ss_c1[idx] = phantom::arith::add_uint64_uint64_mod(d_ss_c1[idx], sum1_ss, Q);
}

__global__ void kernel_MultAddUpdate_HTSS_PermuteC1_SoA_Smem(BasicInteger* d_next_c0, BasicInteger* d_next_c1,
                                                            BasicInteger* d_ss_c0, BasicInteger* d_ss_c1,
                                                            const BasicInteger* d_digits, const BasicInteger* d_HTKeys,
                                                            const BasicInteger* d_HTKeys_shoup, const BasicInteger* d_SSKeys,
                                                            const BasicInteger* d_SSKeys_shoup, const BasicInteger* d_curr_c0,
                                                            const BasicInteger* d_curr_c1, const uint32_t* d_map,
                                                            const DModulus* modulus, uint32_t numDigits, size_t N, uint32_t tiles,
                                                            size_t numLUT, size_t total_size) {
    (void)tiles;
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const size_t digit_stride = numLUT * N;

    const size_t tile = coeff_idx >> 5;
    const size_t lane = coeff_idx & 31u;
    const size_t base = (tile * numDigits) * 2 * 32 + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 8u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, d_HTKeys + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, d_HTKeys_shoup + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, d_HTKeys + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, d_HTKeys_shoup + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 4u * 32u, d_SSKeys + key_base);
        cirbts_cp_async_8(buf + lane + 5u * 32u, d_SSKeys_shoup + key_base);
        cirbts_cp_async_8(buf + lane + 6u * 32u, d_SSKeys + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 7u * 32u, d_SSKeys_shoup + key_base + 32u);
        cirbts_cp_async_commit();
    };

    BasicInteger sum0_ht = 0;
    BasicInteger sum1_ht = 0;
    BasicInteger sum0_ss = 0;
    BasicInteger sum1_ss = 0;
    if (numDigits > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < numDigits; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < numDigits) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger digit_val = d_digits[static_cast<size_t>(d) * digit_stride + idx];

            const BasicInteger ht_k0 = cur[lane + 0u * 32u];
            const BasicInteger ht_k0_shoup = cur[lane + 1u * 32u];
            const BasicInteger ht_k1 = cur[lane + 2u * 32u];
            const BasicInteger ht_k1_shoup = cur[lane + 3u * 32u];

            const BasicInteger ss_k0 = cur[lane + 4u * 32u];
            const BasicInteger ss_k0_shoup = cur[lane + 5u * 32u];
            const BasicInteger ss_k1 = cur[lane + 6u * 32u];
            const BasicInteger ss_k1_shoup = cur[lane + 7u * 32u];

            const BasicInteger ht_prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, ht_k0, ht_k0_shoup, Q);
            sum0_ht = phantom::arith::add_uint64_uint64_mod(sum0_ht, ht_prod0, Q);

            const BasicInteger ht_prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, ht_k1, ht_k1_shoup, Q);
            sum1_ht = phantom::arith::add_uint64_uint64_mod(sum1_ht, ht_prod1, Q);

            const BasicInteger ss_prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, ss_k0, ss_k0_shoup, Q);
            sum0_ss = phantom::arith::add_uint64_uint64_mod(sum0_ss, ss_prod0, Q);

            const BasicInteger ss_prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, ss_k1, ss_k1_shoup, Q);
            sum1_ss = phantom::arith::add_uint64_uint64_mod(sum1_ss, ss_prod1, Q);

            buf ^= 1u;
        }
    }

    d_next_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_curr_c0[idx], sum0_ht, Q);

    uint32_t raw_map_idx = d_map[coeff_idx];
    bool negate          = false;
    uint32_t src_idx     = raw_map_idx;
    if (src_idx >= N) {
        src_idx -= static_cast<uint32_t>(N);
        negate = true;
    }

    BasicInteger perm_val = d_curr_c1[lut * N + src_idx];
    if (negate && perm_val != 0) {
        perm_val = Q - perm_val;
    }

    const BasicInteger temp = phantom::arith::add_uint64_uint64_mod(d_curr_c1[idx], sum1_ht, Q);
    d_next_c1[idx]          = phantom::arith::add_uint64_uint64_mod(temp, perm_val, Q);

    d_ss_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_ss_c0[idx], sum0_ss, Q);
    d_ss_c1[idx] = phantom::arith::add_uint64_uint64_mod(d_ss_c1[idx], sum1_ss, Q);
}

__global__ void kernel_Update_HT_PermuteC1(BasicInteger* d_next_c0, BasicInteger* d_next_c1, const BasicInteger* d_ks_c0,
                                          const BasicInteger* d_ks_c1, const BasicInteger* d_curr_c0,
                                          const BasicInteger* d_curr_c1, const uint32_t* d_map, const DModulus* modulus,
                                          size_t N, size_t total_size) {
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q = modulus[0].value();

    // next_c0 = curr_c0 + ks_c0
    d_next_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_curr_c0[idx], d_ks_c0[idx], Q);

    // next_c1 = curr_c1 + ks_c1 + permute(curr_c1)
    uint32_t raw_map_idx = d_map[coeff_idx];
    bool negate          = false;
    uint32_t src_idx     = raw_map_idx;
    if (src_idx >= N) {
        src_idx -= static_cast<uint32_t>(N);
        negate = true;
    }

    BasicInteger perm_val = d_curr_c1[lut * N + src_idx];
    if (negate && perm_val != 0) {
        perm_val = Q - perm_val;
    }

    const BasicInteger temp = phantom::arith::add_uint64_uint64_mod(d_curr_c1[idx], d_ks_c1[idx], Q);
    d_next_c1[idx]          = phantom::arith::add_uint64_uint64_mod(temp, perm_val, Q);
}

__global__ void kernel_MultAddUpdate_SS(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_digits,
                                       const BasicInteger* d_SSKeys, const BasicInteger* d_SSKeys_shoup,
                                       const BasicInteger* d_backup_c1, const DModulus* modulus, uint32_t numDigits, size_t N,
                                       size_t numLUT, size_t total_size) {
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q          = modulus[0].value();
    const size_t digit_stride     = numLUT * N;

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;
    for (uint32_t d = 0; d < numDigits; ++d) {
        const BasicInteger digit_val = d_digits[static_cast<size_t>(d) * digit_stride + idx];
        const size_t key_base        = static_cast<size_t>(d) * (2 * N);
        const BasicInteger k0        = d_SSKeys[key_base + coeff_idx];
        const BasicInteger k0_shoup  = d_SSKeys_shoup[key_base + coeff_idx];
        const BasicInteger k1        = d_SSKeys[key_base + N + coeff_idx];
        const BasicInteger k1_shoup  = d_SSKeys_shoup[key_base + N + coeff_idx];

        const BasicInteger prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, k0, k0_shoup, Q);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, prod0, Q);

        const BasicInteger prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, k1, k1_shoup, Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, prod1, Q);
    }

    // c0 = backup_c1 + ks_c0
    d_out_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_backup_c1[idx], sum0, Q);
    // c1 = ks_c1
    d_out_c1[idx] = sum1;
}

__global__ void kernel_MultAddUpdate_SS_Save(BasicInteger* d_out_c0, BasicInteger* d_out_c1, BasicInteger* d_RGSW,
                                             const BasicInteger* d_digits, const BasicInteger* d_SSKeys,
                                             const BasicInteger* d_SSKeys_shoup, const BasicInteger* d_backup_c1,
                                             const DModulus* modulus, uint32_t numDigits, size_t N, size_t numLUT,
                                             size_t total_size) {
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q      = modulus[0].value();
    const size_t digit_stride = numLUT * N;

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;
    for (uint32_t d = 0; d < numDigits; ++d) {
        const BasicInteger digit_val = d_digits[static_cast<size_t>(d) * digit_stride + idx];
        const size_t key_base        = static_cast<size_t>(d) * (2 * N);
        const BasicInteger k0        = d_SSKeys[key_base + coeff_idx];
        const BasicInteger k0_shoup  = d_SSKeys_shoup[key_base + coeff_idx];
        const BasicInteger k1        = d_SSKeys[key_base + N + coeff_idx];
        const BasicInteger k1_shoup  = d_SSKeys_shoup[key_base + N + coeff_idx];

        const BasicInteger prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, k0, k0_shoup, Q);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, prod0, Q);

        const BasicInteger prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, k1, k1_shoup, Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, prod1, Q);
    }

    const BasicInteger out0 = phantom::arith::add_uint64_uint64_mod(d_backup_c1[idx], sum0, Q);
    const BasicInteger out1 = sum1;

    d_out_c0[idx] = out0;
    d_out_c1[idx] = out1;

    const size_t rgsw_base = lut * (4 * N);
    d_RGSW[rgsw_base + coeff_idx] = out0;
    d_RGSW[rgsw_base + N + coeff_idx] = out1;
}

__global__ void kernel_MultAddUpdate_SS_SoA(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_digits,
                                           const BasicInteger* d_SSKeys, const BasicInteger* d_SSKeys_shoup,
                                           const BasicInteger* d_backup_c1, const DModulus* modulus, uint32_t numDigits, size_t N,
                                           uint32_t tiles, size_t numLUT, size_t total_size) {
    (void)tiles;
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const size_t digit_stride = numLUT * N;

    const size_t tile = coeff_idx >> 5;
    const size_t lane = coeff_idx & 31u;
    const size_t base = (tile * numDigits) * 2 * 32 + lane;

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;
    for (uint32_t d = 0; d < numDigits; ++d) {
        const BasicInteger digit_val = d_digits[static_cast<size_t>(d) * digit_stride + idx];
        const size_t key_base = base + static_cast<size_t>(d) * 2 * 32;
        const BasicInteger k0 = d_SSKeys[key_base];
        const BasicInteger k0_shoup = d_SSKeys_shoup[key_base];
        const BasicInteger k1 = d_SSKeys[key_base + 32];
        const BasicInteger k1_shoup = d_SSKeys_shoup[key_base + 32];

        const BasicInteger prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, k0, k0_shoup, Q);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, prod0, Q);

        const BasicInteger prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, k1, k1_shoup, Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, prod1, Q);
    }

    d_out_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_backup_c1[idx], sum0, Q);
    d_out_c1[idx] = sum1;
}

__global__ void kernel_MultAddUpdate_SS_SoA_Smem(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_digits,
                                                const BasicInteger* d_SSKeys, const BasicInteger* d_SSKeys_shoup,
                                                const BasicInteger* d_backup_c1, const DModulus* modulus, uint32_t numDigits,
                                                size_t N, uint32_t tiles, size_t numLUT, size_t total_size) {
    (void)tiles;
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const size_t digit_stride = numLUT * N;

    const size_t tile = coeff_idx >> 5;
    const size_t lane = coeff_idx & 31u;
    const size_t base = (tile * numDigits) * 2 * 32 + lane;

    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x >> 5);
    const uint32_t warps_per_block = (blockDim.x + 31) >> 5;
    const size_t warp_stride = 4u * 32u;
    const size_t buf_stride = static_cast<size_t>(warps_per_block) * warp_stride;
    extern __shared__ BasicInteger smem[];
    BasicInteger* buf0 = smem + static_cast<size_t>(warp_id) * warp_stride;
    BasicInteger* buf1 = smem + buf_stride + static_cast<size_t>(warp_id) * warp_stride;

    auto prefetch = [&](uint32_t d, BasicInteger* buf) {
        const size_t key_base = base + static_cast<size_t>(d) * 2u * 32u;
        cirbts_cp_async_8(buf + lane + 0u * 32u, d_SSKeys + key_base);
        cirbts_cp_async_8(buf + lane + 1u * 32u, d_SSKeys_shoup + key_base);
        cirbts_cp_async_8(buf + lane + 2u * 32u, d_SSKeys + key_base + 32u);
        cirbts_cp_async_8(buf + lane + 3u * 32u, d_SSKeys_shoup + key_base + 32u);
        cirbts_cp_async_commit();
    };

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;
    if (numDigits > 0) {
        uint32_t buf = 0;
        prefetch(0, buf0);
        for (uint32_t d = 0; d < numDigits; ++d) {
            cirbts_cp_async_wait();
            BasicInteger* cur = (buf == 0) ? buf0 : buf1;
            if (d + 1 < numDigits) {
                prefetch(d + 1, (buf == 0) ? buf1 : buf0);
            }

            const BasicInteger digit_val = d_digits[static_cast<size_t>(d) * digit_stride + idx];
            const BasicInteger k0 = cur[lane + 0u * 32u];
            const BasicInteger k0_shoup = cur[lane + 1u * 32u];
            const BasicInteger k1 = cur[lane + 2u * 32u];
            const BasicInteger k1_shoup = cur[lane + 3u * 32u];

            const BasicInteger prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, k0, k0_shoup, Q);
            sum0 = phantom::arith::add_uint64_uint64_mod(sum0, prod0, Q);

            const BasicInteger prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, k1, k1_shoup, Q);
            sum1 = phantom::arith::add_uint64_uint64_mod(sum1, prod1, Q);

            buf ^= 1u;
        }
    }

    d_out_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_backup_c1[idx], sum0, Q);
    d_out_c1[idx] = sum1;
}

__global__ void kernel_MultAddUpdate_SS_Accumulate(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_digits,
                                                   const BasicInteger* d_SSKeys, const BasicInteger* d_SSKeys_shoup,
                                                   const DModulus* modulus, uint32_t numDigits, size_t N, size_t numLUT,
                                                   size_t total_size) {
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q      = modulus[0].value();
    const size_t digit_stride = numLUT * N;

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;
    for (uint32_t d = 0; d < numDigits; ++d) {
        const BasicInteger digit_val = d_digits[static_cast<size_t>(d) * digit_stride + idx];
        const size_t key_base        = static_cast<size_t>(d) * (2 * N);
        const BasicInteger k0        = d_SSKeys[key_base + coeff_idx];
        const BasicInteger k0_shoup  = d_SSKeys_shoup[key_base + coeff_idx];
        const BasicInteger k1        = d_SSKeys[key_base + N + coeff_idx];
        const BasicInteger k1_shoup  = d_SSKeys_shoup[key_base + N + coeff_idx];

        const BasicInteger prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, k0, k0_shoup, Q);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, prod0, Q);

        const BasicInteger prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, k1, k1_shoup, Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, prod1, Q);
    }

    d_out_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_out_c0[idx], sum0, Q);
    d_out_c1[idx] = phantom::arith::add_uint64_uint64_mod(d_out_c1[idx], sum1, Q);
}

__global__ void kernel_MultAddUpdate_SS_Accumulate_SoA(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_digits,
                                                      const BasicInteger* d_SSKeys, const BasicInteger* d_SSKeys_shoup,
                                                      const DModulus* modulus, uint32_t numDigits, size_t N, uint32_t tiles,
                                                      size_t numLUT, size_t total_size) {
    (void)tiles;
    const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= N) {
        return;
    }

    const size_t lut = blockIdx.y;
    const size_t idx = lut * N + coeff_idx;
    if (idx >= total_size) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const size_t digit_stride = numLUT * N;

    const size_t tile = coeff_idx >> 5;
    const size_t lane = coeff_idx & 31u;
    const size_t base = (tile * numDigits) * 2 * 32 + lane;

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;
    for (uint32_t d = 0; d < numDigits; ++d) {
        const BasicInteger digit_val = d_digits[static_cast<size_t>(d) * digit_stride + idx];
        const size_t key_base = base + static_cast<size_t>(d) * 2 * 32;
        const BasicInteger k0 = d_SSKeys[key_base];
        const BasicInteger k0_shoup = d_SSKeys_shoup[key_base];
        const BasicInteger k1 = d_SSKeys[key_base + 32];
        const BasicInteger k1_shoup = d_SSKeys_shoup[key_base + 32];

        const BasicInteger prod0 = phantom::arith::multiply_and_reduce_shoup(digit_val, k0, k0_shoup, Q);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, prod0, Q);

        const BasicInteger prod1 = phantom::arith::multiply_and_reduce_shoup(digit_val, k1, k1_shoup, Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, prod1, Q);
    }

    d_out_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_out_c0[idx], sum0, Q);
    d_out_c1[idx] = phantom::arith::add_uint64_uint64_mod(d_out_c1[idx], sum1, Q);
}

__global__ void kernel_Update_SS(BasicInteger *d_out_c0, BasicInteger *d_out_c1,
                                 const BasicInteger *d_ks_c0, const BasicInteger *d_ks_c1,
                                 const BasicInteger *d_backup_c1,
                                 const DModulus *modulus,
                                 size_t N, size_t total_size) {
    size_t coeff = blockIdx.x * blockDim.x + threadIdx.x;
    size_t lut   = blockIdx.y; 
    if (coeff >= N || lut * N + coeff >= total_size) return;

    size_t idx = lut * N + coeff;
    BasicInteger Q = modulus[0].value();

    // c0 = backup_c1 + ks_c0
    d_out_c0[idx] = phantom::arith::add_uint64_uint64_mod(d_backup_c1[idx], d_ks_c0[idx], Q);
    // c1 = ks_c1
    d_out_c1[idx] = d_ks_c1[idx];
}

template __global__ void kernel_EvalAccCore_Binary_Swizzle<2>(BasicInteger* acc, const BasicInteger* dct,
                                                              const BasicInteger* ek_swizzle,
                                                              const BasicInteger* ek_shoup_swizzle,
                                                              const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                              const DModulus* mod, size_t lweIndex,
                                                              const uint32_t* indexPos);
template __global__ void kernel_EvalAccCore_Binary_Swizzle_Delta<2>(BasicInteger* acc, BasicInteger* delta,
                                                                    const BasicInteger* dct,
                                                                    const BasicInteger* ek_swizzle,
                                                                    const BasicInteger* ek_shoup_swizzle,
                                                                    const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                    const DModulus* mod, size_t lweIndex,
                                                                    const uint32_t* indexPos);
template __global__ void kernel_EvalAccCore_Binary_Batch_Swizzle<2>(BasicInteger* acc, const BasicInteger* dct,
                                                                    const BasicInteger* ek_swizzle,
                                                                    const BasicInteger* ek_shoup_swizzle,
                                                                    const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                    const DModulus* mod, uint32_t lweIndex,
                                                                    const uint32_t* d_indexPos, uint32_t indexStride);
template __global__ void kernel_EvalAccCore_Binary_Grouped_Swizzle<2>(BasicInteger* acc, const BasicInteger* dct,
                                                                      const BasicInteger* ek_swizzle,
                                                                      const BasicInteger* ek_shoup_swizzle,
                                                                      const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                      const DModulus* mod, size_t key_group,
                                                                      size_t lweIndex, const uint32_t* indexPos);
template __global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Swizzle<2>(BasicInteger* acc, const BasicInteger* dct,
                                                                          const BasicInteger* ek0_swizzle,
                                                                          const BasicInteger* ek0_shoup_swizzle,
                                                                          const BasicInteger* ek1_swizzle,
                                                                          const BasicInteger* ek1_shoup_swizzle,
                                                                          const BasicInteger* ek_pair_swizzle,
                                                                          const BasicInteger* ek_pair_shoup_swizzle,
                                                                          const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                          const DModulus* mod, size_t key_group,
                                                                          size_t lweIndex0, size_t lweIndex1,
                                                                          const uint32_t* indexPos);
template __global__ void kernel_EvalAccCore_Binary_Grouped_Batch_Swizzle<2>(BasicInteger* acc, const BasicInteger* dct,
                                                                            const BasicInteger* ek_swizzle,
                                                                            const BasicInteger* ek_shoup_swizzle,
                                                                            const BasicInteger* monic_polys, size_t N, uint32_t tiles,
                                                                            const DModulus* mod, uint32_t key_group,
                                                                            uint32_t lweIndex, const uint32_t* d_indexPos,
                                                                            uint32_t indexStride);
template __global__ void kernel_EvalAccCore_Binary_MB2_Grouped_Batch_Swizzle<2>(BasicInteger* acc, const BasicInteger* dct,
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
template __global__ void kernel_EvalAccCore_Binary_SoA_T<2>(BasicInteger* acc, const BasicInteger* dct,
                                                           const BasicInteger* ek_soa,
                                                           const BasicInteger* ek_shoup_soa,
                                                           const BasicInteger* monic_polys, size_t N,
                                                           uint32_t tiles, const DModulus* mod,
                                                           size_t lweIndex, const uint32_t* indexPos);
template __global__ void kernel_EvalAccCore_Binary_SoA_T<4>(BasicInteger* acc, const BasicInteger* dct,
                                                           const BasicInteger* ek_soa,
                                                           const BasicInteger* ek_shoup_soa,
                                                           const BasicInteger* monic_polys, size_t N,
                                                           uint32_t tiles, const DModulus* mod,
                                                           size_t lweIndex, const uint32_t* indexPos);
template __global__ void kernel_EvalAccCore_Binary_SoA_T_Smem<2>(BasicInteger* acc, const BasicInteger* dct,
                                                                 const BasicInteger* ek_soa,
                                                                 const BasicInteger* ek_shoup_soa,
                                                                 const BasicInteger* monic_polys, size_t N,
                                                                 uint32_t tiles, const DModulus* mod,
                                                                 size_t lweIndex, const uint32_t* indexPos);
template __global__ void kernel_EvalAccCore_Binary_SoA_T_Smem<4>(BasicInteger* acc, const BasicInteger* dct,
                                                                 const BasicInteger* ek_soa,
                                                                 const BasicInteger* ek_shoup_soa,
                                                                 const BasicInteger* monic_polys, size_t N,
                                                                 uint32_t tiles, const DModulus* mod,
                                                                 size_t lweIndex, const uint32_t* indexPos);
