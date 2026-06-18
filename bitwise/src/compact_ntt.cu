#include "compact_ntt.cuh"
#include "modarith.cuh"

#include "arith/nbtheory.h"
#include "arith/big_integer_modop.h"

__global__ void kernel_multiply(BasicInteger* output, const BasicInteger* input1, const BasicInteger* input2,
                                BasicInteger mod, BasicInteger mu_lo, BasicInteger mu_hi) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = modMulBarrett(input1[idx], input2[idx], mod, wide_type<BasicInteger>(mu_lo, mu_hi));
}

void FourStepNTT::multiply(BasicInteger* output, const BasicInteger* input1, const BasicInteger* input2,
                           const cudaStream_t& stream) {
    kernel_multiply<<<n_ / 256, 256, 0, stream>>>(
        output, input1, input2, q_, mu_[0], mu_[1]);
}

__global__ void kernel_multiply_and_accumulate(BasicInteger* output,
                                               const BasicInteger* input1, const BasicInteger* input2,
                                               BasicInteger mod, BasicInteger mu_lo, BasicInteger mu_hi) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto tmp = modMulBarrett(input1[idx], input2[idx], mod, wide_type<BasicInteger>(mu_lo, mu_hi));
    tmp += output[idx];
    output[idx] = modFast(tmp, mod);
}

void FourStepNTT::multiply_and_accumulate(BasicInteger* output, const BasicInteger* input1, const BasicInteger* input2,
                                          const cudaStream_t& stream) const
{
    kernel_multiply_and_accumulate<<<n_ / 256, 256, 0, stream>>>(
        output, input1, input2, q_, mu_[0], mu_[1]);
}

__global__ void kernel_multiply_scalar(BasicInteger* output,
                                       const BasicInteger* input, BasicInteger scalar,
                                       BasicInteger mod, BasicInteger mu_lo, BasicInteger mu_hi) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = modMulBarrett(input[idx], scalar, mod, wide_type<BasicInteger>(mu_lo, mu_hi));
}

void FourStepNTT::multiply_scalar(BasicInteger* output, const BasicInteger* input, BasicInteger scalar,
                                  const cudaStream_t& stream) {
    kernel_multiply_scalar<<<n_ / 256, 256, 0, stream>>>(
        output, input, scalar, q_, mu_[0], mu_[1]);
}

__global__ void kernel_add(BasicInteger* output, const BasicInteger* input1, const BasicInteger* input2,
                           const BasicInteger mod) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = modFast(input1[idx] + input2[idx], mod);
}

void FourStepNTT::add(BasicInteger* output, const BasicInteger* input1, const BasicInteger* input2,
                      const cudaStream_t& stream) {
    kernel_add<<<n_ / 256, 256, 0, stream>>>(output, input1, input2, q_);
}

/**
  * This struct contains a op and a precomputed quotient: (op << 64) / mod, for a specific mod.
  * When passed to multiply_uint_mod, a faster variant of Barrett reduction will be performed.
  * Operand must be less than mod.
  */
static uint64_t compute_shoup(const uint64_t op, const uint64_t mod) {
    // Using __uint128_t to avoid overflow during multiplication
    __uint128_t temp = op;
    temp <<= 64; // multiplying by 2^64
    return temp / mod;
}

static uint32_t compute_shoup(const uint32_t op, const uint32_t mod) {
    uint64_t temp = op;
    temp <<= 32; // multiplying by 2^32
    return temp / mod;
}

void FourStepNTT::gen_tw_table(BasicInteger* tw_root, BasicInteger* tw_root_shoup, BasicInteger root,
                               size_t len, size_t log_len, BasicInteger mod, const cudaStream_t& stream) {
    std::vector<BasicInteger> host_tw_root(len);
    std::vector<BasicInteger> host_tw_root_shoup(len);
    host_tw_root[0] = 1;
    host_tw_root_shoup[0] = compute_shoup(1, mod);
    BasicInteger power = root;
    for (size_t i = 1; i < len; i++) {
        host_tw_root[isecfhe::ReverseBits(i, log_len)] = power;
        host_tw_root_shoup[isecfhe::ReverseBits(i, log_len)] = compute_shoup(power, mod);
        power = isecfhe::util::ModMul(power, root, mod);
    }
    cudaMemcpyAsync(tw_root, host_tw_root.data(), len * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tw_root_shoup, host_tw_root_shoup.data(), len * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
}

FourStepNTT::FourStepNTT(size_t n, BasicInteger mod, const cudaStream_t& stream) {
    // set n, n1, n2, log_n, log_n1, log_n2
    n_ = n;
    if (n_ == 1024) {
        n1_ = 32;
        n2_ = 32;
        log_n_ = 10;
        log_n1_ = 5;
        log_n2_ = 5;
    } else if (n_ == 2048) {
        n1_ = 64;
        n2_ = 32;
        log_n_ = 11;
        log_n1_ = 6;
        log_n2_ = 5;
    } else if (n_ == 4096) {
        n1_ = 64;
        n2_ = 64;
        log_n_ = 12;
        log_n1_ = 6;
        log_n2_ = 6;
    } else {
        throw std::invalid_argument("Current 4step NTT implementation requires n to be 1024, 2048, or 4096");
    }

    // set q
    q_ = mod;
    mu_ = (isecfhe::BigInteger<BasicInteger>(1) << (8 * sizeof(BasicInteger) * 2)).DividedBy(q_).first.GetValue();

    // set root_2n and inv_root_2n
    root_2n_ = isecfhe::RootOfUnity(2 * n_, isecfhe::BigInteger<BasicInteger>(q_)).ConvertToInt<BasicInteger>();
    inv_root_2n_ = isecfhe::util::ModInverse(root_2n_, q_);

    // set montgomery constants
    const auto big_q = isecfhe::BigInteger<BasicInteger>(q_);
    const auto big_B = isecfhe::BigInteger<BasicInteger>(1) << (8 * sizeof(BasicInteger));
    const auto big_B_mod_q = big_B.Mod(big_q);
    B_mod_q_ = big_B_mod_q.ConvertToInt<BasicInteger>();
    B2_mod_q_ = (big_B_mod_q * big_B_mod_q).Mod(big_q).ConvertToInt<BasicInteger>();
    q_inv_mod_B_ = isecfhe::util::ModInverse(big_q, big_B).ConvertToInt<BasicInteger>();

    // set inv_n and inv_n_shoup
    inv_n_ = isecfhe::util::ModInverse(static_cast<BasicInteger>(n_), q_);
    inv_n_shoup_ = compute_shoup(inv_n_, q_);
    const auto big_inv_n_R2_mod_q = isecfhe::BigInteger<BasicInteger>(inv_n_) * isecfhe::BigInteger<
                                        BasicInteger>(B2_mod_q_);
    const auto t = (big_inv_n_R2_mod_q * isecfhe::BigInteger<BasicInteger>(q_inv_mod_B_)).getValue()[0];
    inv_n_mont_ = (big_inv_n_R2_mod_q >> (8 * sizeof(BasicInteger))).getValue()[0] + q_ -
                  ((isecfhe::BigInteger<BasicInteger>(t) * big_q) >> (8 * sizeof(BasicInteger))).getValue()[0];

    /*******************************************************************************************************************
     * Precompute twiddle factors for negacyclic convolution in 4-step NTT
     ******************************************************************************************************************/

    // root_2n1 = root_2n ^ n2
    BasicInteger root_2n1 = isecfhe::util::ModExp(root_2n_, static_cast<BasicInteger>(n2_), q_);
    BasicInteger inv_root_2n1 = isecfhe::util::ModInverse(root_2n1, q_);

    tw_root_2n1_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    tw_root_2n1_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    tw_root_2n1_mont_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    gen_tw_table(tw_root_2n1_.get(), tw_root_2n1_shoup_.get(), root_2n1,
                 n1_, log_n1_, q_, stream);
    to_mont(tw_root_2n1_mont_.get(), tw_root_2n1_.get(), 256, 1, stream);

    tw_inv_root_2n1_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    tw_inv_root_2n1_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    tw_inv_root_2n1_mont_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    gen_tw_table(tw_inv_root_2n1_.get(), tw_inv_root_2n1_shoup_.get(), inv_root_2n1,
                 n1_, log_n1_, q_, stream);
    to_mont(tw_inv_root_2n1_mont_.get(), tw_inv_root_2n1_.get(), 256, 1, stream);

    /*******************************************************************************************************************
     * Precompute twiddle factors for cyclic convolution in 4-step NTT
     ******************************************************************************************************************/

    // root_n2 = root_2n ^ (2 * n1)
    BasicInteger root_n2 = isecfhe::util::ModExp(root_2n_, static_cast<BasicInteger>(2 * n1_), q_);
    BasicInteger inv_root_n2 = isecfhe::util::ModInverse(root_n2, q_);

    tw_root_n2_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    tw_root_n2_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    tw_root_n2_mont_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    gen_tw_table(tw_root_n2_.get(), tw_root_n2_shoup_.get(), root_n2,
                 n2_ / 2, log_n2_ - 1, q_, stream);
    to_mont(tw_root_n2_mont_.get(), tw_root_n2_.get(), 256, 1, stream);

    tw_inv_root_n2_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    tw_inv_root_n2_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    tw_inv_root_n2_mont_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    gen_tw_table(tw_inv_root_n2_.get(), tw_inv_root_n2_shoup_.get(), inv_root_n2,
                 n2_ / 2, log_n2_ - 1, q_, stream);
    to_mont(tw_inv_root_n2_mont_.get(), tw_inv_root_n2_.get(), 256, 1, stream);

    /*******************************************************************************************************************
     * Precompute twiddle factors for correction step in 4-step NTT
     ******************************************************************************************************************/

    tw_root_2n_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_root_2n_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_root_2n_mont_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    std::vector<BasicInteger> host_tw_root_2n(n_);
    std::vector<BasicInteger> host_tw_root_2n_shoup(n_);
    for (size_t i = 0; i < n2_; i++) {
        for (size_t j = 0; j < n1_; j++) {
            // calculate exponent
            size_t brev_j = isecfhe::ReverseBits(j, log_n1_);
            size_t exp = 2 * brev_j * i + i;

            // root_ij = root_2n ^ exp
            BasicInteger root_ij = isecfhe::util::ModExp(root_2n_, static_cast<BasicInteger>(exp), q_);
            host_tw_root_2n[i * n1_ + j] = root_ij;
            host_tw_root_2n_shoup[i * n1_ + j] = compute_shoup(root_ij, q_);
        }
    }
    cudaMemcpyAsync(tw_root_2n_.get(), host_tw_root_2n.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tw_root_2n_shoup_.get(), host_tw_root_2n_shoup.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    to_mont(tw_root_2n_mont_.get(), tw_root_2n_.get(), 256, 1, stream);

    tw_inv_root_2n_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_inv_root_2n_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_inv_root_2n_mont_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    std::vector<BasicInteger> host_tw_inv_root_2n(n_);
    std::vector<BasicInteger> host_tw_inv_root_2n_shoup(n_);
    for (size_t i = 0; i < n1_; i++) {
        size_t brev_i = isecfhe::ReverseBits(i, log_n1_);
        for (size_t j = 0; j < n2_; j++) {
            // calculate exponent
            size_t exp = 2 * brev_i * j + j;

            // inv_root_ij = inv_root_2n ^ exp
            BasicInteger inv_root_ij = isecfhe::util::ModExp(inv_root_2n_, static_cast<BasicInteger>(exp), q_);
            host_tw_inv_root_2n[i * n2_ + j] = inv_root_ij;
            host_tw_inv_root_2n_shoup[i * n2_ + j] = compute_shoup(inv_root_ij, q_);
        }
    }
    cudaMemcpyAsync(tw_inv_root_2n_.get(), host_tw_inv_root_2n.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tw_inv_root_2n_shoup_.get(), host_tw_inv_root_2n_shoup.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    to_mont(tw_inv_root_2n_mont_.get(), tw_inv_root_2n_.get(), 256, 1, stream);

    cudaStreamSynchronize(stream);
}

FourStepNTT::FourStepNTT(size_t n, BasicInteger q1, BasicInteger q2, const cudaStream_t& stream) {
    // set n, n1, n2, log_n, log_n1, log_n2
    n_ = n;
    if (n_ == 1024) {
        n1_ = 32;
        n2_ = 32;
        log_n_ = 10;
        log_n1_ = 5;
        log_n2_ = 5;
    } else if (n_ == 2048) {
        n1_ = 64;
        n2_ = 32;
        log_n_ = 11;
        log_n1_ = 6;
        log_n2_ = 5;
    } else if (n_ == 4096) {
        n1_ = 64;
        n2_ = 64;
        log_n_ = 12;
        log_n1_ = 6;
        log_n2_ = 6;
    } else {
        throw std::invalid_argument("Current 4step NTT implementation requires n to be 1024, 2048, or 4096");
    }

    // set q
    BasicInteger q = q1 * q2;
    q_ = q;
    mu_ = (isecfhe::BigInteger<BasicInteger>(1) << (8 * sizeof(BasicInteger) * 2)).DividedBy(q_).first.GetValue();

    // compute root_2n_q1 and root_2n_q2
    auto root_2n_q1 = isecfhe::RootOfUnity(2 * n_, isecfhe::BigInteger<BasicInteger>(q1)).ConvertToInt<BasicInteger>();
    auto root_2n_q2 = isecfhe::RootOfUnity(2 * n_, isecfhe::BigInteger<BasicInteger>(q2)).ConvertToInt<BasicInteger>();

    // root_2n_q = root_2n_q1 * q2 * Integer(1/Zq1(q2)) + root_2n_q2 * q1 * Integer(1/Zq2(q1))
    BasicInteger inv_q1_q2 = isecfhe::util::ModInverse(q1, q2);
    BasicInteger inv_q2_q1 = isecfhe::util::ModInverse(q2, q1);
    BasicInteger root_2n_q = isecfhe::util::ModAddFast(
        isecfhe::util::ModMul(isecfhe::util::ModMul(root_2n_q1, q2, q), inv_q2_q1, q),
        isecfhe::util::ModMul(isecfhe::util::ModMul(root_2n_q2, q1, q), inv_q1_q2, q), q);

    // set root_2n and inv_root_2n
    root_2n_ = root_2n_q;
    inv_root_2n_ = isecfhe::util::ModInverse(root_2n_, q_);

    // set montgomery constants
    const auto big_q = isecfhe::BigInteger<BasicInteger>(q_);
    const auto big_B = isecfhe::BigInteger<BasicInteger>(1) << (8 * sizeof(BasicInteger));
    const auto big_B_mod_q = big_B.Mod(big_q);
    B_mod_q_ = big_B_mod_q.ConvertToInt<BasicInteger>();
    B2_mod_q_ = (big_B_mod_q * big_B_mod_q).Mod(big_q).ConvertToInt<BasicInteger>();
    q_inv_mod_B_ = isecfhe::util::ModInverse(big_q, big_B).ConvertToInt<BasicInteger>();

    // set inv_n and inv_n_shoup
    inv_n_ = isecfhe::util::ModInverse(static_cast<BasicInteger>(n_), q_);
    inv_n_shoup_ = compute_shoup(inv_n_, q_);
    const auto big_inv_n_R2_mod_q = isecfhe::BigInteger<BasicInteger>(inv_n_) * isecfhe::BigInteger<
                                        BasicInteger>(B2_mod_q_);
    const auto t = (big_inv_n_R2_mod_q * isecfhe::BigInteger<BasicInteger>(q_inv_mod_B_)).getValue()[0];
    inv_n_mont_ = (big_inv_n_R2_mod_q >> (8 * sizeof(BasicInteger))).getValue()[0] + q_ -
                  ((isecfhe::BigInteger<BasicInteger>(t) * big_q) >> (8 * sizeof(BasicInteger))).getValue()[0];

    /*******************************************************************************************************************
     * Precompute twiddle factors for negacyclic convolution in 4-step NTT
     ******************************************************************************************************************/

    // root_2n1 = root_2n ^ n2
    BasicInteger root_2n1 = isecfhe::util::ModExp(root_2n_, static_cast<BasicInteger>(n2_), q_);
    BasicInteger inv_root_2n1 = isecfhe::util::ModInverse(root_2n1, q_);

    tw_root_2n1_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    tw_root_2n1_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    tw_root_2n1_mont_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    gen_tw_table(tw_root_2n1_.get(), tw_root_2n1_shoup_.get(), root_2n1,
                 n1_, log_n1_, q_, stream);
    to_mont(tw_root_2n1_mont_.get(), tw_root_2n1_.get(), 256, 1, stream);

    tw_inv_root_2n1_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    tw_inv_root_2n1_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    tw_inv_root_2n1_mont_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    gen_tw_table(tw_inv_root_2n1_.get(), tw_inv_root_2n1_shoup_.get(), inv_root_2n1,
                 n1_, log_n1_, q_, stream);
    to_mont(tw_inv_root_2n1_mont_.get(), tw_inv_root_2n1_.get(), 256, 1, stream);

    /*******************************************************************************************************************
     * Precompute twiddle factors for cyclic convolution in 4-step NTT
     ******************************************************************************************************************/

    // root_n2 = root_2n ^ (2 * n1)
    BasicInteger root_n2 = isecfhe::util::ModExp(root_2n_, static_cast<BasicInteger>(2 * n1_), q_);
    BasicInteger inv_root_n2 = isecfhe::util::ModInverse(root_n2, q_);

    tw_root_n2_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    tw_root_n2_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    tw_root_n2_mont_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    gen_tw_table(tw_root_n2_.get(), tw_root_n2_shoup_.get(), root_n2,
                 n2_ / 2, log_n2_ - 1, q_, stream);
    to_mont(tw_root_n2_mont_.get(), tw_root_n2_.get(), 256, 1, stream);

    tw_inv_root_n2_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    tw_inv_root_n2_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    tw_inv_root_n2_mont_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    gen_tw_table(tw_inv_root_n2_.get(), tw_inv_root_n2_shoup_.get(), inv_root_n2,
                 n2_ / 2, log_n2_ - 1, q_, stream);
    to_mont(tw_inv_root_n2_mont_.get(), tw_inv_root_n2_.get(), 256, 1, stream);

    /*******************************************************************************************************************
     * Precompute twiddle factors for correction step in 4-step NTT
     ******************************************************************************************************************/

    tw_root_2n_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_root_2n_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_root_2n_mont_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    std::vector<BasicInteger> host_tw_root_2n(n_);
    std::vector<BasicInteger> host_tw_root_2n_shoup(n_);
    for (size_t i = 0; i < n2_; i++) {
        for (size_t j = 0; j < n1_; j++) {
            // calculate exponent
            size_t brev_j = isecfhe::ReverseBits(j, log_n1_);
            size_t exp = 2 * brev_j * i + i;

            // root_ij = root_2n ^ exp
            BasicInteger root_ij = isecfhe::util::ModExp(root_2n_, static_cast<BasicInteger>(exp), q_);
            host_tw_root_2n[i * n1_ + j] = root_ij;
            host_tw_root_2n_shoup[i * n1_ + j] = compute_shoup(root_ij, q_);
        }
    }
    cudaMemcpyAsync(tw_root_2n_.get(), host_tw_root_2n.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tw_root_2n_shoup_.get(), host_tw_root_2n_shoup.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    to_mont(tw_root_2n_mont_.get(), tw_root_2n_.get(), 256, 1, stream);

    tw_inv_root_2n_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_inv_root_2n_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_inv_root_2n_mont_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    std::vector<BasicInteger> host_tw_inv_root_2n(n_);
    std::vector<BasicInteger> host_tw_inv_root_2n_shoup(n_);
    for (size_t i = 0; i < n1_; i++) {
        size_t brev_i = isecfhe::ReverseBits(i, log_n1_);
        for (size_t j = 0; j < n2_; j++) {
            // calculate exponent
            size_t exp = 2 * brev_i * j + j;

            // inv_root_ij = inv_root_2n ^ exp
            BasicInteger inv_root_ij = isecfhe::util::ModExp(inv_root_2n_, static_cast<BasicInteger>(exp), q_);
            host_tw_inv_root_2n[i * n2_ + j] = inv_root_ij;
            host_tw_inv_root_2n_shoup[i * n2_ + j] = compute_shoup(inv_root_ij, q_);
        }
    }
    cudaMemcpyAsync(tw_inv_root_2n_.get(), host_tw_inv_root_2n.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tw_inv_root_2n_shoup_.get(), host_tw_inv_root_2n_shoup.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    to_mont(tw_inv_root_2n_mont_.get(), tw_inv_root_2n_.get(), 256, 1, stream);

    cudaStreamSynchronize(stream);
}
