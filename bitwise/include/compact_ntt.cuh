#pragma once

#include <cuda_runtime.h>

#include "openfhe.h"

#include "cuda_wrapper.cuh"

class FourStepNTT {
    BasicInteger q_{};
    std::vector<BasicInteger> mu_{};
    BasicInteger root_2n_{};
    BasicInteger inv_root_2n_{};
    BasicInteger inv_n_{};
    BasicInteger inv_n_shoup_{};
    BasicInteger inv_n_mont_{};
    size_t n_{};
    size_t n1_{};
    size_t n2_{};
    size_t log_n_{};
    size_t log_n1_{};
    size_t log_n2_{};

    // R = 2 ^ 32 or 2 ^ 64
    BasicInteger B_mod_q_{};
    BasicInteger B2_mod_q_{};
    BasicInteger q_inv_mod_B_{};

    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_2n1_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_2n1_shoup_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_2n1_mont_;

    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_2n1_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_2n1_shoup_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_2n1_mont_;

    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_2n_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_2n_shoup_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_2n_mont_;

    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_2n_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_2n_shoup_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_2n_mont_;

    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_n2_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_n2_shoup_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_n2_mont_;

    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_n2_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_n2_shoup_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_n2_mont_;

    static void gen_tw_table(BasicInteger* tw_root, BasicInteger* tw_root_shoup, BasicInteger root,
                             size_t len, size_t log_len, BasicInteger mod, const cudaStream_t& stream);

public:
    /**
     * Constructor for 4step NTT
     * @param n ring dimension
     * @param mod modulus
     * @param stream CUDA stream
     */
    FourStepNTT(size_t n, BasicInteger mod, const cudaStream_t& stream);

    /**
     * Constructor for composite 4step NTT
     * @param n ring dimension
     * @param q1 modulus 1
     * @param q2 modulus 2
     * @param stream CUDA stream
     */
    FourStepNTT(size_t n, BasicInteger q1, BasicInteger q2, const cudaStream_t& stream);

    void forward_shoup(BasicInteger* output, const BasicInteger* input, size_t threadsPerBlock, size_t n_poly,
                       const cudaStream_t& stream) const;

    void forward_mont(BasicInteger* output, const BasicInteger* input, size_t threadsPerBlock, size_t n_poly,
                 const cudaStream_t& stream);

    void inverse_shoup(BasicInteger* output, const BasicInteger* input, size_t threadsPerBlock, size_t n_poly,
                       const cudaStream_t& stream) const;

    void inverse_mont(BasicInteger* output, const BasicInteger* input, size_t threadsPerBlock, size_t n_poly,
                 const cudaStream_t& stream);

    void multiply(BasicInteger* output, const BasicInteger* input1, const BasicInteger* input2,
                  const cudaStream_t& stream);

    void multiply_and_accumulate(BasicInteger* output, const BasicInteger* input1, const BasicInteger* input2,
                                 const cudaStream_t& stream) const;

    void multiply_scalar(BasicInteger* output, const BasicInteger* input, BasicInteger scalar,
                         const cudaStream_t& stream);

    void add(BasicInteger* output, const BasicInteger* input1, const BasicInteger* input2, const cudaStream_t& stream);

    void to_mont(BasicInteger* output, const BasicInteger* input, size_t threadsPerBlock, size_t n_poly,
                 const cudaStream_t& stream) const;

    void from_mont(BasicInteger* output, const BasicInteger* input, size_t threadsPerBlock, size_t n_poly,
                   const cudaStream_t& stream) const;

    [[nodiscard]] auto getMod() const -> BasicInteger {
        return q_;
    }

    [[nodiscard]] auto getMu() const -> const std::vector<BasicInteger>& {
        return mu_;
    }

    [[nodiscard]] auto getMont() const -> BasicInteger {
        return q_inv_mod_B_;
    }

    [[nodiscard]] auto getRoot2n() const -> BasicInteger {
        return root_2n_;
    }

    [[nodiscard]] auto getInvRoot2n() const -> BasicInteger {
        return inv_root_2n_;
    }

    [[nodiscard]] auto getInvn() const -> BasicInteger {
        return inv_n_;
    }

    [[nodiscard]] auto getInvnShoup() const -> BasicInteger {
        return inv_n_shoup_;
    }

    [[nodiscard]] auto getInvnMont() const -> BasicInteger {
        return inv_n_mont_;
    }

    [[nodiscard]] auto getTwRoot2n1() const -> const BasicInteger* {
        return tw_root_2n1_.get();
    }

    [[nodiscard]] auto getTwRoot2n1Shoup() const -> const BasicInteger* {
        return tw_root_2n1_shoup_.get();
    }

    [[nodiscard]] auto getTwRoot2n1Mont() const -> const BasicInteger* {
        return tw_root_2n1_mont_.get();
    }

    [[nodiscard]] auto getTwInvRoot2n1() const -> const BasicInteger* {
        return tw_inv_root_2n1_.get();
    }

    [[nodiscard]] auto getTwInvRoot2n1Shoup() const -> const BasicInteger* {
        return tw_inv_root_2n1_shoup_.get();
    }

    [[nodiscard]] auto getTwInvRoot2n1Mont() const -> const BasicInteger* {
        return tw_inv_root_2n1_mont_.get();
    }

    [[nodiscard]] auto getTwRoot2n() const -> const BasicInteger* {
        return tw_root_2n_.get();
    }

    [[nodiscard]] auto getTwRoot2nShoup() const -> const BasicInteger* {
        return tw_root_2n_shoup_.get();
    }

    [[nodiscard]] auto getTwRoot2nMont() const -> const BasicInteger* {
        return tw_root_2n_mont_.get();
    }

    [[nodiscard]] auto getTwInvRoot2n() const -> const BasicInteger* {
        return tw_inv_root_2n_.get();
    }

    [[nodiscard]] auto getTwInvRoot2nShoup() const -> const BasicInteger* {
        return tw_inv_root_2n_shoup_.get();
    }

    [[nodiscard]] auto getTwInvRoot2nMont() const -> const BasicInteger* {
        return tw_inv_root_2n_mont_.get();
    }

    [[nodiscard]] auto getTwRootn2() const -> const BasicInteger* {
        return tw_root_n2_.get();
    }

    [[nodiscard]] auto getTwRootn2Shoup() const -> const BasicInteger* {
        return tw_root_n2_shoup_.get();
    }

    [[nodiscard]] auto getTwRootn2Mont() const -> const BasicInteger* {
        return tw_root_n2_mont_.get();
    }

    [[nodiscard]] auto getTwInvRootn2() const -> const BasicInteger* {
        return tw_inv_root_n2_.get();
    }

    [[nodiscard]] auto getTwInvRootn2Shoup() const -> const BasicInteger* {
        return tw_inv_root_n2_shoup_.get();
    }

    [[nodiscard]] auto getTwInvRootn2Mont() const -> const BasicInteger* {
        return tw_inv_root_n2_mont_.get();
    }
};
