#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_wrapper.cuh"

namespace phantom::cirbts::experimental {

// Experimental 32-bit negacyclic NTT plan for an RNS base.
//
// This is intentionally isolated from the production CBS path. The current CBS
// modulus is a 54-bit prime, so a 32-bit backend must either use RNS or a limb
// representation. This class provides the first building block: exact NTTs over
// 32-bit NTT primes q_i = 1 mod 2N.
class NTT32RNSPlan {
public:
    NTT32RNSPlan(std::size_t n, const std::vector<std::uint32_t>& moduli, cudaStream_t stream);

    [[nodiscard]] std::size_t n() const {
        return n_;
    }

    [[nodiscard]] std::size_t limb_count() const {
        return primes_.size();
    }

    [[nodiscard]] std::uint32_t modulus(std::size_t limb) const {
        return primes_.at(limb).q;
    }

    void forward_limb(std::uint32_t* out, const std::uint32_t* in, std::size_t batch, std::size_t limb,
                      cudaStream_t stream) const;

    void inverse_limb(std::uint32_t* out, const std::uint32_t* in, std::size_t batch, std::size_t limb,
                      cudaStream_t stream) const;

    // Convert coefficient-domain polynomials modulo the production 64-bit Q into
    // per-limb 32-bit negacyclic NTT form. Layout is [limb][poly][N].
    void native_coeffs_to_rns_eval(std::uint32_t* out_eval, std::uint32_t* out_eval_shoup,
                                   const std::uint64_t* in_coeffs_mod_q, std::size_t poly_count,
                                   cudaStream_t stream) const;

    [[nodiscard]] bool roundtrip_self_test(std::size_t batch, cudaStream_t stream) const;

    // Verify the RNS32 eval-domain external product primitive for one
    // RGSW-style row layout: dct[digits][N] times key[digits][2][N].
    [[nodiscard]] bool external_product_self_test(std::size_t digits, cudaStream_t stream) const;

    [[nodiscard]] float benchmark_roundtrip_ms(std::size_t batch, std::size_t repeat, cudaStream_t stream) const;

    // Measures only the eval-domain multiply-accumulate kernel, not the forward
    // or inverse NTTs around it.
    [[nodiscard]] float benchmark_external_product_eval_ms(std::size_t digits, std::size_t repeat,
                                                           cudaStream_t stream) const;

    [[nodiscard]] static std::vector<std::uint32_t> default_moduli();

private:
    struct PrimeData {
        std::uint32_t q{};
        std::uint32_t psi{};
        std::uint32_t omega{};
        std::uint32_t inv_omega{};
        std::uint32_t inv_n{};
        std::uint32_t inv_n_shoup{};
        phantom::util::cuda_auto_ptr<std::uint32_t> d_omega_pows;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_omega_pows_shoup;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_inv_omega_pows;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_inv_omega_pows_shoup;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_psi_pows;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_psi_pows_shoup;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_inv_psi_inv_n_pows;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_inv_psi_inv_n_pows_shoup;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_tw_root_2n1;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_tw_root_2n1_shoup;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_tw_inv_root_2n1;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_tw_inv_root_2n1_shoup;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_tw_root_2n;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_tw_root_2n_shoup;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_tw_inv_root_2n;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_tw_inv_root_2n_shoup;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_tw_root_n2;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_tw_root_n2_shoup;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_tw_inv_root_n2;
        phantom::util::cuda_auto_ptr<std::uint32_t> d_tw_inv_root_n2_shoup;
    };

    std::size_t n_{};
    std::uint32_t log_n_{};
    std::vector<PrimeData> primes_;
};

} // namespace phantom::cirbts::experimental
