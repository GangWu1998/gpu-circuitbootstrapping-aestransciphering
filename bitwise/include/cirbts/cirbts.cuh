/*
 * =============================================================================
 * File: cirbts.cuh
 * Purpose: Shared declarations for the GPU CirBTS context, scratch workspaces,
 *          and internal helper interfaces used across cirbts-*.cu files.
 * Key parameters:
 *   - max_batch_size_: sizes per-batch scratch and limits batched APIs.
 *   - stream/copy_stream: compute vs. async H2D pipelines.
 *   - *_graph_* fields: cached CUDA graphs for launch overhead reduction.
 * Key points:
 *   - Central contract for GPUCirBTSContext and related scratch structs.
 *   - Owns pinned host buffers and H2D events for pipeline staging.
 * =============================================================================
 */
#pragma once

#include "openfhe.h"
#include "phantom.h"
#include "rlwe-ske.h"
#include "cirbtscontext.h"
#include "cirbts/cirbts-fft.cuh"
#include "cirbts/cirbts-backend.cuh"
#include "cirbts/ntt32_rns.cuh"

#include <cstdint>
#include <memory>
#include <vector>

using namespace lbcrypto;

class GPUCirBTSContext : public lbcrypto::CirBTSContext, public lbcrypto::CirBTSScheme {
    using NativeInt = BasicInteger;

private:
    struct BatchScratchView {
        cudaStream_t stream{};
        cudaStream_t copy_stream{};

        NativeInt* d_monomial_inv{};
        NativeInt* d_bootstrap_acc{};
        NativeInt* d_evalacc_ct{};
        NativeInt* d_evalacc_dct{};
        uint32_t*  d_evalacc_indexPos{};
        uint32_t*  d_evalacc_b_idx{};

        NativeInt* h_monomial_inv_pinned{};
        uint32_t*  h_indexPos_pinned{};
        cudaEvent_t* h2d_monomial_event{};
        cudaEvent_t* h2d_index_event{};

        NativeInt* d_ht_c0{};
        NativeInt* d_ht_c1{};
        NativeInt* d_perm_c0{};
        NativeInt* d_ht_next_c0{};
        NativeInt* d_ht_next_c1{};
        NativeInt* d_ks_c0{};
        NativeInt* d_ks_c1{};
        NativeInt* d_digits{};
        NativeInt* d_ss_c0{};
        NativeInt* d_ss_c1{};
        NativeInt* d_rgsw_out{};

        size_t   scratch_lut_size{};
        uint32_t scratch_digits_max{};

        cudaGraphExec_t* evalacc_batch_graph_exec{};
        bool*            evalacc_batch_graph_failed{};
        uint32_t*        evalacc_batch_graph_ntt_tpb{};
        uint32_t*        evalacc_batch_graph_block_x{};
        uint32_t*        evalacc_batch_graph_batch{};
        uint32_t*        evalacc_batch_graph_multibit{};
        bool*            evalacc_batch_graph_soa{};
        bool*            evalacc_batch_graph_smem{};

        cudaGraphExec_t* htss_graph_exec{};
        bool*            htss_graph_failed{};
        uint32_t*        htss_graph_block_x{};
        uint32_t*        htss_graph_num_luts{};
        bool*            htss_graph_fuse_inwt_decomp{};
        bool*            htss_graph_fuse_decomp_fnwt{};
        bool*            htss_graph_pipeline{};
        bool*            htss_graph_smem{};
    };

    struct BatchWorkspace {
        phantom::util::cuda_stream_wrapper stream;
        phantom::util::cuda_stream_wrapper copy_stream;

        phantom::util::cuda_auto_ptr<NativeInt> d_monomial_inv;
        phantom::util::cuda_auto_ptr<NativeInt> d_bootstrap_acc;
        phantom::util::cuda_auto_ptr<NativeInt> d_evalacc_ct;
        phantom::util::cuda_auto_ptr<NativeInt> d_evalacc_dct;
        phantom::util::cuda_auto_ptr<uint32_t>  d_evalacc_indexPos;
        phantom::util::cuda_auto_ptr<uint32_t>  d_evalacc_b_idx;

        phantom::util::cuda_auto_ptr<NativeInt> d_ht_c0;
        phantom::util::cuda_auto_ptr<NativeInt> d_ht_c1;
        phantom::util::cuda_auto_ptr<NativeInt> d_perm_c0;
        phantom::util::cuda_auto_ptr<NativeInt> d_ht_next_c0;
        phantom::util::cuda_auto_ptr<NativeInt> d_ht_next_c1;
        phantom::util::cuda_auto_ptr<NativeInt> d_ks_c0;
        phantom::util::cuda_auto_ptr<NativeInt> d_ks_c1;
        phantom::util::cuda_auto_ptr<NativeInt> d_digits;
        phantom::util::cuda_auto_ptr<NativeInt> d_ss_c0;
        phantom::util::cuda_auto_ptr<NativeInt> d_ss_c1;
        phantom::util::cuda_auto_ptr<NativeInt> d_rgsw_out;

        size_t   scratch_lut_size{};
        uint32_t scratch_digits_max{};
        uint32_t max_batch_size{};

        NativeInt* h_monomial_inv_pinned{};
        size_t     h_monomial_inv_pinned_capacity{};
        uint32_t*  h_indexPos_pinned{};
        size_t     h_indexPos_pinned_capacity{};
        cudaEvent_t h2d_monomial_event{};
        cudaEvent_t h2d_index_event{};

        NativeInt* h_rgsw_pinned{};
        size_t     h_rgsw_pinned_capacity{};

        BatchWorkspace() {
            PHANTOM_CHECK_CUDA(cudaEventCreateWithFlags(&h2d_monomial_event, cudaEventDisableTiming));
            PHANTOM_CHECK_CUDA(cudaEventCreateWithFlags(&h2d_index_event, cudaEventDisableTiming));
        }
        BatchWorkspace(const BatchWorkspace&) = delete;
        BatchWorkspace& operator=(const BatchWorkspace&) = delete;

        ~BatchWorkspace() {
            if (h_monomial_inv_pinned) {
                cudaFreeHost(h_monomial_inv_pinned);
                h_monomial_inv_pinned = nullptr;
            }
            if (h_indexPos_pinned) {
                cudaFreeHost(h_indexPos_pinned);
                h_indexPos_pinned = nullptr;
            }
            if (h_rgsw_pinned) {
                cudaFreeHost(h_rgsw_pinned);
                h_rgsw_pinned = nullptr;
            }
            if (h2d_monomial_event) {
                cudaEventDestroy(h2d_monomial_event);
                h2d_monomial_event = nullptr;
            }
            if (h2d_index_event) {
                cudaEventDestroy(h2d_index_event);
                h2d_index_event = nullptr;
            }
        }
    };

    phantom::util::cuda_stream_wrapper stream_wrapper_;
    phantom::util::cuda_stream_wrapper copy_stream_wrapper_;

    // NativeInt *d_RFkey_{};
    phantom::util::cuda_auto_ptr<NativeInt> d_RFkey_;
    phantom::util::cuda_auto_ptr<NativeInt> d_RFkey_shoup_;
    size_t size_RFkey_{};
    size_t size_RFRGSWkey_{};
    uint32_t rfkey_dim2_{1};
    size_t rfkey_groups_{};
    bool rfkey_grouped_{false};
    phantom::util::cuda_auto_ptr<NativeInt> d_RFkey_soa_;
    phantom::util::cuda_auto_ptr<NativeInt> d_RFkey_shoup_soa_;
    size_t rfkey_soa_tiles_{};

    // Experimental 32-bit RNS NTT layout. These buffers are built only when
    // CIRBTS_BACKEND=ntt32_rns and are not consumed by the production kernels yet.
    bool use_ntt32_rns_{false};
    uint32_t ntt32_rns_limbs_{0};
    std::unique_ptr<phantom::cirbts::experimental::NTT32RNSPlan> ntt32_rns_plan_;
    phantom::util::cuda_auto_ptr<uint32_t> d_RFkey_rns32_;
    phantom::util::cuda_auto_ptr<uint32_t> d_RFkey_rns32_shoup_;
    phantom::util::cuda_auto_ptr<uint32_t> d_HTkey_rns32_;
    phantom::util::cuda_auto_ptr<uint32_t> d_HTkey_rns32_shoup_;
    phantom::util::cuda_auto_ptr<uint32_t> d_SSkey_rns32_;
    phantom::util::cuda_auto_ptr<uint32_t> d_SSkey_rns32_shoup_;
    phantom::util::cuda_auto_ptr<uint32_t> d_monic_polys_rns32_;
    phantom::util::cuda_auto_ptr<uint32_t> d_monic_polys_rns32_shoup_;
    size_t rfkey_rns32_polys_{};
    size_t htkey_rns32_polys_{};
    size_t sskey_rns32_polys_{};
    size_t monic_rns32_polys_{};

    // LMKCDEY refresh key layout:
    // - eval keys:   [n][digitsG2][2][N]
    // - auto keys:   [numAutoKeys+1][digitsG][2][N]  (index 0 is k = 2N-gen, index i is k = gen^i)
    phantom::util::cuda_auto_ptr<NativeInt> d_LMK_evalkey_;
    phantom::util::cuda_auto_ptr<NativeInt> d_LMK_autokey_;
    phantom::util::cuda_auto_ptr<uint32_t>  d_LMK_autoMaps_;
    uint32_t                                lmk_numAutoKeys_{};
    uint32_t                                lmk_auto_key_count_{};
    uint32_t                                lmk_auto_window_{};
    phantom::util::cuda_auto_ptr<NativeInt> d_lmk_ct_;
    phantom::util::cuda_auto_ptr<NativeInt> d_lmk_dct_;
    phantom::util::cuda_auto_ptr<NativeInt> d_lmk_perm_c0_;
    phantom::util::cuda_auto_ptr<NativeInt> d_lmk_perm_c1_;
    phantom::util::cuda_auto_ptr<NativeInt> d_lmk_dcta_;
    phantom::util::cuda_auto_ptr<NativeInt> d_lmk_ks_c0_;
    phantom::util::cuda_auto_ptr<NativeInt> d_lmk_ks_c1_;
    uint32_t                                lmk_digitsG_auto_{};
    uint32_t                                lmk_digitsG2_eval_{};
    uint32_t                                lmk_N_{};

    // NativeInt *d_HTkey_{};
    phantom::util::cuda_auto_ptr<NativeInt> d_HTkey_;
    phantom::util::cuda_auto_ptr<NativeInt> d_HTkey_shoup_;
    size_t size_HTkey_{};
    size_t size_HTRGSWkey_{};
    phantom::util::cuda_auto_ptr<NativeInt> d_HTkey_soa_;
    phantom::util::cuda_auto_ptr<NativeInt> d_HTkey_shoup_soa_;
    size_t htkey_soa_tiles_{};

    // NativeInt *d_SSkey_{};
    phantom::util::cuda_auto_ptr<NativeInt> d_SSkey_;
    phantom::util::cuda_auto_ptr<NativeInt> d_SSkey_shoup_;
    size_t size_SSkey_{};
    size_t size_SSRGSWkey_{};
    phantom::util::cuda_auto_ptr<NativeInt> d_SSkey_soa_;
    phantom::util::cuda_auto_ptr<NativeInt> d_SSkey_shoup_soa_;
    size_t sskey_soa_tiles_{};

    // Optional L2 persistence window configuration (applied per-phase via stream attributes).
    size_t l2_persist_bytes_{};
    float l2_persist_hit_ratio_{0.6f};
    bool l2_persist_enabled_{false};
    mutable bool l2_persist_failed_{false};

    // Cached constants / tables to avoid per-call H2D copies and kernel rebuilds.
    phantom::util::cuda_auto_ptr<NativeInt> d_monic_polys_;          // [2N][N] (GINX EvalAcc monic table)
    phantom::util::cuda_auto_ptr<NativeInt> d_monomial_inv_table_;   // [2N][N] CirBTS X^{-i} bootstrap monomials
    phantom::util::cuda_auto_ptr<NativeInt> d_monomials_;            // [numLUT-1][N] used in MV-RLWE generation
    phantom::util::cuda_auto_ptr<NativeInt> d_gpow_;         // [numLUT] used in ModAddGpowScaled
    phantom::util::cuda_auto_ptr<uint32_t>  d_auto_maps_;    // [logN][N] coefficient-domain automorphism maps

    size_t   ring_dim_{};
    uint32_t log_ring_dim_{};
    uint32_t num_luts_{};
    size_t   ntt_batch_size_{};
    uint32_t lwe_n_{};
    uint32_t digits_ga_{};
    uint32_t digits_ht_{};
    uint32_t digits_ss_{};
    BINFHE_METHOD method_{BINFHE_METHOD::GINX};
    phantom::bitwise::CirBTSBackend backend_{phantom::bitwise::CirBTSBackend::kNTT};

    phantom::util::cuda_auto_ptr<DModulus>  d_modulus_;
    phantom::util::cuda_auto_ptr<NativeInt> d_twiddles_;
    phantom::util::cuda_auto_ptr<NativeInt> d_twiddles_shoup_;
    phantom::util::cuda_auto_ptr<NativeInt> d_itwiddles_;
    phantom::util::cuda_auto_ptr<NativeInt> d_itwiddles_shoup_;
    phantom::util::cuda_auto_ptr<NativeInt> d_n_inv_mod_q_;
    phantom::util::cuda_auto_ptr<NativeInt> d_n_inv_mod_q_shoup_;

    // Cached LUT for circuit bootstrapping (evaluation form), size [N].
    phantom::util::cuda_auto_ptr<NativeInt> d_lut_;

    // Scratch buffer for X^{-b_ms} (used in BootstrapManyLUT); reused across calls.
    phantom::util::cuda_auto_ptr<NativeInt> d_monomial_inv_;

    // Scratch buffer for BootstrapManyLUT accumulator (c0||c1), size [2N].
    phantom::util::cuda_auto_ptr<NativeInt> d_bootstrap_acc_;

    // Scratch buffers for CGGI EvalAcc (GINX) to avoid per-call allocations.
    phantom::util::cuda_auto_ptr<NativeInt> d_evalacc_ct_;        // [2N]
    phantom::util::cuda_auto_ptr<NativeInt> d_evalacc_dct_;       // [digitsG2][N]
    phantom::util::cuda_auto_ptr<uint32_t>  d_evalacc_indexPos_;  // [n]
    phantom::util::cuda_auto_ptr<uint32_t>  d_evalacc_b_idx_;     // [batch]
    // Experimental digit+residual EvalAcc path (GINX, K-block).
    phantom::util::cuda_auto_ptr<int64_t>   d_evalacc_digits_s_;   // [digitsG2][N] signed digits
    phantom::util::cuda_auto_ptr<int64_t>   d_evalacc_residual_s_; // [2N] signed residual
    phantom::util::cuda_auto_ptr<NativeInt> d_evalacc_delta_;      // [2N] per-step delta (eval domain)
    phantom::util::cuda_auto_ptr<uint32_t>  d_evalacc_carry_flag_; // [1] overflow flag
    phantom::util::cuda_auto_ptr<NativeInt> d_evalacc_acc_backup_; // [2N] block-start accumulator backup
    bool   use_swizzle_{false};
    size_t rfkey_swizzle_tiles_{0};
    phantom::util::cuda_auto_ptr<NativeInt> d_RFkey_swizzle_;
    phantom::util::cuda_auto_ptr<NativeInt> d_RFkey_shoup_swizzle_;

    // Optional split-FFT EvalAcc (GINX) state.
    bool use_split_fft_{false};
    uint32_t split_fft_bits_{0};
    uint32_t split_fft_limbs_{0};
    size_t split_fft_fft_len_{0};
    size_t split_fft_key_polys_{0};
    size_t split_fft_key_stride_{0};
    std::vector<NativeInt> split_fft_base_pows_;
    std::vector<NativeInt> split_fft_base_pows_shoup_;
    std::vector<phantom::util::cuda_auto_ptr<SplitFFTComplex>> d_RFkey_fft_limbs_;
    phantom::util::cuda_auto_ptr<SplitFFTComplex> d_monic_fft_;
    uint32_t split_fft_fpb_{1};
    std::shared_ptr<phantom::bitwise::cuFFTDxWrapperCirBTS<double, 2048, 1, 16, PHANTOM_CUFFTDX_SM>> fft_2048_fpb1_;
    std::shared_ptr<phantom::bitwise::cuFFTDxWrapperCirBTS<double, 2048, 2, 16, PHANTOM_CUFFTDX_SM>> fft_2048_fpb2_;
    std::shared_ptr<phantom::bitwise::cuFFTDxWrapperCirBTS<double, 2048, 4, 16, PHANTOM_CUFFTDX_SM>> fft_2048_fpb4_;
    phantom::util::cuda_auto_ptr<int64_t> d_evalacc_dct_i64_;
    phantom::util::cuda_auto_ptr<SplitFFTComplex> d_evalacc_dct_fft_;
    phantom::util::cuda_auto_ptr<SplitFFTComplex> d_evalacc_acc_fft_;

    // Maximum number of independent ciphertexts supported by batched APIs (scratch is sized accordingly).
    uint32_t max_batch_size_{1};

    // Cached block size for batched EvalAcc core (auto-tuned once per batch size unless overridden).
    mutable uint32_t evalacc_batch_block_x_{0};
    mutable uint32_t evalacc_batch_block_batch_{0};
    mutable uint32_t evalacc_ntt_tpb_{0};
    mutable uint32_t evalacc_block_x_{0};
    mutable bool evalacc_autotuned_{false};

    // Optional CUDA graph for the CGGI EvalAcc loop (reduces ~O(n) kernel launch overhead).
    mutable cudaGraphExec_t cggi_evalacc_graph_exec_{};
    mutable bool            cggi_evalacc_graph_failed_{false};
    mutable NativeInt*      cggi_evalacc_graph_acc_{nullptr};
    mutable uint32_t        cggi_evalacc_graph_ntt_tpb_{0};
    mutable uint32_t        cggi_evalacc_graph_block_x_{0};
    mutable bool            cggi_evalacc_graph_swizzle_{false};
    mutable bool            cggi_evalacc_graph_soa_{false};
    mutable bool            cggi_evalacc_graph_smem_{false};
    mutable cudaGraphExec_t cggi_evalacc_mb2_graph_exec_{};
    mutable bool            cggi_evalacc_mb2_graph_failed_{false};
    mutable NativeInt*      cggi_evalacc_mb2_graph_acc_{nullptr};
    mutable uint32_t        cggi_evalacc_mb2_graph_ntt_tpb_{0};
    mutable uint32_t        cggi_evalacc_mb2_graph_block_x_{0};
    mutable bool            cggi_evalacc_mb2_graph_soa_{false};
    mutable bool            cggi_evalacc_mb2_graph_smem_{false};
    mutable cudaGraphExec_t cggi_evalacc_pbsmb2_graph_exec_{};
    mutable bool            cggi_evalacc_pbsmb2_graph_failed_{false};
    mutable NativeInt*      cggi_evalacc_pbsmb2_graph_acc_{nullptr};
    mutable uint32_t        cggi_evalacc_pbsmb2_graph_ntt_tpb_{0};
    mutable uint32_t        cggi_evalacc_pbsmb2_graph_block_x_{0};
    mutable bool            cggi_evalacc_pbsmb2_graph_swizzle_{false};
    mutable bool            cggi_evalacc_pbsmb2_graph_soa_{false};
    mutable bool            cggi_evalacc_pbsmb2_graph_smem_{false};

    // Optional CUDA graph for split-FFT EvalAcc (reduces launch overhead in FFT path).
    mutable cudaGraphExec_t split_fft_evalacc_graph_exec_{};
    mutable bool            split_fft_evalacc_graph_failed_{false};
    mutable NativeInt*      split_fft_evalacc_graph_acc_{nullptr};
    mutable uint32_t        split_fft_evalacc_graph_limbs_{0};

    // Optional CUDA graph for batched CGGI EvalAcc (reduces ~O(n) kernel launch overhead).
    mutable cudaGraphExec_t cggi_evalacc_batch_graph_exec_{};
    mutable bool            cggi_evalacc_batch_graph_failed_{false};
    mutable uint32_t        cggi_evalacc_batch_graph_ntt_tpb_{0};
    mutable uint32_t        cggi_evalacc_batch_graph_block_x_{0};
    mutable uint32_t        cggi_evalacc_batch_graph_batch_{0};
    mutable uint32_t        cggi_evalacc_batch_graph_multibit_{0};
    mutable bool            cggi_evalacc_batch_graph_soa_{false};
    mutable bool            cggi_evalacc_batch_graph_smem_{false};

    // Optional CUDA graph for HomTrace+SchemeSwitch (reduces kernel launch overhead in the HT/SS stage).
    mutable cudaGraphExec_t htss_graph_exec_{};
    mutable bool            htss_graph_failed_{false};
    mutable uint32_t        htss_graph_block_x_{0};
    mutable uint32_t        htss_graph_num_luts_{0};
    mutable bool            htss_graph_fuse_inwt_decomp_{false};
    mutable bool            htss_graph_fuse_decomp_fnwt_{false};
    mutable bool            htss_graph_pipeline_{false};
    mutable bool            htss_graph_smem_{false};
    mutable bool            htss_graph_save_fusion_{false};

    // Scratch buffers for GINX circuit bootstrapping (reused across calls to avoid cudaMallocAsync/free overhead).
    phantom::util::cuda_auto_ptr<NativeInt> d_ht_c0_;
    phantom::util::cuda_auto_ptr<NativeInt> d_ht_c1_;
    phantom::util::cuda_auto_ptr<NativeInt> d_perm_c0_;  // permute(c0)+INTT output (coefficient domain), [numLUT][N]
    phantom::util::cuda_auto_ptr<NativeInt> d_ht_next_c0_;
    phantom::util::cuda_auto_ptr<NativeInt> d_ht_next_c1_;
    phantom::util::cuda_auto_ptr<NativeInt> d_ks_c0_;
    phantom::util::cuda_auto_ptr<NativeInt> d_ks_c1_;
    phantom::util::cuda_auto_ptr<NativeInt> d_digits_;    // [max(digitsHT,digitsSS)][numLUT][N]
    phantom::util::cuda_auto_ptr<NativeInt> d_ss_c0_;
    phantom::util::cuda_auto_ptr<NativeInt> d_ss_c1_;
    phantom::util::cuda_auto_ptr<NativeInt> d_rgsw_out_;  // [numLUT][4][N]
    size_t                                  scratch_lut_size_{};
    uint32_t                                scratch_digits_max_{};

    mutable NativeInt* h_rgsw_pinned_{};
    mutable size_t     h_rgsw_pinned_capacity_{};
    mutable NativeInt* h_monomial_inv_pinned_{};
    mutable size_t     h_monomial_inv_pinned_capacity_{};
    mutable uint32_t*  h_indexPos_pinned_{};
    mutable size_t     h_indexPos_pinned_capacity_{};
    mutable cudaEvent_t h2d_monomial_event_{};
    mutable cudaEvent_t h2d_index_event_{};

    mutable std::vector<std::unique_ptr<BatchWorkspace>> extra_workspaces_;

    void gpu_init_cirbsk(CirBTSContext& cc);
    void gpu_init_ntt(CirBTSContext& cc);
    void gpu_BootstrapLUT_inplace(const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params, ConstRingGSWACCKey& ek,
                                  lbcrypto::ConstLWECiphertext& ct, uint64_t bitwidth) const;
    void gpu_BootstrapLUT_inplace_batch(const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params, ConstRingGSWACCKey& ek,
                                        const std::vector<lbcrypto::LWECiphertext>& cts, uint64_t bitwidth,
                                        const BatchScratchView& scratch, uint32_t batch) const;
    void gpu_BootstrapLUT_inplace_batch_device(const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params, ConstRingGSWACCKey& ek,
                                               const NativeInt* d_a, const NativeInt* d_b, uint32_t batch, uint32_t a_stride,
                                               uint64_t bitwidth, const BatchScratchView& scratch) const;
    BatchScratchView make_batch_scratch_view() const;
    BatchScratchView make_batch_scratch_view(BatchWorkspace& workspace) const;
    void ensure_batch_workspaces(uint32_t count) const;
    void ensure_h2d_events() const;
    NativeInt* ensure_pinned_monomial_inv(size_t elems) const;
	    uint32_t* ensure_pinned_index_pos(size_t elems) const;

	public:
	    // Non-owning view of a batched CircuitBootstrapping output in device memory.
	    // Layout per ciphertext: [digitsCC*2 rows][2 polys][N coeffs] flattened as
	    //   ((row * 2 + poly) * N + coeff).
	    // The underlying storage is owned by GPUCirBTSContext scratch and is valid until the
	    // next call that reuses the same scratch buffers.
	    struct DeviceRGSWBatchView {
	        const NativeInt* d_rgsw{};
	        uint32_t batch{};
	        uint32_t digitsCC{};
	        uint32_t N{};
	        size_t per_ciphertext_elems{};
	        cudaStream_t stream{};
	    };

	    explicit GPUCirBTSContext(const CirBTSContext& cc, uint32_t maxBatchSize = 1) : max_batch_size_(maxBatchSize) {
	        if (max_batch_size_ == 0) {
	            max_batch_size_ = 1;
	        }
	        gpu_init_cirbsk(const_cast<CirBTSContext&>(cc));
        gpu_init_ntt(const_cast<CirBTSContext&>(cc));
    }

    ~GPUCirBTSContext();

    // NTT tables/modulus accessors (read-only) for advanced kernels (e.g., GPU-side CMUX chains).
    [[nodiscard]] const DModulus* d_modulus() const {
        return d_modulus_.get();
    }
    [[nodiscard]] const NativeInt* d_twiddles() const {
        return d_twiddles_.get();
    }
    [[nodiscard]] const NativeInt* d_twiddles_shoup() const {
        return d_twiddles_shoup_.get();
    }
    [[nodiscard]] const NativeInt* d_itwiddles() const {
        return d_itwiddles_.get();
    }
    [[nodiscard]] const NativeInt* d_itwiddles_shoup() const {
        return d_itwiddles_shoup_.get();
    }
    [[nodiscard]] const NativeInt* d_n_inv_mod_q() const {
        return d_n_inv_mod_q_.get();
    }
    [[nodiscard]] const NativeInt* d_n_inv_mod_q_shoup() const {
        return d_n_inv_mod_q_shoup_.get();
    }
    [[nodiscard]] cudaStream_t stream() const {
        return stream_wrapper_.get_stream();
    }

    [[nodiscard]] RGSWCiphertext gpu_CircuitBootstrapping(const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params,
                                                          const lbcrypto::RingGSWCirBTKey& ek,
                                                          lbcrypto::ConstLWECiphertext& ct) const;

	    [[nodiscard]] std::vector<RGSWCiphertext> gpu_CircuitBootstrappingBatch(const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params,
	                                                                            const lbcrypto::RingGSWCirBTKey& ek,
	                                                                            const std::vector<lbcrypto::LWECiphertext>& cts) const;

	    [[nodiscard]] DeviceRGSWBatchView gpu_CircuitBootstrappingBatchToDevice(const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params,
	                                                                            const lbcrypto::RingGSWCirBTKey& ek,
	                                                                            const std::vector<lbcrypto::LWECiphertext>& cts) const;

	    // Device-input variant: TLWE ciphertexts provided as device arrays (a,b) with modulus q.
	    [[nodiscard]] DeviceRGSWBatchView gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE(
	        const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params,
	        const lbcrypto::RingGSWCirBTKey& ek,
	        const NativeInt* d_a,
	        const NativeInt* d_b,
	        uint32_t batch,
	        uint32_t a_stride) const;


	    [[nodiscard]] phantom::util::cuda_auto_ptr<uint64_t> gpu_BootstrapManyLUT(const std::shared_ptr<CirBTSCryptoParams>& params,
	                                                                              ConstRingGSWACCKey& ek,
	                                                                              ConstLWECiphertext& ct,
	                                                                              const NativePoly& LUT,
                                                                              uint64_t bitwidth) const;

    [[nodiscard]] NativeInteger gpu_SpecilMS(const NativeInteger& v, const NativeInteger& q,
                                             const NativeInteger& Q, uint64_t bitwidth) const;

    void CGGI_EvalAcc(const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params,
                      const NativeVector& a,
                      const phantom::util::cuda_auto_ptr<NativeInt>& d_acc) const;

    void LMKCDEY_EvalAcc(const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params,
                         const NativeVector& a,
                         const phantom::util::cuda_auto_ptr<NativeInt>& d_acc) const;

    void GenerateAllAutoMaps(uint32_t* d_AllMaps, RLWECryptoParams* params, cudaStream_t s) const;

    bool MaybeEnableL2Persist(const void* base_ptr, size_t bytes, cudaStream_t s) const;
    void DisableL2Persist(cudaStream_t s) const;

};
