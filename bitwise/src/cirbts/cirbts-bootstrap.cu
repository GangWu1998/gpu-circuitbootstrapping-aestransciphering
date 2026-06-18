/*
 * =============================================================================
 * File: cirbts-bootstrap.cu
 * Purpose: Core single-ciphertext bootstrapping path, including EvalAcc and
 *          the HT/SS stages, with optional CUDA graph execution.
 * Key parameters:
 *   - block sizes (env-configured) for EvalAcc/HT/SS kernels.
 *   - graph enable flags to reduce kernel launch overhead.
 *   - per-call scratch buffers for accumulators and digits.
 * Key points:
 *   - Hot path for latency; launches the main kernel sequence.
 *   - Stages monomial/index data to GPU and synchronizes only when needed.
 * =============================================================================
 */
#include <cassert>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sstream>
#include <mutex>
#include <utility>
#include "openfhe.h"
#include "cirbts/cirbts.cuh"
#include "cirbtscontext.h"
#include "cirbts/kernel.cuh"
#include "ntt.cuh"
#include "rlwe-homtrace.h"
#include "math/nbtheory.h"
#include "host/uintarithsmallmod.h"

using namespace lbcrypto;

namespace {
class L2PersistScope {
public:
    L2PersistScope(const GPUCirBTSContext* ctx, cudaStream_t stream) : ctx_(ctx), stream_(stream) {}

    void reset(const void* base_ptr, size_t bytes) {
        if (active_) {
            ctx_->DisableL2Persist(stream_);
        }
        active_ = ctx_->MaybeEnableL2Persist(base_ptr, bytes, stream_);
    }

    ~L2PersistScope() {
        if (active_) {
            ctx_->DisableL2Persist(stream_);
        }
    }

private:
    const GPUCirBTSContext* ctx_{};
    cudaStream_t stream_{};
    bool active_{false};
};

struct HtssProfileStats {
    std::mutex mu;
    double ht_ms{0.0};
    double ss_ms{0.0};
    uint64_t samples{0};
    bool registered{false};
};

HtssProfileStats& GetHtssProfileStats() {
    static HtssProfileStats stats;
    return stats;
}

void PrintHtssProfileStats() {
    auto& stats = GetHtssProfileStats();
    std::lock_guard<std::mutex> lock(stats.mu);
    if (stats.samples == 0) {
        return;
    }
    const double avg_ht = stats.ht_ms / static_cast<double>(stats.samples);
    const double avg_ss = stats.ss_ms / static_cast<double>(stats.samples);
    std::cout << "[CIRBTS_PROFILE_HTSS] avg_ht=" << avg_ht << "ms"
              << " avg_ss=" << avg_ss << "ms"
              << " samples=" << stats.samples << std::endl;
}
}  // namespace

RGSWCiphertext GPUCirBTSContext::gpu_CircuitBootstrapping(const std::shared_ptr<CirBTSCryptoParams>& params,
                                                          const RingGSWCirBTKey& ek,
                                                          ConstLWECiphertext& ct) const {
    const auto method = params->GetRingGSWParams1()->GetMethod();
    if (method != BINFHE_METHOD::GINX && method != BINFHE_METHOD::LMKCDEY) {
        std::ostringstream oss;
        oss << "gpu_CircuitBootstrapping (GPU) supports only method=GINX or method=LMKCDEY; got method=" << method;
        OPENFHE_THROW(config_error, oss.str());
    }

    const uint32_t numLUT = params->GetDigitsCC();
    const size_t N        = params->GetRingGSWParams1()->GetN();
    const uint32_t n      = params->GetLWEParams()->Getn();
    const NativeInt Q     = params->GetRingGSWParams1()->GetQ().ConvertToInt();
    if (N != ring_dim_ || numLUT != num_luts_) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrapping: GPU context params mismatch (recreate GPUCirBTSContext)");
    }
    if (!d_auto_maps_.get() || log_ring_dim_ == 0) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrapping: automorphism maps are not initialized on GPU");
    }
    if (!d_gpow_.get()) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrapping: Gpow is not initialized on GPU");
    }
    if (numLUT > 1 && !d_monomials_.get()) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrapping: monomials are not initialized on GPU");
    }

    const uint64_t bitwidth = static_cast<uint64_t>(std::ceil(std::log2(static_cast<double>(numLUT))));
    auto& s               = stream_wrapper_.get_stream();
    const bool profile    = (std::getenv("CIRBTS_PROFILE") != nullptr);
    const bool profile_htss = (std::getenv("CIRBTS_PROFILE_HTSS") != nullptr);
    const auto& rlweParams = params->GetRLWEParams();
    const uint32_t traceShift = rlweParams->GetTraceShift();
    if (traceShift >= log_ring_dim_) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrapping: TraceShift must be smaller than log2(N)");
    }
    const uint32_t traceRounds = log_ring_dim_ - traceShift;
    auto polyparams        = rlweParams->GetPolyParams();
    const uint32_t baseHT  = rlweParams->GetBaseHT();
    const uint32_t baseSS  = rlweParams->GetBaseSS();
    const uint32_t digitsHT = rlweParams->GetDigitsHTA();
    const uint32_t digitsSS = rlweParams->GetDigitsSSA();
    const uint64_t numRows = static_cast<uint64_t>(numLUT) * 2;
    const size_t lutSize   = static_cast<size_t>(numLUT) * N;

    if (scratch_lut_size_ < lutSize) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrapping: scratch buffer too small (recreate GPUCirBTSContext)");
    }
    if (scratch_digits_max_ < std::max(digitsHT, digitsSS)) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrapping: scratch digits mismatch (recreate GPUCirBTSContext)");
    }
    if (!d_ht_c0_.get() || !d_ht_c1_.get() || !d_perm_c0_.get() || !d_ht_next_c0_.get() || !d_ht_next_c1_.get() ||
        !d_ks_c0_.get() || !d_ks_c1_.get() || !d_digits_.get() || !d_ss_c0_.get() || !d_ss_c1_.get() || !d_rgsw_out_.get()) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrapping: scratch buffers are not initialized on GPU");
    }

    NativeInt* ht_c0      = d_ht_c0_.get();
    NativeInt* ht_c1      = d_ht_c1_.get();
    NativeInt* perm_c0    = d_perm_c0_.get();
    NativeInt* ht_next_c0 = d_ht_next_c0_.get();
    NativeInt* ht_next_c1 = d_ht_next_c1_.get();
    NativeInt* ks_c0      = d_ks_c0_.get();
    NativeInt* ks_c1      = d_ks_c1_.get();
    NativeInt* digits     = d_digits_.get();
    NativeInt* ss_c0      = d_ss_c0_.get();
    NativeInt* ss_c1      = d_ss_c1_.get();
    NativeInt* rgsw_out   = d_rgsw_out_.get();

    cudaEvent_t ev_start{}, ev_boot_end{}, ev_mvr_end{}, ev_htss_end{}, ev_d2h_start{}, ev_d2h_end{};
    cudaEvent_t ev_ht_start{}, ev_ht_end{}, ev_ss_start{}, ev_ss_end{};
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_start));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_boot_end));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_mvr_end));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_htss_end));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_d2h_start));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_d2h_end));
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev_start, s));
    }
    if (profile_htss) {
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_ht_start));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_ht_end));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_ss_start));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_ss_end));
    }

    // 1) Bootstrap (ACC) on GPU.
    if (!d_bootstrap_acc_.get()) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrapping: bootstrap accumulator scratch is not initialized on GPU");
    }
    gpu_BootstrapLUT_inplace(params, ek.RFkey, ct, bitwidth);
    const NativeInt* d_acc = d_bootstrap_acc_.get();
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev_boot_end, s));
    }

    // 2) Generate MV-RLWEs directly into HT accumulator buffers (evaluation form).
    {
        const size_t total = lutSize * 2;
        constexpr int blockSize = 256;
        const int numBlocks = static_cast<int>((total + blockSize - 1) / blockSize);
        kernel_GenMVRLWEs<<<numBlocks, blockSize, 0, s>>>(ht_c0, ht_c1, d_acc, d_monomials_.get(), d_modulus_.get(), N, numLUT,
                                                          /*batch=*/1);
    }
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev_mvr_end, s));
    }

    const bool htssFuseInwtDecomp = (std::getenv("CIRBTS_HTSS_FUSE_INWT_DECOMP") != nullptr);
    const bool htssFuseDecompFnwt = (!htssFuseInwtDecomp && (std::getenv("CIRBTS_HTSS_FUSE_DECOMP_FNWT") != nullptr));
    const bool htss_soa_requested = (std::getenv("CIRBTS_HTSS_SOA") != nullptr);
    const bool htss_smem_requested = (std::getenv("CIRBTS_HTSS_SMEM") != nullptr);
    const bool htss_smem_verbose = (std::getenv("CIRBTS_HTSS_SMEM_VERBOSE") != nullptr);
    const bool htss_pipeline      = (std::getenv("CIRBTS_HTSS_PIPELINE") != nullptr);
    const bool has_htss_soa = d_HTkey_soa_.get() && d_HTkey_shoup_soa_.get() && d_SSkey_soa_.get() && d_SSkey_shoup_soa_.get() &&
                              (htkey_soa_tiles_ > 0) && (sskey_soa_tiles_ > 0);
    if ((htss_soa_requested || htss_smem_requested) && !has_htss_soa) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrapping: HT/SS SoA requested but not initialized (set CIRBTS_HTSS_SOA before init)");
    }
    if (htss_pipeline && (baseHT != baseSS || digitsHT != digitsSS)) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrapping: HT/SS pipeline requires baseHT==baseSS and digitsHT==digitsSS");
    }
    if (htss_pipeline && std::getenv("CIRBTS_HTSS_PIPELINE_VERBOSE")) {
        std::cout << "[CIRBTS] HT/SS pipeline enabled (experimental, skips Step10)." << std::endl;
    }
    uint32_t htss_block_x         = 256;
    if (const char* v = std::getenv("CIRBTS_HTSS_BLOCK_X"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0' && parsed > 0) {
            htss_block_x = static_cast<uint32_t>(std::min<unsigned long>(parsed, 1024ul));
        }
    }

    size_t ht_smem_bytes = 0;
    size_t htss_smem_bytes = 0;
    size_t ss_smem_bytes = 0;
    bool ht_smem_enabled = false;
    bool htss_smem_enabled = false;
    bool ss_smem_enabled = false;
    if (htss_smem_requested && has_htss_soa) {
        int dev = 0;
        int max_default = 0;
        int max_opt = 0;
        PHANTOM_CHECK_CUDA(cudaGetDevice(&dev));
        PHANTOM_CHECK_CUDA(cudaDeviceGetAttribute(&max_default, cudaDevAttrMaxSharedMemoryPerBlock, dev));
        (void)cudaDeviceGetAttribute(&max_opt, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        const size_t max_smem = (max_opt > 0) ? static_cast<size_t>(max_opt) : static_cast<size_t>(max_default);
        const uint32_t warps = (htss_block_x + 31u) >> 5;
        ht_smem_bytes = static_cast<size_t>(warps) * 2u * 4u * 32u * sizeof(NativeInt);
        htss_smem_bytes = static_cast<size_t>(warps) * 2u * 8u * 32u * sizeof(NativeInt);
        ss_smem_bytes = ht_smem_bytes;

        if (ht_smem_bytes <= max_smem) {
            if (ht_smem_bytes > static_cast<size_t>(max_default)) {
                const auto st = cudaFuncSetAttribute(kernel_MultAddUpdate_HT_PermuteC1_SoA_Smem,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    static_cast<int>(ht_smem_bytes));
                ht_smem_enabled = (st == cudaSuccess);
            } else {
                ht_smem_enabled = true;
            }
        } else if (htss_smem_verbose) {
            std::cerr << "[CIRBTS] HT SMEM disabled (requested " << ht_smem_bytes
                      << " bytes > max " << max_smem << ")." << std::endl;
        }

        if (htss_smem_bytes <= max_smem) {
            if (htss_smem_bytes > static_cast<size_t>(max_default)) {
                const auto st = cudaFuncSetAttribute(kernel_MultAddUpdate_HTSS_PermuteC1_SoA_Smem,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    static_cast<int>(htss_smem_bytes));
                htss_smem_enabled = (st == cudaSuccess);
            } else {
                htss_smem_enabled = true;
            }
        } else if (htss_smem_verbose) {
            std::cerr << "[CIRBTS] HTSS SMEM disabled (requested " << htss_smem_bytes
                      << " bytes > max " << max_smem << ")." << std::endl;
        }

        if (ss_smem_bytes <= max_smem) {
            if (ss_smem_bytes > static_cast<size_t>(max_default)) {
                const auto st = cudaFuncSetAttribute(kernel_MultAddUpdate_SS_SoA_Smem,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    static_cast<int>(ss_smem_bytes));
                ss_smem_enabled = (st == cudaSuccess);
            } else {
                ss_smem_enabled = true;
            }
        } else if (htss_smem_verbose) {
            std::cerr << "[CIRBTS] SS SMEM disabled (requested " << ss_smem_bytes
                      << " bytes > max " << max_smem << ")." << std::endl;
        }
    }

    const bool htss_graph_smem = htss_pipeline ? htss_smem_enabled : ht_smem_enabled;
    const bool disableHtssGraph = profile_htss || (std::getenv("CIRBTS_DISABLE_HTSS_CUDA_GRAPH") != nullptr);
    const bool canUseHtssGraph  = !disableHtssGraph && !htss_graph_failed_;
    const bool htss_l2_switch   = l2_persist_enabled_ && !canUseHtssGraph && !htss_pipeline;

    auto runHomTraceAndSchemeSwitch = [&]() {
        // 3) HomTrace (EvalHT) for all LUT ciphertexts.
        NativeInt* local_ht_c0      = ht_c0;
        NativeInt* local_ht_c1      = ht_c1;
        NativeInt* local_ht_next_c0 = ht_next_c0;
        NativeInt* local_ht_next_c1 = ht_next_c1;

        dim3 block(htss_block_x);
        dim3 grid((N + block.x - 1) / block.x, numLUT);

        const bool use_htss_soa = (htss_soa_requested || htss_smem_requested) && has_htss_soa;
        const bool use_ht_smem = use_htss_soa && ht_smem_enabled;
        const bool use_htss_smem = use_htss_soa && htss_smem_enabled;
        const bool use_ss_smem = use_htss_soa && ss_smem_enabled;
        const uint32_t htss_tiles = static_cast<uint32_t>(N / 32u);
        const NativeInt* ht_key_base = use_htss_soa ? d_HTkey_soa_.get() : d_HTkey_.get();
        const NativeInt* ht_key_shoup_base = use_htss_soa ? d_HTkey_shoup_soa_.get() : d_HTkey_shoup_.get();
        const NativeInt* ss_key_base = use_htss_soa ? d_SSkey_soa_.get() : d_SSkey_.get();
        const NativeInt* ss_key_shoup_base = use_htss_soa ? d_SSkey_shoup_soa_.get() : d_SSkey_shoup_.get();

        L2PersistScope l2_scope(this, s);
        if (htss_l2_switch) {
            l2_scope.reset(ht_key_base, size_HTkey_ * size_HTRGSWkey_ * sizeof(NativeInt));
        }

        if (htss_pipeline) {
            cudaMemsetAsync(ss_c0, 0, sizeof(NativeInt) * lutSize, s);
            cudaMemsetAsync(ss_c1, 0, sizeof(NativeInt) * lutSize, s);
        }

        if (profile_htss) {
            PHANTOM_CHECK_CUDA(cudaEventRecord(ev_ht_start, s));
        }
        for (uint32_t k = 0; k < traceRounds; ++k) {
            const uint32_t* d_map = d_auto_maps_.get() + static_cast<size_t>(k) * N;
            if (htssFuseInwtDecomp) {
                // fuse: permute(evaluation) + INTT + decompose (coefficient)
                const size_t inwtShmem = N * sizeof(NativeInt);
                kernel_INWT_Permute_SignedDigitDecompose<<<numLUT, N / 2, inwtShmem, s>>>(
                    digits, local_ht_c0, d_map, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                    d_n_inv_mod_q_shoup_.get(), baseHT, digitsHT, N, numLUT);
                fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                    static_cast<size_t>(numLUT) * digitsHT, s);
            }
            else {
                // baseline: permute(evaluation) + INTT -> coefficient, then decompose.
                inwt_1d_opt_permute_batched(perm_c0, local_ht_c0, d_map, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(),
                                            d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N, numLUT, s);
                if (htssFuseDecompFnwt) {
                    const size_t ht_batch   = static_cast<size_t>(numLUT) * digitsHT;
                    const size_t htNttShmem = N * sizeof(NativeInt);
                    kernel_SignedDigitDecompose_FusedFNWT<<<ht_batch, N / 2, htNttShmem, s>>>(
                        digits, perm_c0, Q, baseHT, digitsHT, N, numLUT, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());
                }
                else {
                    kernel_Fused_Permute_Decompose<<<grid, block, 0, s>>>(digits, perm_c0, Q, baseHT, digitsHT, N, numLUT);
                    fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                        static_cast<size_t>(numLUT) * digitsHT, s);
                }
            }

            const size_t key_stride = size_HTRGSWkey_;  // digitsHT * 2 * N
            const NativeInt* d_key  = ht_key_base + static_cast<size_t>(k) * key_stride;
            const NativeInt* d_key_shoup = ht_key_shoup_base + static_cast<size_t>(k) * key_stride;
            // next = identity + automorphism + key-switch (fused to avoid ks temp buffers).
            if (htss_pipeline) {
                if (use_htss_soa) {
                    if (use_htss_smem) {
                        kernel_MultAddUpdate_HTSS_PermuteC1_SoA_Smem<<<grid, block, htss_smem_bytes, s>>>(
                            local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                            ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N, htss_tiles,
                            numLUT, lutSize);
                    }
                    else {
                        kernel_MultAddUpdate_HTSS_PermuteC1_SoA<<<grid, block, 0, s>>>(
                            local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                            ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N, htss_tiles,
                            numLUT, lutSize);
                    }
                }
                else {
                    kernel_MultAddUpdate_HTSS_PermuteC1<<<grid, block, 0, s>>>(
                        local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                        ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N, numLUT, lutSize);
                }
            }
            else if (use_htss_soa) {
                if (use_ht_smem) {
                    kernel_MultAddUpdate_HT_PermuteC1_SoA_Smem<<<grid, block, ht_smem_bytes, s>>>(
                        local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                        d_modulus_.get(), digitsHT, N, htss_tiles, numLUT, lutSize);
                }
                else {
                    kernel_MultAddUpdate_HT_PermuteC1_SoA<<<grid, block, 0, s>>>(
                        local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                        d_modulus_.get(), digitsHT, N, htss_tiles, numLUT, lutSize);
                }
            } else {
                kernel_MultAddUpdate_HT_PermuteC1<<<grid, block, 0, s>>>(
                    local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                    d_modulus_.get(), digitsHT, N, numLUT, lutSize);
            }
            std::swap(local_ht_c0, local_ht_next_c0);
            std::swap(local_ht_c1, local_ht_next_c1);
        }

        // 4) Save HT rows (odd rows) to output.
        kernel_SaveToRGSW<<<grid, block, 0, s>>>(rgsw_out, local_ht_c0, local_ht_c1, /*row_type=*/1, N, numLUT);
        if (profile_htss) {
            PHANTOM_CHECK_CUDA(cudaEventRecord(ev_ht_end, s));
            PHANTOM_CHECK_CUDA(cudaEventRecord(ev_ss_start, s));
        }

        if (htss_pipeline) {
            constexpr int addBlock = 256;
            const int addBlocks = static_cast<int>((lutSize + addBlock - 1) / addBlock);
            kernel_AddInPlace<<<addBlocks, addBlock, 0, s>>>(ss_c0, local_ht_c1, d_modulus_.get(), lutSize);
            kernel_SaveToRGSW<<<grid, block, 0, s>>>(rgsw_out, ss_c0, ss_c1, /*row_type=*/0, N, numLUT);
            if (profile_htss) {
                PHANTOM_CHECK_CUDA(cudaEventRecord(ev_ss_end, s));
            }
            return;
        }

        // 5) SchemeSwitch (EvalSS): use c0 in coefficient form, key-switch with SS key, then update.
        if (htssFuseInwtDecomp) {
            const size_t inwtShmem = N * sizeof(NativeInt);
            kernel_INWT_SignedDigitDecompose<<<numLUT, N / 2, inwtShmem, s>>>(
                digits, local_ht_c0, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                d_n_inv_mod_q_shoup_.get(), baseSS, digitsSS, N, numLUT);
            fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                static_cast<size_t>(numLUT) * digitsSS, s);
        }
        else {
            inwt_1d_opt_batched(local_ht_c0, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                d_n_inv_mod_q_shoup_.get(), N, numLUT, s);
            if (htssFuseDecompFnwt) {
                const size_t ss_batch   = static_cast<size_t>(numLUT) * digitsSS;
                const size_t ssNttShmem = N * sizeof(NativeInt);
                kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, ssNttShmem, s>>>(
                    digits, local_ht_c0, Q, baseSS, digitsSS, N, numLUT, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());
            }
            else {
                kernel_Fused_Permute_Decompose<<<grid, block, 0, s>>>(digits, local_ht_c0, Q, baseSS, digitsSS, N, numLUT);
                fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                    static_cast<size_t>(numLUT) * digitsSS, s);
            }
        }
        if (htss_l2_switch) {
            l2_scope.reset(ss_key_base, size_SSkey_ * N * sizeof(NativeInt));
        }
        if (use_htss_soa) {
            if (use_ss_smem) {
                kernel_MultAddUpdate_SS_SoA_Smem<<<grid, block, ss_smem_bytes, s>>>(
                    ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1, d_modulus_.get(), digitsSS, N,
                    htss_tiles, numLUT, lutSize);
            }
            else {
                kernel_MultAddUpdate_SS_SoA<<<grid, block, 0, s>>>(
                    ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1, d_modulus_.get(), digitsSS, N,
                    htss_tiles, numLUT, lutSize);
            }
        } else {
            kernel_MultAddUpdate_SS<<<grid, block, 0, s>>>(ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1,
                                                           d_modulus_.get(), digitsSS, N, numLUT, lutSize);
        }

        // 6) Save SS rows (even rows) to output and copy back.
        kernel_SaveToRGSW<<<grid, block, 0, s>>>(rgsw_out, ss_c0, ss_c1, /*row_type=*/0, N, numLUT);
        if (profile_htss) {
            PHANTOM_CHECK_CUDA(cudaEventRecord(ev_ss_end, s));
        }
    };

    if (canUseHtssGraph) {
        if (!htss_graph_exec_ || htss_graph_block_x_ != htss_block_x || htss_graph_num_luts_ != numLUT ||
            htss_graph_fuse_inwt_decomp_ != htssFuseInwtDecomp || htss_graph_fuse_decomp_fnwt_ != htssFuseDecompFnwt ||
            htss_graph_pipeline_ != htss_pipeline || htss_graph_smem_ != htss_graph_smem) {
            if (htss_graph_exec_) {
                cudaGraphExecDestroy(htss_graph_exec_);
                htss_graph_exec_ = nullptr;
            }
            htss_graph_block_x_ = htss_block_x;
            htss_graph_num_luts_ = numLUT;
            htss_graph_fuse_inwt_decomp_ = htssFuseInwtDecomp;
            htss_graph_fuse_decomp_fnwt_ = htssFuseDecompFnwt;
            htss_graph_pipeline_ = htss_pipeline;
            htss_graph_smem_ = htss_graph_smem;

            // Build graph once; subsequent launches reuse it.
            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
            cudaGraph_t graph{};
            PHANTOM_CHECK_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            runHomTraceAndSchemeSwitch();
            PHANTOM_CHECK_CUDA(cudaStreamEndCapture(s, &graph));

            cudaGraphExec_t exec{};
            const auto inst = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
            cudaGraphDestroy(graph);
            if (inst != cudaSuccess) {
                htss_graph_failed_ = true;
            }
            else {
                htss_graph_exec_ = exec;
            }
        }

        if (htss_graph_exec_) {
            L2PersistScope l2_scope(this, s);
            if (l2_persist_enabled_) {
                const bool use_htss_soa = (htss_soa_requested || htss_smem_requested) && has_htss_soa;
                const NativeInt* ht_key_base = use_htss_soa ? d_HTkey_soa_.get() : d_HTkey_.get();
                l2_scope.reset(ht_key_base, size_HTkey_ * size_HTRGSWkey_ * sizeof(NativeInt));
            }
            PHANTOM_CHECK_CUDA(cudaGraphLaunch(htss_graph_exec_, s));
        }
        else {
            runHomTraceAndSchemeSwitch();
        }
    }
    else {
        runHomTraceAndSchemeSwitch();
    }
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev_htss_end, s));
    }

    std::vector<NativeInt> h_RGSW(static_cast<size_t>(numRows) * 2 * N);
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev_d2h_start, s));
    }
    cudaMemcpyAsync(h_RGSW.data(), rgsw_out, h_RGSW.size() * sizeof(NativeInt), cudaMemcpyDeviceToHost, s);
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev_d2h_end, s));
    }
    cudaStreamSynchronize(s);

    double cpu_pack_ms = 0.0;
    std::chrono::high_resolution_clock::time_point cpu_pack_start;
    if (profile) {
        cpu_pack_start = std::chrono::high_resolution_clock::now();
    }
    RGSWCiphertextImpl res_gpu(numRows, 2);
    for (uint64_t row = 0; row < numRows; ++row) {
        NativePoly c0(polyparams, Format::EVALUATION, true);
        NativePoly c1(polyparams, Format::EVALUATION, true);
        const NativeInt* base = h_RGSW.data() + row * 2 * N;
        for (size_t i = 0; i < N; ++i) {
            c0[i] = NativeInteger(base[i]);
            c1[i] = NativeInteger(base[N + i]);
        }
        res_gpu[row] = {c0, c1};
    }
    if (profile) {
        const auto cpu_pack_end = std::chrono::high_resolution_clock::now();
        cpu_pack_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cpu_pack_end - cpu_pack_start).count();
    }

    if (profile) {
        float ms_boot = 0.0f;
        float ms_mvr  = 0.0f;
        float ms_htss = 0.0f;
        float ms_d2h  = 0.0f;
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_boot, ev_start, ev_boot_end));
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_mvr, ev_boot_end, ev_mvr_end));
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_htss, ev_mvr_end, ev_htss_end));
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_d2h, ev_d2h_start, ev_d2h_end));

        std::cout << "[CIRBTS_PROFILE] bootstrap=" << ms_boot << "ms"
                  << " mvr=" << ms_mvr << "ms"
                  << " htss=" << ms_htss << "ms"
                  << " d2h=" << ms_d2h << "ms"
                  << " cpu_pack=" << cpu_pack_ms << "ms" << std::endl;

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_boot_end);
        cudaEventDestroy(ev_mvr_end);
        cudaEventDestroy(ev_htss_end);
        cudaEventDestroy(ev_d2h_start);
        cudaEventDestroy(ev_d2h_end);
    }
    if (profile_htss) {
        float ms_ht = 0.0f;
        float ms_ss = 0.0f;
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_ht, ev_ht_start, ev_ht_end));
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_ss, ev_ss_start, ev_ss_end));
        auto& stats = GetHtssProfileStats();
        {
            std::lock_guard<std::mutex> lock(stats.mu);
            stats.ht_ms += static_cast<double>(ms_ht);
            stats.ss_ms += static_cast<double>(ms_ss);
            ++stats.samples;
            if (!stats.registered) {
                stats.registered = true;
                std::atexit(PrintHtssProfileStats);
            }
        }
        cudaEventDestroy(ev_ht_start);
        cudaEventDestroy(ev_ht_end);
        cudaEventDestroy(ev_ss_start);
        cudaEventDestroy(ev_ss_end);
    }

    return std::make_shared<RGSWCiphertextImpl>(std::move(res_gpu));
}

void GPUCirBTSContext::gpu_BootstrapLUT_inplace(const std::shared_ptr<CirBTSCryptoParams>& params, ConstRingGSWACCKey& ek,
                                                ConstLWECiphertext& ct, uint64_t bitwidth) const {
    if (!ek) {
        OPENFHE_THROW(config_error,
                      "Bootstrapping keys have not been generated. Please call BTKeyGen before calling bootstrapping.");
    }
    if (!d_bootstrap_acc_.get() || !d_lut_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace: bootstrap scratch buffers are not initialized on GPU");
    }

    const uint32_t numLUT = params->GetDigitsCC();
    const auto& lweParams = params->GetLWEParams();
    const auto q          = lweParams->Getq();
    const auto n          = lweParams->Getn();

    const auto& rgswParams1 = params->GetRingGSWParams1();
    const auto polyParams   = rgswParams1->GetPolyParams();
    const auto Q            = rgswParams1->GetQ();
    const size_t N          = polyParams->GetRingDimension();
    const auto N_inv        = NativeInteger(N).ModInverse(Q);

    if (N != ring_dim_ || numLUT != num_luts_) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace: GPU context params mismatch (recreate GPUCirBTSContext)");
    }
    if (!d_gpow_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace: Gpow is not initialized on GPU");
    }
    if (!d_monomial_inv_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace: monomial scratch buffer is not initialized on GPU");
    }

    // Special modulus switching (CPU-side; output modulus is 2N).
    const auto b             = ct->GetB();
    const NativeInteger b_ms = gpu_SpecilMS(b, NativeInteger(2 * N), q, bitwidth);
    const auto a             = ct->GetA();
    NativeVector a_ms(n, NativeInteger(2 * N));
    const bool isLMKCDEY = (rgswParams1->GetMethod() == BINFHE_METHOD::LMKCDEY);
    for (usint i = 0; i < n; ++i) {
        a_ms[i] = isLMKCDEY ? a[i] : gpu_SpecilMS(a[i], NativeInteger(2 * N), q, bitwidth);
    }
    auto ct_ms = LWECiphertextImpl(a_ms, b_ms);
    ct_ms.SetModulus(NativeInteger(2 * N));

    auto& s          = stream_wrapper_.get_stream();
    NativeInt* d_res = d_bootstrap_acc_.get();

    const bool profile = (std::getenv("CIRBTS_PROFILE") != nullptr);
    cudaEvent_t ev0{}, ev1{}, ev2{}, ev3{}, ev4{}, ev5{}, ev6{};
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev0));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev1));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev2));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev3));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev4));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev5));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev6));
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev0, s));
    }

    // ACC starts as (0, LUT) in evaluation form.
    cudaMemsetAsync(d_res, 0, sizeof(NativeInt) * N, s);
    cudaMemcpyAsync(d_res + N, d_lut_.get(), sizeof(NativeInt) * N, cudaMemcpyDeviceToDevice, s);

    // Multiply with X^{-b_MS}.
    const uint32_t b_idx     = b_ms.ConvertToInt<uint32_t>();
    const auto b_monomial    = params->GetMonomial(b_idx);
    ensure_h2d_events();
    NativeInt* h_mono = ensure_pinned_monomial_inv(N);
    std::memcpy(h_mono, &b_monomial.GetValues().at(0), sizeof(NativeInt) * N);
    cudaMemcpyAsync(d_monomial_inv_.get(), h_mono, sizeof(NativeInt) * N, cudaMemcpyHostToDevice, copy_stream_wrapper_.get_stream());
    PHANTOM_CHECK_CUDA(cudaEventRecord(h2d_monomial_event_, copy_stream_wrapper_.get_stream()));

    constexpr int blockSize = 256;
    const int numBlocks     = static_cast<int>((N + blockSize - 1) / blockSize);
    PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_monomial_event_, 0));
    kernel_ModMulScalar<<<numBlocks, blockSize, 0, s>>>(d_res + N, d_monomial_inv_.get(), d_modulus_.get(), N);
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev1, s));
    }

    if (rgswParams1->GetMethod() == BINFHE_METHOD::GINX) {
        CGGI_EvalAcc(params, ct_ms.GetA(), d_bootstrap_acc_);
    }
    else if (rgswParams1->GetMethod() == BINFHE_METHOD::LMKCDEY) {
        // CirBTSScheme::BootstrapManyLUT negates `a` before LMKCDEY EvalAcc to align sign conventions.
        // Mirror that behavior here so GPU/CPU agree.
        auto& a_ct          = ct_ms.GetA();
        const auto modMS    = ct_ms.GetModulus();
        for (usint i = 0; i < n; ++i) {
            a_ct[i] = NativeInteger(0).ModSubFast(a_ct[i], modMS);
        }
        LMKCDEY_EvalAcc(params, ct_ms.GetA(), d_bootstrap_acc_);
    }
    else {
        std::ostringstream oss;
        oss << "gpu_BootstrapLUT_inplace: unsupported method=" << rgswParams1->GetMethod();
        OPENFHE_THROW(config_error, oss.str());
    }
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev2, s));
    }

    kernel_ModMulConst<<<numBlocks, blockSize, 0, s>>>(d_res, N_inv.ConvertToInt(), d_modulus_.get(), N);
    kernel_ModMulConst<<<numBlocks, blockSize, 0, s>>>(d_res + N, N_inv.ConvertToInt(), d_modulus_.get(), N);
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev3, s));
    }

    inwt_1d_opt_batched(d_res + N, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                        d_n_inv_mod_q_shoup_.get(), N, 1, s);
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev4, s));
    }

    const int numBlocks1 = static_cast<int>((numLUT + blockSize - 1) / blockSize);
    kernel_ModAddGpowScaled<<<numBlocks1, blockSize, 0, s>>>(d_res + N, d_gpow_.get(), N_inv.ConvertToInt(), d_modulus_.get(), numLUT);
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev5, s));
    }
    fnwt_1d_opt_batched(d_res + N, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N, 1, s);

    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev6, s));
        PHANTOM_CHECK_CUDA(cudaEventSynchronize(ev6));

        float ms_init = 0.0f;
        float ms_eval = 0.0f;
        float ms_norm = 0.0f;
        float ms_inwt = 0.0f;
        float ms_gpow = 0.0f;
        float ms_fnwt = 0.0f;
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_init, ev0, ev1));
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_eval, ev1, ev2));
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_norm, ev2, ev3));
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_inwt, ev3, ev4));
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_gpow, ev4, ev5));
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_fnwt, ev5, ev6));

        std::cout << "[CIRBTS_PROFILE_BOOT] init+mono=" << ms_init << "ms"
                  << " evalacc=" << ms_eval << "ms"
                  << " norm=" << ms_norm << "ms"
                  << " inwt=" << ms_inwt << "ms"
                  << " gpow=" << ms_gpow << "ms"
                  << " fnwt=" << ms_fnwt << "ms" << std::endl;

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaEventDestroy(ev2);
        cudaEventDestroy(ev3);
        cudaEventDestroy(ev4);
        cudaEventDestroy(ev5);
        cudaEventDestroy(ev6);
    }
}

phantom::util::cuda_auto_ptr<uint64_t> GPUCirBTSContext::gpu_BootstrapManyLUT(const std::shared_ptr<CirBTSCryptoParams>& params, ConstRingGSWACCKey& ek,
                                                                             ConstLWECiphertext& ct, const NativePoly& LUT,
                                                                             uint64_t bitwidth) const {
    if (!ek) {
        OPENFHE_THROW(config_error,
                      "Bootstrapping keys have not been generated. Please call BTKeyGen before calling bootstrapping.");
    }

    const uint32_t numLUT = params->GetDigitsCC();
    const auto& lweParams = params->GetLWEParams();
    const auto q          = lweParams->Getq();
    const auto n          = lweParams->Getn();

    const auto& rgswParams1 = params->GetRingGSWParams1();
    const auto polyParams   = rgswParams1->GetPolyParams();
    const auto Q            = rgswParams1->GetQ();
    const size_t N          = polyParams->GetRingDimension();
    const auto N_inv        = NativeInteger(N).ModInverse(Q);

    if (N != ring_dim_ || numLUT != num_luts_) {
        OPENFHE_THROW(config_error, "gpu_BootstrapManyLUT: GPU context params mismatch (recreate GPUCirBTSContext)");
    }
    if (!d_gpow_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapManyLUT: Gpow is not initialized on GPU");
    }

    // Special modulus switching (CPU-side; output modulus is 2N).
    const auto b             = ct->GetB();
    const NativeInteger b_ms = gpu_SpecilMS(b, NativeInteger(2 * N), q, bitwidth);
    const auto a             = ct->GetA();
    NativeVector a_ms(n, NativeInteger(2 * N));
    const bool isLMKCDEY = (rgswParams1->GetMethod() == BINFHE_METHOD::LMKCDEY);
    for (usint i = 0; i < n; ++i) {
        a_ms[i] = isLMKCDEY ? a[i] : gpu_SpecilMS(a[i], NativeInteger(2 * N), q, bitwidth);
    }
    auto ct_ms = LWECiphertextImpl(a_ms, b_ms);
    ct_ms.SetModulus(NativeInteger(2 * N));

    auto& s    = stream_wrapper_.get_stream();
    auto d_res = phantom::util::make_cuda_auto_ptr<NativeInt>(2 * N, s);

    // ACC starts as (0, LUT) in evaluation form.
    cudaMemsetAsync(d_res.get(), 0, sizeof(NativeInt) * N, s);
    cudaMemcpyAsync(d_res.get() + N, &LUT.GetValues()[0], sizeof(NativeInt) * N, cudaMemcpyHostToDevice, s);

    // Multiply with X^{-b_MS}.
    if (!d_monomial_inv_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapManyLUT: monomial scratch buffer is not initialized on GPU");
    }
    const uint32_t b_idx = b_ms.ConvertToInt<uint32_t>();
    const auto b_monomial = params->GetMonomial(b_idx);
    cudaMemcpyAsync(d_monomial_inv_.get(), &b_monomial.GetValues().at(0), sizeof(NativeInt) * N, cudaMemcpyHostToDevice, s);

    constexpr int blockSize = 256;
    const int numBlocks = static_cast<int>((N + blockSize - 1) / blockSize);
    kernel_ModMulScalar<<<numBlocks, blockSize, 0, s>>>(d_res.get() + N, d_monomial_inv_.get(), d_modulus_.get(), N);

    if (rgswParams1->GetMethod() == BINFHE_METHOD::GINX) {
        CGGI_EvalAcc(params, ct_ms.GetA(), d_res);
    }
    else if (rgswParams1->GetMethod() == BINFHE_METHOD::LMKCDEY) {
        // Keep sign conventions consistent with the OpenFHE CPU reference (see CirBTSScheme::BootstrapManyLUT).
        auto& a_ct          = ct_ms.GetA();
        const auto modMS    = ct_ms.GetModulus();
        for (usint i = 0; i < n; ++i) {
            a_ct[i] = NativeInteger(0).ModSubFast(a_ct[i], modMS);
        }
        LMKCDEY_EvalAcc(params, ct_ms.GetA(), d_res);
    }
    else {
        std::ostringstream oss;
        oss << "gpu_BootstrapManyLUT: unsupported method=" << rgswParams1->GetMethod();
        OPENFHE_THROW(config_error, oss.str());
    }

    kernel_ModMulConst<<<numBlocks, blockSize, 0, s>>>(d_res.get(), N_inv.ConvertToInt(), d_modulus_.get(), N);
    kernel_ModMulConst<<<numBlocks, blockSize, 0, s>>>(d_res.get() + N, N_inv.ConvertToInt(), d_modulus_.get(), N);

    inwt_1d_opt_batched(d_res.get() + N, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                        d_n_inv_mod_q_shoup_.get(), N, 1, s);

    const int numBlocks1 = static_cast<int>((numLUT + blockSize - 1) / blockSize);
    kernel_ModAddGpowScaled<<<numBlocks1, blockSize, 0, s>>>(d_res.get() + N, d_gpow_.get(), N_inv.ConvertToInt(), d_modulus_.get(), numLUT);
    fnwt_1d_opt_batched(d_res.get() + N, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N, 1, s);

    return d_res;
}


void GPUCirBTSContext::CGGI_EvalAcc(const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params, const NativeVector& a,
                                   const phantom::util::cuda_auto_ptr<NativeInt>& d_acc) const {
    const auto& lweParams   = params->GetLWEParams();
    const auto& rgswParams1 = params->GetRingGSWParams1();

    if (rgswParams1->GetMethod() != BINFHE_METHOD::GINX) {
        std::ostringstream oss;
        oss << "CGGI_EvalAcc requires method=GINX; got method=" << rgswParams1->GetMethod();
        OPENFHE_THROW(config_error, oss.str());
    }
    if (!d_monic_polys_.get()) {
        OPENFHE_THROW(config_error, "CGGI_EvalAcc: monic polys are not initialized on GPU");
    }
    if (!d_RFkey_shoup_.get()) {
        OPENFHE_THROW(config_error, "CGGI_EvalAcc: RF key Shoup table is not initialized on GPU");
    }
    if (!d_evalacc_ct_.get() || !d_evalacc_dct_.get() || !d_evalacc_indexPos_.get()) {
        OPENFHE_THROW(config_error, "CGGI_EvalAcc: EvalAcc scratch buffers are not initialized on GPU");
    }

    const size_t n = lweParams->Getn();
    const size_t N = rgswParams1->GetN();
    const auto& mod = a.GetModulus();

    const uint32_t MInt{2 * static_cast<uint32_t>(N)};
    const NativeInteger M{MInt};
    const auto MbyMod{M / mod};

    const uint32_t digitsGA = rgswParams1->GetDigitsGA();
    const uint32_t digitsG2 = digitsGA << 1;
    const NativeInt Q       = rgswParams1->GetQ().ConvertToInt();
    const uint32_t baseG    = rgswParams1->GetBaseG();
    const bool use_swizzle_path = use_swizzle_ && (digitsG2 == 2u) && rfkey_swizzle_tiles_ &&
                                  d_RFkey_swizzle_.get() && d_RFkey_shoup_swizzle_.get();

    auto& s = stream_wrapper_.get_stream();

    bool fusedNttOverride = false;
    uint32_t fusedNttTpb = static_cast<uint32_t>(N / 2);
    if (const char* v = std::getenv("CIRBTS_FUSED_NTT_TPB"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0' && parsed > 0) {
            const uint32_t maxTpb = static_cast<uint32_t>(std::min<size_t>(N / 2, 1024));
            fusedNttTpb           = static_cast<uint32_t>(std::min<unsigned long>(parsed, maxTpb));
            fusedNttTpb           = (fusedNttTpb / 32) * 32;
            if (fusedNttTpb == 0) {
                fusedNttTpb = 32;
            }
            fusedNttOverride = true;
        }
    }

    const bool disableGraph = (std::getenv("CIRBTS_DISABLE_CUDA_GRAPH") != nullptr);

    auto get_env_u32_default = [](const char* name, uint32_t defval) -> uint32_t {
        const char* v = std::getenv(name);
        if (!v || !*v) {
            return defval;
        }
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0') {
            return static_cast<uint32_t>(parsed);
        }
        return defval;
    };
    uint32_t pbs_multibit = get_env_u32_default("CIRBTS_PBS_MULTIBIT", 0u);
    uint32_t evalacc_multibit = get_env_u32_default("CIRBTS_EVALACC_MULTIBIT", 2u);
    const bool use_pbs_multibit2 = (pbs_multibit == 2u);
    const bool use_evalacc_multibit2 = (!use_pbs_multibit2 && evalacc_multibit == 2u);
    const bool use_multibit2 = use_pbs_multibit2 || use_evalacc_multibit2;
    if (use_multibit2 && use_split_fft_) {
        OPENFHE_THROW(config_error, "CGGI_EvalAcc: multibit=2 is not supported with split FFT");
    }

    // Precompute all monic polynomial indices for this ciphertext and upload once.
    ensure_h2d_events();
    uint32_t* h_indexPos = ensure_pinned_index_pos(n);
    for (size_t lweIndex = 0; lweIndex < n; ++lweIndex) {
        const NativeInteger ai = a[lweIndex].ModMulFast(MbyMod, mod);
        h_indexPos[lweIndex]   = ai.ConvertToInt<uint32_t>();
    }
    cudaMemcpyAsync(d_evalacc_indexPos_.get(), h_indexPos, n * sizeof(uint32_t), cudaMemcpyHostToDevice,
                    copy_stream_wrapper_.get_stream());
    PHANTOM_CHECK_CUDA(cudaEventRecord(h2d_index_event_, copy_stream_wrapper_.get_stream()));

    const bool evalacc_digit_k_verbose = (std::getenv("CIRBTS_EVALACC_DIGIT_K_VERBOSE") != nullptr);
    if (use_split_fft_) {
#if defined(PHANTOM_ENABLE_CUFFTDX)
        if (evalacc_digit_k_verbose) {
            static bool printed = false;
            if (!printed) {
                std::cout << "[CIRBTS] EvalAcc digit-K disabled: split FFT backend in use" << std::endl;
                printed = true;
            }
        }
        const bool has_fft =
            (split_fft_fpb_ == 1 && fft_2048_fpb1_) || (split_fft_fpb_ == 2 && fft_2048_fpb2_) ||
            (split_fft_fpb_ == 4 && fft_2048_fpb4_);
        if (!has_fft || d_RFkey_fft_limbs_.empty() || !d_monic_fft_.get() || split_fft_base_pows_shoup_.empty() ||
            !d_evalacc_dct_i64_.get() || !d_evalacc_dct_fft_.get() || !d_evalacc_acc_fft_.get()) {
            OPENFHE_THROW(config_error, "CGGI_EvalAcc: split FFT requested but not initialized (set CIRBTS_USE_SPLIT_FFT before init)");
        }
        if (split_fft_fft_len_ != N / 2 || split_fft_key_stride_ == 0 || split_fft_limbs_ == 0) {
            OPENFHE_THROW(config_error, "CGGI_EvalAcc: split FFT configuration mismatch");
        }

        const dim3 decompBlock(256);
        const dim3 decompGrid((N + decompBlock.x - 1) / decompBlock.x, 2);
        const size_t fftN = split_fft_fft_len_;
        const size_t keyStride = split_fft_key_stride_;
        uint32_t fftBlockX = 256;
        if (const char* v = std::getenv("CIRBTS_EVALACC_FFT_BLOCK_X"); v && *v) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(v, &end, 10);
            if (end != v && end && *end == '\0' && parsed > 0) {
                fftBlockX = static_cast<uint32_t>(std::min<unsigned long>(parsed, 1024ul));
                fftBlockX = (fftBlockX / 32) * 32;
                if (fftBlockX == 0) {
                    fftBlockX = 32;
                }
            }
        }
        const dim3 fftBlock(fftBlockX);
        const uint32_t blocksPerPoly = static_cast<uint32_t>((fftN + fftBlock.x - 1) / fftBlock.x);
        const dim3 fftGrid(blocksPerPoly * 2);

        auto fft_i2c_forward = [&](SplitFFTComplex* out, const int64_t* in, size_t batch_size) {
            if (split_fft_fpb_ == 1) {
                fft_2048_fpb1_->i2c_forward(out, in, batch_size, s);
            } else if (split_fft_fpb_ == 2) {
                fft_2048_fpb2_->i2c_forward(out, in, batch_size, s);
            } else {
                fft_2048_fpb4_->i2c_forward(out, in, batch_size, s);
            }
        };
        auto fft_c2i_inverse_add = [&](NativeInt* acc, const SplitFFTComplex* in, NativeInt scale, NativeInt scale_shoup,
                                       size_t batch_size) {
            if (split_fft_fpb_ == 1) {
                fft_2048_fpb1_->c2i_inverse_add(acc, in, Q, scale, scale_shoup, batch_size, s);
            } else if (split_fft_fpb_ == 2) {
                fft_2048_fpb2_->c2i_inverse_add(acc, in, Q, scale, scale_shoup, batch_size, s);
            } else {
                fft_2048_fpb4_->c2i_inverse_add(acc, in, Q, scale, scale_shoup, batch_size, s);
            }
        };

        const bool canUseGraph = !disableGraph && !split_fft_evalacc_graph_failed_ && (d_acc.get() == d_bootstrap_acc_.get());
        if (canUseGraph) {
            if (split_fft_evalacc_graph_exec_ && split_fft_evalacc_graph_limbs_ != split_fft_limbs_) {
                cudaGraphExecDestroy(split_fft_evalacc_graph_exec_);
                split_fft_evalacc_graph_exec_ = nullptr;
                split_fft_evalacc_graph_acc_  = nullptr;
            }
            if (!split_fft_evalacc_graph_exec_) {
                split_fft_evalacc_graph_acc_   = d_bootstrap_acc_.get();
                split_fft_evalacc_graph_limbs_ = split_fft_limbs_;

                PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
                cudaGraph_t graph{};
                PHANTOM_CHECK_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));

                inwt_1d_opt_batched(split_fft_evalacc_graph_acc_, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(),
                                    d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N, /*batch=*/2, s);

                for (size_t lweIndex = 0; lweIndex < n; ++lweIndex) {
                    kernel_SignedDigitDecompose2_Int64<<<decompGrid, decompBlock, 0, s>>>(
                        d_evalacc_dct_i64_.get(), split_fft_evalacc_graph_acc_, Q, baseG, digitsGA, N);

                    fft_i2c_forward(d_evalacc_dct_fft_.get(), d_evalacc_dct_i64_.get(), digitsG2);

                    const size_t keyOffset = lweIndex * keyStride;
                    for (uint32_t limb = 0; limb < split_fft_limbs_; ++limb) {
                        const SplitFFTComplex* key_fft = d_RFkey_fft_limbs_[limb].get() + keyOffset;
                        kernel_EvalAccCoreCGGI_fft_monic<<<fftGrid, fftBlock, 0, s>>>(
                            d_evalacc_acc_fft_.get(), d_evalacc_dct_fft_.get(), key_fft, d_monic_fft_.get(),
                            fftN, digitsG2, blocksPerPoly, d_evalacc_indexPos_.get(), static_cast<uint32_t>(lweIndex));
                        const NativeInt scale = split_fft_base_pows_[limb];
                        const NativeInt scale_shoup = split_fft_base_pows_shoup_[limb];
                        fft_c2i_inverse_add(split_fft_evalacc_graph_acc_, d_evalacc_acc_fft_.get(), scale, scale_shoup, 2);
                    }
                }

                fnwt_1d_opt_batched(split_fft_evalacc_graph_acc_, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(),
                                    N, /*batch=*/2, s);

                PHANTOM_CHECK_CUDA(cudaStreamEndCapture(s, &graph));
                cudaGraphExec_t exec{};
                const auto inst = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
                cudaGraphDestroy(graph);
                if (inst != cudaSuccess) {
                    split_fft_evalacc_graph_failed_ = true;
                    split_fft_evalacc_graph_acc_    = nullptr;
                }
                else {
                    split_fft_evalacc_graph_exec_ = exec;
                }
            }

            if (split_fft_evalacc_graph_exec_) {
                PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_index_event_, 0));
                PHANTOM_CHECK_CUDA(cudaGraphLaunch(split_fft_evalacc_graph_exec_, s));
                return;
            }
        }

        // Fallback (no CUDA graph).
        inwt_1d_opt_batched(d_acc.get(), d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(),
                            d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N, /*batch=*/2, s);

        for (size_t lweIndex = 0; lweIndex < n; ++lweIndex) {
            kernel_SignedDigitDecompose2_Int64<<<decompGrid, decompBlock, 0, s>>>(
                d_evalacc_dct_i64_.get(), d_acc.get(), Q, baseG, digitsGA, N);

            fft_i2c_forward(d_evalacc_dct_fft_.get(), d_evalacc_dct_i64_.get(), digitsG2);

            const size_t keyOffset = lweIndex * keyStride;

            if (lweIndex == 0) {
                PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_index_event_, 0));
            }
            for (uint32_t limb = 0; limb < split_fft_limbs_; ++limb) {
                const SplitFFTComplex* key_fft = d_RFkey_fft_limbs_[limb].get() + keyOffset;
                kernel_EvalAccCoreCGGI_fft_monic<<<fftGrid, fftBlock, 0, s>>>(
                    d_evalacc_acc_fft_.get(), d_evalacc_dct_fft_.get(), key_fft, d_monic_fft_.get(),
                    fftN, digitsG2, blocksPerPoly, d_evalacc_indexPos_.get(), static_cast<uint32_t>(lweIndex));
                const NativeInt scale = split_fft_base_pows_[limb];
                const NativeInt scale_shoup = split_fft_base_pows_shoup_[limb];
                fft_c2i_inverse_add(d_acc.get(), d_evalacc_acc_fft_.get(), scale, scale_shoup, 2);
            }
        }

        fnwt_1d_opt_batched(d_acc.get(), d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(),
                            N, /*batch=*/2, s);
        return;
#else
        OPENFHE_THROW(config_error, "CGGI_EvalAcc: split FFT requested but PHANTOM_ENABLE_CUFFTDX is disabled");
#endif
    }

    const char* evalacc_soa_env = std::getenv("CIRBTS_EVALACC_SOA");
    const bool evalacc_soa_explicit = (evalacc_soa_env != nullptr);
    const bool evalacc_soa_requested = evalacc_soa_explicit ? (evalacc_soa_env[0] != '0') : true;
    const bool evalacc_soa_force = (std::getenv("CIRBTS_EVALACC_SOA_FORCE") != nullptr);
    const char* evalacc_smem_env = std::getenv("CIRBTS_EVALACC_SMEM");
    const bool evalacc_smem_explicit = (evalacc_smem_env != nullptr);
    const bool evalacc_smem_requested = evalacc_smem_explicit ? (evalacc_smem_env[0] != '0') : true;
    const bool evalacc_smem_verbose = (std::getenv("CIRBTS_EVALACC_SMEM_VERBOSE") != nullptr);
    const bool has_evalacc_soa = d_RFkey_soa_.get() && d_RFkey_shoup_soa_.get() && (rfkey_soa_tiles_ > 0);
    bool use_evalacc_soa = evalacc_soa_requested && has_evalacc_soa;
    if (evalacc_soa_requested && !has_evalacc_soa) {
        if (evalacc_soa_explicit) {
            OPENFHE_THROW(config_error, "CGGI_EvalAcc: EvalAcc SoA requested but not initialized (set CIRBTS_EVALACC_SOA before init)");
        }
        if (evalacc_smem_verbose) {
            std::cout << "[CIRBTS] EvalAcc SoA disabled: not initialized." << std::endl;
        }
    }
    if (use_evalacc_soa && use_swizzle_path && !evalacc_soa_force) {
        static bool printed = false;
        if (!printed) {
            std::cout << "[CIRBTS] EvalAcc SoA disabled: swizzle path active (digitsG2=2)." << std::endl;
            printed = true;
        }
        use_evalacc_soa = false;
    }
    if (evalacc_smem_requested && !use_evalacc_soa) {
        if (evalacc_smem_explicit) {
            OPENFHE_THROW(config_error,
                          "CGGI_EvalAcc: EvalAcc SMEM requires EvalAcc SoA (set CIRBTS_EVALACC_SOA_FORCE=1)");
        }
        if (evalacc_smem_verbose) {
            std::cout << "[CIRBTS] EvalAcc SMEM disabled: EvalAcc SoA unavailable." << std::endl;
        }
    }
    const bool use_pipeline = (std::getenv("CIRBTS_EVALACC_PIPELINE") != nullptr) && !use_evalacc_soa;
    size_t evalacc_smem_bytes = 0;
    bool evalacc_smem_enabled = false;

    auto launch_evalacc_soa = [&](dim3 grid, dim3 block, BasicInteger* acc_ptr, size_t lweIndex) {
        const size_t shmem = evalacc_smem_enabled ? evalacc_smem_bytes : 0;
        if (evalacc_smem_enabled) {
            if (digitsG2 == 2) {
                kernel_EvalAccCore_Binary_SoA_T_Smem<2><<<grid, block, shmem, s>>>(
                    acc_ptr, d_evalacc_dct_.get(), d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(), d_monic_polys_.get(), N,
                    static_cast<uint32_t>(rfkey_soa_tiles_), d_modulus_.get(), lweIndex, d_evalacc_indexPos_.get());
            } else if (digitsG2 == 4) {
                kernel_EvalAccCore_Binary_SoA_T_Smem<4><<<grid, block, shmem, s>>>(
                    acc_ptr, d_evalacc_dct_.get(), d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(), d_monic_polys_.get(), N,
                    static_cast<uint32_t>(rfkey_soa_tiles_), d_modulus_.get(), lweIndex, d_evalacc_indexPos_.get());
            } else {
                kernel_EvalAccCore_Binary_SoA_Smem<<<grid, block, shmem, s>>>(
                    acc_ptr, d_evalacc_dct_.get(), d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(), d_monic_polys_.get(), N,
                    static_cast<uint32_t>(rfkey_soa_tiles_), d_modulus_.get(), digitsG2, lweIndex, d_evalacc_indexPos_.get());
            }
        } else {
            if (digitsG2 == 2) {
                kernel_EvalAccCore_Binary_SoA_T<2><<<grid, block, 0, s>>>(
                    acc_ptr, d_evalacc_dct_.get(), d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(), d_monic_polys_.get(), N,
                    static_cast<uint32_t>(rfkey_soa_tiles_), d_modulus_.get(), lweIndex, d_evalacc_indexPos_.get());
            } else if (digitsG2 == 4) {
                kernel_EvalAccCore_Binary_SoA_T<4><<<grid, block, 0, s>>>(
                    acc_ptr, d_evalacc_dct_.get(), d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(), d_monic_polys_.get(), N,
                    static_cast<uint32_t>(rfkey_soa_tiles_), d_modulus_.get(), lweIndex, d_evalacc_indexPos_.get());
            } else {
                kernel_EvalAccCore_Binary_SoA<<<grid, block, 0, s>>>(
                    acc_ptr, d_evalacc_dct_.get(), d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(), d_monic_polys_.get(), N,
                    static_cast<uint32_t>(rfkey_soa_tiles_), d_modulus_.get(), digitsG2, lweIndex, d_evalacc_indexPos_.get());
            }
        }
    };

    uint32_t evalacc_digit_k = 0;
    if (const char* v = std::getenv("CIRBTS_EVALACC_DIGIT_K"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0') {
            evalacc_digit_k = static_cast<uint32_t>(parsed);
        }
    }
    const bool k_pow2 = (evalacc_digit_k >= 2) && ((evalacc_digit_k & (evalacc_digit_k - 1)) == 0) && (evalacc_digit_k <= 32);
    bool use_digit_k = k_pow2;
    if (use_digit_k && use_evalacc_soa) {
        use_digit_k = false;
    }
    if (use_digit_k) {
        if (!d_evalacc_digits_s_.get() || !d_evalacc_residual_s_.get() || !d_evalacc_delta_.get() || !d_evalacc_carry_flag_.get()) {
            use_digit_k = false;
        }
        if (Q > static_cast<NativeInt>(std::numeric_limits<int64_t>::max())) {
            use_digit_k = false;
        }
    }
    if (evalacc_digit_k_verbose) {
        static bool printed = false;
        if (!printed) {
            if (use_digit_k) {
                std::cout << "[CIRBTS] EvalAcc digit-K enabled (K=2)." << std::endl;
            } else {
                std::string reason;
                if (evalacc_digit_k == 0) {
                    reason = "CIRBTS_EVALACC_DIGIT_K not set";
                } else if (!k_pow2) {
                    reason = "K must be power-of-two in [2, 32]";
                } else if (use_evalacc_soa) {
                    reason = "EvalAcc SoA enabled";
                } else if (!d_evalacc_digits_s_.get() || !d_evalacc_residual_s_.get() ||
                           !d_evalacc_delta_.get() || !d_evalacc_carry_flag_.get()) {
                    reason = "missing scratch buffers";
                } else if (Q > static_cast<NativeInt>(std::numeric_limits<int64_t>::max())) {
                    reason = "Q > INT64_MAX";
                } else {
                    reason = "unknown gating condition";
                }
                std::cout << "[CIRBTS] EvalAcc digit-K disabled: " << reason << std::endl;
            }
            printed = true;
        }
    }
    if (use_multibit2 && use_digit_k) {
        OPENFHE_THROW(config_error, "CGGI_EvalAcc: multibit=2 is incompatible with EvalAcc digit-K");
    }
    if (use_digit_k) {
        uint32_t evalaccBlockSize = 128;
        if (const char* v = std::getenv("CIRBTS_EVALACC_BLOCK_X"); v && *v) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(v, &end, 10);
            if (end != v && end && *end == '\0' && parsed > 0) {
                evalaccBlockSize = static_cast<uint32_t>(std::min<unsigned long>(parsed, 1024ul));
                evalaccBlockSize = (evalaccBlockSize / 32) * 32;
                if (evalaccBlockSize == 0) {
                    evalaccBlockSize = 32;
                }
            }
        }

        dim3 evalaccBlock(evalaccBlockSize);
        dim3 evalaccGrid((N + evalaccBlock.x - 1) / evalaccBlock.x, 1);
        const dim3 decompBlock(256);
        const dim3 decompGrid((N + decompBlock.x - 1) / decompBlock.x, 2);
        const dim3 updateGrid((static_cast<size_t>(2) * N + decompBlock.x - 1) / decompBlock.x, 1);
        const size_t evalaccNttShmem = N * sizeof(NativeInt);

        L2PersistScope l2_scope(this, s);
        {
            const NativeInt* key_base = use_evalacc_soa ? d_RFkey_soa_.get() :
                                        (use_swizzle_path ? d_RFkey_swizzle_.get() : d_RFkey_.get());
            l2_scope.reset(key_base, size_RFkey_ * size_RFRGSWkey_ * sizeof(NativeInt));
        }

        auto refresh_digits = [&]() {
            inwt_1d_opt_outofplace_batched(d_evalacc_ct_.get(), d_acc.get(), d_itwiddles_.get(), d_itwiddles_shoup_.get(),
                                           d_modulus_.get(), d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N,
                                           /*batch=*/2, s);
            kernel_SignedDigitDecompose2_Residual<<<decompGrid, decompBlock, 0, s>>>(
                d_evalacc_digits_s_.get(), d_evalacc_residual_s_.get(), d_evalacc_ct_.get(), Q, baseG, digitsGA, N);
        };

        refresh_digits();

        uint32_t carry_flag = 0;
        size_t lweIndex = 0;
        while (lweIndex < n) {
            const size_t block_start = lweIndex;
            const size_t block_end = std::min(n, block_start + static_cast<size_t>(evalacc_digit_k));

            cudaMemcpyAsync(d_evalacc_acc_backup_.get(), d_acc.get(), sizeof(NativeInt) * 2 * N,
                            cudaMemcpyDeviceToDevice, s);
            PHANTOM_CHECK_CUDA(cudaMemsetAsync(d_evalacc_carry_flag_.get(), 0, sizeof(uint32_t), s));

            for (size_t idx = block_start; idx < block_end; ++idx) {
                kernel_DigitsSigned_FusedFNWT<<<digitsG2, fusedNttTpb, evalaccNttShmem, s>>>(
                    d_evalacc_dct_.get(), d_evalacc_digits_s_.get(), Q, N, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());

                if (idx == 0) {
                    PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_index_event_, 0));
                }
                if (use_swizzle_path) {
                    kernel_EvalAccCore_Binary_Swizzle_Delta<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                        d_acc.get(), d_evalacc_delta_.get(), d_evalacc_dct_.get(), d_RFkey_swizzle_.get(),
                        d_RFkey_shoup_swizzle_.get(), d_monic_polys_.get(), N, static_cast<uint32_t>(rfkey_swizzle_tiles_),
                        d_modulus_.get(), idx, d_evalacc_indexPos_.get());
                } else {
                    kernel_EvalAccCore_Binary_Delta<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        d_acc.get(), d_evalacc_delta_.get(), d_evalacc_dct_.get(), d_RFkey_.get(), d_RFkey_shoup_.get(),
                        d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, idx, d_evalacc_indexPos_.get());
                }

                inwt_1d_opt_outofplace_batched(d_evalacc_ct_.get(), d_evalacc_delta_.get(), d_itwiddles_.get(),
                                               d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                               d_n_inv_mod_q_shoup_.get(), N, /*batch=*/2, s);

                kernel_AddDelta_ToDigitsResidual<<<updateGrid, decompBlock, 0, s>>>(
                    d_evalacc_digits_s_.get(), d_evalacc_residual_s_.get(), d_evalacc_ct_.get(), Q, baseG, digitsGA, N,
                    d_evalacc_carry_flag_.get());
            }

            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(&carry_flag, d_evalacc_carry_flag_.get(), sizeof(uint32_t),
                                               cudaMemcpyDeviceToHost, s));
            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
            if (carry_flag != 0) {
                cudaMemcpyAsync(d_acc.get(), d_evalacc_acc_backup_.get(), sizeof(NativeInt) * 2 * N,
                                cudaMemcpyDeviceToDevice, s);
                for (size_t idx = block_start; idx < block_end; ++idx) {
                    inwt_1d_opt_outofplace_batched(d_evalacc_ct_.get(), d_acc.get(), d_itwiddles_.get(), d_itwiddles_shoup_.get(),
                                                   d_modulus_.get(), d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N,
                                                   /*batch=*/2, s);

                    kernel_SignedDigitDecompose2_FusedFNWT<<<digitsG2, fusedNttTpb, evalaccNttShmem, s>>>(
                        d_evalacc_dct_.get(), d_evalacc_ct_.get(), Q, baseG, digitsGA, N, d_twiddles_.get(),
                        d_twiddles_shoup_.get(), d_modulus_.get());

                    if (idx == 0) {
                        PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_index_event_, 0));
                    }
                    if (use_swizzle_path) {
                        kernel_EvalAccCore_Binary_Swizzle<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                            d_acc.get(), d_evalacc_dct_.get(), d_RFkey_swizzle_.get(), d_RFkey_shoup_swizzle_.get(),
                            d_monic_polys_.get(), N, static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(), idx,
                            d_evalacc_indexPos_.get());
                    } else {
                        kernel_EvalAccCore_Binary<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            d_acc.get(), d_evalacc_dct_.get(), d_RFkey_.get(), d_RFkey_shoup_.get(), d_monic_polys_.get(), N,
                            d_modulus_.get(), digitsG2, idx, d_evalacc_indexPos_.get());
                    }
                }
                refresh_digits();
                carry_flag = 0;
            }

            lweIndex = block_end;
        }
        return;
    }

    if (use_pbs_multibit2) {
        if (!rfkey_grouped_ || rfkey_dim2_ != 3u || rfkey_groups_ == 0) {
            OPENFHE_THROW(config_error, "CGGI_EvalAcc: PBS multibit=2 requires grouped RFkey (set CIRBTS_PBS_MULTIBIT=2 before keygen)");
        }
        if (use_pipeline && use_swizzle_path) {
            OPENFHE_THROW(config_error, "CGGI_EvalAcc: PBS multibit=2 pipeline does not support swizzle (set CIRBTS_USE_SWIZZLE=0)");
        }

        uint32_t evalaccBlockSize = 128;
        if (const char* v = std::getenv("CIRBTS_EVALACC_BLOCK_X"); v && *v) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(v, &end, 10);
            if (end != v && end && *end == '\0' && parsed > 0) {
                evalaccBlockSize = static_cast<uint32_t>(std::min<unsigned long>(parsed, 1024ul));
                evalaccBlockSize = (evalaccBlockSize / 32) * 32;
                if (evalaccBlockSize == 0) {
                    evalaccBlockSize = 32;
                }
            }
        }

        const dim3 evalaccBlock(evalaccBlockSize);
        const dim3 evalaccGrid((N + evalaccBlock.x - 1) / evalaccBlock.x, 1);
        const size_t evalaccNttShmem = N * sizeof(NativeInt);
        const bool use_multibit_soa = use_evalacc_soa;
        const uint32_t soa_tiles = use_multibit_soa ? static_cast<uint32_t>(rfkey_soa_tiles_) : 0u;
        if (use_swizzle_path && use_multibit_soa) {
            OPENFHE_THROW(config_error, "CGGI_EvalAcc: PBS multibit swizzle is incompatible with EvalAcc SoA");
        }

        L2PersistScope l2_scope(this, s);
        {
            const NativeInt* key_base = use_multibit_soa ? d_RFkey_soa_.get() :
                                        (use_swizzle_path ? d_RFkey_swizzle_.get() : d_RFkey_.get());
            l2_scope.reset(key_base, size_RFkey_ * size_RFRGSWkey_ * sizeof(NativeInt));
        }

        const size_t key_stride = size_RFRGSWkey_;
        const size_t key_block = key_stride * rfkey_groups_;
        const NativeInt* ek0 = use_multibit_soa ? d_RFkey_soa_.get() :
                               (use_swizzle_path ? d_RFkey_swizzle_.get() : d_RFkey_.get());
        const NativeInt* ek1 = ek0 + key_block;
        const NativeInt* ek_pair = ek1 + key_block;
        const NativeInt* ek_shoup0 = use_multibit_soa ? d_RFkey_shoup_soa_.get() :
                                     (use_swizzle_path ? d_RFkey_shoup_swizzle_.get() : d_RFkey_shoup_.get());
        const NativeInt* ek_shoup1 = ek_shoup0 + key_block;
        const NativeInt* ek_shoup_pair = ek_shoup1 + key_block;

        evalacc_smem_enabled = false;
        evalacc_smem_bytes = 0;
        if (use_multibit_soa && evalacc_smem_requested) {
            int dev = 0;
            int max_default = 0;
            int max_opt = 0;
            PHANTOM_CHECK_CUDA(cudaGetDevice(&dev));
            PHANTOM_CHECK_CUDA(cudaDeviceGetAttribute(&max_default, cudaDevAttrMaxSharedMemoryPerBlock, dev));
            (void)cudaDeviceGetAttribute(&max_opt, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            const size_t max_smem = (max_opt > 0) ? static_cast<size_t>(max_opt) : static_cast<size_t>(max_default);
            const uint32_t warps = (evalaccBlockSize + 31u) >> 5;
            const size_t warp_stride = 8u * 32u;
            evalacc_smem_bytes = static_cast<size_t>(warps) * 2u * warp_stride * sizeof(NativeInt);
            if (evalacc_smem_bytes <= max_smem) {
                if (evalacc_smem_bytes > static_cast<size_t>(max_default)) {
                    const cudaError_t st0 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_MB2_Grouped_SoA_Smem,
                                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                                 static_cast<int>(evalacc_smem_bytes));
                    const cudaError_t st1 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_Grouped_SoA_Smem,
                                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                                 static_cast<int>(evalacc_smem_bytes));
                    evalacc_smem_enabled = (st0 == cudaSuccess) && (st1 == cudaSuccess);
                } else {
                    evalacc_smem_enabled = true;
                }
            } else if (evalacc_smem_verbose) {
                std::cerr << "[CIRBTS] EvalAcc SMEM disabled (requested " << evalacc_smem_bytes
                          << " bytes > max " << max_smem << ")." << std::endl;
            }
        }

        if (use_pipeline) {
            uint32_t pipelineTpb = fusedNttTpb;
            const uint32_t maxTpb = static_cast<uint32_t>(std::min<size_t>(N / 2, 1024));
            pipelineTpb = std::max<uint32_t>(32, std::min(pipelineTpb, maxTpb));
            pipelineTpb = (pipelineTpb / 32) * 32;
            if (pipelineTpb == 0) {
                pipelineTpb = 32;
            }
            const size_t pipelineShmem = static_cast<size_t>(2) * N * sizeof(NativeInt);
            {
                int dev = 0;
                int max_default = 0;
                int max_opt = 0;
                PHANTOM_CHECK_CUDA(cudaGetDevice(&dev));
                PHANTOM_CHECK_CUDA(cudaDeviceGetAttribute(&max_default, cudaDevAttrMaxSharedMemoryPerBlock, dev));
                (void)cudaDeviceGetAttribute(&max_opt, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
                const size_t max_smem = (max_opt > 0) ? static_cast<size_t>(max_opt) : static_cast<size_t>(max_default);
                if (pipelineShmem > max_smem) {
                    OPENFHE_THROW(config_error, "CGGI_EvalAcc: pipeline requires more shared memory than supported by the device");
                }
                if (pipelineShmem > static_cast<size_t>(max_default)) {
                    const cudaError_t st0 =
                        cudaFuncSetAttribute(kernel_EvalAccPipeline_Binary_MB2_Grouped,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(pipelineShmem));
                    const cudaError_t st1 =
                        cudaFuncSetAttribute(kernel_EvalAccPipeline_Binary_Grouped,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(pipelineShmem));
                    if (st0 != cudaSuccess || st1 != cudaSuccess) {
                        OPENFHE_THROW(config_error,
                                      "CGGI_EvalAcc: failed to opt-in dynamic shared memory for EvalAcc pipeline kernels");
                    }
                }
            }

            for (size_t group = 0; group < rfkey_groups_; ++group) {
                const size_t idx0 = group * 2u;
                const size_t idx1 = idx0 + 1u;
                if (idx0 == 0) {
                    PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_index_event_, 0));
                }
                if (idx1 < n) {
                    kernel_EvalAccPipeline_Binary_MB2_Grouped<<<1, pipelineTpb, pipelineShmem, s>>>(
                        d_acc.get(), d_evalacc_ct_.get(), ek0, ek_shoup0, ek1, ek_shoup1, ek_pair, ek_shoup_pair,
                        d_monic_polys_.get(), d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_twiddles_.get(),
                        d_twiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(),
                        N, baseG, digitsGA, group, idx0, idx1, d_evalacc_indexPos_.get());
                } else {
                    kernel_EvalAccPipeline_Binary_Grouped<<<1, pipelineTpb, pipelineShmem, s>>>(
                        d_acc.get(), d_evalacc_ct_.get(), ek0, ek_shoup0, d_monic_polys_.get(), d_itwiddles_.get(),
                        d_itwiddles_shoup_.get(), d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(),
                        d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N, baseG, digitsGA, group, idx0,
                        d_evalacc_indexPos_.get());
                }
            }
            return;
        }

        const bool canUseGraph = !disableGraph && !cggi_evalacc_pbsmb2_graph_failed_ &&
                                 (d_acc.get() == d_bootstrap_acc_.get());
        if (canUseGraph) {
            if (cggi_evalacc_pbsmb2_graph_exec_ &&
                (cggi_evalacc_pbsmb2_graph_ntt_tpb_ != fusedNttTpb ||
                 cggi_evalacc_pbsmb2_graph_block_x_ != evalaccBlockSize ||
                 cggi_evalacc_pbsmb2_graph_swizzle_ != use_swizzle_path ||
                 cggi_evalacc_pbsmb2_graph_soa_ != use_multibit_soa ||
                 cggi_evalacc_pbsmb2_graph_smem_ != evalacc_smem_enabled)) {
                cudaGraphExecDestroy(cggi_evalacc_pbsmb2_graph_exec_);
                cggi_evalacc_pbsmb2_graph_exec_ = nullptr;
                cggi_evalacc_pbsmb2_graph_acc_  = nullptr;
            }
            if (!cggi_evalacc_pbsmb2_graph_exec_) {
                cggi_evalacc_pbsmb2_graph_acc_ = d_bootstrap_acc_.get();
                cggi_evalacc_pbsmb2_graph_ntt_tpb_ = fusedNttTpb;
                cggi_evalacc_pbsmb2_graph_block_x_ = evalaccBlockSize;
                cggi_evalacc_pbsmb2_graph_swizzle_ = use_swizzle_path;
                cggi_evalacc_pbsmb2_graph_soa_ = use_multibit_soa;
                cggi_evalacc_pbsmb2_graph_smem_ = evalacc_smem_enabled;

                const cudaError_t sync_st = cudaStreamSynchronize(s);
                if (sync_st != cudaSuccess) {
                    cggi_evalacc_pbsmb2_graph_failed_ = true;
                    cggi_evalacc_pbsmb2_graph_acc_ = nullptr;
                } else {
                    cudaGraph_t graph{};
                    cudaError_t cap_st = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
                    if (cap_st != cudaSuccess) {
                        cggi_evalacc_pbsmb2_graph_failed_ = true;
                        cggi_evalacc_pbsmb2_graph_acc_ = nullptr;
                    } else {

                        for (size_t group = 0; group < rfkey_groups_; ++group) {
                            const size_t idx0 = group * 2u;
                            const size_t idx1 = idx0 + 1u;

                            inwt_1d_opt_outofplace_batched(d_evalacc_ct_.get(), cggi_evalacc_pbsmb2_graph_acc_,
                                                           d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(),
                                                           d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N, /*batch=*/2, s);

                            kernel_SignedDigitDecompose2_FusedFNWT<<<digitsG2, fusedNttTpb, evalaccNttShmem, s>>>(
                                d_evalacc_dct_.get(), d_evalacc_ct_.get(), Q, baseG, digitsGA, N, d_twiddles_.get(),
                                d_twiddles_shoup_.get(), d_modulus_.get());

                            if (idx1 < n) {
                                if (use_swizzle_path) {
                                    kernel_EvalAccCore_Binary_MB2_Grouped_Swizzle<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                                        cggi_evalacc_pbsmb2_graph_acc_, d_evalacc_dct_.get(), ek0, ek_shoup0, ek1, ek_shoup1,
                                        ek_pair, ek_shoup_pair, d_monic_polys_.get(), N,
                                        static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(), group, idx0, idx1,
                                        d_evalacc_indexPos_.get());
                                } else if (use_multibit_soa) {
                                    if (evalacc_smem_enabled) {
                                        kernel_EvalAccCore_Binary_MB2_Grouped_SoA_Smem<<<evalaccGrid, evalaccBlock,
                                                                                        evalacc_smem_bytes, s>>>(
                                            cggi_evalacc_pbsmb2_graph_acc_, d_evalacc_dct_.get(), ek0, ek_shoup0, ek1, ek_shoup1,
                                            ek_pair, ek_shoup_pair, d_monic_polys_.get(), N, soa_tiles, d_modulus_.get(),
                                            digitsG2, group, idx0, idx1, d_evalacc_indexPos_.get());
                                    } else {
                                        kernel_EvalAccCore_Binary_MB2_Grouped_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                            cggi_evalacc_pbsmb2_graph_acc_, d_evalacc_dct_.get(), ek0, ek_shoup0, ek1, ek_shoup1,
                                            ek_pair, ek_shoup_pair, d_monic_polys_.get(), N, soa_tiles, d_modulus_.get(),
                                            digitsG2, group, idx0, idx1, d_evalacc_indexPos_.get());
                                    }
                                } else {
                                    kernel_EvalAccCore_Binary_MB2_Grouped<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                        cggi_evalacc_pbsmb2_graph_acc_, d_evalacc_dct_.get(), ek0, ek_shoup0, ek1, ek_shoup1,
                                        ek_pair, ek_shoup_pair, d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, group,
                                        idx0, idx1, d_evalacc_indexPos_.get());
                                }
                            } else {
                                if (use_swizzle_path) {
                                    kernel_EvalAccCore_Binary_Grouped_Swizzle<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                                        cggi_evalacc_pbsmb2_graph_acc_, d_evalacc_dct_.get(), ek0, ek_shoup0, d_monic_polys_.get(),
                                        N, static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(), group, idx0,
                                        d_evalacc_indexPos_.get());
                                } else if (use_multibit_soa) {
                                    if (evalacc_smem_enabled) {
                                        kernel_EvalAccCore_Binary_Grouped_SoA_Smem<<<evalaccGrid, evalaccBlock,
                                                                                    evalacc_smem_bytes, s>>>(
                                            cggi_evalacc_pbsmb2_graph_acc_, d_evalacc_dct_.get(), ek0, ek_shoup0, d_monic_polys_.get(),
                                            N, soa_tiles, d_modulus_.get(), digitsG2, group, idx0, d_evalacc_indexPos_.get());
                                    } else {
                                        kernel_EvalAccCore_Binary_Grouped_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                            cggi_evalacc_pbsmb2_graph_acc_, d_evalacc_dct_.get(), ek0, ek_shoup0, d_monic_polys_.get(),
                                            N, soa_tiles, d_modulus_.get(), digitsG2, group, idx0, d_evalacc_indexPos_.get());
                                    }
                                } else {
                                    kernel_EvalAccCore_Binary_Grouped<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                        cggi_evalacc_pbsmb2_graph_acc_, d_evalacc_dct_.get(), ek0, ek_shoup0, d_monic_polys_.get(),
                                        N, d_modulus_.get(), digitsG2, group, idx0, d_evalacc_indexPos_.get());
                                }
                            }
                        }

                        cap_st = cudaStreamEndCapture(s, &graph);
                        if (cap_st != cudaSuccess || !graph) {
                            cggi_evalacc_pbsmb2_graph_failed_ = true;
                            cggi_evalacc_pbsmb2_graph_acc_ = nullptr;
                            if (graph) {
                                cudaGraphDestroy(graph);
                            }
                        } else {
                            cudaGraphExec_t exec{};
                            const cudaError_t inst = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
                            cudaGraphDestroy(graph);
                            if (inst != cudaSuccess) {
                                cggi_evalacc_pbsmb2_graph_failed_ = true;
                                cggi_evalacc_pbsmb2_graph_acc_ = nullptr;
                            } else {
                                cggi_evalacc_pbsmb2_graph_exec_ = exec;
                            }
                        }
                    }
                }
            }

            if (cggi_evalacc_pbsmb2_graph_exec_) {
                PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_index_event_, 0));
                PHANTOM_CHECK_CUDA(cudaGraphLaunch(cggi_evalacc_pbsmb2_graph_exec_, s));
                return;
            }
        }

        for (size_t group = 0; group < rfkey_groups_; ++group) {
            const size_t idx0 = group * 2u;
            const size_t idx1 = idx0 + 1u;

            inwt_1d_opt_outofplace_batched(d_evalacc_ct_.get(), d_acc.get(), d_itwiddles_.get(), d_itwiddles_shoup_.get(),
                                           d_modulus_.get(), d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N,
                                           /*batch=*/2, s);

            kernel_SignedDigitDecompose2_FusedFNWT<<<digitsG2, fusedNttTpb, evalaccNttShmem, s>>>(
                d_evalacc_dct_.get(), d_evalacc_ct_.get(), Q, baseG, digitsGA, N, d_twiddles_.get(),
                d_twiddles_shoup_.get(), d_modulus_.get());

            if (idx0 == 0) {
                PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_index_event_, 0));
            }

            if (idx1 < n) {
                if (use_swizzle_path) {
                    kernel_EvalAccCore_Binary_MB2_Grouped_Swizzle<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                        d_acc.get(), d_evalacc_dct_.get(), ek0, ek_shoup0, ek1, ek_shoup1, ek_pair, ek_shoup_pair,
                        d_monic_polys_.get(), N, static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(), group,
                        idx0, idx1, d_evalacc_indexPos_.get());
                } else if (use_multibit_soa) {
                    if (evalacc_smem_enabled) {
                        kernel_EvalAccCore_Binary_MB2_Grouped_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem_bytes, s>>>(
                            d_acc.get(), d_evalacc_dct_.get(), ek0, ek_shoup0, ek1, ek_shoup1, ek_pair, ek_shoup_pair,
                            d_monic_polys_.get(), N, soa_tiles, d_modulus_.get(), digitsG2, group, idx0, idx1,
                            d_evalacc_indexPos_.get());
                    } else {
                        kernel_EvalAccCore_Binary_MB2_Grouped_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            d_acc.get(), d_evalacc_dct_.get(), ek0, ek_shoup0, ek1, ek_shoup1, ek_pair, ek_shoup_pair,
                            d_monic_polys_.get(), N, soa_tiles, d_modulus_.get(), digitsG2, group, idx0, idx1,
                            d_evalacc_indexPos_.get());
                    }
                } else {
                    kernel_EvalAccCore_Binary_MB2_Grouped<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        d_acc.get(), d_evalacc_dct_.get(), ek0, ek_shoup0, ek1, ek_shoup1, ek_pair, ek_shoup_pair,
                        d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, group, idx0, idx1,
                        d_evalacc_indexPos_.get());
                }
            } else {
                if (use_swizzle_path) {
                    kernel_EvalAccCore_Binary_Grouped_Swizzle<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                        d_acc.get(), d_evalacc_dct_.get(), ek0, ek_shoup0, d_monic_polys_.get(), N,
                        static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(), group, idx0,
                        d_evalacc_indexPos_.get());
                } else if (use_multibit_soa) {
                    if (evalacc_smem_enabled) {
                        kernel_EvalAccCore_Binary_Grouped_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem_bytes, s>>>(
                            d_acc.get(), d_evalacc_dct_.get(), ek0, ek_shoup0, d_monic_polys_.get(), N, soa_tiles,
                            d_modulus_.get(), digitsG2, group, idx0, d_evalacc_indexPos_.get());
                    } else {
                        kernel_EvalAccCore_Binary_Grouped_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            d_acc.get(), d_evalacc_dct_.get(), ek0, ek_shoup0, d_monic_polys_.get(), N, soa_tiles,
                            d_modulus_.get(), digitsG2, group, idx0, d_evalacc_indexPos_.get());
                    }
                } else {
                    kernel_EvalAccCore_Binary_Grouped<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        d_acc.get(), d_evalacc_dct_.get(), ek0, ek_shoup0, d_monic_polys_.get(), N, d_modulus_.get(),
                        digitsG2, group, idx0, d_evalacc_indexPos_.get());
                }
            }
        }
        return;
    }

    if (use_evalacc_multibit2) {
        if (rfkey_dim2_ < 2) {
            OPENFHE_THROW(config_error, "CGGI_EvalAcc: multibit=2 requires RFkey dim2=2 (set during keygen)");
        }
        if (use_swizzle_path) {
            OPENFHE_THROW(config_error, "CGGI_EvalAcc: multibit=2 does not support swizzle (set CIRBTS_USE_SWIZZLE=0)");
        }
        if (use_pipeline) {
            OPENFHE_THROW(config_error, "CGGI_EvalAcc: multibit=2 does not support the fused pipeline");
        }

        uint32_t evalaccBlockSize = 128;
        if (const char* v = std::getenv("CIRBTS_EVALACC_BLOCK_X"); v && *v) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(v, &end, 10);
            if (end != v && end && *end == '\0' && parsed > 0) {
                evalaccBlockSize = static_cast<uint32_t>(std::min<unsigned long>(parsed, 1024ul));
                evalaccBlockSize = (evalaccBlockSize / 32) * 32;
                if (evalaccBlockSize == 0) {
                    evalaccBlockSize = 32;
                }
            }
        }

        const dim3 evalaccBlock(evalaccBlockSize);
        const dim3 evalaccGrid((N + evalaccBlock.x - 1) / evalaccBlock.x, 1);
        const size_t evalaccNttShmem = N * sizeof(NativeInt);
        const bool use_multibit_soa = use_evalacc_soa;
        const uint32_t soa_tiles = use_multibit_soa ? static_cast<uint32_t>(rfkey_soa_tiles_) : 0u;

        L2PersistScope l2_scope(this, s);
        {
            const NativeInt* key_base = use_multibit_soa ? d_RFkey_soa_.get() : d_RFkey_.get();
            l2_scope.reset(key_base, size_RFkey_ * size_RFRGSWkey_ * sizeof(NativeInt));
        }

        const size_t key_stride = size_RFRGSWkey_;
        const size_t key_block = key_stride * n;
        const NativeInt* ek_base = use_multibit_soa ? d_RFkey_soa_.get() : d_RFkey_.get();
        const NativeInt* ek_pair = ek_base + key_block;
        const NativeInt* ek_shoup_base = use_multibit_soa ? d_RFkey_shoup_soa_.get() : d_RFkey_shoup_.get();
        const NativeInt* ek_shoup_pair = ek_shoup_base + key_block;

        evalacc_smem_enabled = false;
        evalacc_smem_bytes = 0;
        if (use_multibit_soa && evalacc_smem_requested) {
            int dev = 0;
            int max_default = 0;
            int max_opt = 0;
            PHANTOM_CHECK_CUDA(cudaGetDevice(&dev));
            PHANTOM_CHECK_CUDA(cudaDeviceGetAttribute(&max_default, cudaDevAttrMaxSharedMemoryPerBlock, dev));
            (void)cudaDeviceGetAttribute(&max_opt, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            const size_t max_smem = (max_opt > 0) ? static_cast<size_t>(max_opt) : static_cast<size_t>(max_default);
            const uint32_t warps = (evalaccBlockSize + 31u) >> 5;
            const size_t warp_stride = 8u * 32u;
            evalacc_smem_bytes = static_cast<size_t>(warps) * 2u * warp_stride * sizeof(NativeInt);
            if (evalacc_smem_bytes <= max_smem) {
                if (evalacc_smem_bytes > static_cast<size_t>(max_default)) {
                    cudaError_t st = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_MB2_SoA_Smem,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          static_cast<int>(evalacc_smem_bytes));
                    evalacc_smem_enabled = (st == cudaSuccess);
                } else {
                    evalacc_smem_enabled = true;
                }
            } else if (evalacc_smem_verbose) {
                std::cerr << "[CIRBTS] EvalAcc SMEM disabled (requested " << evalacc_smem_bytes
                          << " bytes > max " << max_smem << ")." << std::endl;
            }
        }

        size_t lweIndex = 0;
        const bool canUseGraph = !disableGraph && !cggi_evalacc_mb2_graph_failed_ && (d_acc.get() == d_bootstrap_acc_.get());
        if (canUseGraph) {
            if (cggi_evalacc_mb2_graph_exec_ &&
                (cggi_evalacc_mb2_graph_ntt_tpb_ != fusedNttTpb || cggi_evalacc_mb2_graph_block_x_ != evalaccBlockSize ||
                 cggi_evalacc_mb2_graph_soa_ != use_multibit_soa || cggi_evalacc_mb2_graph_smem_ != evalacc_smem_enabled)) {
                cudaGraphExecDestroy(cggi_evalacc_mb2_graph_exec_);
                cggi_evalacc_mb2_graph_exec_ = nullptr;
                cggi_evalacc_mb2_graph_acc_  = nullptr;
            }
            if (!cggi_evalacc_mb2_graph_exec_) {
                cggi_evalacc_mb2_graph_acc_ = d_bootstrap_acc_.get();
                cggi_evalacc_mb2_graph_ntt_tpb_ = fusedNttTpb;
                cggi_evalacc_mb2_graph_block_x_ = evalaccBlockSize;
                cggi_evalacc_mb2_graph_soa_ = use_multibit_soa;
                cggi_evalacc_mb2_graph_smem_ = evalacc_smem_enabled;

                const cudaError_t sync_st = cudaStreamSynchronize(s);
                if (sync_st != cudaSuccess) {
                    cggi_evalacc_mb2_graph_failed_ = true;
                    cggi_evalacc_mb2_graph_acc_ = nullptr;
                } else {
                    cudaGraph_t graph{};
                    cudaError_t cap_st = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
                    if (cap_st != cudaSuccess) {
                        cggi_evalacc_mb2_graph_failed_ = true;
                        cggi_evalacc_mb2_graph_acc_ = nullptr;
                    } else {

                        const size_t smem = evalacc_smem_enabled ? evalacc_smem_bytes : 0;
                        for (size_t i = 0; i + 1 < n; i += 2) {
                            inwt_1d_opt_outofplace_batched(d_evalacc_ct_.get(), cggi_evalacc_mb2_graph_acc_, d_itwiddles_.get(),
                                                           d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                                           d_n_inv_mod_q_shoup_.get(), N, /*batch=*/2, s);

                            kernel_SignedDigitDecompose2_FusedFNWT<<<digitsG2, fusedNttTpb, evalaccNttShmem, s>>>(
                                d_evalacc_dct_.get(), d_evalacc_ct_.get(), Q, baseG, digitsGA, N, d_twiddles_.get(),
                                d_twiddles_shoup_.get(), d_modulus_.get());

                            if (use_multibit_soa) {
                                if (evalacc_smem_enabled) {
                                    kernel_EvalAccCore_Binary_MB2_SoA_Smem<<<evalaccGrid, evalaccBlock, smem, s>>>(
                                        cggi_evalacc_mb2_graph_acc_, d_evalacc_dct_.get(), ek_base, ek_shoup_base, ek_pair,
                                        ek_shoup_pair, d_monic_polys_.get(), N, soa_tiles, d_modulus_.get(), digitsG2, i, i + 1,
                                        d_evalacc_indexPos_.get());
                                } else {
                                    kernel_EvalAccCore_Binary_MB2_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                        cggi_evalacc_mb2_graph_acc_, d_evalacc_dct_.get(), ek_base, ek_shoup_base, ek_pair,
                                        ek_shoup_pair, d_monic_polys_.get(), N, soa_tiles, d_modulus_.get(), digitsG2, i, i + 1,
                                        d_evalacc_indexPos_.get());
                                }
                            } else {
                                kernel_EvalAccCore_Binary_MB2<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                    cggi_evalacc_mb2_graph_acc_, d_evalacc_dct_.get(), ek_base, ek_shoup_base, ek_pair,
                                    ek_shoup_pair, d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, i, i + 1,
                                    d_evalacc_indexPos_.get());
                            }
                        }

                        if ((n & 1u) != 0u) {
                            const size_t i = n - 1;
                            inwt_1d_opt_outofplace_batched(d_evalacc_ct_.get(), cggi_evalacc_mb2_graph_acc_, d_itwiddles_.get(),
                                                           d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                                           d_n_inv_mod_q_shoup_.get(), N, /*batch=*/2, s);

                            kernel_SignedDigitDecompose2_FusedFNWT<<<digitsG2, fusedNttTpb, evalaccNttShmem, s>>>(
                                d_evalacc_dct_.get(), d_evalacc_ct_.get(), Q, baseG, digitsGA, N, d_twiddles_.get(),
                                d_twiddles_shoup_.get(), d_modulus_.get());

                            if (use_multibit_soa) {
                                kernel_EvalAccCore_Binary_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                    cggi_evalacc_mb2_graph_acc_, d_evalacc_dct_.get(), ek_base, ek_shoup_base,
                                    d_monic_polys_.get(), N, soa_tiles, d_modulus_.get(), digitsG2, i, d_evalacc_indexPos_.get());
                            } else {
                                kernel_EvalAccCore_Binary<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                    cggi_evalacc_mb2_graph_acc_, d_evalacc_dct_.get(), ek_base, ek_shoup_base, d_monic_polys_.get(),
                                    N, d_modulus_.get(), digitsG2, i, d_evalacc_indexPos_.get());
                            }
                        }

                        cap_st = cudaStreamEndCapture(s, &graph);
                        if (cap_st != cudaSuccess || !graph) {
                            cggi_evalacc_mb2_graph_failed_ = true;
                            cggi_evalacc_mb2_graph_acc_ = nullptr;
                            if (graph) {
                                cudaGraphDestroy(graph);
                            }
                        } else {
                            cudaGraphExec_t exec{};
                            const cudaError_t inst = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
                            cudaGraphDestroy(graph);
                            if (inst != cudaSuccess) {
                                cggi_evalacc_mb2_graph_failed_ = true;
                                cggi_evalacc_mb2_graph_acc_ = nullptr;
                            } else {
                                cggi_evalacc_mb2_graph_exec_ = exec;
                            }
                        }
                    }
                }
            }

            if (cggi_evalacc_mb2_graph_exec_) {
                PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_index_event_, 0));
                PHANTOM_CHECK_CUDA(cudaGraphLaunch(cggi_evalacc_mb2_graph_exec_, s));
                return;
            }
        }
        for (; lweIndex + 1 < n; lweIndex += 2) {
            inwt_1d_opt_outofplace_batched(d_evalacc_ct_.get(), d_acc.get(), d_itwiddles_.get(), d_itwiddles_shoup_.get(),
                                           d_modulus_.get(), d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N,
                                           /*batch=*/2, s);

            kernel_SignedDigitDecompose2_FusedFNWT<<<digitsG2, fusedNttTpb, evalaccNttShmem, s>>>(
                d_evalacc_dct_.get(), d_evalacc_ct_.get(), Q, baseG, digitsGA, N, d_twiddles_.get(),
                d_twiddles_shoup_.get(), d_modulus_.get());

            if (lweIndex == 0) {
                PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_index_event_, 0));
            }

            if (use_multibit_soa) {
                if (evalacc_smem_enabled) {
                    kernel_EvalAccCore_Binary_MB2_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem_bytes, s>>>(
                        d_acc.get(), d_evalacc_dct_.get(), ek_base, ek_shoup_base, ek_pair, ek_shoup_pair,
                        d_monic_polys_.get(), N, soa_tiles, d_modulus_.get(), digitsG2, lweIndex, lweIndex + 1,
                        d_evalacc_indexPos_.get());
                } else {
                    kernel_EvalAccCore_Binary_MB2_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        d_acc.get(), d_evalacc_dct_.get(), ek_base, ek_shoup_base, ek_pair, ek_shoup_pair,
                        d_monic_polys_.get(), N, soa_tiles, d_modulus_.get(), digitsG2, lweIndex, lweIndex + 1,
                        d_evalacc_indexPos_.get());
                }
            } else {
                kernel_EvalAccCore_Binary_MB2<<<evalaccGrid, evalaccBlock, 0, s>>>(
                    d_acc.get(), d_evalacc_dct_.get(), ek_base, ek_shoup_base, ek_pair, ek_shoup_pair, d_monic_polys_.get(),
                    N, d_modulus_.get(), digitsG2, lweIndex, lweIndex + 1, d_evalacc_indexPos_.get());
            }
        }

        if (lweIndex < n) {
            inwt_1d_opt_outofplace_batched(d_evalacc_ct_.get(), d_acc.get(), d_itwiddles_.get(), d_itwiddles_shoup_.get(),
                                           d_modulus_.get(), d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N,
                                           /*batch=*/2, s);

            kernel_SignedDigitDecompose2_FusedFNWT<<<digitsG2, fusedNttTpb, evalaccNttShmem, s>>>(
                d_evalacc_dct_.get(), d_evalacc_ct_.get(), Q, baseG, digitsGA, N, d_twiddles_.get(),
                d_twiddles_shoup_.get(), d_modulus_.get());

            if (lweIndex == 0) {
                PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_index_event_, 0));
            }
            if (use_multibit_soa) {
                kernel_EvalAccCore_Binary_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                    d_acc.get(), d_evalacc_dct_.get(), ek_base, ek_shoup_base, d_monic_polys_.get(), N, soa_tiles,
                    d_modulus_.get(), digitsG2, lweIndex, d_evalacc_indexPos_.get());
            } else {
                kernel_EvalAccCore_Binary<<<evalaccGrid, evalaccBlock, 0, s>>>(
                    d_acc.get(), d_evalacc_dct_.get(), ek_base, ek_shoup_base, d_monic_polys_.get(), N, d_modulus_.get(),
                    digitsG2, lweIndex, d_evalacc_indexPos_.get());
            }
        }
        return;
    }

    L2PersistScope l2_scope(this, s);
    {
        const NativeInt* key_base = use_evalacc_soa ? d_RFkey_soa_.get() :
                                    (use_swizzle_path ? d_RFkey_swizzle_.get() : d_RFkey_.get());
        l2_scope.reset(key_base, size_RFkey_ * size_RFRGSWkey_ * sizeof(NativeInt));
    }

    if (use_pipeline && !use_swizzle_path) {
        uint32_t pipelineTpb = fusedNttTpb;
        const uint32_t maxTpb = static_cast<uint32_t>(std::min<size_t>(N / 2, 1024));
        pipelineTpb = std::max<uint32_t>(32, std::min(pipelineTpb, maxTpb));
        pipelineTpb = (pipelineTpb / 32) * 32;
        if (pipelineTpb == 0) {
            pipelineTpb = 32;
        }
        const size_t pipelineShmem = static_cast<size_t>(2) * N * sizeof(NativeInt);
        {
            int dev = 0;
            int max_default = 0;
            int max_opt = 0;
            PHANTOM_CHECK_CUDA(cudaGetDevice(&dev));
            PHANTOM_CHECK_CUDA(cudaDeviceGetAttribute(&max_default, cudaDevAttrMaxSharedMemoryPerBlock, dev));
            (void)cudaDeviceGetAttribute(&max_opt, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            const size_t max_smem = (max_opt > 0) ? static_cast<size_t>(max_opt) : static_cast<size_t>(max_default);
            if (pipelineShmem > max_smem) {
                OPENFHE_THROW(config_error, "CGGI_EvalAcc: pipeline requires more shared memory than supported by the device");
            }
            if (pipelineShmem > static_cast<size_t>(max_default)) {
                const cudaError_t st =
                    cudaFuncSetAttribute(kernel_EvalAccPipeline_Binary_GINX,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(pipelineShmem));
                if (st != cudaSuccess) {
                    OPENFHE_THROW(config_error,
                                  "CGGI_EvalAcc: failed to opt-in dynamic shared memory for EvalAcc pipeline kernel");
                }
            }
        }

        for (size_t lweIndex = 0; lweIndex < n; ++lweIndex) {
            if (lweIndex == 0) {
                PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_index_event_, 0));
            }
            kernel_EvalAccPipeline_Binary_GINX<<<1, pipelineTpb, pipelineShmem, s>>>(
                d_acc.get(), d_evalacc_ct_.get(), d_RFkey_.get(), d_RFkey_shoup_.get(), d_monic_polys_.get(),
                d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_twiddles_.get(), d_twiddles_shoup_.get(),
                d_modulus_.get(), d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N, baseG, digitsGA, lweIndex,
                d_evalacc_indexPos_.get());
        }
        return;
    }

    bool evalaccBlockOverride = false;
    uint32_t evalaccBlockSize = 128;
    if (const char* v = std::getenv("CIRBTS_EVALACC_BLOCK_X"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0' && parsed > 0) {
            evalaccBlockSize = static_cast<uint32_t>(std::min<unsigned long>(parsed, 1024ul));
            evalaccBlockSize = (evalaccBlockSize / 32) * 32;
            if (evalaccBlockSize == 0) {
                evalaccBlockSize = 32;
            }
            evalaccBlockOverride = true;
        }
    }

    const bool autotune_evalacc = (std::getenv("CIRBTS_AUTOTUNE_EVALACC") != nullptr);
    if (autotune_evalacc && !evalacc_autotuned_ && !fusedNttOverride && !evalaccBlockOverride) {
        const uint32_t testIters = std::min<uint32_t>(static_cast<uint32_t>(n), 8u);
        const std::array<uint32_t, 4> nttCandidates{128u, 256u, 512u, static_cast<uint32_t>(std::min<size_t>(N / 2, 1024))};
        const std::array<uint32_t, 3> blockCandidates{64u, 128u, 256u};
        float bestMs = std::numeric_limits<float>::infinity();
        uint32_t bestNtt = fusedNttTpb;
        uint32_t bestBlock = evalaccBlockSize;

        const size_t accSize = static_cast<size_t>(2) * N;
        auto acc_backup = phantom::util::make_cuda_auto_ptr<NativeInt>(accSize, s);
        cudaMemcpyAsync(acc_backup.get(), d_acc.get(), accSize * sizeof(NativeInt), cudaMemcpyDeviceToDevice, s);
        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));

        cudaEvent_t start{}, stop{};
        PHANTOM_CHECK_CUDA(cudaEventCreate(&start));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&stop));

        const size_t evalaccNttShmem = N * sizeof(NativeInt);
        for (uint32_t candNtt : nttCandidates) {
            if (candNtt == 0) {
                continue;
            }
            for (uint32_t candBlock : blockCandidates) {
                cudaMemcpyAsync(d_acc.get(), acc_backup.get(), accSize * sizeof(NativeInt), cudaMemcpyDeviceToDevice, s);
                PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));

                dim3 block(candBlock);
                dim3 grid((N + block.x - 1) / block.x, 1);

                PHANTOM_CHECK_CUDA(cudaEventRecord(start, s));
                for (uint32_t lweIndex = 0; lweIndex < testIters; ++lweIndex) {
                    inwt_1d_opt_outofplace_batched(d_evalacc_ct_.get(), d_acc.get(), d_itwiddles_.get(), d_itwiddles_shoup_.get(),
                                                   d_modulus_.get(), d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N,
                                                   /*batch=*/2, s);

                    kernel_SignedDigitDecompose2_FusedFNWT<<<digitsG2, candNtt, evalaccNttShmem, s>>>(
                        d_evalacc_dct_.get(), d_evalacc_ct_.get(), Q, baseG, digitsGA, N, d_twiddles_.get(),
                        d_twiddles_shoup_.get(), d_modulus_.get());

                    if (use_evalacc_soa) {
                        launch_evalacc_soa(grid, block, d_acc.get(), lweIndex);
                    } else if (use_swizzle_path) {
                        kernel_EvalAccCore_Binary_Swizzle<2><<<grid, block, 0, s>>>(
                            d_acc.get(), d_evalacc_dct_.get(), d_RFkey_swizzle_.get(), d_RFkey_shoup_swizzle_.get(),
                            d_monic_polys_.get(), N, static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(), lweIndex,
                            d_evalacc_indexPos_.get());
                    } else {
                        kernel_EvalAccCore_Binary<<<grid, block, 0, s>>>(d_acc.get(), d_evalacc_dct_.get(), d_RFkey_.get(),
                                                                         d_RFkey_shoup_.get(), d_monic_polys_.get(), N,
                                                                         d_modulus_.get(), digitsG2, lweIndex, d_evalacc_indexPos_.get());
                    }
                }
                PHANTOM_CHECK_CUDA(cudaEventRecord(stop, s));
                PHANTOM_CHECK_CUDA(cudaEventSynchronize(stop));

                float ms = 0.0f;
                PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
                if (ms < bestMs) {
                    bestMs = ms;
                    bestNtt = candNtt;
                    bestBlock = candBlock;
                }
            }
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpyAsync(d_acc.get(), acc_backup.get(), accSize * sizeof(NativeInt), cudaMemcpyDeviceToDevice, s);
        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));

        evalacc_ntt_tpb_ = bestNtt;
        evalacc_block_x_ = bestBlock;
        evalacc_autotuned_ = true;
    }

    if (evalacc_autotuned_ && !fusedNttOverride && !evalaccBlockOverride) {
        fusedNttTpb = evalacc_ntt_tpb_;
        evalaccBlockSize = evalacc_block_x_;
    }

    if (evalacc_smem_requested && use_evalacc_soa) {
        int dev = 0;
        int max_default = 0;
        int max_opt = 0;
        PHANTOM_CHECK_CUDA(cudaGetDevice(&dev));
        PHANTOM_CHECK_CUDA(cudaDeviceGetAttribute(&max_default, cudaDevAttrMaxSharedMemoryPerBlock, dev));
        (void)cudaDeviceGetAttribute(&max_opt, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        const size_t max_smem = (max_opt > 0) ? static_cast<size_t>(max_opt) : static_cast<size_t>(max_default);
        const uint32_t warps = (evalaccBlockSize + 31u) >> 5;
        evalacc_smem_bytes = static_cast<size_t>(warps) * 2u * 4u * 32u * sizeof(NativeInt);
        if (evalacc_smem_bytes <= max_smem) {
            if (evalacc_smem_bytes > static_cast<size_t>(max_default)) {
                cudaError_t st = cudaSuccess;
                if (digitsG2 == 2) {
                    st = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_SoA_T_Smem<2>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(evalacc_smem_bytes));
                } else if (digitsG2 == 4) {
                    st = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_SoA_T_Smem<4>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(evalacc_smem_bytes));
                } else {
                    st = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_SoA_Smem,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(evalacc_smem_bytes));
                }
                evalacc_smem_enabled = (st == cudaSuccess);
            } else {
                evalacc_smem_enabled = true;
            }
        } else if (evalacc_smem_verbose) {
            std::cerr << "[CIRBTS] EvalAcc SMEM disabled (requested " << evalacc_smem_bytes
                      << " bytes > max " << max_smem << ")." << std::endl;
        }
    }

    const bool canUseGraph  = !disableGraph && !cggi_evalacc_graph_failed_ && (d_acc.get() == d_bootstrap_acc_.get());
    const dim3 evalaccBlock(evalaccBlockSize);
    const dim3 evalaccGrid((N + evalaccBlock.x - 1) / evalaccBlock.x, 1);
    const size_t evalaccNttShmem = N * sizeof(NativeInt);

    if (canUseGraph) {
        if (cggi_evalacc_graph_exec_ &&
            (cggi_evalacc_graph_ntt_tpb_ != fusedNttTpb || cggi_evalacc_graph_block_x_ != evalaccBlockSize ||
             cggi_evalacc_graph_swizzle_ != use_swizzle_path || cggi_evalacc_graph_soa_ != use_evalacc_soa ||
             cggi_evalacc_graph_smem_ != evalacc_smem_enabled)) {
            cudaGraphExecDestroy(cggi_evalacc_graph_exec_);
            cggi_evalacc_graph_exec_ = nullptr;
            cggi_evalacc_graph_acc_  = nullptr;
        }
        if (!cggi_evalacc_graph_exec_) {
            cggi_evalacc_graph_acc_ = d_bootstrap_acc_.get();
            cggi_evalacc_graph_ntt_tpb_ = fusedNttTpb;
            cggi_evalacc_graph_block_x_ = evalaccBlockSize;
            cggi_evalacc_graph_swizzle_ = use_swizzle_path;
            cggi_evalacc_graph_soa_ = use_evalacc_soa;
            cggi_evalacc_graph_smem_ = evalacc_smem_enabled;

            // Build graph once; subsequent launches reuse it with updated d_evalacc_indexPos_ content.
            const cudaError_t sync_st = cudaStreamSynchronize(s);
            if (sync_st != cudaSuccess) {
                cggi_evalacc_graph_failed_ = true;
                cggi_evalacc_graph_acc_    = nullptr;
            } else {
                cudaGraph_t graph{};
                cudaError_t cap_st = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
                if (cap_st != cudaSuccess) {
                    cggi_evalacc_graph_failed_ = true;
                    cggi_evalacc_graph_acc_    = nullptr;
                } else {

                    for (size_t lweIndex = 0; lweIndex < n; ++lweIndex) {
                        inwt_1d_opt_outofplace_batched(d_evalacc_ct_.get(), cggi_evalacc_graph_acc_, d_itwiddles_.get(),
                                                       d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                                       d_n_inv_mod_q_shoup_.get(), N, /*batch=*/2, s);

                        kernel_SignedDigitDecompose2_FusedFNWT<<<digitsG2, fusedNttTpb, evalaccNttShmem, s>>>(
                            d_evalacc_dct_.get(), d_evalacc_ct_.get(), Q, baseG, digitsGA, N, d_twiddles_.get(),
                            d_twiddles_shoup_.get(), d_modulus_.get());

                        if (use_evalacc_soa) {
                            launch_evalacc_soa(evalaccGrid, evalaccBlock, cggi_evalacc_graph_acc_, lweIndex);
                        } else if (use_swizzle_path) {
                            kernel_EvalAccCore_Binary_Swizzle<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                                cggi_evalacc_graph_acc_, d_evalacc_dct_.get(), d_RFkey_swizzle_.get(),
                                d_RFkey_shoup_swizzle_.get(), d_monic_polys_.get(), N,
                                static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(), lweIndex,
                                d_evalacc_indexPos_.get());
                        } else {
                            kernel_EvalAccCore_Binary<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                cggi_evalacc_graph_acc_, d_evalacc_dct_.get(), d_RFkey_.get(), d_RFkey_shoup_.get(),
                                d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, lweIndex, d_evalacc_indexPos_.get());
                        }
                    }

                    cap_st = cudaStreamEndCapture(s, &graph);
                    if (cap_st != cudaSuccess || !graph) {
                        cggi_evalacc_graph_failed_ = true;
                        cggi_evalacc_graph_acc_    = nullptr;
                        if (graph) {
                            cudaGraphDestroy(graph);
                        }
                    } else {
                        cudaGraphExec_t exec{};
                        const cudaError_t inst = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
                        cudaGraphDestroy(graph);
                        if (inst != cudaSuccess) {
                            cggi_evalacc_graph_failed_ = true;
                            cggi_evalacc_graph_acc_    = nullptr;
                        } else {
                            cggi_evalacc_graph_exec_ = exec;
                        }
                    }
                }
            }
        }

        if (cggi_evalacc_graph_exec_) {
            PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_index_event_, 0));
            PHANTOM_CHECK_CUDA(cudaGraphLaunch(cggi_evalacc_graph_exec_, s));
            return;
        }
    }

    // Fallback (no CUDA graph): keep using scratch buffers to avoid per-call allocations.
    for (size_t lweIndex = 0; lweIndex < n; ++lweIndex) {
        inwt_1d_opt_outofplace_batched(d_evalacc_ct_.get(), d_acc.get(), d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(),
                                       d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N, /*batch=*/2, s);

        kernel_SignedDigitDecompose2_FusedFNWT<<<digitsG2, fusedNttTpb, evalaccNttShmem, s>>>(
            d_evalacc_dct_.get(), d_evalacc_ct_.get(), Q, baseG, digitsGA, N, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());

        if (lweIndex == 0) {
            PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, h2d_index_event_, 0));
        }
        if (use_evalacc_soa) {
            launch_evalacc_soa(evalaccGrid, evalaccBlock, d_acc.get(), lweIndex);
        } else if (use_swizzle_path) {
            kernel_EvalAccCore_Binary_Swizzle<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                d_acc.get(), d_evalacc_dct_.get(), d_RFkey_swizzle_.get(), d_RFkey_shoup_swizzle_.get(), d_monic_polys_.get(),
                N, static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(), lweIndex, d_evalacc_indexPos_.get());
        } else {
            kernel_EvalAccCore_Binary<<<evalaccGrid, evalaccBlock, 0, s>>>(
                d_acc.get(), d_evalacc_dct_.get(), d_RFkey_.get(), d_RFkey_shoup_.get(), d_monic_polys_.get(), N,
                d_modulus_.get(), digitsG2, lweIndex, d_evalacc_indexPos_.get());
        }
    }
}
