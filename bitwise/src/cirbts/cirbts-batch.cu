/*
 * =============================================================================
 * File: cirbts-batch.cu
 * Purpose: Batched circuit bootstrapping for throughput, with per-stream
 *          workspaces and optional sub-batching to overlap H2D and compute.
 * Key parameters:
 *   - batch size and batch_streams (sub-batch count).
 *   - graph enable flags for EvalAcc and HT/SS.
 *   - per-stream scratch buffers for batched pipelines.
 * Key points:
 *   - Focused on throughput and overlap rather than single-call latency.
 *   - Uses copy stream + events to stage indices/monomials asynchronously.
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

inline void GetHtssFuseFlags(bool* fuse_inwt_decomp, bool* fuse_decomp_fnwt) {
    const bool env_inwt = (std::getenv("CIRBTS_HTSS_FUSE_INWT_DECOMP") != nullptr);
    const bool env_fnwt = (std::getenv("CIRBTS_HTSS_FUSE_DECOMP_FNWT") != nullptr);
    const bool campaign = (std::getenv("CIRBTS_BATCH_CAMPAIGN") != nullptr);
    if (env_inwt) {
        *fuse_inwt_decomp = true;
        *fuse_decomp_fnwt = false;
    } else if (env_fnwt) {
        *fuse_inwt_decomp = false;
        *fuse_decomp_fnwt = true;
    } else if (campaign) {
        *fuse_inwt_decomp = false;
        *fuse_decomp_fnwt = true;
    } else {
        *fuse_inwt_decomp = false;
        *fuse_decomp_fnwt = false;
    }
}
}  // namespace

std::vector<RGSWCiphertext> GPUCirBTSContext::gpu_CircuitBootstrappingBatch(
    const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params,
    const lbcrypto::RingGSWCirBTKey& ek,
    const std::vector<lbcrypto::LWECiphertext>& cts) const {
    if (cts.empty()) {
        return {};
    }

    const auto method = params->GetRingGSWParams1()->GetMethod();
    if (method != BINFHE_METHOD::GINX) {
        std::ostringstream oss;
        oss << "gpu_CircuitBootstrappingBatch (GPU) supports only method=GINX; got method=" << method;
        OPENFHE_THROW(config_error, oss.str());
    }

    const uint32_t batch = static_cast<uint32_t>(cts.size());
    if (batch > std::max<uint32_t>(1u, max_batch_size_)) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatch: batch size exceeds max_batch_size_ (recreate GPUCirBTSContext)");
    }

    const uint32_t numLUT = params->GetDigitsCC();
    const size_t N        = params->GetRingGSWParams1()->GetN();
    const uint32_t n      = params->GetLWEParams()->Getn();
    const NativeInt Q     = params->GetRingGSWParams1()->GetQ().ConvertToInt();
    if (N != ring_dim_ || numLUT != num_luts_) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatch: GPU context params mismatch (recreate GPUCirBTSContext)");
    }
    if (!d_auto_maps_.get() || log_ring_dim_ == 0) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatch: automorphism maps are not initialized on GPU");
    }
    if (!d_gpow_.get()) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatch: Gpow is not initialized on GPU");
    }
    if (numLUT > 1 && !d_monomials_.get()) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatch: monomials are not initialized on GPU");
    }

    const uint64_t bitwidth = static_cast<uint64_t>(std::ceil(std::log2(static_cast<double>(numLUT))));
    auto& s                = stream_wrapper_.get_stream();
    const bool profile     = (std::getenv("CIRBTS_PROFILE") != nullptr);
    const auto& rlweParams = params->GetRLWEParams();
    const uint32_t traceShift = rlweParams->GetTraceShift();
    if (traceShift >= log_ring_dim_) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatch: TraceShift must be smaller than log2(N)");
    }
    const uint32_t traceRounds = log_ring_dim_ - traceShift;
    auto polyparams        = rlweParams->GetPolyParams();
    const uint32_t baseHT  = rlweParams->GetBaseHT();
    const uint32_t baseSS  = rlweParams->GetBaseSS();
    const uint32_t digitsHT = rlweParams->GetDigitsHTA();
    const uint32_t digitsSS = rlweParams->GetDigitsSSA();
    const bool htss_soa_requested = (std::getenv("CIRBTS_HTSS_SOA") != nullptr);
    const bool htss_smem_requested = (std::getenv("CIRBTS_HTSS_SMEM") != nullptr);
    const bool htss_smem_verbose = (std::getenv("CIRBTS_HTSS_SMEM_VERBOSE") != nullptr);
    const bool htss_pipeline      = (std::getenv("CIRBTS_HTSS_PIPELINE") != nullptr);
    const bool has_htss_soa = d_HTkey_soa_.get() && d_HTkey_shoup_soa_.get() && d_SSkey_soa_.get() && d_SSkey_shoup_soa_.get() &&
                              (htkey_soa_tiles_ > 0) && (sskey_soa_tiles_ > 0);
    if ((htss_soa_requested || htss_smem_requested) && !has_htss_soa) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatch: HT/SS SoA requested but not initialized (set CIRBTS_HTSS_SOA before init)");
    }
    if (htss_pipeline && (baseHT != baseSS || digitsHT != digitsSS)) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatch: HT/SS pipeline requires baseHT==baseSS and digitsHT==digitsSS");
    }
    if (htss_pipeline && std::getenv("CIRBTS_HTSS_PIPELINE_VERBOSE")) {
        std::cout << "[CIRBTS] HT/SS pipeline enabled (experimental, skips Step10)." << std::endl;
    }

    uint32_t htss_block_x = 256;
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

    uint32_t batch_streams = 1;
    bool batch_streams_explicit = false;
    if (const char* v = std::getenv("CIRBTS_BATCH_STREAMS"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0' && parsed > 0) {
            batch_streams = static_cast<uint32_t>(parsed);
            batch_streams_explicit = true;
        }
    }
    const bool batch_campaign = (std::getenv("CIRBTS_BATCH_CAMPAIGN") != nullptr);
    const bool batch_streams_auto = (!batch_streams_explicit &&
                                     (batch_campaign || (std::getenv("CIRBTS_AUTOTUNE_BATCH_STREAMS") != nullptr)));
    if (batch_streams_auto) {
        uint32_t target = 128;
        if (const char* v = std::getenv("CIRBTS_AUTOTUNE_SUBBATCH"); v && *v) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(v, &end, 10);
            if (end != v && end && *end == '\0' && parsed > 0) {
                target = static_cast<uint32_t>(parsed);
            }
        }
        if (target > 0 && target < batch) {
            const uint32_t needed = (batch + target - 1) / target;
            batch_streams = std::max(1u, std::min(batch, needed));
        }
    }
    if (!batch_streams_explicit && !batch_streams_auto) {
        if (const char* v = std::getenv("CIRBTS_PIPELINE_SUBBATCH"); v && *v) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(v, &end, 10);
            if (end != v && end && *end == '\0' && parsed > 0) {
                const uint32_t sub_batch = static_cast<uint32_t>(parsed);
                if (sub_batch > 0 && sub_batch < batch) {
                    const uint32_t needed = (batch + sub_batch - 1) / sub_batch;
                    batch_streams = std::max(1u, std::min(batch, needed));
                }
            }
        }
    }
    if (batch_streams > batch) {
        batch_streams = batch;
    }

    if (batch_streams > 1) {
        ensure_batch_workspaces(batch_streams - 1);

        const uint64_t numRowsPer  = static_cast<uint64_t>(numLUT) * 2;
        bool htssFuseInwtDecomp = false;
        bool htssFuseDecompFnwt = false;
        GetHtssFuseFlags(&htssFuseInwtDecomp, &htssFuseDecompFnwt);

        auto ensure_pinned_buffer = [&](NativeInt*& ptr, size_t& capacity, size_t elems) {
            if (capacity >= elems && ptr) {
                return ptr;
            }
            if (ptr) {
                PHANTOM_CHECK_CUDA(cudaFreeHost(ptr));
                ptr = nullptr;
            }
            PHANTOM_CHECK_CUDA(cudaMallocHost(&ptr, elems * sizeof(NativeInt)));
            capacity = elems;
            return ptr;
        };

        auto ensure_h2d_buffers = [&](NativeInt*& mono_ptr, size_t& mono_cap, uint32_t*& idx_ptr, size_t& idx_cap,
                                      size_t mono_elems, size_t idx_elems) {
            if (!mono_ptr || mono_cap < mono_elems) {
                if (mono_ptr) {
                    PHANTOM_CHECK_CUDA(cudaFreeHost(mono_ptr));
                }
                PHANTOM_CHECK_CUDA(cudaMallocHost(&mono_ptr, mono_elems * sizeof(NativeInt)));
                mono_cap = mono_elems;
            }
            if (!idx_ptr || idx_cap < idx_elems) {
                if (idx_ptr) {
                    PHANTOM_CHECK_CUDA(cudaFreeHost(idx_ptr));
                }
                PHANTOM_CHECK_CUDA(cudaMallocHost(&idx_ptr, idx_elems * sizeof(uint32_t)));
                idx_cap = idx_elems;
            }
        };

        ensure_h2d_events();

        struct SubBatchJob {
            uint32_t offset{};
            uint32_t batch{};
            NativeInt* h_rgsw{};
            size_t h_rgsw_elems{};
            cudaEvent_t d2h_done{};
        };

        std::vector<SubBatchJob> jobs;
        jobs.reserve(batch_streams);

        uint32_t offset = 0;
        for (uint32_t i = 0; i < batch_streams; ++i) {
            const uint32_t sub_batch = batch / batch_streams + ((i < (batch % batch_streams)) ? 1u : 0u);
            if (sub_batch == 0) {
                continue;
            }
            const size_t mono_elems = static_cast<size_t>(sub_batch) * N;
            const size_t idx_elems  = static_cast<size_t>(sub_batch) * n;
            if (i == 0) {
                ensure_h2d_buffers(h_monomial_inv_pinned_, h_monomial_inv_pinned_capacity_, h_indexPos_pinned_,
                                   h_indexPos_pinned_capacity_, mono_elems, idx_elems);
            } else {
                auto& ws = *extra_workspaces_[i - 1];
                ensure_h2d_buffers(ws.h_monomial_inv_pinned, ws.h_monomial_inv_pinned_capacity, ws.h_indexPos_pinned,
                                   ws.h_indexPos_pinned_capacity, mono_elems, idx_elems);
            }

            BatchScratchView scratch = (i == 0) ? make_batch_scratch_view() : make_batch_scratch_view(*extra_workspaces_[i - 1]);
            const uint32_t numLUTTotal = numLUT * sub_batch;
            const size_t lutSize       = static_cast<size_t>(numLUTTotal) * N;
            if (scratch.scratch_lut_size < lutSize) {
                OPENFHE_THROW(config_error,
                              "gpu_CircuitBootstrappingBatch: scratch buffer too small for sub-batch (recreate GPUCirBTSContext)");
            }
            if (scratch.scratch_digits_max < std::max(digitsHT, digitsSS)) {
                OPENFHE_THROW(config_error,
                              "gpu_CircuitBootstrappingBatch: scratch digits mismatch for sub-batch (recreate GPUCirBTSContext)");
            }

            std::vector<lbcrypto::LWECiphertext> sub_cts;
            sub_cts.reserve(sub_batch);
            for (uint32_t b = 0; b < sub_batch; ++b) {
                sub_cts.emplace_back(cts[static_cast<size_t>(offset + b)]);
            }

            gpu_BootstrapLUT_inplace_batch(params, ek.RFkey, sub_cts, bitwidth, scratch, sub_batch);

            const NativeInt* d_acc = scratch.d_bootstrap_acc;
            {
                const size_t total = lutSize * 2;
                constexpr int blockSize = 256;
                const int numBlocks = static_cast<int>((total + blockSize - 1) / blockSize);
                kernel_GenMVRLWEs<<<numBlocks, blockSize, 0, scratch.stream>>>(scratch.d_ht_c0, scratch.d_ht_c1, d_acc,
                                                                              d_monomials_.get(), d_modulus_.get(), N, numLUT, sub_batch);
            }

            NativeInt* ht_c0      = scratch.d_ht_c0;
            NativeInt* ht_c1      = scratch.d_ht_c1;
            NativeInt* perm_c0    = scratch.d_perm_c0;
            NativeInt* ht_next_c0 = scratch.d_ht_next_c0;
            NativeInt* ht_next_c1 = scratch.d_ht_next_c1;
            NativeInt* digits     = scratch.d_digits;
            NativeInt* ss_c0      = scratch.d_ss_c0;
            NativeInt* ss_c1      = scratch.d_ss_c1;
            NativeInt* rgsw_out   = scratch.d_rgsw_out;

            const bool disableHtssGraph = (std::getenv("CIRBTS_DISABLE_HTSS_CUDA_GRAPH") != nullptr);
            const bool canUseHtssGraph  = !disableHtssGraph && scratch.htss_graph_exec && scratch.htss_graph_failed &&
                                          scratch.htss_graph_block_x && scratch.htss_graph_num_luts &&
                                          scratch.htss_graph_fuse_inwt_decomp && scratch.htss_graph_fuse_decomp_fnwt &&
                                          scratch.htss_graph_pipeline && scratch.htss_graph_smem;
            const bool htss_graph_ok = canUseHtssGraph && !(*scratch.htss_graph_failed);
            const bool htss_l2_switch = l2_persist_enabled_ && !htss_graph_ok && !htss_pipeline;

            auto runHomTraceAndSchemeSwitch = [&]() {
                NativeInt* local_ht_c0      = ht_c0;
                NativeInt* local_ht_c1      = ht_c1;
                NativeInt* local_ht_next_c0 = ht_next_c0;
                NativeInt* local_ht_next_c1 = ht_next_c1;

                dim3 block(htss_block_x);
                dim3 grid((N + block.x - 1) / block.x, numLUTTotal);

                const bool use_htss_soa = (htss_soa_requested || htss_smem_requested) && has_htss_soa;
                const bool use_ht_smem = use_htss_soa && ht_smem_enabled;
                const bool use_htss_smem = use_htss_soa && htss_smem_enabled;
                const bool use_ss_smem = use_htss_soa && ss_smem_enabled;
                const uint32_t htss_tiles = static_cast<uint32_t>(N / 32u);
                const NativeInt* ht_key_base = use_htss_soa ? d_HTkey_soa_.get() : d_HTkey_.get();
                const NativeInt* ht_key_shoup_base = use_htss_soa ? d_HTkey_shoup_soa_.get() : d_HTkey_shoup_.get();
                const NativeInt* ss_key_base = use_htss_soa ? d_SSkey_soa_.get() : d_SSkey_.get();
                const NativeInt* ss_key_shoup_base = use_htss_soa ? d_SSkey_shoup_soa_.get() : d_SSkey_shoup_.get();

                L2PersistScope l2_scope(this, scratch.stream);
                if (htss_l2_switch) {
                    l2_scope.reset(ht_key_base, size_HTkey_ * size_HTRGSWkey_ * sizeof(NativeInt));
                }

                if (htss_pipeline) {
                    cudaMemsetAsync(ss_c0, 0, sizeof(NativeInt) * lutSize, scratch.stream);
                    cudaMemsetAsync(ss_c1, 0, sizeof(NativeInt) * lutSize, scratch.stream);
                }

                for (uint32_t k = 0; k < traceRounds; ++k) {
                    const uint32_t* d_map = d_auto_maps_.get() + static_cast<size_t>(k) * N;
                    if (htssFuseInwtDecomp) {
                        const size_t inwtShmem = N * sizeof(NativeInt);
                        kernel_INWT_Permute_SignedDigitDecompose<<<numLUTTotal, N / 2, inwtShmem, scratch.stream>>>(
                            digits, local_ht_c0, d_map, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                            d_n_inv_mod_q_shoup_.get(), baseHT, digitsHT, N, numLUTTotal);
                        fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                            static_cast<size_t>(numLUTTotal) * digitsHT, scratch.stream);
                    }
                    else {
                        inwt_1d_opt_permute_batched(perm_c0, local_ht_c0, d_map, d_itwiddles_.get(), d_itwiddles_shoup_.get(),
                                                    d_modulus_.get(), d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(),
                                                    N, numLUTTotal, scratch.stream);
                        if (htssFuseDecompFnwt) {
                            const size_t ht_batch   = static_cast<size_t>(numLUTTotal) * digitsHT;
                            const size_t htNttShmem = N * sizeof(NativeInt);
                            kernel_SignedDigitDecompose_FusedFNWT<<<ht_batch, N / 2, htNttShmem, scratch.stream>>>(
                                digits, perm_c0, Q, baseHT, digitsHT, N, numLUTTotal, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());
                        }
                        else {
                            kernel_Fused_Permute_Decompose<<<grid, block, 0, scratch.stream>>>(digits, perm_c0, Q, baseHT, digitsHT, N, numLUTTotal);
                            fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                                static_cast<size_t>(numLUTTotal) * digitsHT, scratch.stream);
                        }
                    }

                    const size_t key_stride = size_HTRGSWkey_;
                    const NativeInt* d_key  = ht_key_base + static_cast<size_t>(k) * key_stride;
                    const NativeInt* d_key_shoup = ht_key_shoup_base + static_cast<size_t>(k) * key_stride;
                    if (htss_pipeline) {
                        if (use_htss_soa) {
                            if (use_htss_smem) {
                                kernel_MultAddUpdate_HTSS_PermuteC1_SoA_Smem<<<grid, block, htss_smem_bytes, scratch.stream>>>(
                                    local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                                    ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N,
                                    htss_tiles, numLUTTotal, lutSize);
                            }
                            else {
                                kernel_MultAddUpdate_HTSS_PermuteC1_SoA<<<grid, block, 0, scratch.stream>>>(
                                    local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                                    ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N,
                                    htss_tiles, numLUTTotal, lutSize);
                            }
                        }
                        else {
                            kernel_MultAddUpdate_HTSS_PermuteC1<<<grid, block, 0, scratch.stream>>>(
                                local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                                ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N,
                                numLUTTotal, lutSize);
                        }
                    }
                    else if (use_htss_soa) {
                        if (use_ht_smem) {
                            kernel_MultAddUpdate_HT_PermuteC1_SoA_Smem<<<grid, block, ht_smem_bytes, scratch.stream>>>(
                                local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                                d_modulus_.get(), digitsHT, N, htss_tiles, numLUTTotal, lutSize);
                        }
                        else {
                            kernel_MultAddUpdate_HT_PermuteC1_SoA<<<grid, block, 0, scratch.stream>>>(
                                local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                                d_modulus_.get(), digitsHT, N, htss_tiles, numLUTTotal, lutSize);
                        }
                    } else {
                        kernel_MultAddUpdate_HT_PermuteC1<<<grid, block, 0, scratch.stream>>>(
                            local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                            d_modulus_.get(), digitsHT, N, numLUTTotal, lutSize);
                    }
                    std::swap(local_ht_c0, local_ht_next_c0);
                    std::swap(local_ht_c1, local_ht_next_c1);
                }

                kernel_SaveToRGSW<<<grid, block, 0, scratch.stream>>>(rgsw_out, local_ht_c0, local_ht_c1, /*row_type=*/1, N, numLUTTotal);

                if (htss_pipeline) {
                    constexpr int addBlock = 256;
                    const int addBlocks = static_cast<int>((lutSize + addBlock - 1) / addBlock);
                    kernel_AddInPlace<<<addBlocks, addBlock, 0, scratch.stream>>>(ss_c0, local_ht_c1, d_modulus_.get(), lutSize);
                    kernel_SaveToRGSW<<<grid, block, 0, scratch.stream>>>(rgsw_out, ss_c0, ss_c1, /*row_type=*/0, N, numLUTTotal);
                    return;
                }

                if (htssFuseInwtDecomp) {
                    const size_t inwtShmem = N * sizeof(NativeInt);
                    kernel_INWT_SignedDigitDecompose<<<numLUTTotal, N / 2, inwtShmem, scratch.stream>>>(
                        digits, local_ht_c0, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                        d_n_inv_mod_q_shoup_.get(), baseSS, digitsSS, N, numLUTTotal);
                    fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                        static_cast<size_t>(numLUTTotal) * digitsSS, scratch.stream);
                }
                else {
                    inwt_1d_opt_batched(local_ht_c0, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(),
                                        d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N, numLUTTotal, scratch.stream);
                    if (htssFuseDecompFnwt) {
                        const size_t ss_batch   = static_cast<size_t>(numLUTTotal) * digitsSS;
                        const size_t ssNttShmem = N * sizeof(NativeInt);
                        kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, ssNttShmem, scratch.stream>>>(
                            digits, local_ht_c0, Q, baseSS, digitsSS, N, numLUTTotal, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());
                    }
                    else {
                        kernel_Fused_Permute_Decompose<<<grid, block, 0, scratch.stream>>>(digits, local_ht_c0, Q, baseSS, digitsSS, N, numLUTTotal);
                        fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                            static_cast<size_t>(numLUTTotal) * digitsSS, scratch.stream);
                    }
                }

                if (htss_l2_switch) {
                    l2_scope.reset(ss_key_base, size_SSkey_ * N * sizeof(NativeInt));
                }
                if (use_htss_soa) {
                    if (use_ss_smem) {
                        kernel_MultAddUpdate_SS_SoA_Smem<<<grid, block, ss_smem_bytes, scratch.stream>>>(
                            ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1, d_modulus_.get(), digitsSS, N,
                            htss_tiles, numLUTTotal, lutSize);
                    }
                    else {
                        kernel_MultAddUpdate_SS_SoA<<<grid, block, 0, scratch.stream>>>(
                            ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1, d_modulus_.get(), digitsSS, N,
                            htss_tiles, numLUTTotal, lutSize);
                    }
                } else {
                    kernel_MultAddUpdate_SS<<<grid, block, 0, scratch.stream>>>(
                        ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1, d_modulus_.get(), digitsSS, N,
                        numLUTTotal, lutSize);
                }

                kernel_SaveToRGSW<<<grid, block, 0, scratch.stream>>>(rgsw_out, ss_c0, ss_c1, /*row_type=*/0, N, numLUTTotal);
            };

            if (htss_graph_ok) {
                if (!*scratch.htss_graph_exec || *scratch.htss_graph_block_x != htss_block_x || *scratch.htss_graph_num_luts != numLUTTotal ||
                    *scratch.htss_graph_fuse_inwt_decomp != htssFuseInwtDecomp || *scratch.htss_graph_fuse_decomp_fnwt != htssFuseDecompFnwt ||
                    *scratch.htss_graph_pipeline != htss_pipeline || *scratch.htss_graph_smem != htss_graph_smem) {
                    if (*scratch.htss_graph_exec) {
                        cudaGraphExecDestroy(*scratch.htss_graph_exec);
                        *scratch.htss_graph_exec = nullptr;
                    }
                    *scratch.htss_graph_block_x = htss_block_x;
                    *scratch.htss_graph_num_luts = numLUTTotal;
                    *scratch.htss_graph_fuse_inwt_decomp = htssFuseInwtDecomp;
                    *scratch.htss_graph_fuse_decomp_fnwt = htssFuseDecompFnwt;
                    *scratch.htss_graph_pipeline = htss_pipeline;
                    *scratch.htss_graph_smem = htss_graph_smem;

                    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(scratch.stream));
                    cudaGraph_t graph{};
                    PHANTOM_CHECK_CUDA(cudaStreamBeginCapture(scratch.stream, cudaStreamCaptureModeThreadLocal));
                    runHomTraceAndSchemeSwitch();
                    PHANTOM_CHECK_CUDA(cudaStreamEndCapture(scratch.stream, &graph));

                    cudaGraphExec_t exec{};
                    const auto inst = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
                    cudaGraphDestroy(graph);
                    if (inst != cudaSuccess) {
                        *scratch.htss_graph_failed = true;
                    }
                    else {
                        *scratch.htss_graph_exec = exec;
                    }
                }

                if (*scratch.htss_graph_exec) {
                    L2PersistScope l2_scope(this, scratch.stream);
                    if (l2_persist_enabled_) {
                        const bool use_htss_soa = (htss_soa_requested || htss_smem_requested) && has_htss_soa;
                        const NativeInt* ht_key_base = use_htss_soa ? d_HTkey_soa_.get() : d_HTkey_.get();
                        l2_scope.reset(ht_key_base, size_HTkey_ * size_HTRGSWkey_ * sizeof(NativeInt));
                    }
                    PHANTOM_CHECK_CUDA(cudaGraphLaunch(*scratch.htss_graph_exec, scratch.stream));
                }
                else {
                    runHomTraceAndSchemeSwitch();
                }
            }
            else {
                runHomTraceAndSchemeSwitch();
            }

            const size_t host_elems = static_cast<size_t>(numLUTTotal) * 4 * N;
            NativeInt* host_ptr = nullptr;
            if (i == 0) {
                host_ptr = ensure_pinned_buffer(h_rgsw_pinned_, h_rgsw_pinned_capacity_, host_elems);
            }
            else {
                host_ptr = ensure_pinned_buffer(extra_workspaces_[i - 1]->h_rgsw_pinned, extra_workspaces_[i - 1]->h_rgsw_pinned_capacity,
                                                host_elems);
            }

            cudaEvent_t ev_d2h{};
            PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_d2h));
            cudaMemcpyAsync(host_ptr, rgsw_out, host_elems * sizeof(NativeInt), cudaMemcpyDeviceToHost, scratch.stream);
            PHANTOM_CHECK_CUDA(cudaEventRecord(ev_d2h, scratch.stream));

            jobs.push_back({offset, sub_batch, host_ptr, host_elems, ev_d2h});
            offset += sub_batch;
        }

        std::vector<RGSWCiphertext> res(batch);
        auto pack_batch = [&](const NativeInt* h_rgsw, uint32_t sub_batch, uint32_t out_offset) {
            for (uint32_t b = 0; b < sub_batch; ++b) {
                RGSWCiphertextImpl ct_gpu(numRowsPer, 2);
                for (uint32_t lut = 0; lut < numLUT; ++lut) {
                    const uint32_t lut_global = b * numLUT + lut;
                    for (uint32_t row_type = 0; row_type < 2; ++row_type) {
                        const uint64_t row_local  = static_cast<uint64_t>(lut) * 2 + row_type;
                        const uint64_t row_global = static_cast<uint64_t>(lut_global) * 2 + row_type;
                        const NativeInt* base = h_rgsw + row_global * 2 * N;
                        NativePoly c0(polyparams, Format::EVALUATION, true);
                        NativePoly c1(polyparams, Format::EVALUATION, true);
                        for (size_t i = 0; i < N; ++i) {
                            c0[i] = NativeInteger(base[i]);
                            c1[i] = NativeInteger(base[N + i]);
                        }
                        ct_gpu[row_local] = {c0, c1};
                    }
                }
                res[out_offset + b] = std::make_shared<RGSWCiphertextImpl>(std::move(ct_gpu));
            }
        };

        for (const auto& job : jobs) {
            PHANTOM_CHECK_CUDA(cudaEventSynchronize(job.d2h_done));
            pack_batch(job.h_rgsw, job.batch, job.offset);
            cudaEventDestroy(job.d2h_done);
        }

        return res;
    }

    const uint32_t numLUTTotal = numLUT * batch;
    const size_t lutSize       = static_cast<size_t>(numLUTTotal) * N;
    const uint64_t numRowsPer  = static_cast<uint64_t>(numLUT) * 2;
    const uint64_t numRowsTotal = static_cast<uint64_t>(numLUTTotal) * 2;

    if (scratch_lut_size_ < lutSize) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatch: scratch buffer too small (recreate GPUCirBTSContext)");
    }
    if (scratch_digits_max_ < std::max(digitsHT, digitsSS)) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatch: scratch digits mismatch (recreate GPUCirBTSContext)");
    }
    if (!d_ht_c0_.get() || !d_ht_c1_.get() || !d_perm_c0_.get() || !d_ht_next_c0_.get() || !d_ht_next_c1_.get() ||
        !d_ks_c0_.get() || !d_ks_c1_.get() || !d_digits_.get() || !d_ss_c0_.get() || !d_ss_c1_.get() ||
        !d_rgsw_out_.get()) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatch: scratch buffers are not initialized on GPU");
    }

    ensure_h2d_events();
    ensure_pinned_monomial_inv(static_cast<size_t>(batch) * N);
    ensure_pinned_index_pos(static_cast<size_t>(batch) * n);

    const BatchScratchView scratch = make_batch_scratch_view();

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
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_start));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_boot_end));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_mvr_end));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_htss_end));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_d2h_start));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_d2h_end));
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev_start, s));
    }

    // 1) Bootstrap (ACC) on GPU for the whole batch.
    if (!d_bootstrap_acc_.get()) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatch: bootstrap accumulator scratch is not initialized on GPU");
    }
    gpu_BootstrapLUT_inplace_batch(params, ek.RFkey, cts, bitwidth, scratch, batch);
    const NativeInt* d_acc = scratch.d_bootstrap_acc;
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev_boot_end, s));
    }

    // 2) Generate MV-RLWEs directly into HT accumulator buffers (evaluation form).
    {
        const size_t total = lutSize * 2;
        constexpr int blockSize = 256;
        const int numBlocks = static_cast<int>((total + blockSize - 1) / blockSize);
        kernel_GenMVRLWEs<<<numBlocks, blockSize, 0, s>>>(ht_c0, ht_c1, d_acc, d_monomials_.get(), d_modulus_.get(), N, numLUT,
                                                          batch);
    }
    if (profile) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev_mvr_end, s));
    }

    bool htssFuseInwtDecomp = false;
    bool htssFuseDecompFnwt = false;
    GetHtssFuseFlags(&htssFuseInwtDecomp, &htssFuseDecompFnwt);

    const bool disableHtssGraph = (std::getenv("CIRBTS_DISABLE_HTSS_CUDA_GRAPH") != nullptr);
    const bool canUseHtssGraph  = !disableHtssGraph && !htss_graph_failed_;
    const bool htss_l2_switch   = l2_persist_enabled_ && !canUseHtssGraph && !htss_pipeline;

    auto runHomTraceAndSchemeSwitch = [&]() {
        // 3) HomTrace (EvalHT) for all LUT ciphertexts.
        NativeInt* local_ht_c0      = ht_c0;
        NativeInt* local_ht_c1      = ht_c1;
        NativeInt* local_ht_next_c0 = ht_next_c0;
        NativeInt* local_ht_next_c1 = ht_next_c1;

        dim3 block(htss_block_x);
        dim3 grid((N + block.x - 1) / block.x, numLUTTotal);

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

        for (uint32_t k = 0; k < traceRounds; ++k) {
            const uint32_t* d_map = d_auto_maps_.get() + static_cast<size_t>(k) * N;
            if (htssFuseInwtDecomp) {
                const size_t inwtShmem = N * sizeof(NativeInt);
                kernel_INWT_Permute_SignedDigitDecompose<<<numLUTTotal, N / 2, inwtShmem, s>>>(
                    digits, local_ht_c0, d_map, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                    d_n_inv_mod_q_shoup_.get(), baseHT, digitsHT, N, numLUTTotal);
                fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                    static_cast<size_t>(numLUTTotal) * digitsHT, s);
            }
            else {
                inwt_1d_opt_permute_batched(perm_c0, local_ht_c0, d_map, d_itwiddles_.get(), d_itwiddles_shoup_.get(),
                                            d_modulus_.get(), d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N,
                                            numLUTTotal, s);
                if (htssFuseDecompFnwt) {
                    const size_t ht_batch   = static_cast<size_t>(numLUTTotal) * digitsHT;
                    const size_t htNttShmem = N * sizeof(NativeInt);
                    kernel_SignedDigitDecompose_FusedFNWT<<<ht_batch, N / 2, htNttShmem, s>>>(
                        digits, perm_c0, Q, baseHT, digitsHT, N, numLUTTotal, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());
                }
                else {
                    kernel_Fused_Permute_Decompose<<<grid, block, 0, s>>>(digits, perm_c0, Q, baseHT, digitsHT, N, numLUTTotal);
                    fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                        static_cast<size_t>(numLUTTotal) * digitsHT, s);
                }
            }

            const size_t key_stride = size_HTRGSWkey_;
            const NativeInt* d_key  = ht_key_base + static_cast<size_t>(k) * key_stride;
            const NativeInt* d_key_shoup = ht_key_shoup_base + static_cast<size_t>(k) * key_stride;
            if (htss_pipeline) {
                if (use_htss_soa) {
                    if (use_htss_smem) {
                        kernel_MultAddUpdate_HTSS_PermuteC1_SoA_Smem<<<grid, block, htss_smem_bytes, s>>>(
                            local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                            ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N, htss_tiles,
                            numLUTTotal, lutSize);
                    }
                    else {
                        kernel_MultAddUpdate_HTSS_PermuteC1_SoA<<<grid, block, 0, s>>>(
                            local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                            ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N, htss_tiles,
                            numLUTTotal, lutSize);
                    }
                }
                else {
                    kernel_MultAddUpdate_HTSS_PermuteC1<<<grid, block, 0, s>>>(
                        local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                        ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N, numLUTTotal,
                        lutSize);
                }
            }
            else if (use_htss_soa) {
                if (use_ht_smem) {
                    kernel_MultAddUpdate_HT_PermuteC1_SoA_Smem<<<grid, block, ht_smem_bytes, s>>>(
                        local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                        d_modulus_.get(), digitsHT, N, htss_tiles, numLUTTotal, lutSize);
                }
                else {
                    kernel_MultAddUpdate_HT_PermuteC1_SoA<<<grid, block, 0, s>>>(
                        local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                        d_modulus_.get(), digitsHT, N, htss_tiles, numLUTTotal, lutSize);
                }
            } else {
                kernel_MultAddUpdate_HT_PermuteC1<<<grid, block, 0, s>>>(
                    local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                    d_modulus_.get(), digitsHT, N, numLUTTotal, lutSize);
            }
            std::swap(local_ht_c0, local_ht_next_c0);
            std::swap(local_ht_c1, local_ht_next_c1);
        }

        // 4) Save HT rows (odd rows) to output.
        kernel_SaveToRGSW<<<grid, block, 0, s>>>(rgsw_out, local_ht_c0, local_ht_c1, /*row_type=*/1, N, numLUTTotal);

        if (htss_pipeline) {
            constexpr int addBlock = 256;
            const int addBlocks = static_cast<int>((lutSize + addBlock - 1) / addBlock);
            kernel_AddInPlace<<<addBlocks, addBlock, 0, s>>>(ss_c0, local_ht_c1, d_modulus_.get(), lutSize);
            kernel_SaveToRGSW<<<grid, block, 0, s>>>(rgsw_out, ss_c0, ss_c1, /*row_type=*/0, N, numLUTTotal);
            return;
        }

        // 5) SchemeSwitch (EvalSS).
        if (htssFuseInwtDecomp) {
            const size_t inwtShmem = N * sizeof(NativeInt);
            kernel_INWT_SignedDigitDecompose<<<numLUTTotal, N / 2, inwtShmem, s>>>(
                digits, local_ht_c0, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                d_n_inv_mod_q_shoup_.get(), baseSS, digitsSS, N, numLUTTotal);
            fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                static_cast<size_t>(numLUTTotal) * digitsSS, s);
        }
        else {
            inwt_1d_opt_batched(local_ht_c0, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(),
                                d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N, numLUTTotal, s);
            if (htssFuseDecompFnwt) {
                const size_t ss_batch   = static_cast<size_t>(numLUTTotal) * digitsSS;
                const size_t ssNttShmem = N * sizeof(NativeInt);
                kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, ssNttShmem, s>>>(
                    digits, local_ht_c0, Q, baseSS, digitsSS, N, numLUTTotal, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());
            }
            else {
                kernel_Fused_Permute_Decompose<<<grid, block, 0, s>>>(digits, local_ht_c0, Q, baseSS, digitsSS, N, numLUTTotal);
                fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                    static_cast<size_t>(numLUTTotal) * digitsSS, s);
            }
        }

        if (htss_l2_switch) {
            l2_scope.reset(ss_key_base, size_SSkey_ * N * sizeof(NativeInt));
        }
        if (use_htss_soa) {
            if (use_ss_smem) {
                kernel_MultAddUpdate_SS_SoA_Smem<<<grid, block, ss_smem_bytes, s>>>(
                    ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1, d_modulus_.get(), digitsSS, N,
                    htss_tiles, numLUTTotal, lutSize);
            }
            else {
                kernel_MultAddUpdate_SS_SoA<<<grid, block, 0, s>>>(
                    ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1, d_modulus_.get(), digitsSS, N,
                    htss_tiles, numLUTTotal, lutSize);
            }
        } else {
            kernel_MultAddUpdate_SS<<<grid, block, 0, s>>>(ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1,
                                                           d_modulus_.get(), digitsSS, N, numLUTTotal, lutSize);
        }

        // 6) Save SS rows (even rows) to output.
        kernel_SaveToRGSW<<<grid, block, 0, s>>>(rgsw_out, ss_c0, ss_c1, /*row_type=*/0, N, numLUTTotal);
    };

    if (canUseHtssGraph) {
        if (!htss_graph_exec_ || htss_graph_block_x_ != htss_block_x || htss_graph_num_luts_ != numLUTTotal ||
            htss_graph_fuse_inwt_decomp_ != htssFuseInwtDecomp || htss_graph_fuse_decomp_fnwt_ != htssFuseDecompFnwt ||
            htss_graph_pipeline_ != htss_pipeline || htss_graph_smem_ != htss_graph_smem) {
            if (htss_graph_exec_) {
                cudaGraphExecDestroy(htss_graph_exec_);
                htss_graph_exec_ = nullptr;
            }
            htss_graph_block_x_            = htss_block_x;
            htss_graph_num_luts_           = numLUTTotal;
            htss_graph_fuse_inwt_decomp_   = htssFuseInwtDecomp;
            htss_graph_fuse_decomp_fnwt_   = htssFuseDecompFnwt;
            htss_graph_pipeline_           = htss_pipeline;
            htss_graph_smem_               = htss_graph_smem;

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

    std::vector<NativeInt> h_RGSW(static_cast<size_t>(numRowsTotal) * 2 * N);
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

    std::vector<RGSWCiphertext> res;
    res.reserve(batch);
    for (uint32_t b = 0; b < batch; ++b) {
        RGSWCiphertextImpl ct_gpu(numRowsPer, 2);
        for (uint32_t lut = 0; lut < numLUT; ++lut) {
            const uint32_t lut_global = b * numLUT + lut;
            for (uint32_t row_type = 0; row_type < 2; ++row_type) {
                const uint64_t row_local  = static_cast<uint64_t>(lut) * 2 + row_type;
                const uint64_t row_global = static_cast<uint64_t>(lut_global) * 2 + row_type;
                const NativeInt* base     = h_RGSW.data() + row_global * 2 * N;
                NativePoly c0(polyparams, Format::EVALUATION, true);
                NativePoly c1(polyparams, Format::EVALUATION, true);
                for (size_t i = 0; i < N; ++i) {
                    c0[i] = NativeInteger(base[i]);
                    c1[i] = NativeInteger(base[N + i]);
                }
                ct_gpu[row_local] = {c0, c1};
            }
        }
        res.emplace_back(std::make_shared<RGSWCiphertextImpl>(std::move(ct_gpu)));
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

        std::cout << "[CIRBTS_PROFILE_BATCH] batch=" << batch
                  << " bootstrap=" << ms_boot << "ms"
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

    return res;
}

GPUCirBTSContext::DeviceRGSWBatchView GPUCirBTSContext::gpu_CircuitBootstrappingBatchToDevice(
    const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params,
    const lbcrypto::RingGSWCirBTKey& ek,
    const std::vector<lbcrypto::LWECiphertext>& cts) const {
    if (cts.empty()) {
        return {};
    }

    const auto method = params->GetRingGSWParams1()->GetMethod();
    if (method != BINFHE_METHOD::GINX) {
        std::ostringstream oss;
        oss << "gpu_CircuitBootstrappingBatchToDevice (GPU) supports only method=GINX; got method=" << method;
        OPENFHE_THROW(config_error, oss.str());
    }

    const uint32_t batch = static_cast<uint32_t>(cts.size());
    if (batch > std::max<uint32_t>(1u, max_batch_size_)) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatchToDevice: batch size exceeds max_batch_size_ (recreate GPUCirBTSContext)");
    }

    const uint32_t numLUT = params->GetDigitsCC();
    const size_t N        = params->GetRingGSWParams1()->GetN();
    const uint32_t n      = params->GetLWEParams()->Getn();
    const NativeInt Q     = params->GetRingGSWParams1()->GetQ().ConvertToInt();
    if (N != ring_dim_ || numLUT != num_luts_) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatchToDevice: GPU context params mismatch (recreate GPUCirBTSContext)");
    }
    if (!d_auto_maps_.get() || log_ring_dim_ == 0) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDevice: automorphism maps are not initialized on GPU");
    }
    if (!d_gpow_.get()) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDevice: Gpow is not initialized on GPU");
    }
    if (numLUT > 1 && !d_monomials_.get()) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDevice: monomials are not initialized on GPU");
    }

    // This device-output path is meant for low-latency chaining; keep it single-stream for now.
    if (const char* v = std::getenv("CIRBTS_BATCH_STREAMS"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0' && parsed > 1) {
            OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDevice: CIRBTS_BATCH_STREAMS>1 is not supported (set to 1)");
        }
    }

    const uint64_t bitwidth = static_cast<uint64_t>(std::ceil(std::log2(static_cast<double>(numLUT))));
    auto& s                = stream_wrapper_.get_stream();
    const auto& rlweParams = params->GetRLWEParams();
    const uint32_t traceShift = rlweParams->GetTraceShift();
    if (traceShift >= log_ring_dim_) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDevice: TraceShift must be smaller than log2(N)");
    }
    const uint32_t traceRounds = log_ring_dim_ - traceShift;
    const uint32_t baseHT  = rlweParams->GetBaseHT();
    const uint32_t baseSS  = rlweParams->GetBaseSS();
    const uint32_t digitsHT = rlweParams->GetDigitsHTA();
    const uint32_t digitsSS = rlweParams->GetDigitsSSA();
    const bool htss_soa_requested = (std::getenv("CIRBTS_HTSS_SOA") != nullptr);
    const bool htss_smem_requested = (std::getenv("CIRBTS_HTSS_SMEM") != nullptr);
    const bool htss_smem_verbose = (std::getenv("CIRBTS_HTSS_SMEM_VERBOSE") != nullptr);
    const bool htss_pipeline      = (std::getenv("CIRBTS_HTSS_PIPELINE") != nullptr);
    const bool has_htss_soa = d_HTkey_soa_.get() && d_HTkey_shoup_soa_.get() && d_SSkey_soa_.get() && d_SSkey_shoup_soa_.get() &&
                              (htkey_soa_tiles_ > 0) && (sskey_soa_tiles_ > 0);
    if ((htss_soa_requested || htss_smem_requested) && !has_htss_soa) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatchToDevice: HT/SS SoA requested but not initialized (set CIRBTS_HTSS_SOA before init)");
    }
    if (htss_pipeline && (baseHT != baseSS || digitsHT != digitsSS)) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDevice: HT/SS pipeline requires baseHT==baseSS and digitsHT==digitsSS");
    }

    uint32_t htss_block_x = 256;
    if (const char* v = std::getenv("CIRBTS_HTSS_BLOCK_X"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0' && parsed > 0) {
            htss_block_x = static_cast<uint32_t>(std::min<unsigned long>(parsed, 1024ul));
        }
    }

    bool htssFuseInwtDecomp = false;
    bool htssFuseDecompFnwt = false;
    GetHtssFuseFlags(&htssFuseInwtDecomp, &htssFuseDecompFnwt);

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

    const uint32_t numLUTTotal  = numLUT * batch;
    const size_t lutSize        = static_cast<size_t>(numLUTTotal) * N;
    const uint64_t numRowsTotal = static_cast<uint64_t>(numLUTTotal) * 2;

    if (scratch_lut_size_ < lutSize) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatchToDevice: scratch buffer too small (recreate GPUCirBTSContext)");
    }
    if (scratch_digits_max_ < std::max(digitsHT, digitsSS)) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatchToDevice: scratch digits mismatch (recreate GPUCirBTSContext)");
    }
    if (!d_ht_c0_.get() || !d_ht_c1_.get() || !d_perm_c0_.get() || !d_ht_next_c0_.get() || !d_ht_next_c1_.get() ||
        !d_ks_c0_.get() || !d_ks_c1_.get() || !d_digits_.get() || !d_ss_c0_.get() || !d_ss_c1_.get() ||
        !d_rgsw_out_.get()) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDevice: scratch buffers are not initialized on GPU");
    }

    ensure_h2d_events();
    ensure_pinned_monomial_inv(static_cast<size_t>(batch) * N);
    ensure_pinned_index_pos(static_cast<size_t>(batch) * n);

    const BatchScratchView scratch = make_batch_scratch_view();

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

    gpu_BootstrapLUT_inplace_batch(params, ek.RFkey, cts, bitwidth, scratch, batch);

    const NativeInt* d_acc = scratch.d_bootstrap_acc;
    {
        const size_t total = lutSize * 2;
        constexpr int blockSize = 256;
        const int numBlocks = static_cast<int>((total + blockSize - 1) / blockSize);
        kernel_GenMVRLWEs<<<numBlocks, blockSize, 0, s>>>(ht_c0, ht_c1, d_acc, d_monomials_.get(), d_modulus_.get(), N, numLUT, batch);
    }

    const bool disableHtssGraph = (std::getenv("CIRBTS_DISABLE_HTSS_CUDA_GRAPH") != nullptr);
    const bool canUseHtssGraph  = !disableHtssGraph && !htss_graph_failed_;
    const bool htss_l2_switch   = l2_persist_enabled_ && !canUseHtssGraph && !htss_pipeline;

    auto runHomTraceAndSchemeSwitch = [&]() {
        // 3) HomTrace (EvalHT) for all LUT ciphertexts.
        NativeInt* local_ht_c0      = ht_c0;
        NativeInt* local_ht_c1      = ht_c1;
        NativeInt* local_ht_next_c0 = ht_next_c0;
        NativeInt* local_ht_next_c1 = ht_next_c1;

        dim3 block(htss_block_x);
        dim3 grid((N + block.x - 1) / block.x, numLUTTotal);

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

        for (uint32_t k = 0; k < traceRounds; ++k) {
            const uint32_t* d_map = d_auto_maps_.get() + static_cast<size_t>(k) * N;
            if (htssFuseInwtDecomp) {
                const size_t inwtShmem = N * sizeof(NativeInt);
                kernel_INWT_Permute_SignedDigitDecompose<<<numLUTTotal, N / 2, inwtShmem, s>>>(
                    digits, local_ht_c0, d_map, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                    d_n_inv_mod_q_shoup_.get(), baseHT, digitsHT, N, numLUTTotal);
                fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                    static_cast<size_t>(numLUTTotal) * digitsHT, s);
            } else {
                inwt_1d_opt_permute_batched(perm_c0, local_ht_c0, d_map, d_itwiddles_.get(), d_itwiddles_shoup_.get(),
                                            d_modulus_.get(), d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N,
                                            numLUTTotal, s);
                if (htssFuseDecompFnwt) {
                    const size_t ht_batch   = static_cast<size_t>(numLUTTotal) * digitsHT;
                    const size_t htNttShmem = N * sizeof(NativeInt);
                    kernel_SignedDigitDecompose_FusedFNWT<<<ht_batch, N / 2, htNttShmem, s>>>(
                        digits, perm_c0, Q, baseHT, digitsHT, N, numLUTTotal, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());
                } else {
                    kernel_Fused_Permute_Decompose<<<grid, block, 0, s>>>(digits, perm_c0, Q, baseHT, digitsHT, N, numLUTTotal);
                    fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                        static_cast<size_t>(numLUTTotal) * digitsHT, s);
                }
            }

            const size_t key_stride = size_HTRGSWkey_;
            const NativeInt* d_key  = ht_key_base + static_cast<size_t>(k) * key_stride;
            const NativeInt* d_key_shoup = ht_key_shoup_base + static_cast<size_t>(k) * key_stride;
            if (htss_pipeline) {
                if (use_htss_soa) {
                    if (use_htss_smem) {
                        kernel_MultAddUpdate_HTSS_PermuteC1_SoA_Smem<<<grid, block, htss_smem_bytes, s>>>(
                            local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                            ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N, htss_tiles,
                            numLUTTotal, lutSize);
                    } else {
                        kernel_MultAddUpdate_HTSS_PermuteC1_SoA<<<grid, block, 0, s>>>(
                            local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                            ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N, htss_tiles,
                            numLUTTotal, lutSize);
                    }
                } else {
                    kernel_MultAddUpdate_HTSS_PermuteC1<<<grid, block, 0, s>>>(
                        local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                        ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N, numLUTTotal,
                        lutSize);
                }
            } else if (use_htss_soa) {
                if (use_ht_smem) {
                    kernel_MultAddUpdate_HT_PermuteC1_SoA_Smem<<<grid, block, ht_smem_bytes, s>>>(
                        local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                        d_modulus_.get(), digitsHT, N, htss_tiles, numLUTTotal, lutSize);
                } else {
                    kernel_MultAddUpdate_HT_PermuteC1_SoA<<<grid, block, 0, s>>>(
                        local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                        d_modulus_.get(), digitsHT, N, htss_tiles, numLUTTotal, lutSize);
                }
            } else {
                kernel_MultAddUpdate_HT_PermuteC1<<<grid, block, 0, s>>>(
                    local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                    d_modulus_.get(), digitsHT, N, numLUTTotal, lutSize);
            }
            std::swap(local_ht_c0, local_ht_next_c0);
            std::swap(local_ht_c1, local_ht_next_c1);
        }

        // 4) Save HT rows (odd rows) to output.
        kernel_SaveToRGSW<<<grid, block, 0, s>>>(rgsw_out, local_ht_c0, local_ht_c1, /*row_type=*/1, N, numLUTTotal);

        if (htss_pipeline) {
            constexpr int addBlock = 256;
            const int addBlocks = static_cast<int>((lutSize + addBlock - 1) / addBlock);
            kernel_AddInPlace<<<addBlocks, addBlock, 0, s>>>(ss_c0, local_ht_c1, d_modulus_.get(), lutSize);
            kernel_SaveToRGSW<<<grid, block, 0, s>>>(rgsw_out, ss_c0, ss_c1, /*row_type=*/0, N, numLUTTotal);
            return;
        }

        // 5) SchemeSwitch (EvalSS).
        if (htssFuseInwtDecomp) {
            const size_t inwtShmem = N * sizeof(NativeInt);
            kernel_INWT_SignedDigitDecompose<<<numLUTTotal, N / 2, inwtShmem, s>>>(
                digits, local_ht_c0, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                d_n_inv_mod_q_shoup_.get(), baseSS, digitsSS, N, numLUTTotal);
            fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                static_cast<size_t>(numLUTTotal) * digitsSS, s);
        } else {
            inwt_1d_opt_batched(local_ht_c0, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(),
                                d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N, numLUTTotal, s);
            if (htssFuseDecompFnwt) {
                const size_t ss_batch   = static_cast<size_t>(numLUTTotal) * digitsSS;
                const size_t ssNttShmem = N * sizeof(NativeInt);
                kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, ssNttShmem, s>>>(
                    digits, local_ht_c0, Q, baseSS, digitsSS, N, numLUTTotal, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());
            } else {
                kernel_Fused_Permute_Decompose<<<grid, block, 0, s>>>(digits, local_ht_c0, Q, baseSS, digitsSS, N, numLUTTotal);
                fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                    static_cast<size_t>(numLUTTotal) * digitsSS, s);
            }
        }

        if (htss_l2_switch) {
            l2_scope.reset(ss_key_base, size_SSkey_ * N * sizeof(NativeInt));
        }
        if (use_htss_soa) {
            if (use_ss_smem) {
                kernel_MultAddUpdate_SS_SoA_Smem<<<grid, block, ss_smem_bytes, s>>>(
                    ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1, d_modulus_.get(), digitsSS, N,
                    htss_tiles, numLUTTotal, lutSize);
            } else {
                kernel_MultAddUpdate_SS_SoA<<<grid, block, 0, s>>>(
                    ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1, d_modulus_.get(), digitsSS, N,
                    htss_tiles, numLUTTotal, lutSize);
            }
        } else {
            kernel_MultAddUpdate_SS<<<grid, block, 0, s>>>(ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1,
                                                           d_modulus_.get(), digitsSS, N, numLUTTotal, lutSize);
        }

        // 6) Save SS rows (even rows) to output.
        kernel_SaveToRGSW<<<grid, block, 0, s>>>(rgsw_out, ss_c0, ss_c1, /*row_type=*/0, N, numLUTTotal);
    };

    if (canUseHtssGraph) {
        if (!htss_graph_exec_ || htss_graph_block_x_ != htss_block_x || htss_graph_num_luts_ != numLUTTotal ||
            htss_graph_fuse_inwt_decomp_ != htssFuseInwtDecomp || htss_graph_fuse_decomp_fnwt_ != htssFuseDecompFnwt ||
            htss_graph_pipeline_ != htss_pipeline || htss_graph_smem_ != htss_graph_smem) {
            if (htss_graph_exec_) {
                cudaGraphExecDestroy(htss_graph_exec_);
                htss_graph_exec_ = nullptr;
            }
            htss_graph_block_x_            = htss_block_x;
            htss_graph_num_luts_           = numLUTTotal;
            htss_graph_fuse_inwt_decomp_   = htssFuseInwtDecomp;
            htss_graph_fuse_decomp_fnwt_   = htssFuseDecompFnwt;
            htss_graph_pipeline_           = htss_pipeline;
            htss_graph_smem_               = htss_graph_smem;

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
            } else {
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
        } else {
            runHomTraceAndSchemeSwitch();
        }
    } else {
        runHomTraceAndSchemeSwitch();
    }

    DeviceRGSWBatchView view{};
    view.d_rgsw = rgsw_out;
    view.batch = batch;
    view.digitsCC = numLUT;
    view.N = static_cast<uint32_t>(N);
    view.per_ciphertext_elems = static_cast<size_t>(numLUT) * 4 * N;
    view.stream = s;
    return view;
}

GPUCirBTSContext::DeviceRGSWBatchView GPUCirBTSContext::gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE(
    const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params,
    const lbcrypto::RingGSWCirBTKey& ek,
    const NativeInt* d_a,
    const NativeInt* d_b,
    uint32_t batch,
    uint32_t a_stride) const {
    if (!d_a || !d_b || batch == 0) {
        return {};
    }

    const auto method = params->GetRingGSWParams1()->GetMethod();
    if (method != BINFHE_METHOD::GINX) {
        std::ostringstream oss;
        oss << "gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE (GPU) supports only method=GINX; got method=" << method;
        OPENFHE_THROW(config_error, oss.str());
    }

    if (batch > std::max<uint32_t>(1u, max_batch_size_)) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE: batch size exceeds max_batch_size_ (recreate GPUCirBTSContext)");
    }

    const uint32_t numLUT = params->GetDigitsCC();
    const size_t N        = params->GetRingGSWParams1()->GetN();
    const uint32_t n      = params->GetLWEParams()->Getn();
    const NativeInt Q     = params->GetRingGSWParams1()->GetQ().ConvertToInt();
    if (a_stride < n) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE: a_stride < n");
    }
    if (N != ring_dim_ || numLUT != num_luts_) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE: GPU context params mismatch (recreate GPUCirBTSContext)");
    }
    if (!d_auto_maps_.get() || log_ring_dim_ == 0) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE: automorphism maps are not initialized on GPU");
    }
    if (!d_gpow_.get()) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE: Gpow is not initialized on GPU");
    }
    if (numLUT > 1 && !d_monomials_.get()) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE: monomials are not initialized on GPU");
    }

    const uint64_t bitwidth = static_cast<uint64_t>(std::ceil(std::log2(static_cast<double>(numLUT))));
    auto& s                = stream_wrapper_.get_stream();
    const auto& rlweParams = params->GetRLWEParams();
    const uint32_t traceShift = rlweParams->GetTraceShift();
    if (traceShift >= log_ring_dim_) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE: TraceShift must be smaller than log2(N)");
    }
    const uint32_t traceRounds = log_ring_dim_ - traceShift;
    const uint32_t baseHT  = rlweParams->GetBaseHT();
    const uint32_t baseSS  = rlweParams->GetBaseSS();
    const uint32_t digitsHT = rlweParams->GetDigitsHTA();
    const uint32_t digitsSS = rlweParams->GetDigitsSSA();
    const bool htss_soa_requested = (std::getenv("CIRBTS_HTSS_SOA") != nullptr);
    const bool htss_smem_requested = (std::getenv("CIRBTS_HTSS_SMEM") != nullptr);
    const bool htss_smem_verbose = (std::getenv("CIRBTS_HTSS_SMEM_VERBOSE") != nullptr);
    const bool htss_pipeline      = (std::getenv("CIRBTS_HTSS_PIPELINE") != nullptr);
    const bool has_htss_soa = d_HTkey_soa_.get() && d_HTkey_shoup_soa_.get() && d_SSkey_soa_.get() && d_SSkey_shoup_soa_.get() &&
                              (htkey_soa_tiles_ > 0) && (sskey_soa_tiles_ > 0);
    if ((htss_soa_requested || htss_smem_requested) && !has_htss_soa) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE: HT/SS SoA requested but not initialized (set CIRBTS_HTSS_SOA before init)");
    }
    if (htss_pipeline && (baseHT != baseSS || digitsHT != digitsSS)) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE: HT/SS pipeline requires baseHT==baseSS and digitsHT==digitsSS");
    }

    uint32_t htss_block_x = 256;
    if (const char* v = std::getenv("CIRBTS_HTSS_BLOCK_X"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0' && parsed > 0) {
            htss_block_x = static_cast<uint32_t>(std::min<unsigned long>(parsed, 1024ul));
        }
    }

    bool htssFuseInwtDecomp = false;
    bool htssFuseDecompFnwt = false;
    GetHtssFuseFlags(&htssFuseInwtDecomp, &htssFuseDecompFnwt);

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

    const uint32_t numLUTTotal  = numLUT * batch;
    const size_t lutSize        = static_cast<size_t>(numLUTTotal) * N;
    const uint64_t numRowsTotal = static_cast<uint64_t>(numLUTTotal) * 2;

    if (scratch_lut_size_ < lutSize) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE: scratch buffer too small (recreate GPUCirBTSContext)");
    }
    if (scratch_digits_max_ < std::max(digitsHT, digitsSS)) {
        OPENFHE_THROW(config_error,
                      "gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE: scratch digits mismatch (recreate GPUCirBTSContext)");
    }
    if (!d_ht_c0_.get() || !d_ht_c1_.get() || !d_perm_c0_.get() || !d_ht_next_c0_.get() || !d_ht_next_c1_.get() ||
        !d_ks_c0_.get() || !d_ks_c1_.get() || !d_digits_.get() || !d_ss_c0_.get() || !d_ss_c1_.get() ||
        !d_rgsw_out_.get()) {
        OPENFHE_THROW(config_error, "gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE: scratch buffers are not initialized on GPU");
    }

    const BatchScratchView scratch = make_batch_scratch_view();

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

    const bool profile_device = (std::getenv("CIRBTS_PROFILE_DEVICE") != nullptr);
    const char* htss_save_fusion_env = std::getenv("CIRBTS_HTSS_SAVE_FUSION");
    const bool htss_save_fusion = !(htss_save_fusion_env && htss_save_fusion_env[0] == '0');
    cudaEvent_t ev_start{}, ev_boot_end{}, ev_mvr_end{}, ev_htss_end{};
    if (profile_device) {
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_start));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_boot_end));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_mvr_end));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&ev_htss_end));
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev_start, s));
    }

    gpu_BootstrapLUT_inplace_batch_device(params, ek.RFkey, d_a, d_b, batch, a_stride, bitwidth, scratch);
    if (profile_device) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev_boot_end, s));
    }

    const NativeInt* d_acc = scratch.d_bootstrap_acc;
    {
        const size_t total = lutSize * 2;
        constexpr int blockSize = 256;
        const int numBlocks = static_cast<int>((total + blockSize - 1) / blockSize);
        kernel_GenMVRLWEs<<<numBlocks, blockSize, 0, s>>>(ht_c0, ht_c1, d_acc, d_monomials_.get(), d_modulus_.get(), N, numLUT, batch);
        PHANTOM_CHECK_CUDA_LAST();
    }
    if (profile_device) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev_mvr_end, s));
    }

    const bool disableHtssGraph = (std::getenv("CIRBTS_DISABLE_HTSS_CUDA_GRAPH") != nullptr);
    const bool canUseHtssGraph  = !disableHtssGraph && !htss_graph_failed_;
    const bool htss_l2_switch   = l2_persist_enabled_ && !canUseHtssGraph && !htss_pipeline;

    auto runHomTraceAndSchemeSwitch = [&]() {
        NativeInt* local_ht_c0      = ht_c0;
        NativeInt* local_ht_c1      = ht_c1;
        NativeInt* local_ht_next_c0 = ht_next_c0;
        NativeInt* local_ht_next_c1 = ht_next_c1;

        dim3 block(htss_block_x);
        dim3 grid((N + block.x - 1) / block.x, numLUTTotal);

        const bool use_htss_soa = (htss_soa_requested || htss_smem_requested) && has_htss_soa;
        const bool use_default_save_fusion = htss_save_fusion && !htss_pipeline && !use_htss_soa;
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

        for (uint32_t k = 0; k < traceRounds; ++k) {
            const uint32_t* d_map = d_auto_maps_.get() + static_cast<size_t>(k) * N;
            if (htssFuseInwtDecomp) {
                const size_t inwtShmem = N * sizeof(NativeInt);
                kernel_INWT_Permute_SignedDigitDecompose<<<numLUTTotal, N / 2, inwtShmem, s>>>(
                    digits, local_ht_c0, d_map, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                    d_n_inv_mod_q_shoup_.get(), baseHT, digitsHT, N, numLUTTotal);
                fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                    static_cast<size_t>(numLUTTotal) * digitsHT, s);
            } else {
                inwt_1d_opt_permute_batched(perm_c0, local_ht_c0, d_map, d_itwiddles_.get(), d_itwiddles_shoup_.get(),
                                            d_modulus_.get(), d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N,
                                            numLUTTotal, s);
                if (htssFuseDecompFnwt) {
                    const size_t ht_batch   = static_cast<size_t>(numLUTTotal) * digitsHT;
                    const size_t htNttShmem = N * sizeof(NativeInt);
                    kernel_SignedDigitDecompose_FusedFNWT<<<ht_batch, N / 2, htNttShmem, s>>>(
                        digits, perm_c0, Q, baseHT, digitsHT, N, numLUTTotal, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());
                } else {
                    kernel_Fused_Permute_Decompose<<<grid, block, 0, s>>>(digits, perm_c0, Q, baseHT, digitsHT, N, numLUTTotal);
                    fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                        static_cast<size_t>(numLUTTotal) * digitsHT, s);
                }
            }

            const size_t key_stride = size_HTRGSWkey_;
            const NativeInt* d_key  = ht_key_base + static_cast<size_t>(k) * key_stride;
            const NativeInt* d_key_shoup = ht_key_shoup_base + static_cast<size_t>(k) * key_stride;
            if (htss_pipeline) {
                if (use_htss_soa) {
                    if (use_htss_smem) {
                        kernel_MultAddUpdate_HTSS_PermuteC1_SoA_Smem<<<grid, block, htss_smem_bytes, s>>>(
                            local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                            ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N, htss_tiles,
                            numLUTTotal, lutSize);
                    } else {
                        kernel_MultAddUpdate_HTSS_PermuteC1_SoA<<<grid, block, 0, s>>>(
                            local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                            ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N, htss_tiles,
                            numLUTTotal, lutSize);
                    }
                } else {
                    kernel_MultAddUpdate_HTSS_PermuteC1<<<grid, block, 0, s>>>(
                        local_ht_next_c0, local_ht_next_c1, ss_c0, ss_c1, digits, d_key, d_key_shoup, ss_key_base,
                        ss_key_shoup_base, local_ht_c0, local_ht_c1, d_map, d_modulus_.get(), digitsHT, N, numLUTTotal,
                        lutSize);
                }
            } else if (use_htss_soa) {
                if (use_ht_smem) {
                    kernel_MultAddUpdate_HT_PermuteC1_SoA_Smem<<<grid, block, ht_smem_bytes, s>>>(
                        local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                        d_modulus_.get(), digitsHT, N, htss_tiles, numLUTTotal, lutSize);
                } else {
                    kernel_MultAddUpdate_HT_PermuteC1_SoA<<<grid, block, 0, s>>>(
                        local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                        d_modulus_.get(), digitsHT, N, htss_tiles, numLUTTotal, lutSize);
                }
            } else {
                if (use_default_save_fusion && (k + 1u == traceRounds)) {
                    kernel_MultAddUpdate_HT_PermuteC1_Save<<<grid, block, 0, s>>>(
                        local_ht_next_c0, local_ht_next_c1, rgsw_out, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                        d_modulus_.get(), digitsHT, N, numLUTTotal, lutSize);
                } else {
                    kernel_MultAddUpdate_HT_PermuteC1<<<grid, block, 0, s>>>(
                        local_ht_next_c0, local_ht_next_c1, digits, d_key, d_key_shoup, local_ht_c0, local_ht_c1, d_map,
                        d_modulus_.get(), digitsHT, N, numLUTTotal, lutSize);
                }
            }
            std::swap(local_ht_c0, local_ht_next_c0);
            std::swap(local_ht_c1, local_ht_next_c1);
        }

        if (!use_default_save_fusion) {
            kernel_SaveToRGSW<<<grid, block, 0, s>>>(rgsw_out, local_ht_c0, local_ht_c1, /*row_type=*/1, N, numLUTTotal);
        }

        if (htss_pipeline) {
            constexpr int addBlock = 256;
            const int addBlocks = static_cast<int>((lutSize + addBlock - 1) / addBlock);
            kernel_AddInPlace<<<addBlocks, addBlock, 0, s>>>(ss_c0, local_ht_c1, d_modulus_.get(), lutSize);
            kernel_SaveToRGSW<<<grid, block, 0, s>>>(rgsw_out, ss_c0, ss_c1, /*row_type=*/0, N, numLUTTotal);
            return;
        }

        if (htssFuseInwtDecomp) {
            const size_t inwtShmem = N * sizeof(NativeInt);
            kernel_INWT_SignedDigitDecompose<<<numLUTTotal, N / 2, inwtShmem, s>>>(
                digits, local_ht_c0, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                d_n_inv_mod_q_shoup_.get(), baseSS, digitsSS, N, numLUTTotal);
            fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                static_cast<size_t>(numLUTTotal) * digitsSS, s);
        } else {
            inwt_1d_opt_batched(local_ht_c0, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(),
                                d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N, numLUTTotal, s);
            if (htssFuseDecompFnwt) {
                const size_t ss_batch   = static_cast<size_t>(numLUTTotal) * digitsSS;
                const size_t ssNttShmem = N * sizeof(NativeInt);
                kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, ssNttShmem, s>>>(
                    digits, local_ht_c0, Q, baseSS, digitsSS, N, numLUTTotal, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());
            } else {
                kernel_Fused_Permute_Decompose<<<grid, block, 0, s>>>(digits, local_ht_c0, Q, baseSS, digitsSS, N, numLUTTotal);
                fnwt_1d_opt_batched(digits, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                                    static_cast<size_t>(numLUTTotal) * digitsSS, s);
            }
        }

        if (htss_l2_switch) {
            l2_scope.reset(ss_key_base, size_SSkey_ * N * sizeof(NativeInt));
        }
        if (use_htss_soa) {
            if (use_ss_smem) {
                kernel_MultAddUpdate_SS_SoA_Smem<<<grid, block, ss_smem_bytes, s>>>(
                    ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1, d_modulus_.get(), digitsSS, N,
                    htss_tiles, numLUTTotal, lutSize);
            } else {
                kernel_MultAddUpdate_SS_SoA<<<grid, block, 0, s>>>(
                    ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1, d_modulus_.get(), digitsSS, N,
                    htss_tiles, numLUTTotal, lutSize);
            }
        } else {
            if (use_default_save_fusion) {
                kernel_MultAddUpdate_SS_Save<<<grid, block, 0, s>>>(ss_c0, ss_c1, rgsw_out, digits, ss_key_base, ss_key_shoup_base,
                                                                    local_ht_c1, d_modulus_.get(), digitsSS, N, numLUTTotal, lutSize);
            } else {
                kernel_MultAddUpdate_SS<<<grid, block, 0, s>>>(ss_c0, ss_c1, digits, ss_key_base, ss_key_shoup_base, local_ht_c1,
                                                               d_modulus_.get(), digitsSS, N, numLUTTotal, lutSize);
            }
        }

        if (!use_default_save_fusion) {
            kernel_SaveToRGSW<<<grid, block, 0, s>>>(rgsw_out, ss_c0, ss_c1, /*row_type=*/0, N, numLUTTotal);
        }
    };

    if (canUseHtssGraph) {
        const bool use_htss_soa_for_key = (htss_soa_requested || htss_smem_requested) && has_htss_soa;
        const bool use_default_save_fusion_for_key = htss_save_fusion && !htss_pipeline && !use_htss_soa_for_key;
        if (!htss_graph_exec_ || htss_graph_block_x_ != htss_block_x || htss_graph_num_luts_ != numLUTTotal ||
            htss_graph_fuse_inwt_decomp_ != htssFuseInwtDecomp || htss_graph_fuse_decomp_fnwt_ != htssFuseDecompFnwt ||
            htss_graph_pipeline_ != htss_pipeline || htss_graph_smem_ != htss_graph_smem ||
            htss_graph_save_fusion_ != use_default_save_fusion_for_key) {
            if (htss_graph_exec_) {
                cudaGraphExecDestroy(htss_graph_exec_);
                htss_graph_exec_ = nullptr;
            }
            htss_graph_block_x_            = htss_block_x;
            htss_graph_num_luts_           = numLUTTotal;
            htss_graph_fuse_inwt_decomp_   = htssFuseInwtDecomp;
            htss_graph_fuse_decomp_fnwt_   = htssFuseDecompFnwt;
            htss_graph_pipeline_           = htss_pipeline;
            htss_graph_smem_               = htss_graph_smem;
            htss_graph_save_fusion_        = use_default_save_fusion_for_key;

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
            } else {
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
        } else {
            runHomTraceAndSchemeSwitch();
        }
    } else {
        runHomTraceAndSchemeSwitch();
    }

    if (profile_device) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(ev_htss_end, s));
        PHANTOM_CHECK_CUDA(cudaEventSynchronize(ev_htss_end));
        float ms_boot = 0.0f;
        float ms_mvr = 0.0f;
        float ms_htss = 0.0f;
        float ms_total = 0.0f;
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_boot, ev_start, ev_boot_end));
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_mvr, ev_boot_end, ev_mvr_end));
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_htss, ev_mvr_end, ev_htss_end));
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms_total, ev_start, ev_htss_end));
        std::cout << "[CIRBTS_PROFILE_DEVICE] batch=" << batch
                  << " num_lut_total=" << numLUTTotal
                  << " bootstrap_ms=" << ms_boot
                  << " mvr_ms=" << ms_mvr
                  << " htss_ms=" << ms_htss
                  << " total_ms=" << ms_total
                  << " htss_graph=" << (htss_graph_exec_ ? 1 : 0)
                  << " htss_soa=" << (((htss_soa_requested || htss_smem_requested) && has_htss_soa) ? 1 : 0)
                  << " htss_smem=" << (htss_graph_smem ? 1 : 0)
                  << " htss_pipeline=" << (htss_pipeline ? 1 : 0)
                  << " htss_save_fusion=" << (htss_save_fusion ? 1 : 0)
                  << std::endl;
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_boot_end);
        cudaEventDestroy(ev_mvr_end);
        cudaEventDestroy(ev_htss_end);
    }

    DeviceRGSWBatchView view{};
    view.d_rgsw = rgsw_out;
    view.batch = batch;
    view.digitsCC = numLUT;
    view.N = static_cast<uint32_t>(N);
    view.per_ciphertext_elems = static_cast<size_t>(numLUT) * 4 * N;
    view.stream = s;
    return view;
}


void GPUCirBTSContext::gpu_BootstrapLUT_inplace_batch(const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params,
                                                      ConstRingGSWACCKey& ek,
                                                      const std::vector<lbcrypto::LWECiphertext>& cts,
                                                      uint64_t bitwidth,
                                                      const BatchScratchView& scratch,
                                                      uint32_t batch) const {
    if (cts.empty() || batch == 0) {
        return;
    }
    if (!ek) {
        OPENFHE_THROW(config_error,
                      "Bootstrapping keys have not been generated. Please call BTKeyGen before calling bootstrapping.");
    }
    const uint32_t actual_batch = static_cast<uint32_t>(cts.size());
    if (batch != actual_batch) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: batch size mismatch with ciphertext vector");
    }
    if (batch > std::max<uint32_t>(1u, max_batch_size_)) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: batch size exceeds max_batch_size_ (recreate GPUCirBTSContext)");
    }
    if (!scratch.d_bootstrap_acc || !d_lut_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: bootstrap scratch buffers are not initialized on GPU");
    }
    if (!d_gpow_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: Gpow is not initialized on GPU");
    }
    if (!scratch.d_monomial_inv) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: monomial scratch buffer is not initialized on GPU");
    }

    const auto& rgswParams1 = params->GetRingGSWParams1();
    if (rgswParams1->GetMethod() != BINFHE_METHOD::GINX) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: currently supports method=GINX only");
    }
    if (!d_RFkey_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: RF key is not initialized on GPU");
    }
    if (!d_RFkey_shoup_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: RF key Shoup table is not initialized on GPU");
    }
    if (!d_monic_polys_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: monic polys are not initialized on GPU");
    }
    if (!scratch.d_evalacc_ct || !scratch.d_evalacc_dct || !scratch.d_evalacc_indexPos) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: EvalAcc scratch buffers are not initialized on GPU");
    }

    const uint32_t numLUT = params->GetDigitsCC();
    const auto& lweParams = params->GetLWEParams();
    const auto q          = lweParams->Getq();
    const uint32_t n      = lweParams->Getn();

    const auto polyParams = rgswParams1->GetPolyParams();
    const auto Q          = rgswParams1->GetQ();
    const size_t N        = polyParams->GetRingDimension();
    const auto N_inv      = NativeInteger(N).ModInverse(Q);
    const uint32_t twoN = static_cast<uint32_t>(N * 2);
    const NativeInt Q_lwe = q.ConvertToInt<NativeInt>();
    const uint32_t bitwidth_u32 = static_cast<uint32_t>(bitwidth);
    const NativeInt* d_a = nullptr;
    const uint32_t a_stride = 0;
    bool use_evalacc_fuse_ms = false;
    const char* index_nmajor_env = std::getenv("CIRBTS_BATCH_INDEX_NMAJOR");
    const bool index_nmajor_requested = (index_nmajor_env != nullptr) && (index_nmajor_env[0] != '0');

    if (N != ring_dim_ || numLUT != num_luts_) {
        OPENFHE_THROW(config_error,
                      "gpu_BootstrapLUT_inplace_batch: GPU context params mismatch (recreate GPUCirBTSContext)");
    }
    if (!d_modulus_.get() || !d_twiddles_.get() || !d_twiddles_shoup_.get() || !d_itwiddles_.get() || !d_itwiddles_shoup_.get() ||
        !d_n_inv_mod_q_.get() || !d_n_inv_mod_q_shoup_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: NTT tables are not initialized on GPU");
    }

    auto& s = scratch.stream;

    const uint32_t digitsGA = rgswParams1->GetDigitsGA();
    const uint32_t digitsG2 = digitsGA << 1;
    const NativeInt Qint    = Q.ConvertToInt();
    const uint32_t baseG    = rgswParams1->GetBaseG();
    const bool use_swizzle_path = use_swizzle_ && (digitsG2 == 2u) && rfkey_swizzle_tiles_ &&
                                  d_RFkey_swizzle_.get() && d_RFkey_shoup_swizzle_.get();
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
            OPENFHE_THROW(config_error,
                          "gpu_BootstrapLUT_inplace_batch: EvalAcc SoA requested but not initialized (set CIRBTS_EVALACC_SOA before init)");
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
                          "gpu_BootstrapLUT_inplace_batch: EvalAcc SMEM requires EvalAcc SoA (set CIRBTS_EVALACC_SOA_FORCE=1)");
        }
        if (evalacc_smem_verbose) {
            std::cout << "[CIRBTS] EvalAcc SMEM disabled: EvalAcc SoA unavailable." << std::endl;
        }
    }
    bool use_swizzle_evalacc = use_swizzle_path && !use_evalacc_soa;
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
    if (pbs_multibit != 0u && pbs_multibit != 2u && pbs_multibit != 3u && pbs_multibit != 4u) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: unsupported CIRBTS_PBS_MULTIBIT (supported: 0,2,3,4)");
    }
    const bool use_pbs_multibit2 = (pbs_multibit == 2u);
    const bool use_pbs_multibit3 = (pbs_multibit == 3u);
    const bool use_pbs_multibit4 = (pbs_multibit == 4u);
    const bool use_evalacc_multibit2 = (pbs_multibit == 0u && evalacc_multibit == 2u);
    const bool use_multibit2 = use_pbs_multibit2 || use_evalacc_multibit2;
    const bool use_multibit3 = use_pbs_multibit3;
    const bool use_multibit4 = use_pbs_multibit4;
    if (use_swizzle_evalacc || use_pbs_multibit2 || use_pbs_multibit3 || use_pbs_multibit4) {
        use_evalacc_fuse_ms = false;
    }
    if (use_multibit3 && use_evalacc_multibit2) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: conflicting multibit configuration");
    }
    if (use_multibit4 && use_evalacc_multibit2) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: conflicting multibit configuration");
    }
    if (use_multibit2) {
        if (use_pbs_multibit2) {
            if (!rfkey_grouped_ || rfkey_dim2_ != 3u || rfkey_groups_ == 0) {
                OPENFHE_THROW(config_error,
                              "gpu_BootstrapLUT_inplace_batch: PBS multibit=2 requires grouped RFkey (set CIRBTS_PBS_MULTIBIT=2 before keygen)");
            }
        } else if (rfkey_dim2_ < 2) {
            OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: multibit=2 requires RFkey dim2=2");
        }
        if (!use_pbs_multibit2 && use_swizzle_path) {
            OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: multibit=2 does not support swizzle");
        }
    }
    bool use_multibit3_experimental = false;
    if (use_multibit3) {
        if (!rfkey_grouped_ || rfkey_dim2_ != 7u || rfkey_groups_ == 0) {
            OPENFHE_THROW(config_error,
                          "gpu_BootstrapLUT_inplace_batch: PBS multibit=3 requires grouped RFkey dim2=7 (set CIRBTS_PBS_MULTIBIT=3 before keygen)");
        }
        const char* mb3_exp_env = std::getenv("CIRBTS_MB3_EXPERIMENTAL");
        use_multibit3_experimental = (mb3_exp_env != nullptr) && (mb3_exp_env[0] != '0');
        if (!use_multibit3_experimental) {
            static bool printed = false;
            if (!printed) {
                std::cout << "[CIRBTS] multibit=3 baseline: forcing non-swizzle/non-SoA/non-SMEM/non-NMajor EvalAcc path."
                          << std::endl;
                printed = true;
            }
            use_evalacc_soa = false;
            use_swizzle_evalacc = false;
        } else {
            if (use_swizzle_evalacc) {
                static bool printed = false;
                if (!printed) {
                    std::cout << "[CIRBTS] multibit=3: forcing non-swizzle EvalAcc path." << std::endl;
                    printed = true;
                }
                use_swizzle_evalacc = false;
            }
            if (evalacc_smem_requested) {
                static bool printed_smem = false;
                if (!printed_smem) {
                    std::cout << "[CIRBTS] multibit=3: enabling experimental EvalAcc SMEM (main-key staging)." << std::endl;
                    printed_smem = true;
                }
            }
        }
    }
    bool use_multibit4_experimental = false;
    if (use_multibit4) {
        if (!rfkey_grouped_ || rfkey_dim2_ != 8u || rfkey_groups_ == 0) {
            OPENFHE_THROW(config_error,
                          "gpu_BootstrapLUT_inplace_batch: PBS multibit=4 requires grouped RFkey dim2=8 (set CIRBTS_PBS_MULTIBIT=4 before keygen)");
        }
        const char* mb4_exp_env = std::getenv("CIRBTS_MB4_EXPERIMENTAL");
        use_multibit4_experimental = (mb4_exp_env == nullptr) || (mb4_exp_env[0] != '0');
        if (!use_multibit4_experimental) {
            static bool printed = false;
            if (!printed) {
                std::cout << "[CIRBTS] multibit=4 baseline: forcing non-swizzle/non-SoA/non-SMEM/non-NMajor EvalAcc path."
                          << std::endl;
                printed = true;
            }
            use_evalacc_soa = false;
            use_swizzle_evalacc = false;
        } else {
            if (use_swizzle_evalacc) {
                static bool printed = false;
                if (!printed) {
                    std::cout << "[CIRBTS] multibit=4: forcing non-swizzle EvalAcc path." << std::endl;
                    printed = true;
                }
                use_swizzle_evalacc = false;
            }
            if (evalacc_smem_requested) {
                static bool printed_smem = false;
                if (!printed_smem) {
                    std::cout << "[CIRBTS] multibit=4: enabling experimental EvalAcc SMEM (main-key staging)." << std::endl;
                    printed_smem = true;
                }
            }
        }
    }

    bool index_nmajor = index_nmajor_requested;
    if (use_multibit3 && !use_multibit3_experimental && index_nmajor) {
        static bool printed = false;
        if (!printed) {
            std::cout << "[CIRBTS] multibit=3 baseline: disabling batch index N-major." << std::endl;
            printed = true;
        }
        index_nmajor = false;
    }
    if (use_multibit4 && !use_multibit4_experimental && index_nmajor) {
        static bool printed = false;
        if (!printed) {
            std::cout << "[CIRBTS] multibit=4 baseline: disabling batch index N-major." << std::endl;
            printed = true;
        }
        index_nmajor = false;
    }
    if (index_nmajor) {
        if (use_evalacc_fuse_ms || use_swizzle_evalacc) {
            static bool printed = false;
            if (!printed) {
                std::cout << "[CIRBTS] Batch index N-major disabled: incompatible with current EvalAcc options." << std::endl;
                printed = true;
            }
            index_nmajor = false;
        }
    }
    const uint32_t index_stride = index_nmajor ? batch : n;

    // Special modulus switching (CPU-side) + precompute:
    // - monomial_inv[b] = X^{-b_ms} (evaluation)
    // - indexPos[b][i]  = a_ms[i] in [0..2N)
    const size_t mono_elems  = static_cast<size_t>(batch) * N;
    const size_t index_elems = static_cast<size_t>(batch) * n;
    const bool use_h2d_pipeline =
        scratch.copy_stream && scratch.h2d_monomial_event && scratch.h2d_index_event &&
        *scratch.h2d_monomial_event && *scratch.h2d_index_event &&
        scratch.h_monomial_inv_pinned && scratch.h_indexPos_pinned;

    std::vector<NativeInt> host_monomial_inv;
    std::vector<uint32_t> host_indexPos;
    NativeInt* host_monomial_inv_ptr = nullptr;
    uint32_t* host_indexPos_ptr      = nullptr;
    if (use_h2d_pipeline) {
        host_monomial_inv_ptr = scratch.h_monomial_inv_pinned;
        host_indexPos_ptr     = scratch.h_indexPos_pinned;
    }
    else {
        host_monomial_inv.resize(mono_elems);
        host_indexPos.resize(index_elems);
        host_monomial_inv_ptr = host_monomial_inv.data();
        host_indexPos_ptr     = host_indexPos.data();
    }

    const bool profile_h2d = (std::getenv("CIRBTS_PROFILE_H2D") != nullptr);
    const auto h2d_prep0 = std::chrono::steady_clock::now();
    for (uint32_t b = 0; b < batch; ++b) {
        const auto& ct = cts[b];
        if (!ct) {
            OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch: ciphertext contains null pointer");
        }
        const auto b_ms = gpu_SpecilMS(ct->GetB(), NativeInteger(2 * N), q, bitwidth);
        const uint32_t b_idx = b_ms.ConvertToInt<uint32_t>();
        const auto b_mono = params->GetMonomial(b_idx);
        std::memcpy(host_monomial_inv_ptr + static_cast<size_t>(b) * N, &b_mono.GetValues().at(0), N * sizeof(NativeInt));

        const auto a = ct->GetA();
        for (uint32_t i = 0; i < n; ++i) {
            const NativeInteger ai_ms = gpu_SpecilMS(a[i], NativeInteger(2 * N), q, bitwidth);
            const size_t idx = index_nmajor ? (static_cast<size_t>(i) * batch + b)
                                            : (static_cast<size_t>(b) * n + i);
            host_indexPos_ptr[idx] = ai_ms.ConvertToInt<uint32_t>();
        }
    }
    const auto h2d_prep1 = std::chrono::steady_clock::now();

    cudaEvent_t h2d_start_ev{};
    cudaEvent_t h2d_end_ev{};
    cudaStream_t h2d_stream = use_h2d_pipeline ? scratch.copy_stream : s;
    if (profile_h2d) {
        PHANTOM_CHECK_CUDA(cudaEventCreate(&h2d_start_ev));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&h2d_end_ev));
        PHANTOM_CHECK_CUDA(cudaEventRecord(h2d_start_ev, h2d_stream));
    }

    if (use_h2d_pipeline) {
        cudaMemcpyAsync(scratch.d_monomial_inv, host_monomial_inv_ptr, mono_elems * sizeof(NativeInt), cudaMemcpyHostToDevice,
                        scratch.copy_stream);
        PHANTOM_CHECK_CUDA(cudaEventRecord(*scratch.h2d_monomial_event, scratch.copy_stream));
        cudaMemcpyAsync(scratch.d_evalacc_indexPos, host_indexPos_ptr, index_elems * sizeof(uint32_t), cudaMemcpyHostToDevice,
                        scratch.copy_stream);
        PHANTOM_CHECK_CUDA(cudaEventRecord(*scratch.h2d_index_event, scratch.copy_stream));
    }
    else {
        cudaMemcpyAsync(scratch.d_monomial_inv, host_monomial_inv_ptr, mono_elems * sizeof(NativeInt), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(scratch.d_evalacc_indexPos, host_indexPos_ptr, index_elems * sizeof(uint32_t), cudaMemcpyHostToDevice, s);
    }

    if (profile_h2d) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(h2d_end_ev, h2d_stream));
        PHANTOM_CHECK_CUDA(cudaEventSynchronize(h2d_end_ev));
        float h2d_ms = 0.0f;
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&h2d_ms, h2d_start_ev, h2d_end_ev));
        std::cout << "[CIRBTS_PROFILE_H2D] batch=" << batch
                  << " prep_ms=" << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(h2d_prep1 - h2d_prep0).count()
                  << " h2d_ms=" << static_cast<double>(h2d_ms)
                  << " monomial_bytes=" << (mono_elems * sizeof(NativeInt))
                  << " index_bytes=" << (index_elems * sizeof(uint32_t))
                  << std::endl;
        cudaEventDestroy(h2d_start_ev);
        cudaEventDestroy(h2d_end_ev);
    }

    // ACC starts as (0, LUT) in evaluation form for each ciphertext in the batch.
    constexpr uint32_t initBlockX = 256;
    dim3 initBlock(initBlockX);
    dim3 initGrid((N + initBlock.x - 1) / initBlock.x, batch);
    kernel_InitBootstrapAccBatch<<<initGrid, initBlock, 0, s>>>(scratch.d_bootstrap_acc, d_lut_.get(), N, batch);

    // Multiply with X^{-b_MS} (per ciphertext).
    if (use_h2d_pipeline) {
        PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, *scratch.h2d_monomial_event, 0));
    }
    kernel_ModMulScalarBatch<<<initGrid, initBlock, 0, s>>>(scratch.d_bootstrap_acc, scratch.d_monomial_inv, d_modulus_.get(), N, batch);
    const bool use_grouped_multibit = use_pbs_multibit2 || use_pbs_multibit3 || use_pbs_multibit4;
    const size_t key_block = size_RFRGSWkey_ * static_cast<size_t>(use_grouped_multibit ? rfkey_groups_ : n);
    const NativeInt* rfkey_base = use_evalacc_soa ? d_RFkey_soa_.get() :
                                  (use_swizzle_evalacc ? d_RFkey_swizzle_.get() : d_RFkey_.get());
    const NativeInt* rfkey_shoup_base = use_evalacc_soa ? d_RFkey_shoup_soa_.get() :
                                        (use_swizzle_evalacc ? d_RFkey_shoup_swizzle_.get() : d_RFkey_shoup_.get());
    const NativeInt* ek_pair = nullptr;
    const NativeInt* ek_pair_shoup = nullptr;
    const NativeInt* ek_pair_soa = nullptr;
    const NativeInt* ek_pair_shoup_soa = nullptr;
    const NativeInt* ek0 = nullptr;
    const NativeInt* ek1 = nullptr;
    const NativeInt* ek2 = nullptr;
    const NativeInt* ek3 = nullptr;
    const NativeInt* ek01 = nullptr;
    const NativeInt* ek02 = nullptr;
    const NativeInt* ek12 = nullptr;
    const NativeInt* ek012 = nullptr;
    const NativeInt* ek0_shoup = nullptr;
    const NativeInt* ek1_shoup = nullptr;
    const NativeInt* ek2_shoup = nullptr;
    const NativeInt* ek3_shoup = nullptr;
    const NativeInt* ek01_shoup = nullptr;
    const NativeInt* ek02_shoup = nullptr;
    const NativeInt* ek12_shoup = nullptr;
    const NativeInt* ek012_shoup = nullptr;
    const NativeInt* ek0_soa = nullptr;
    const NativeInt* ek1_soa = nullptr;
    const NativeInt* ek2_soa = nullptr;
    const NativeInt* ek3_soa = nullptr;
    const NativeInt* ek01_soa = nullptr;
    const NativeInt* ek02_soa = nullptr;
    const NativeInt* ek12_soa = nullptr;
    const NativeInt* ek012_soa = nullptr;
    const NativeInt* ek0_shoup_soa = nullptr;
    const NativeInt* ek1_shoup_soa = nullptr;
    const NativeInt* ek2_shoup_soa = nullptr;
    const NativeInt* ek3_shoup_soa = nullptr;
    const NativeInt* ek01_shoup_soa = nullptr;
    const NativeInt* ek02_shoup_soa = nullptr;
    const NativeInt* ek12_shoup_soa = nullptr;
    const NativeInt* ek012_shoup_soa = nullptr;
    if (use_multibit2) {
        if (use_pbs_multibit2) {
            ek0 = rfkey_base;
            ek1 = rfkey_base + key_block;
            ek_pair = rfkey_base + 2 * key_block;
            ek0_shoup = rfkey_shoup_base;
            ek1_shoup = rfkey_shoup_base + key_block;
            ek_pair_shoup = rfkey_shoup_base + 2 * key_block;
            if (use_evalacc_soa) {
                ek0_soa = d_RFkey_soa_.get();
                ek1_soa = d_RFkey_soa_.get() + key_block;
                ek_pair_soa = d_RFkey_soa_.get() + 2 * key_block;
                ek0_shoup_soa = d_RFkey_shoup_soa_.get();
                ek1_shoup_soa = d_RFkey_shoup_soa_.get() + key_block;
                ek_pair_shoup_soa = d_RFkey_shoup_soa_.get() + 2 * key_block;
            }
        } else {
            ek_pair = rfkey_base + key_block;
            ek_pair_shoup = rfkey_shoup_base + key_block;
            if (use_evalacc_soa) {
                ek_pair_soa = d_RFkey_soa_.get() + key_block;
                ek_pair_shoup_soa = d_RFkey_shoup_soa_.get() + key_block;
            }
        }
    }
    if (use_multibit3) {
        ek0 = rfkey_base;
        ek1 = rfkey_base + key_block;
        ek2 = rfkey_base + 2 * key_block;
        ek01 = rfkey_base + 3 * key_block;
        ek02 = rfkey_base + 4 * key_block;
        ek12 = rfkey_base + 5 * key_block;
        ek012 = rfkey_base + 6 * key_block;
        ek0_shoup = rfkey_shoup_base;
        ek1_shoup = rfkey_shoup_base + key_block;
        ek2_shoup = rfkey_shoup_base + 2 * key_block;
        ek01_shoup = rfkey_shoup_base + 3 * key_block;
        ek02_shoup = rfkey_shoup_base + 4 * key_block;
        ek12_shoup = rfkey_shoup_base + 5 * key_block;
        ek012_shoup = rfkey_shoup_base + 6 * key_block;
        if (use_evalacc_soa) {
            ek0_soa = d_RFkey_soa_.get();
            ek1_soa = d_RFkey_soa_.get() + key_block;
            ek2_soa = d_RFkey_soa_.get() + 2 * key_block;
            ek01_soa = d_RFkey_soa_.get() + 3 * key_block;
            ek02_soa = d_RFkey_soa_.get() + 4 * key_block;
            ek12_soa = d_RFkey_soa_.get() + 5 * key_block;
            ek012_soa = d_RFkey_soa_.get() + 6 * key_block;
            ek0_shoup_soa = d_RFkey_shoup_soa_.get();
            ek1_shoup_soa = d_RFkey_shoup_soa_.get() + key_block;
            ek2_shoup_soa = d_RFkey_shoup_soa_.get() + 2 * key_block;
            ek01_shoup_soa = d_RFkey_shoup_soa_.get() + 3 * key_block;
            ek02_shoup_soa = d_RFkey_shoup_soa_.get() + 4 * key_block;
            ek12_shoup_soa = d_RFkey_shoup_soa_.get() + 5 * key_block;
            ek012_shoup_soa = d_RFkey_shoup_soa_.get() + 6 * key_block;
        }
    }
    if (use_multibit4) {
        ek0 = rfkey_base;
        ek1 = rfkey_base + key_block;
        ek2 = rfkey_base + 2 * key_block;
        ek3 = rfkey_base + 3 * key_block;
        ek01 = rfkey_base + 4 * key_block;
        ek02 = rfkey_base + 5 * key_block;
        ek12 = rfkey_base + 6 * key_block;
        ek012 = rfkey_base + 7 * key_block;
        ek0_shoup = rfkey_shoup_base;
        ek1_shoup = rfkey_shoup_base + key_block;
        ek2_shoup = rfkey_shoup_base + 2 * key_block;
        ek3_shoup = rfkey_shoup_base + 3 * key_block;
        ek01_shoup = rfkey_shoup_base + 4 * key_block;
        ek02_shoup = rfkey_shoup_base + 5 * key_block;
        ek12_shoup = rfkey_shoup_base + 6 * key_block;
        ek012_shoup = rfkey_shoup_base + 7 * key_block;
        if (use_evalacc_soa) {
            ek0_soa = d_RFkey_soa_.get();
            ek1_soa = d_RFkey_soa_.get() + key_block;
            ek2_soa = d_RFkey_soa_.get() + 2 * key_block;
            ek3_soa = d_RFkey_soa_.get() + 3 * key_block;
            ek01_soa = d_RFkey_soa_.get() + 4 * key_block;
            ek02_soa = d_RFkey_soa_.get() + 5 * key_block;
            ek12_soa = d_RFkey_soa_.get() + 6 * key_block;
            ek012_soa = d_RFkey_soa_.get() + 7 * key_block;
            ek0_shoup_soa = d_RFkey_shoup_soa_.get();
            ek1_shoup_soa = d_RFkey_shoup_soa_.get() + key_block;
            ek2_shoup_soa = d_RFkey_shoup_soa_.get() + 2 * key_block;
            ek3_shoup_soa = d_RFkey_shoup_soa_.get() + 3 * key_block;
            ek01_shoup_soa = d_RFkey_shoup_soa_.get() + 4 * key_block;
            ek02_shoup_soa = d_RFkey_shoup_soa_.get() + 5 * key_block;
            ek12_shoup_soa = d_RFkey_shoup_soa_.get() + 6 * key_block;
            ek012_shoup_soa = d_RFkey_shoup_soa_.get() + 7 * key_block;
        }
    }

    L2PersistScope l2_scope(this, s);
    {
        const NativeInt* key_base = use_evalacc_soa ? d_RFkey_soa_.get() :
                                    (use_swizzle_path ? d_RFkey_swizzle_.get() : d_RFkey_.get());
        l2_scope.reset(key_base, size_RFkey_ * size_RFRGSWkey_ * sizeof(NativeInt));
    }

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
        }
    }

    uint32_t evalaccBlockX = 256;
    if (const char* v = std::getenv("CIRBTS_EVALACC_BATCH_BLOCK_X"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0' && parsed > 0) {
            evalaccBlockX = static_cast<uint32_t>(std::min<unsigned long>(parsed, 1024ul));
            evalaccBlockX = (evalaccBlockX / 32) * 32;
            if (evalaccBlockX == 0) {
                evalaccBlockX = 32;
            }
        }
    }
    else {
        if (use_pbs_multibit2 || use_pbs_multibit3 || use_pbs_multibit4) {
            // MB3 has higher per-thread state than MB2; prefer a smaller default block
            // to improve occupancy while still keeping warp-aligned scheduling.
            evalaccBlockX = (use_pbs_multibit3 || use_pbs_multibit4) ? 128u : 256u;
        } else {
            if (evalacc_batch_block_x_ == 0 || evalacc_batch_block_batch_ != batch) {
            phantom::util::cuda_auto_ptr<NativeInt> acc_backup =
                phantom::util::make_cuda_auto_ptr<NativeInt>(static_cast<size_t>(2) * N * batch, s);
            cudaMemcpyAsync(acc_backup.get(), scratch.d_bootstrap_acc, static_cast<size_t>(2) * N * batch * sizeof(NativeInt),
                            cudaMemcpyDeviceToDevice, s);

            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
            cudaEvent_t start{}, stop{};
            PHANTOM_CHECK_CUDA(cudaEventCreate(&start));
            PHANTOM_CHECK_CUDA(cudaEventCreate(&stop));

            const uint32_t testIters = std::min<uint32_t>(n, 8u);
            const std::array<uint32_t, 3> candidates{64u, 128u, 256u};
            float bestMs = std::numeric_limits<float>::infinity();
            uint32_t bestBlock = candidates[1];
            const uint32_t evalacc_soa_tiles = use_evalacc_soa ? static_cast<uint32_t>(rfkey_soa_tiles_) : 0u;

            const size_t evalaccNttShmem = N * sizeof(NativeInt);
            for (uint32_t cand : candidates) {
                cudaMemcpyAsync(scratch.d_bootstrap_acc, acc_backup.get(), static_cast<size_t>(2) * N * batch * sizeof(NativeInt),
                                cudaMemcpyDeviceToDevice, s);
                PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));

                dim3 block(cand);
                dim3 grid((N + block.x - 1) / block.x, batch);

                PHANTOM_CHECK_CUDA(cudaEventRecord(start, s));
                for (uint32_t lweIndex = 0; lweIndex < testIters; ) {
                    inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                                   d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                                   d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

                    kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                        scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                        d_twiddles_shoup_.get(), d_modulus_.get());

                    if (use_multibit2 && (lweIndex + 1 < testIters)) {
                        if (use_evalacc_soa) {
                            if (index_nmajor) {
                                kernel_EvalAccCore_Binary_MB2_Batch_SoA_NMajor<<<grid, block, 0, s>>>(
                                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                                    ek_pair_soa, ek_pair_shoup_soa,
                                    d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2,
                                    lweIndex, lweIndex + 1, scratch.d_evalacc_indexPos, index_stride);
                            } else {
                                kernel_EvalAccCore_Binary_MB2_Batch_SoA<<<grid, block, 0, s>>>(
                                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                                    ek_pair_soa, ek_pair_shoup_soa,
                                    d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2,
                                    lweIndex, lweIndex + 1, scratch.d_evalacc_indexPos, index_stride);
                            }
                        } else {
                            if (index_nmajor) {
                                kernel_EvalAccCore_Binary_MB2_Batch_NMajor<<<grid, block, 0, s>>>(
                                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_.get(), d_RFkey_shoup_.get(), ek_pair,
                                    ek_pair_shoup, d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, lweIndex, lweIndex + 1,
                                    scratch.d_evalacc_indexPos, index_stride);
                            } else {
                                kernel_EvalAccCore_Binary_MB2_Batch<<<grid, block, 0, s>>>(
                                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_.get(), d_RFkey_shoup_.get(), ek_pair,
                                    ek_pair_shoup, d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, lweIndex, lweIndex + 1,
                                    scratch.d_evalacc_indexPos, index_stride);
                            }
                        }
                        lweIndex += 2;
                    }
                    else if (use_swizzle_evalacc) {
                        kernel_EvalAccCore_Binary_Batch_Swizzle<2><<<grid, block, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_swizzle_.get(), d_RFkey_shoup_swizzle_.get(),
                            d_monic_polys_.get(), N, static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(), lweIndex,
                            scratch.d_evalacc_indexPos, index_stride);
                        ++lweIndex;
                    } else if (use_evalacc_soa) {
                        if (index_nmajor) {
                            kernel_EvalAccCore_Binary_Batch_SoA_NMajor<<<grid, block, 0, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                                d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, lweIndex,
                                scratch.d_evalacc_indexPos, index_stride);
                        } else {
                            kernel_EvalAccCore_Binary_Batch_SoA<<<grid, block, 0, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                                d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, lweIndex,
                                scratch.d_evalacc_indexPos, index_stride);
                        }
                        ++lweIndex;
                    } else {
                        kernel_EvalAccCore_Binary_Batch<<<grid, block, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_.get(), d_RFkey_shoup_.get(), d_monic_polys_.get(), N,
                            d_modulus_.get(), digitsG2, lweIndex, scratch.d_evalacc_indexPos, index_stride);
                        ++lweIndex;
                    }
                }
                PHANTOM_CHECK_CUDA(cudaEventRecord(stop, s));
                PHANTOM_CHECK_CUDA(cudaEventSynchronize(stop));

                float ms = 0.0f;
                PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
                if (ms < bestMs) {
                    bestMs = ms;
                    bestBlock = cand;
                }
            }

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            evalacc_batch_block_x_ = bestBlock;
            evalacc_batch_block_batch_ = batch;

            cudaMemcpyAsync(scratch.d_bootstrap_acc, acc_backup.get(), static_cast<size_t>(2) * N * batch * sizeof(NativeInt),
                            cudaMemcpyDeviceToDevice, s);
            }
            evalaccBlockX = evalacc_batch_block_x_;
        }
    }

    const uint32_t evalacc_soa_tiles = use_evalacc_soa ? static_cast<uint32_t>(rfkey_soa_tiles_) : 0u;
    size_t evalacc_smem_bytes = 0;
    bool evalacc_smem_enabled = false;
    if (use_evalacc_soa && evalacc_smem_requested) {
        int dev = 0;
        int max_default = 0;
        int max_opt = 0;
        PHANTOM_CHECK_CUDA(cudaGetDevice(&dev));
        PHANTOM_CHECK_CUDA(cudaDeviceGetAttribute(&max_default, cudaDevAttrMaxSharedMemoryPerBlock, dev));
        (void)cudaDeviceGetAttribute(&max_opt, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        const size_t max_smem = (max_opt > 0) ? static_cast<size_t>(max_opt) : static_cast<size_t>(max_default);
        const uint32_t warps = (evalaccBlockX + 31u) >> 5;
        const size_t warp_stride = (use_multibit3 || use_multibit4) ? (12u * 32u) : (use_multibit2 ? (8u * 32u) : (4u * 32u));
        evalacc_smem_bytes = static_cast<size_t>(warps) * 2u * warp_stride * sizeof(NativeInt);
        if (evalacc_smem_bytes <= max_smem) {
            if (evalacc_smem_bytes > static_cast<size_t>(max_default)) {
                cudaError_t st0 = cudaErrorUnknown;
                cudaError_t st1 = cudaErrorUnknown;
                if (use_multibit3 || use_multibit4) {
                    st0 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(evalacc_smem_bytes));
                    if (index_nmajor) {
                        st1 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem_NMajor,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   static_cast<int>(evalacc_smem_bytes));
                        evalacc_smem_enabled = (st0 == cudaSuccess) && (st1 == cudaSuccess);
                    } else {
                        evalacc_smem_enabled = (st0 == cudaSuccess);
                    }
                } else if (use_multibit2) {
                    if (use_pbs_multibit2) {
                        st0 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_Smem,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   static_cast<int>(evalacc_smem_bytes));
                        st1 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   static_cast<int>(evalacc_smem_bytes));
                        if (index_nmajor) {
                            cudaError_t st2 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_Smem_NMajor,
                                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                                   static_cast<int>(evalacc_smem_bytes));
                            cudaError_t st3 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem_NMajor,
                                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                                   static_cast<int>(evalacc_smem_bytes));
                            evalacc_smem_enabled = (st0 == cudaSuccess) && (st1 == cudaSuccess) &&
                                                   (st2 == cudaSuccess) && (st3 == cudaSuccess);
                        } else {
                            evalacc_smem_enabled = (st0 == cudaSuccess) && (st1 == cudaSuccess);
                        }
                    } else {
                        st0 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   static_cast<int>(evalacc_smem_bytes));
                        if (index_nmajor) {
                            cudaError_t st2 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem_NMajor,
                                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                                   static_cast<int>(evalacc_smem_bytes));
                            evalacc_smem_enabled = (st0 == cudaSuccess) && (st2 == cudaSuccess);
                        } else {
                            evalacc_smem_enabled = (st0 == cudaSuccess);
                        }
                    }
                } else {
                    st0 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_Batch_SoA_Smem,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(evalacc_smem_bytes));
                    if (index_nmajor) {
                        cudaError_t st2 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_Batch_SoA_Smem_NMajor,
                                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                               static_cast<int>(evalacc_smem_bytes));
                        evalacc_smem_enabled = (st0 == cudaSuccess) && (st2 == cudaSuccess);
                    } else {
                        evalacc_smem_enabled = (st0 == cudaSuccess);
                    }
                }
            } else {
                evalacc_smem_enabled = true;
            }
        } else if (evalacc_smem_verbose) {
            std::cerr << "[CIRBTS] EvalAcc SMEM disabled (requested " << evalacc_smem_bytes
                      << " bytes > max " << max_smem << ")." << std::endl;
        }
    }

    const dim3 evalaccBlock(evalaccBlockX);
    const dim3 evalaccGrid((N + evalaccBlock.x - 1) / evalaccBlock.x, batch);
    const size_t evalaccNttShmem = N * sizeof(NativeInt);
    const size_t evalacc_smem = evalacc_smem_enabled ? evalacc_smem_bytes : 0;
    auto finalize_bootstrap_acc = [&]() {
        // Normalize: multiply by N^{-1} mod Q for all coefficients in all polynomials.
        constexpr uint32_t normBlockX = 256;
        const size_t totalCoeffs = static_cast<size_t>(2) * N * batch;
        const dim3 normBlock(normBlockX);
        const dim3 normGrid((totalCoeffs + normBlock.x - 1) / normBlock.x);
        kernel_ModMulConst<<<normGrid, normBlock, 0, s>>>(scratch.d_bootstrap_acc, N_inv.ConvertToInt(), d_modulus_.get(), totalCoeffs);

        // Add B^i/(2N) (implemented as gpow[i] >> 1 scaled by N^{-1}) on c1 in coefficient domain, then return to evaluation domain.
        inwt_1d_opt_batched(scratch.d_bootstrap_acc, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(),
                            d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N,
                            /*batch=*/static_cast<size_t>(2) * batch, s);

        dim3 gpowBlock(256);
        dim3 gpowGrid((numLUT + gpowBlock.x - 1) / gpowBlock.x, batch);
        kernel_ModAddGpowScaledBatch<<<gpowGrid, gpowBlock, 0, s>>>(scratch.d_bootstrap_acc, d_gpow_.get(), N_inv.ConvertToInt(), d_modulus_.get(),
                                                                    static_cast<uint32_t>(N), numLUT, batch);

        fnwt_1d_opt_batched(scratch.d_bootstrap_acc, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                            /*batch=*/static_cast<size_t>(2) * batch, s);
    };

    auto run_evalacc_mb3 = [&](bool wait_index_event_first_group) {
        for (size_t group = 0; group < rfkey_groups_; ++group) {
            const uint32_t idx0 = static_cast<uint32_t>(group * 3u);
            const uint32_t idx1 = idx0 + 1u;
            const uint32_t idx2 = idx0 + 2u;

            inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                           d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                           d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

            kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                d_twiddles_shoup_.get(), d_modulus_.get());
            PHANTOM_CHECK_CUDA_LAST();

            if (group == 0 && wait_index_event_first_group && use_h2d_pipeline) {
                PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, *scratch.h2d_index_event, 0));
            }

            if (idx2 < n) {
                if (use_evalacc_soa) {
                    if (evalacc_smem_enabled) {
                        if (index_nmajor) {
                            kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                                ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                                d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1, idx2,
                                scratch.d_evalacc_indexPos, index_stride);
                        } else {
                            kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                            ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                            d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1, idx2,
                            scratch.d_evalacc_indexPos, index_stride);
                        }
                    } else {
                        if (index_nmajor) {
                            kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                                ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                                d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1, idx2,
                                scratch.d_evalacc_indexPos, index_stride);
                        } else {
                            kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                                ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                                d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1, idx2,
                                scratch.d_evalacc_indexPos, index_stride);
                        }
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB3_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek2, ek2_shoup,
                            ek01, ek01_shoup, ek02, ek02_shoup, ek12, ek12_shoup, ek012, ek012_shoup, d_monic_polys_.get(),
                            N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1, idx2,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB3_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek2, ek2_shoup,
                            ek01, ek01_shoup, ek02, ek02_shoup, ek12, ek12_shoup, ek012, ek012_shoup, d_monic_polys_.get(),
                            N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1, idx2,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else if (idx1 < n) {
                if (use_evalacc_soa) {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek01_soa, ek01_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                            digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek01_soa, ek01_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                            digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek01, ek01_shoup,
                            d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek01, ek01_shoup,
                            d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else {
                if (use_evalacc_soa) {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, d_monic_polys_.get(),
                            N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, d_monic_polys_.get(),
                            N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, d_monic_polys_.get(), N,
                            d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, d_monic_polys_.get(), N,
                            d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            }
        }
    };

    auto run_evalacc_mb4 = [&](bool wait_index_event_first_group) {
        auto launch_group_mb3 = [&](uint32_t key_group, uint32_t idx0, uint32_t idx1, uint32_t idx2) {
            if (use_evalacc_soa) {
                if (evalacc_smem_enabled) {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                            ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                            d_modulus_.get(), digitsG2, key_group, idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                            ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                            d_modulus_.get(), digitsG2, key_group, idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride);
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                            ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                            d_modulus_.get(), digitsG2, key_group, idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                            ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                            d_modulus_.get(), digitsG2, key_group, idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else {
                if (index_nmajor) {
                    kernel_EvalAccCore_Binary_MB3_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek2, ek2_shoup,
                        ek01, ek01_shoup, ek02, ek02_shoup, ek12, ek12_shoup, ek012, ek012_shoup, d_monic_polys_.get(),
                        N, d_modulus_.get(), digitsG2, key_group, idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride);
                } else {
                    kernel_EvalAccCore_Binary_MB3_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek2, ek2_shoup,
                        ek01, ek01_shoup, ek02, ek02_shoup, ek12, ek12_shoup, ek012, ek012_shoup, d_monic_polys_.get(),
                        N, d_modulus_.get(), digitsG2, key_group, idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride);
                }
            }
        };

        auto launch_group_single = [&](const NativeInt* k, const NativeInt* k_shoup,
                                       const NativeInt* k_soa, const NativeInt* k_shoup_soa,
                                       uint32_t key_group, uint32_t idx) {
            if (use_evalacc_soa) {
                if (evalacc_smem_enabled) {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, k_soa, k_shoup_soa, d_monic_polys_.get(),
                            N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, key_group, idx,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, k_soa, k_shoup_soa, d_monic_polys_.get(),
                            N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, key_group, idx,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, k_soa, k_shoup_soa, d_monic_polys_.get(),
                            N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, key_group, idx,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, k_soa, k_shoup_soa, d_monic_polys_.get(),
                            N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, key_group, idx,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else {
                if (index_nmajor) {
                    kernel_EvalAccCore_Binary_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, k, k_shoup, d_monic_polys_.get(), N,
                        d_modulus_.get(), digitsG2, key_group, idx, scratch.d_evalacc_indexPos, index_stride);
                } else {
                    kernel_EvalAccCore_Binary_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, k, k_shoup, d_monic_polys_.get(), N,
                        d_modulus_.get(), digitsG2, key_group, idx, scratch.d_evalacc_indexPos, index_stride);
                }
            }
        };

        for (size_t group = 0; group < rfkey_groups_; ++group) {
            const uint32_t idx0 = static_cast<uint32_t>(group * 4u);
            const uint32_t idx1 = idx0 + 1u;
            const uint32_t idx2 = idx0 + 2u;
            const uint32_t idx3 = idx0 + 3u;
            if (idx0 >= n) {
                break;
            }

            // Phase A: fused 3-way update on {idx0, idx1, idx2}
            inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                           d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                           d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

            kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                d_twiddles_shoup_.get(), d_modulus_.get());
            PHANTOM_CHECK_CUDA_LAST();

            if (group == 0 && wait_index_event_first_group && use_h2d_pipeline) {
                PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, *scratch.h2d_index_event, 0));
            }

            if (idx2 < n) {
                launch_group_mb3(static_cast<uint32_t>(group), idx0, idx1, idx2);
            } else if (idx1 < n) {
                if (use_evalacc_soa) {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek01_soa, ek01_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                            digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek01_soa, ek01_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                            digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek01, ek01_shoup,
                            d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek01, ek01_shoup,
                            d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else {
                launch_group_single(ek0, ek0_shoup, ek0_soa, ek0_shoup_soa, static_cast<uint32_t>(group), idx0);
            }

            // Phase B: exact tail update for idx3 (if present), with fresh decomposition.
            if (idx3 < n) {
                inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                               d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                               d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

                kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                    scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                    d_twiddles_shoup_.get(), d_modulus_.get());

                launch_group_single(ek3, ek3_shoup, ek3_soa, ek3_shoup_soa, static_cast<uint32_t>(group), idx3);
            }
        }
    };

    if (use_pbs_multibit2) {
        for (size_t group = 0; group < rfkey_groups_; ++group) {
            const uint32_t idx0 = static_cast<uint32_t>(group * 2u);
            const uint32_t idx1 = idx0 + 1u;

            inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                           d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                           d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

            kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                d_twiddles_shoup_.get(), d_modulus_.get());

            if (group == 0 && use_h2d_pipeline) {
                PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, *scratch.h2d_index_event, 0));
            }

            if (idx1 < n) {
                if (use_swizzle_evalacc) {
                    kernel_EvalAccCore_Binary_MB2_Grouped_Batch_Swizzle<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek_pair,
                        ek_pair_shoup, d_monic_polys_.get(), N, static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(),
                        static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                } else if (use_evalacc_soa) {
                    if (evalacc_smem_enabled) {
                        if (index_nmajor) {
                            kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                                digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                        } else {
                            kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                                digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                        }
                    } else {
                        if (index_nmajor) {
                            kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                                digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                        } else {
                            kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                                digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                        }
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek_pair,
                            ek_pair_shoup, d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group),
                            idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek_pair,
                            ek_pair_shoup, d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group),
                            idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else {
                if (use_swizzle_evalacc) {
                    kernel_EvalAccCore_Binary_Grouped_Batch_Swizzle<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, d_monic_polys_.get(), N,
                        static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(), static_cast<uint32_t>(group), idx0,
                        scratch.d_evalacc_indexPos, index_stride);
                } else if (use_evalacc_soa) {
                    if (evalacc_smem_enabled) {
                        if (index_nmajor) {
                            kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, d_monic_polys_.get(),
                                N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0,
                                scratch.d_evalacc_indexPos, index_stride);
                        } else {
                            kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, d_monic_polys_.get(),
                                N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0,
                                scratch.d_evalacc_indexPos, index_stride);
                        }
                    } else {
                        if (index_nmajor) {
                            kernel_EvalAccCore_Binary_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, d_monic_polys_.get(),
                                N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0,
                                scratch.d_evalacc_indexPos, index_stride);
                        } else {
                            kernel_EvalAccCore_Binary_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, d_monic_polys_.get(),
                                N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0,
                                scratch.d_evalacc_indexPos, index_stride);
                        }
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, d_monic_polys_.get(), N,
                            d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, d_monic_polys_.get(), N,
                            d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            }
        }
        finalize_bootstrap_acc();
        return;
    }

    auto launch_evalacc_pair = [&](uint32_t idx0, uint32_t idx1) {
        if (use_evalacc_soa) {
            if (evalacc_smem_enabled) {
                if (use_evalacc_fuse_ms) {
                    kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem_MS<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                        ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                        digitsG2, idx0, idx1, d_a, a_stride, twoN, Q_lwe, bitwidth_u32);
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                            ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                            digitsG2, idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                            ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                            digitsG2, idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else {
                if (use_evalacc_fuse_ms) {
                    kernel_EvalAccCore_Binary_MB2_Batch_SoA_MS<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                        ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                        digitsG2, idx0, idx1, d_a, a_stride, twoN, Q_lwe, bitwidth_u32);
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB2_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                            ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                            digitsG2, idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB2_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                            ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                            digitsG2, idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            }
        } else {
            if (use_evalacc_fuse_ms) {
                kernel_EvalAccCore_Binary_MB2_Batch_MS<<<evalaccGrid, evalaccBlock, 0, s>>>(
                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_.get(), d_RFkey_shoup_.get(), ek_pair,
                    ek_pair_shoup, d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, idx0, idx1,
                    d_a, a_stride, twoN, Q_lwe, bitwidth_u32);
            } else {
                if (index_nmajor) {
                    kernel_EvalAccCore_Binary_MB2_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_.get(), d_RFkey_shoup_.get(), ek_pair,
                        ek_pair_shoup, d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, idx0, idx1,
                        scratch.d_evalacc_indexPos, index_stride);
                } else {
                    kernel_EvalAccCore_Binary_MB2_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_.get(), d_RFkey_shoup_.get(), ek_pair,
                        ek_pair_shoup, d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, idx0, idx1,
                        scratch.d_evalacc_indexPos, index_stride);
                }
            }
        }
    };

    auto launch_evalacc_single = [&](uint32_t idx) {
        if (use_swizzle_evalacc) {
            kernel_EvalAccCore_Binary_Batch_Swizzle<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_swizzle_.get(), d_RFkey_shoup_swizzle_.get(),
                d_monic_polys_.get(), N, static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(), idx,
                scratch.d_evalacc_indexPos, index_stride);
        } else if (use_evalacc_soa) {
            if (evalacc_smem_enabled) {
                if (use_evalacc_fuse_ms) {
                    kernel_EvalAccCore_Binary_Batch_SoA_Smem_MS<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                        d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, idx,
                        d_a, a_stride, twoN, Q_lwe, bitwidth_u32);
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                            d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, idx,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                            d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, idx,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else {
                if (use_evalacc_fuse_ms) {
                    kernel_EvalAccCore_Binary_Batch_SoA_MS<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                        d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, idx,
                        d_a, a_stride, twoN, Q_lwe, bitwidth_u32);
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                            d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, idx,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                            d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, idx,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            }
        } else {
            if (use_evalacc_fuse_ms) {
                kernel_EvalAccCore_Binary_Batch_MS<<<evalaccGrid, evalaccBlock, 0, s>>>(
                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_.get(), d_RFkey_shoup_.get(), d_monic_polys_.get(),
                    N, d_modulus_.get(), digitsG2, idx, d_a, a_stride, twoN, Q_lwe, bitwidth_u32);
            } else {
                if (index_nmajor) {
                    kernel_EvalAccCore_Binary_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_.get(), d_RFkey_shoup_.get(), d_monic_polys_.get(),
                        N, d_modulus_.get(), digitsG2, idx, scratch.d_evalacc_indexPos, index_stride);
                } else {
                    kernel_EvalAccCore_Binary_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_.get(), d_RFkey_shoup_.get(), d_monic_polys_.get(),
                        N, d_modulus_.get(), digitsG2, idx, scratch.d_evalacc_indexPos, index_stride);
                }
            }
        }
    };

    const uint32_t graph_multibit_mode = use_pbs_multibit4 ? 4u : (use_pbs_multibit3 ? 3u : (use_multibit2 ? 2u : 1u));
    const bool disableBatchGraph = (std::getenv("CIRBTS_DISABLE_EVALACC_BATCH_CUDA_GRAPH") != nullptr) || use_evalacc_fuse_ms;
    const bool canUseBatchGraph = !disableBatchGraph && scratch.evalacc_batch_graph_exec && scratch.evalacc_batch_graph_failed &&
                                  scratch.evalacc_batch_graph_ntt_tpb && scratch.evalacc_batch_graph_block_x &&
                                  scratch.evalacc_batch_graph_batch && scratch.evalacc_batch_graph_multibit &&
                                  scratch.evalacc_batch_graph_soa && scratch.evalacc_batch_graph_smem;
    if (canUseBatchGraph && !(*scratch.evalacc_batch_graph_failed)) {
        if (*scratch.evalacc_batch_graph_exec &&
            (*scratch.evalacc_batch_graph_ntt_tpb != fusedNttTpb || *scratch.evalacc_batch_graph_block_x != evalaccBlockX ||
             *scratch.evalacc_batch_graph_batch != batch ||
             *scratch.evalacc_batch_graph_multibit != graph_multibit_mode ||
             *scratch.evalacc_batch_graph_soa != use_evalacc_soa ||
             *scratch.evalacc_batch_graph_smem != evalacc_smem_enabled)) {
            cudaGraphExecDestroy(*scratch.evalacc_batch_graph_exec);
            *scratch.evalacc_batch_graph_exec = nullptr;
        }
        if (!*scratch.evalacc_batch_graph_exec) {
            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
            cudaGraph_t graph{};
            PHANTOM_CHECK_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            if (use_pbs_multibit4) {
                run_evalacc_mb4(/*wait_index_event_first_group=*/false);
            } else if (use_pbs_multibit3) {
                run_evalacc_mb3(/*wait_index_event_first_group=*/false);
            } else {
                for (uint32_t lweIndex = 0; lweIndex < n; ) {
                    inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                                   d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                                   d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

                    kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                        scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                        d_twiddles_shoup_.get(), d_modulus_.get());

                    if (use_multibit2 && (lweIndex + 1 < n)) {
                        launch_evalacc_pair(lweIndex, lweIndex + 1);
                        lweIndex += 2;
                    } else {
                        launch_evalacc_single(lweIndex);
                        ++lweIndex;
                    }
                }
            }
            PHANTOM_CHECK_CUDA(cudaStreamEndCapture(s, &graph));
            cudaGraphExec_t exec{};
            const auto inst = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
            cudaGraphDestroy(graph);
            if (inst != cudaSuccess) {
                *scratch.evalacc_batch_graph_failed = true;
            }
            else {
                *scratch.evalacc_batch_graph_exec = exec;
                *scratch.evalacc_batch_graph_ntt_tpb = fusedNttTpb;
                *scratch.evalacc_batch_graph_block_x = evalaccBlockX;
                *scratch.evalacc_batch_graph_batch = batch;
                *scratch.evalacc_batch_graph_multibit = graph_multibit_mode;
                *scratch.evalacc_batch_graph_soa = use_evalacc_soa;
                *scratch.evalacc_batch_graph_smem = evalacc_smem_enabled;
            }
        }

        if (*scratch.evalacc_batch_graph_exec) {
            if (use_h2d_pipeline) {
                PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, *scratch.h2d_index_event, 0));
            }
            PHANTOM_CHECK_CUDA(cudaGraphLaunch(*scratch.evalacc_batch_graph_exec, s));
        }
        else {
            if (use_pbs_multibit4) {
                run_evalacc_mb4(/*wait_index_event_first_group=*/true);
            } else if (use_pbs_multibit3) {
                run_evalacc_mb3(/*wait_index_event_first_group=*/true);
            } else {
                for (uint32_t lweIndex = 0; lweIndex < n; ) {
                    inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                                   d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                                   d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

                    kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                        scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                        d_twiddles_shoup_.get(), d_modulus_.get());

                    if (use_h2d_pipeline && lweIndex == 0) {
                        PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, *scratch.h2d_index_event, 0));
                    }
                    if (use_multibit2 && (lweIndex + 1 < n)) {
                        launch_evalacc_pair(lweIndex, lweIndex + 1);
                        lweIndex += 2;
                    } else {
                        launch_evalacc_single(lweIndex);
                        ++lweIndex;
                    }
                }
            }
        }
    }
    else {
        if (use_pbs_multibit4) {
            run_evalacc_mb4(/*wait_index_event_first_group=*/true);
        } else if (use_pbs_multibit3) {
            run_evalacc_mb3(/*wait_index_event_first_group=*/true);
        } else {
            for (uint32_t lweIndex = 0; lweIndex < n; ) {
                inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                               d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                               d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

                kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                    scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                    d_twiddles_shoup_.get(), d_modulus_.get());

                if (use_h2d_pipeline && lweIndex == 0) {
                    PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(s, *scratch.h2d_index_event, 0));
                }
                if (use_multibit2 && (lweIndex + 1 < n)) {
                    launch_evalacc_pair(lweIndex, lweIndex + 1);
                    lweIndex += 2;
                } else {
                    launch_evalacc_single(lweIndex);
                    ++lweIndex;
                }
            }
        }
    }

    finalize_bootstrap_acc();
}

void GPUCirBTSContext::gpu_BootstrapLUT_inplace_batch_device(const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params,
                                                             ConstRingGSWACCKey& ek,
                                                             const NativeInt* d_a,
                                                             const NativeInt* d_b,
                                                             uint32_t batch,
                                                             uint32_t a_stride,
                                                             uint64_t bitwidth,
                                                             const BatchScratchView& scratch) const {
    if (!d_a || !d_b || batch == 0) {
        return;
    }
    if (!ek) {
        OPENFHE_THROW(config_error,
                      "Bootstrapping keys have not been generated. Please call BTKeyGen before calling bootstrapping.");
    }
    if (batch > std::max<uint32_t>(1u, max_batch_size_)) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: batch size exceeds max_batch_size_ (recreate GPUCirBTSContext)");
    }
    if (!scratch.d_bootstrap_acc || !d_lut_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: bootstrap scratch buffers are not initialized on GPU");
    }
    if (!d_gpow_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: Gpow is not initialized on GPU");
    }
    if (!scratch.d_monomial_inv) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: monomial scratch buffer is not initialized on GPU");
    }
    if (!scratch.d_evalacc_indexPos || !scratch.d_evalacc_b_idx) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: EvalAcc index buffers are not initialized on GPU");
    }

    const auto& rgswParams1 = params->GetRingGSWParams1();
    if (rgswParams1->GetMethod() != BINFHE_METHOD::GINX) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: currently supports method=GINX only");
    }
    if (!d_RFkey_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: RF key is not initialized on GPU");
    }
    if (!d_RFkey_shoup_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: RF key Shoup table is not initialized on GPU");
    }
    if (!d_monic_polys_.get() || !d_monomial_inv_table_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: monomial tables are not initialized on GPU");
    }
    if (!scratch.d_evalacc_ct || !scratch.d_evalacc_dct || !scratch.d_evalacc_indexPos) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: EvalAcc scratch buffers are not initialized on GPU");
    }

    const uint32_t numLUT = params->GetDigitsCC();
    const auto& lweParams = params->GetLWEParams();
    const auto q          = lweParams->Getq();
    const uint32_t n      = lweParams->Getn();
    if (a_stride < n) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: a_stride < n");
    }

    const auto polyParams = rgswParams1->GetPolyParams();
    const auto Q          = rgswParams1->GetQ();
    const size_t N        = polyParams->GetRingDimension();
    const auto N_inv      = NativeInteger(N).ModInverse(Q);

    if (N != ring_dim_ || numLUT != num_luts_) {
        OPENFHE_THROW(config_error,
                      "gpu_BootstrapLUT_inplace_batch_device: GPU context params mismatch (recreate GPUCirBTSContext)");
    }
    if (!d_modulus_.get() || !d_twiddles_.get() || !d_twiddles_shoup_.get() || !d_itwiddles_.get() || !d_itwiddles_shoup_.get() ||
        !d_n_inv_mod_q_.get() || !d_n_inv_mod_q_shoup_.get()) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: NTT tables are not initialized on GPU");
    }

    auto& s = scratch.stream;
    const bool profile_evalacc_detail =
        (std::getenv("CIRBTS_PROFILE_EVALACC_DETAIL") != nullptr) ||
        (std::getenv("CIRBTS_PROFILE_DEVICE_BOOT") != nullptr);
    cudaEvent_t prof_begin_ev{}, prof_end_ev{};
    float prof_ms_index = 0.0f;
    float prof_ms_acc_init = 0.0f;
    float prof_ms_group_inwt = 0.0f;
    float prof_ms_group_decomp_fnwt = 0.0f;
    float prof_ms_group_external = 0.0f;
    float prof_ms_final_norm = 0.0f;
    float prof_ms_final_inwt = 0.0f;
    float prof_ms_final_gpow = 0.0f;
    float prof_ms_final_fnwt = 0.0f;
    uint32_t prof_group_steps = 0;
    if (profile_evalacc_detail) {
        PHANTOM_CHECK_CUDA(cudaEventCreate(&prof_begin_ev));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&prof_end_ev));
    }
    auto prof_begin = [&]() {
        if (profile_evalacc_detail) {
            PHANTOM_CHECK_CUDA(cudaEventRecord(prof_begin_ev, s));
        }
    };
    auto prof_end = [&](float& bucket) {
        if (profile_evalacc_detail) {
            PHANTOM_CHECK_CUDA(cudaEventRecord(prof_end_ev, s));
            PHANTOM_CHECK_CUDA(cudaEventSynchronize(prof_end_ev));
            float ms = 0.0f;
            PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms, prof_begin_ev, prof_end_ev));
            bucket += ms;
        }
    };

    const uint32_t twoN = static_cast<uint32_t>(N * 2);
    const NativeInt Q_lwe = q.ConvertToInt<NativeInt>();
    const uint32_t bitwidth_u32 = static_cast<uint32_t>(bitwidth);

    const dim3 ms_block(256);
    const dim3 ms_grid_a((n + ms_block.x - 1) / ms_block.x, batch);
    const dim3 ms_grid_b((batch + ms_block.x - 1) / ms_block.x);
    const dim3 mono_grid((N + ms_block.x - 1) / ms_block.x, batch);

    uint32_t fuse_ms = 1u;
    if (const char* v = std::getenv("CIRBTS_EVALACC_FUSE_MS"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0') {
            fuse_ms = static_cast<uint32_t>(parsed);
        }
    }
    bool use_evalacc_fuse_ms = (fuse_ms != 0u);
    if (const char* v = std::getenv("CIRBTS_PBS_MULTIBIT"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0' && parsed != 0ul) {
            use_evalacc_fuse_ms = false;
        }
    }
    const char* index_nmajor_env = std::getenv("CIRBTS_BATCH_INDEX_NMAJOR");
    const bool index_nmajor_requested = (index_nmajor_env != nullptr) && (index_nmajor_env[0] != '0');

    if (use_evalacc_fuse_ms) {
        prof_begin();
        constexpr uint32_t initBlockX = 256;
        dim3 initBlock(initBlockX);
        dim3 initGrid((N + initBlock.x - 1) / initBlock.x, batch);
        kernel_InitBootstrapAccBatch_FusedB<<<initGrid, initBlock, 0, s>>>(
            scratch.d_bootstrap_acc, d_lut_.get(), d_monomial_inv_table_.get(), d_b, batch, twoN, Q_lwe, bitwidth_u32,
            d_modulus_.get(), static_cast<uint32_t>(N));
        PHANTOM_CHECK_CUDA_LAST();
        prof_end(prof_ms_acc_init);
    } else {
        prof_begin();
        if (index_nmajor_requested) {
            kernel_SpecialMS_IndexPos_Batch_NMajor<<<ms_grid_a, ms_block, 0, s>>>(
                d_a, scratch.d_evalacc_indexPos, n, batch, a_stride, twoN, Q_lwe, bitwidth_u32);
        } else {
            kernel_SpecialMS_IndexPos_Batch<<<ms_grid_a, ms_block, 0, s>>>(
                d_a, scratch.d_evalacc_indexPos, n, batch, a_stride, twoN, Q_lwe, bitwidth_u32);
        }
        PHANTOM_CHECK_CUDA_LAST();

        kernel_SpecialMS_B_Batch<<<ms_grid_b, ms_block, 0, s>>>(d_b, scratch.d_evalacc_b_idx, batch, twoN, Q_lwe, bitwidth_u32);
        PHANTOM_CHECK_CUDA_LAST();

        kernel_GatherMonomialInv_Batch<<<mono_grid, ms_block, 0, s>>>(scratch.d_monomial_inv, d_monomial_inv_table_.get(),
                                                                      scratch.d_evalacc_b_idx, static_cast<uint32_t>(N), batch);
        PHANTOM_CHECK_CUDA_LAST();
        prof_end(prof_ms_index);

        prof_begin();
        constexpr uint32_t initBlockX = 256;
        dim3 initBlock(initBlockX);
        dim3 initGrid((N + initBlock.x - 1) / initBlock.x, batch);
        kernel_InitBootstrapAccBatch<<<initGrid, initBlock, 0, s>>>(scratch.d_bootstrap_acc, d_lut_.get(), N, batch);
        PHANTOM_CHECK_CUDA_LAST();
        kernel_ModMulScalarBatch<<<initGrid, initBlock, 0, s>>>(scratch.d_bootstrap_acc, scratch.d_monomial_inv, d_modulus_.get(), N, batch);
        PHANTOM_CHECK_CUDA_LAST();
        prof_end(prof_ms_acc_init);
    }

    const uint32_t digitsGA = rgswParams1->GetDigitsGA();
    const uint32_t digitsG2 = digitsGA << 1;
    const NativeInt Qint    = Q.ConvertToInt();
    const uint32_t baseG    = rgswParams1->GetBaseG();
    const bool use_swizzle_path = use_swizzle_ && (digitsG2 == 2u) && rfkey_swizzle_tiles_ &&
                                  d_RFkey_swizzle_.get() && d_RFkey_shoup_swizzle_.get();
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
            OPENFHE_THROW(config_error,
                          "gpu_BootstrapLUT_inplace_batch_device: EvalAcc SoA requested but not initialized (set CIRBTS_EVALACC_SOA before init)");
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
                          "gpu_BootstrapLUT_inplace_batch_device: EvalAcc SMEM requires EvalAcc SoA (set CIRBTS_EVALACC_SOA_FORCE=1)");
        }
        if (evalacc_smem_verbose) {
            std::cout << "[CIRBTS] EvalAcc SMEM disabled: EvalAcc SoA unavailable." << std::endl;
        }
    }
    bool use_swizzle_evalacc = use_swizzle_path && !use_evalacc_soa;
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
    if (pbs_multibit != 0u && pbs_multibit != 2u && pbs_multibit != 3u && pbs_multibit != 4u) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: unsupported CIRBTS_PBS_MULTIBIT (supported: 0,2,3,4)");
    }

    const uint32_t evalacc_block_x_env = [](uint32_t n_in) {
        const char* v = std::getenv("CIRBTS_EVALACC_BLOCK_X");
        if (!v || !*v) {
            return n_in;
        }
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0' && parsed > 0) {
            return static_cast<uint32_t>(std::min<unsigned long>(parsed, 1024ul));
        }
        return n_in;
    }(evalacc_batch_block_x_);

    const bool use_pbs_multibit2 = (pbs_multibit == 2u);
    const bool use_pbs_multibit3 = (pbs_multibit == 3u);
    const bool use_pbs_multibit4 = (pbs_multibit == 4u);
    const bool use_evalacc_multibit2 = (pbs_multibit == 0u && evalacc_multibit == 2u);
    const bool use_multibit2 = use_pbs_multibit2 || use_evalacc_multibit2;
    const bool use_multibit3 = use_pbs_multibit3;
    const bool use_multibit4 = use_pbs_multibit4;
    if (use_swizzle_evalacc || use_pbs_multibit2 || use_pbs_multibit3 || use_pbs_multibit4) {
        use_evalacc_fuse_ms = false;
    }
    if (use_multibit3 && use_evalacc_multibit2) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: conflicting multibit configuration");
    }
    if (use_multibit4 && use_evalacc_multibit2) {
        OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: conflicting multibit configuration");
    }
    if (use_multibit2) {
        if (use_pbs_multibit2) {
            if (!rfkey_grouped_ || rfkey_dim2_ != 3u || rfkey_groups_ == 0) {
                OPENFHE_THROW(config_error,
                              "gpu_BootstrapLUT_inplace_batch_device: PBS multibit=2 requires grouped RFkey (set CIRBTS_PBS_MULTIBIT=2 before keygen)");
            }
        } else if (rfkey_dim2_ < 2u) {
            OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: multibit=2 requires RFkey dim2=2");
        }
        if (!use_pbs_multibit2 && use_swizzle_evalacc) {
            OPENFHE_THROW(config_error, "gpu_BootstrapLUT_inplace_batch_device: multibit=2 does not support swizzle");
        }
    }
    bool use_multibit3_experimental = false;
    if (use_multibit3) {
        if (!rfkey_grouped_ || rfkey_dim2_ != 7u || rfkey_groups_ == 0) {
            OPENFHE_THROW(config_error,
                          "gpu_BootstrapLUT_inplace_batch_device: PBS multibit=3 requires grouped RFkey dim2=7 (set CIRBTS_PBS_MULTIBIT=3 before keygen)");
        }
        const char* mb3_exp_env = std::getenv("CIRBTS_MB3_EXPERIMENTAL");
        use_multibit3_experimental = (mb3_exp_env != nullptr) && (mb3_exp_env[0] != '0');
        if (!use_multibit3_experimental) {
            static bool printed = false;
            if (!printed) {
                std::cout << "[CIRBTS] multibit=3 baseline: forcing non-swizzle/non-SoA/non-SMEM/non-NMajor EvalAcc path."
                          << std::endl;
                printed = true;
            }
            use_evalacc_soa = false;
            use_swizzle_evalacc = false;
        } else {
            if (use_swizzle_evalacc) {
                static bool printed = false;
                if (!printed) {
                    std::cout << "[CIRBTS] multibit=3: forcing non-swizzle EvalAcc path." << std::endl;
                    printed = true;
                }
                use_swizzle_evalacc = false;
            }
            if (evalacc_smem_requested) {
                static bool printed_smem = false;
                if (!printed_smem) {
                    std::cout << "[CIRBTS] multibit=3: enabling experimental EvalAcc SMEM (main-key staging)." << std::endl;
                    printed_smem = true;
                }
            }
        }
    }
    bool use_multibit4_experimental = false;
    if (use_multibit4) {
        if (!rfkey_grouped_ || rfkey_dim2_ != 8u || rfkey_groups_ == 0) {
            OPENFHE_THROW(config_error,
                          "gpu_BootstrapLUT_inplace_batch_device: PBS multibit=4 requires grouped RFkey dim2=8 (set CIRBTS_PBS_MULTIBIT=4 before keygen)");
        }
        const char* mb4_exp_env = std::getenv("CIRBTS_MB4_EXPERIMENTAL");
        use_multibit4_experimental = (mb4_exp_env == nullptr) || (mb4_exp_env[0] != '0');
        if (!use_multibit4_experimental) {
            static bool printed = false;
            if (!printed) {
                std::cout << "[CIRBTS] multibit=4 baseline: forcing non-swizzle/non-SoA/non-SMEM/non-NMajor EvalAcc path."
                          << std::endl;
                printed = true;
            }
            use_evalacc_soa = false;
            use_swizzle_evalacc = false;
        } else {
            if (use_swizzle_evalacc) {
                static bool printed = false;
                if (!printed) {
                    std::cout << "[CIRBTS] multibit=4: forcing non-swizzle EvalAcc path." << std::endl;
                    printed = true;
                }
                use_swizzle_evalacc = false;
            }
            if (evalacc_smem_requested) {
                static bool printed_smem = false;
                if (!printed_smem) {
                    std::cout << "[CIRBTS] multibit=4: enabling experimental EvalAcc SMEM (main-key staging)." << std::endl;
                    printed_smem = true;
                }
            }
        }
    }

    bool index_nmajor = index_nmajor_requested;
    if (use_multibit3 && !use_multibit3_experimental && index_nmajor) {
        static bool printed = false;
        if (!printed) {
            std::cout << "[CIRBTS] multibit=3 baseline: disabling batch index N-major." << std::endl;
            printed = true;
        }
        index_nmajor = false;
    }
    if (use_multibit4 && !use_multibit4_experimental && index_nmajor) {
        static bool printed = false;
        if (!printed) {
            std::cout << "[CIRBTS] multibit=4 baseline: disabling batch index N-major." << std::endl;
            printed = true;
        }
        index_nmajor = false;
    }
    if (index_nmajor) {
        if (use_evalacc_fuse_ms || use_swizzle_evalacc) {
            static bool printed = false;
            if (!printed) {
                std::cout << "[CIRBTS] Batch index N-major disabled: incompatible with current EvalAcc options." << std::endl;
                printed = true;
            }
            index_nmajor = false;
        }
    }
    const uint32_t index_stride = index_nmajor ? batch : n;

    bool evalacc_smem_enabled = false;
    size_t evalacc_smem = 0;
    if (evalacc_smem_requested && use_evalacc_soa) {
        int dev = 0;
        int max_default = 0;
        int max_opt = 0;
        PHANTOM_CHECK_CUDA(cudaGetDevice(&dev));
        PHANTOM_CHECK_CUDA(cudaDeviceGetAttribute(&max_default, cudaDevAttrMaxSharedMemoryPerBlock, dev));
        (void)cudaDeviceGetAttribute(&max_opt, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        const size_t max_smem = (max_opt > 0) ? static_cast<size_t>(max_opt) : static_cast<size_t>(max_default);
        const uint32_t evalacc_block_x =
            (evalacc_block_x_env == 0) ? ((use_pbs_multibit3 || use_pbs_multibit4) ? 128u : 256u) : evalacc_block_x_env;
        const uint32_t warps = (evalacc_block_x + 31u) >> 5;
        const size_t warp_stride = (use_multibit3 || use_multibit4) ? (12u * 32u) : (use_multibit2 ? (8u * 32u) : (4u * 32u));
        evalacc_smem = static_cast<size_t>(warps) * 2u * warp_stride * sizeof(NativeInt);
        if (evalacc_smem <= max_smem) {
            if (evalacc_smem > static_cast<size_t>(max_default)) {
                if (use_multibit3 || use_multibit4) {
                    const auto st = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem,
                                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                         static_cast<int>(evalacc_smem));
                    if (index_nmajor) {
                        const auto st2 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem_NMajor,
                                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                              static_cast<int>(evalacc_smem));
                        evalacc_smem_enabled = (st == cudaSuccess) && (st2 == cudaSuccess);
                    } else {
                        evalacc_smem_enabled = (st == cudaSuccess);
                    }
                } else if (use_multibit2) {
                    const auto st = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem,
                                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                         static_cast<int>(evalacc_smem));
                    if (index_nmajor) {
                        const auto st2 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem_NMajor,
                                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                              static_cast<int>(evalacc_smem));
                        evalacc_smem_enabled = (st == cudaSuccess) && (st2 == cudaSuccess);
                    } else {
                        evalacc_smem_enabled = (st == cudaSuccess);
                    }
                } else {
                    const auto st = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_Batch_SoA_Smem,
                                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                         static_cast<int>(evalacc_smem));
                    if (index_nmajor) {
                        const auto st2 = cudaFuncSetAttribute(kernel_EvalAccCore_Binary_Batch_SoA_Smem_NMajor,
                                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                              static_cast<int>(evalacc_smem));
                        evalacc_smem_enabled = (st == cudaSuccess) && (st2 == cudaSuccess);
                    } else {
                        evalacc_smem_enabled = (st == cudaSuccess);
                    }
                }
            } else {
                evalacc_smem_enabled = true;
            }
        } else if (evalacc_smem_verbose) {
            std::cerr << "[CIRBTS] EvalAcc SMEM disabled (requested " << evalacc_smem
                      << " bytes > max " << max_smem << ")." << std::endl;
        }
    }

    uint32_t evalaccBlockX = evalacc_block_x_env;
    if (evalaccBlockX == 0) {
        // Keep device-input MB3/MB4 default aligned with host-input defaults.
        evalaccBlockX = (use_pbs_multibit3 || use_pbs_multibit4) ? 128u : 256u;
    }
    dim3 evalaccBlock(evalaccBlockX);
    dim3 evalaccGrid((N + evalaccBlock.x - 1) / evalaccBlock.x, batch);

    uint32_t fusedNttTpb = static_cast<uint32_t>(std::min<size_t>(N / 2, 256));
    const bool fused_ntt_tpb_env_set = (std::getenv("CIRBTS_FUSED_NTT_TPB") != nullptr);
    if (const char* v = std::getenv("CIRBTS_FUSED_NTT_TPB"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0' && parsed > 0) {
            fusedNttTpb = static_cast<uint32_t>(std::min<unsigned long>(parsed, static_cast<unsigned long>(N / 2)));
            fusedNttTpb = (fusedNttTpb / 32u) * 32u;
            if (fusedNttTpb == 0) {
                fusedNttTpb = 32;
            }
        }
    }
    const size_t evalaccNttShmem = N * sizeof(NativeInt);
    if (evalaccNttShmem > 0) {
        const auto st = cudaFuncSetAttribute(kernel_SignedDigitDecompose2_FusedFNWT_Batch,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(evalaccNttShmem));
        (void)st;
    }

    const uint32_t evalacc_soa_tiles = static_cast<uint32_t>(N / 32u);
    const bool use_grouped_multibit = use_pbs_multibit2 || use_pbs_multibit3 || use_pbs_multibit4;
    const size_t key_block = size_RFRGSWkey_ * static_cast<size_t>(use_grouped_multibit ? rfkey_groups_ : n);
    const NativeInt* rfkey_base = use_evalacc_soa ? d_RFkey_soa_.get() :
                                  (use_swizzle_evalacc ? d_RFkey_swizzle_.get() : d_RFkey_.get());
    const NativeInt* rfkey_shoup_base = use_evalacc_soa ? d_RFkey_shoup_soa_.get() :
                                        (use_swizzle_evalacc ? d_RFkey_shoup_swizzle_.get() : d_RFkey_shoup_.get());
    const NativeInt* ek_pair = nullptr;
    const NativeInt* ek_pair_shoup = nullptr;
    const NativeInt* ek_pair_soa = nullptr;
    const NativeInt* ek_pair_shoup_soa = nullptr;
    const NativeInt* ek0 = nullptr;
    const NativeInt* ek1 = nullptr;
    const NativeInt* ek2 = nullptr;
    const NativeInt* ek3 = nullptr;
    const NativeInt* ek01 = nullptr;
    const NativeInt* ek02 = nullptr;
    const NativeInt* ek12 = nullptr;
    const NativeInt* ek012 = nullptr;
    const NativeInt* ek0_shoup = nullptr;
    const NativeInt* ek1_shoup = nullptr;
    const NativeInt* ek2_shoup = nullptr;
    const NativeInt* ek3_shoup = nullptr;
    const NativeInt* ek01_shoup = nullptr;
    const NativeInt* ek02_shoup = nullptr;
    const NativeInt* ek12_shoup = nullptr;
    const NativeInt* ek012_shoup = nullptr;
    const NativeInt* ek0_soa = nullptr;
    const NativeInt* ek1_soa = nullptr;
    const NativeInt* ek2_soa = nullptr;
    const NativeInt* ek3_soa = nullptr;
    const NativeInt* ek01_soa = nullptr;
    const NativeInt* ek02_soa = nullptr;
    const NativeInt* ek12_soa = nullptr;
    const NativeInt* ek012_soa = nullptr;
    const NativeInt* ek0_shoup_soa = nullptr;
    const NativeInt* ek1_shoup_soa = nullptr;
    const NativeInt* ek2_shoup_soa = nullptr;
    const NativeInt* ek3_shoup_soa = nullptr;
    const NativeInt* ek01_shoup_soa = nullptr;
    const NativeInt* ek02_shoup_soa = nullptr;
    const NativeInt* ek12_shoup_soa = nullptr;
    const NativeInt* ek012_shoup_soa = nullptr;
    if (use_multibit2) {
        if (use_pbs_multibit2) {
            ek0 = rfkey_base;
            ek1 = rfkey_base + key_block;
            ek_pair = rfkey_base + 2 * key_block;
            ek0_shoup = rfkey_shoup_base;
            ek1_shoup = rfkey_shoup_base + key_block;
            ek_pair_shoup = rfkey_shoup_base + 2 * key_block;
            if (use_evalacc_soa) {
                ek0_soa = d_RFkey_soa_.get();
                ek1_soa = d_RFkey_soa_.get() + key_block;
                ek_pair_soa = d_RFkey_soa_.get() + 2 * key_block;
                ek0_shoup_soa = d_RFkey_shoup_soa_.get();
                ek1_shoup_soa = d_RFkey_shoup_soa_.get() + key_block;
                ek_pair_shoup_soa = d_RFkey_shoup_soa_.get() + 2 * key_block;
            }
        } else {
            ek_pair = rfkey_base + key_block;
            ek_pair_shoup = rfkey_shoup_base + key_block;
            if (use_evalacc_soa) {
                ek_pair_soa = d_RFkey_soa_.get() + key_block;
                ek_pair_shoup_soa = d_RFkey_shoup_soa_.get() + key_block;
            }
        }
    }
    if (use_multibit3) {
        ek0 = rfkey_base;
        ek1 = rfkey_base + key_block;
        ek2 = rfkey_base + 2 * key_block;
        ek01 = rfkey_base + 3 * key_block;
        ek02 = rfkey_base + 4 * key_block;
        ek12 = rfkey_base + 5 * key_block;
        ek012 = rfkey_base + 6 * key_block;
        ek0_shoup = rfkey_shoup_base;
        ek1_shoup = rfkey_shoup_base + key_block;
        ek2_shoup = rfkey_shoup_base + 2 * key_block;
        ek01_shoup = rfkey_shoup_base + 3 * key_block;
        ek02_shoup = rfkey_shoup_base + 4 * key_block;
        ek12_shoup = rfkey_shoup_base + 5 * key_block;
        ek012_shoup = rfkey_shoup_base + 6 * key_block;
        if (use_evalacc_soa) {
            ek0_soa = d_RFkey_soa_.get();
            ek1_soa = d_RFkey_soa_.get() + key_block;
            ek2_soa = d_RFkey_soa_.get() + 2 * key_block;
            ek01_soa = d_RFkey_soa_.get() + 3 * key_block;
            ek02_soa = d_RFkey_soa_.get() + 4 * key_block;
            ek12_soa = d_RFkey_soa_.get() + 5 * key_block;
            ek012_soa = d_RFkey_soa_.get() + 6 * key_block;
            ek0_shoup_soa = d_RFkey_shoup_soa_.get();
            ek1_shoup_soa = d_RFkey_shoup_soa_.get() + key_block;
            ek2_shoup_soa = d_RFkey_shoup_soa_.get() + 2 * key_block;
            ek01_shoup_soa = d_RFkey_shoup_soa_.get() + 3 * key_block;
            ek02_shoup_soa = d_RFkey_shoup_soa_.get() + 4 * key_block;
            ek12_shoup_soa = d_RFkey_shoup_soa_.get() + 5 * key_block;
            ek012_shoup_soa = d_RFkey_shoup_soa_.get() + 6 * key_block;
        }
    }
    if (use_multibit4) {
        ek0 = rfkey_base;
        ek1 = rfkey_base + key_block;
        ek2 = rfkey_base + 2 * key_block;
        ek3 = rfkey_base + 3 * key_block;
        ek01 = rfkey_base + 4 * key_block;
        ek02 = rfkey_base + 5 * key_block;
        ek12 = rfkey_base + 6 * key_block;
        ek012 = rfkey_base + 7 * key_block;
        ek0_shoup = rfkey_shoup_base;
        ek1_shoup = rfkey_shoup_base + key_block;
        ek2_shoup = rfkey_shoup_base + 2 * key_block;
        ek3_shoup = rfkey_shoup_base + 3 * key_block;
        ek01_shoup = rfkey_shoup_base + 4 * key_block;
        ek02_shoup = rfkey_shoup_base + 5 * key_block;
        ek12_shoup = rfkey_shoup_base + 6 * key_block;
        ek012_shoup = rfkey_shoup_base + 7 * key_block;
        if (use_evalacc_soa) {
            ek0_soa = d_RFkey_soa_.get();
            ek1_soa = d_RFkey_soa_.get() + key_block;
            ek2_soa = d_RFkey_soa_.get() + 2 * key_block;
            ek3_soa = d_RFkey_soa_.get() + 3 * key_block;
            ek01_soa = d_RFkey_soa_.get() + 4 * key_block;
            ek02_soa = d_RFkey_soa_.get() + 5 * key_block;
            ek12_soa = d_RFkey_soa_.get() + 6 * key_block;
            ek012_soa = d_RFkey_soa_.get() + 7 * key_block;
            ek0_shoup_soa = d_RFkey_shoup_soa_.get();
            ek1_shoup_soa = d_RFkey_shoup_soa_.get() + key_block;
            ek2_shoup_soa = d_RFkey_shoup_soa_.get() + 2 * key_block;
            ek3_shoup_soa = d_RFkey_shoup_soa_.get() + 3 * key_block;
            ek01_shoup_soa = d_RFkey_shoup_soa_.get() + 4 * key_block;
            ek02_shoup_soa = d_RFkey_shoup_soa_.get() + 5 * key_block;
            ek12_shoup_soa = d_RFkey_shoup_soa_.get() + 6 * key_block;
            ek012_shoup_soa = d_RFkey_shoup_soa_.get() + 7 * key_block;
        }
    }

    auto launch_evalacc_pair = [&](uint32_t idx0, uint32_t idx1) {
        if (use_evalacc_soa) {
            if (evalacc_smem_enabled) {
                if (index_nmajor) {
                    kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                        ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                        digitsG2, idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                } else {
                    kernel_EvalAccCore_Binary_MB2_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                        ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                        digitsG2, idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                }
            } else {
                if (index_nmajor) {
                    kernel_EvalAccCore_Binary_MB2_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                        ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                        digitsG2, idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                } else {
                    kernel_EvalAccCore_Binary_MB2_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                        ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                        digitsG2, idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                }
            }
        } else {
            if (index_nmajor) {
                kernel_EvalAccCore_Binary_MB2_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_.get(), d_RFkey_shoup_.get(), ek_pair,
                    ek_pair_shoup, d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, idx0, idx1,
                    scratch.d_evalacc_indexPos, index_stride);
            } else {
                kernel_EvalAccCore_Binary_MB2_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_.get(), d_RFkey_shoup_.get(), ek_pair,
                    ek_pair_shoup, d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, idx0, idx1,
                    scratch.d_evalacc_indexPos, index_stride);
            }
        }
    };

    auto launch_evalacc_single = [&](uint32_t idx) {
        if (use_swizzle_evalacc) {
            kernel_EvalAccCore_Binary_Batch_Swizzle<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_swizzle_.get(), d_RFkey_shoup_swizzle_.get(),
                d_monic_polys_.get(), N, static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(), idx,
                scratch.d_evalacc_indexPos, index_stride);
        } else if (use_evalacc_soa) {
            if (evalacc_smem_enabled) {
                if (index_nmajor) {
                    kernel_EvalAccCore_Binary_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                        d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, idx,
                        scratch.d_evalacc_indexPos, index_stride);
                } else {
                    kernel_EvalAccCore_Binary_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                        d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, idx,
                        scratch.d_evalacc_indexPos, index_stride);
                }
            } else {
                if (index_nmajor) {
                    kernel_EvalAccCore_Binary_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                        d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, idx,
                        scratch.d_evalacc_indexPos, index_stride);
                } else {
                    kernel_EvalAccCore_Binary_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_soa_.get(), d_RFkey_shoup_soa_.get(),
                        d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, idx,
                        scratch.d_evalacc_indexPos, index_stride);
                }
            }
        } else {
            if (index_nmajor) {
                kernel_EvalAccCore_Binary_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_.get(), d_RFkey_shoup_.get(), d_monic_polys_.get(),
                    N, d_modulus_.get(), digitsG2, idx, scratch.d_evalacc_indexPos, index_stride);
            } else {
                kernel_EvalAccCore_Binary_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, d_RFkey_.get(), d_RFkey_shoup_.get(), d_monic_polys_.get(),
                    N, d_modulus_.get(), digitsG2, idx, scratch.d_evalacc_indexPos, index_stride);
            }
        }
    };
    auto finalize_bootstrap_acc = [&]() {
        constexpr uint32_t normBlockX = 256;
        const size_t totalCoeffs = static_cast<size_t>(2) * N * batch;
        const dim3 normBlock(normBlockX);
        const dim3 normGrid((totalCoeffs + normBlock.x - 1) / normBlock.x);
        prof_begin();
        kernel_ModMulConst<<<normGrid, normBlock, 0, s>>>(scratch.d_bootstrap_acc, N_inv.ConvertToInt(), d_modulus_.get(), totalCoeffs);
        prof_end(prof_ms_final_norm);

        prof_begin();
        inwt_1d_opt_batched(scratch.d_bootstrap_acc, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(),
                            d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N,
                            /*batch=*/static_cast<size_t>(2) * batch, s);
        prof_end(prof_ms_final_inwt);

        dim3 gpowBlock(256);
        dim3 gpowGrid((numLUT + gpowBlock.x - 1) / gpowBlock.x, batch);
        prof_begin();
        kernel_ModAddGpowScaledBatch<<<gpowGrid, gpowBlock, 0, s>>>(scratch.d_bootstrap_acc, d_gpow_.get(), N_inv.ConvertToInt(), d_modulus_.get(),
                                                                    static_cast<uint32_t>(N), numLUT, batch);
        prof_end(prof_ms_final_gpow);

        prof_begin();
        fnwt_1d_opt_batched(scratch.d_bootstrap_acc, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N,
                            /*batch=*/static_cast<size_t>(2) * batch, s);
        prof_end(prof_ms_final_fnwt);
    };

    const char* aes_mb3_specialized_env = std::getenv("CIRBTS_AES_MB3_SPECIALIZED");
    const bool aes_mb3_specialized_requested = aes_mb3_specialized_env && aes_mb3_specialized_env[0] != '\0' &&
                                               aes_mb3_specialized_env[0] != '0';
    const char* aes_mb3_lazy_reduce_env = std::getenv("CIRBTS_AES_MB3_LAZY_REDUCE");
    const bool aes_mb3_lazy_reduce_requested = aes_mb3_lazy_reduce_env && aes_mb3_lazy_reduce_env[0] != '\0' &&
                                               aes_mb3_lazy_reduce_env[0] != '0';
    const char* aes_mb3_split_acc_env = std::getenv("CIRBTS_AES_MB3_SPLIT_ACC");
    const bool aes_mb3_split_acc_requested = aes_mb3_split_acc_env && aes_mb3_split_acc_env[0] != '\0' &&
                                             aes_mb3_split_acc_env[0] != '0';
    const char* aes_mb3_fused_dct_ext_env = std::getenv("CIRBTS_AES_MB3_FUSED_DCT_EXT");
    const bool aes_mb3_fused_dct_ext_requested = aes_mb3_fused_dct_ext_env && aes_mb3_fused_dct_ext_env[0] != '\0' &&
                                                 aes_mb3_fused_dct_ext_env[0] != '0';
    const char* aes_fused_inwt_dct_env = std::getenv("CIRBTS_AES_FUSED_INWT_DCT");
    const bool aes_fused_inwt_dct_requested = aes_fused_inwt_dct_env && aes_fused_inwt_dct_env[0] != '\0' &&
                                              aes_fused_inwt_dct_env[0] != '0';
    const bool use_aes_mb3_specialized = aes_mb3_specialized_requested && use_evalacc_soa && index_nmajor &&
                                         !evalacc_smem_enabled && N == 1024 && digitsG2 == 4 &&
                                         evalacc_soa_tiles == 32;
    const bool use_aes_mb3_split_acc = aes_mb3_split_acc_requested && use_evalacc_soa && index_nmajor &&
                                       !evalacc_smem_enabled && N == 1024 && digitsG2 == 4 &&
                                       evalacc_soa_tiles == 32;
    const bool use_aes_fused_inwt_dct = aes_fused_inwt_dct_requested && use_multibit3 && N == 2048 &&
                                        digitsGA == 1 && digitsG2 == 2;
    if (use_aes_fused_inwt_dct && !fused_ntt_tpb_env_set) {
        fusedNttTpb = 384;
    }
    (void)aes_mb3_fused_dct_ext_requested;
    static bool printed_aes_fused_inwt_dct_status = false;
    if (aes_fused_inwt_dct_requested && !printed_aes_fused_inwt_dct_status) {
        printed_aes_fused_inwt_dct_status = true;
        std::cout << "[CIRBTS] CIRBTS_AES_FUSED_INWT_DCT requested active="
                  << (use_aes_fused_inwt_dct ? 1 : 0)
                  << " use_multibit3=" << (use_multibit3 ? 1 : 0)
                  << " N=" << N
                  << " digitsGA=" << digitsGA
                  << " digitsG2=" << digitsG2
                  << " batch=" << batch
                  << " tpb=" << fusedNttTpb
                  << std::endl;
    }

    auto run_group_inwt_dct = [&]() {
        if (use_aes_fused_inwt_dct) {
            prof_begin();
            const size_t fused_smem = static_cast<size_t>(2) * N * sizeof(NativeInt);
            kernel_INWT_Decompose2_FusedFNWT_Batch_AES2048_D1<<<static_cast<size_t>(2) * batch, fusedNttTpb, fused_smem, s>>>(
                scratch.d_evalacc_dct, scratch.d_bootstrap_acc, d_itwiddles_.get(), d_itwiddles_shoup_.get(),
                d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                d_n_inv_mod_q_shoup_.get(), Qint, baseG, batch);
            PHANTOM_CHECK_CUDA_LAST();
            prof_end(prof_ms_group_decomp_fnwt);
            return;
        }

        prof_begin();
        inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                       d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                       d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);
        prof_end(prof_ms_group_inwt);

        prof_begin();
        kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
            scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
            d_twiddles_shoup_.get(), d_modulus_.get());
        PHANTOM_CHECK_CUDA_LAST();
        prof_end(prof_ms_group_decomp_fnwt);
    };

    auto run_evalacc_mb3 = [&]() {
        for (size_t group = 0; group < rfkey_groups_; ++group) {
            const uint32_t idx0 = static_cast<uint32_t>(group * 3u);
            const uint32_t idx1 = idx0 + 1u;
            const uint32_t idx2 = idx0 + 2u;

            run_group_inwt_dct();

            prof_begin();
            if (idx2 < n) {
                if (use_evalacc_soa) {
                    if (evalacc_smem_enabled) {
                        if (index_nmajor) {
                            kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                                ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                                d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1, idx2,
                                scratch.d_evalacc_indexPos, index_stride);
                        } else {
                            kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                                ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                                d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1, idx2,
                                scratch.d_evalacc_indexPos, index_stride);
                        }
                    } else {
                        if (index_nmajor) {
                            if (use_aes_mb3_split_acc) {
                                kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_NMajor_AES1024_D4_SplitAcc<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                    ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                                    ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), d_modulus_.get(),
                                    static_cast<uint32_t>(group), idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride, 0u,
                                    aes_mb3_lazy_reduce_requested);
                                PHANTOM_CHECK_CUDA_LAST();
                                kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_NMajor_AES1024_D4_SplitAcc<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                    ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                                    ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), d_modulus_.get(),
                                    static_cast<uint32_t>(group), idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride, 1u,
                                    aes_mb3_lazy_reduce_requested);
                            } else if (use_aes_mb3_specialized) {
                                kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_NMajor_AES1024_D4<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                    ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                                    ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), d_modulus_.get(),
                                    static_cast<uint32_t>(group), idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride);
                            } else {
                                kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                    scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                    ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                                    ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                                    d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1, idx2,
                                    scratch.d_evalacc_indexPos, index_stride);
                            }
                        } else {
                            kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                                ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                                d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1, idx2,
                                scratch.d_evalacc_indexPos, index_stride);
                        }
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB3_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek2, ek2_shoup,
                            ek01, ek01_shoup, ek02, ek02_shoup, ek12, ek12_shoup, ek012, ek012_shoup, d_monic_polys_.get(),
                            N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1, idx2,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB3_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek2, ek2_shoup,
                            ek01, ek01_shoup, ek02, ek02_shoup, ek12, ek12_shoup, ek012, ek012_shoup, d_monic_polys_.get(),
                            N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1, idx2,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else if (idx1 < n) {
                if (use_evalacc_soa) {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek01_soa, ek01_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                            digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek01_soa, ek01_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                            digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek01, ek01_shoup,
                            d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek01, ek01_shoup,
                            d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else {
                if (use_evalacc_soa) {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, d_monic_polys_.get(),
                            N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, d_monic_polys_.get(),
                            N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, d_monic_polys_.get(), N,
                            d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, d_monic_polys_.get(), N,
                            d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            }
            prof_end(prof_ms_group_external);
            ++prof_group_steps;
        }
    };

    auto run_evalacc_mb4 = [&]() {
        auto launch_group_mb3 = [&](uint32_t key_group, uint32_t idx0, uint32_t idx1, uint32_t idx2) {
            if (use_evalacc_soa) {
                if (evalacc_smem_enabled) {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                            ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                            d_modulus_.get(), digitsG2, key_group, idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                            ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                            d_modulus_.get(), digitsG2, key_group, idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride);
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                            ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                            d_modulus_.get(), digitsG2, key_group, idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB3_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek2_soa, ek2_shoup_soa, ek01_soa, ek01_shoup_soa, ek02_soa, ek02_shoup_soa, ek12_soa,
                            ek12_shoup_soa, ek012_soa, ek012_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles,
                            d_modulus_.get(), digitsG2, key_group, idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else {
                if (index_nmajor) {
                    kernel_EvalAccCore_Binary_MB3_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek2, ek2_shoup,
                        ek01, ek01_shoup, ek02, ek02_shoup, ek12, ek12_shoup, ek012, ek012_shoup, d_monic_polys_.get(),
                        N, d_modulus_.get(), digitsG2, key_group, idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride);
                } else {
                    kernel_EvalAccCore_Binary_MB3_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek2, ek2_shoup,
                        ek01, ek01_shoup, ek02, ek02_shoup, ek12, ek12_shoup, ek012, ek012_shoup, d_monic_polys_.get(),
                        N, d_modulus_.get(), digitsG2, key_group, idx0, idx1, idx2, scratch.d_evalacc_indexPos, index_stride);
                }
            }
        };

        auto launch_group_single = [&](const NativeInt* k, const NativeInt* k_shoup,
                                       const NativeInt* k_soa, const NativeInt* k_shoup_soa,
                                       uint32_t key_group, uint32_t idx) {
            if (use_evalacc_soa) {
                if (evalacc_smem_enabled) {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, k_soa, k_shoup_soa, d_monic_polys_.get(),
                            N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, key_group, idx,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, k_soa, k_shoup_soa, d_monic_polys_.get(),
                            N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, key_group, idx,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, k_soa, k_shoup_soa, d_monic_polys_.get(),
                            N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, key_group, idx,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, k_soa, k_shoup_soa, d_monic_polys_.get(),
                            N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, key_group, idx,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else {
                if (index_nmajor) {
                    kernel_EvalAccCore_Binary_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, k, k_shoup, d_monic_polys_.get(), N,
                        d_modulus_.get(), digitsG2, key_group, idx, scratch.d_evalacc_indexPos, index_stride);
                } else {
                    kernel_EvalAccCore_Binary_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, k, k_shoup, d_monic_polys_.get(), N,
                        d_modulus_.get(), digitsG2, key_group, idx, scratch.d_evalacc_indexPos, index_stride);
                }
            }
        };

        for (size_t group = 0; group < rfkey_groups_; ++group) {
            const uint32_t idx0 = static_cast<uint32_t>(group * 4u);
            const uint32_t idx1 = idx0 + 1u;
            const uint32_t idx2 = idx0 + 2u;
            const uint32_t idx3 = idx0 + 3u;
            if (idx0 >= n) {
                break;
            }

            inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                           d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                           d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

            kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                d_twiddles_shoup_.get(), d_modulus_.get());

            if (idx2 < n) {
                launch_group_mb3(static_cast<uint32_t>(group), idx0, idx1, idx2);
            } else if (idx1 < n) {
                if (use_evalacc_soa) {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek01_soa, ek01_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                            digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                            ek01_soa, ek01_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                            digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek01, ek01_shoup,
                            d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1,
                            scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek01, ek01_shoup,
                            d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, idx1,
                            scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else {
                launch_group_single(ek0, ek0_shoup, ek0_soa, ek0_shoup_soa, static_cast<uint32_t>(group), idx0);
            }

            if (idx3 < n) {
                inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                               d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                               d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

                kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                    scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                    d_twiddles_shoup_.get(), d_modulus_.get());

                launch_group_single(ek3, ek3_shoup, ek3_soa, ek3_shoup_soa, static_cast<uint32_t>(group), idx3);
            }
        }
    };

    auto finish_evalacc_detail_profile = [&]() {
        if (!profile_evalacc_detail) {
            return;
        }
        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
        const char* mode = use_pbs_multibit4 ? "mb4" : (use_pbs_multibit3 ? "mb3" : (use_multibit2 ? "mb2" : "single"));
        const float total_measured = prof_ms_index + prof_ms_acc_init + prof_ms_group_inwt +
                                     prof_ms_group_decomp_fnwt + prof_ms_group_external +
                                     prof_ms_final_norm + prof_ms_final_inwt + prof_ms_final_gpow +
                                     prof_ms_final_fnwt;
        std::cout << "[CIRBTS_PROFILE_EVALACC_DETAIL]"
                  << " batch=" << batch
                  << " n=" << n
                  << " mode=" << mode
                  << " soa=" << (use_evalacc_soa ? 1 : 0)
                  << " smem=" << (evalacc_smem_enabled ? 1 : 0)
                  << " nmajor=" << (index_nmajor ? 1 : 0)
                  << " block_x=" << evalaccBlockX
                  << " groups_profiled=" << prof_group_steps
                  << " index_ms=" << prof_ms_index
                  << " acc_init_ms=" << prof_ms_acc_init
                  << " group_inwt_ms=" << prof_ms_group_inwt
                  << " group_decomp_fnwt_ms=" << prof_ms_group_decomp_fnwt
                  << " group_external_ms=" << prof_ms_group_external
                  << " final_norm_ms=" << prof_ms_final_norm
                  << " final_inwt_ms=" << prof_ms_final_inwt
                  << " final_gpow_ms=" << prof_ms_final_gpow
                  << " final_fnwt_ms=" << prof_ms_final_fnwt
                  << " measured_ms=" << total_measured
                  << std::endl;
        cudaEventDestroy(prof_begin_ev);
        cudaEventDestroy(prof_end_ev);
    };

    if (use_pbs_multibit2) {
        for (size_t group = 0; group < rfkey_groups_; ++group) {
            const uint32_t idx0 = static_cast<uint32_t>(group * 2u);
            const uint32_t idx1 = idx0 + 1u;

            inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                           d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                           d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

            kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                d_twiddles_shoup_.get(), d_modulus_.get());

            if (idx1 < n) {
                if (use_swizzle_evalacc) {
                    kernel_EvalAccCore_Binary_MB2_Grouped_Batch_Swizzle<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek_pair,
                        ek_pair_shoup, d_monic_polys_.get(), N, static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(),
                        static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                } else if (use_evalacc_soa) {
                    if (evalacc_smem_enabled) {
                        if (index_nmajor) {
                            kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                                digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                        } else {
                            kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                                digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                        }
                    } else {
                        if (index_nmajor) {
                            kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                                digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                        } else {
                            kernel_EvalAccCore_Binary_MB2_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, ek1_soa, ek1_shoup_soa,
                                ek_pair_soa, ek_pair_shoup_soa, d_monic_polys_.get(), N, evalacc_soa_tiles, d_modulus_.get(),
                                digitsG2, static_cast<uint32_t>(group), idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                        }
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek_pair,
                            ek_pair_shoup, d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group),
                            idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_MB2_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, ek1, ek1_shoup, ek_pair,
                            ek_pair_shoup, d_monic_polys_.get(), N, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group),
                            idx0, idx1, scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            } else {
                if (use_swizzle_evalacc) {
                    kernel_EvalAccCore_Binary_Grouped_Batch_Swizzle<2><<<evalaccGrid, evalaccBlock, 0, s>>>(
                        scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, d_monic_polys_.get(), N,
                        static_cast<uint32_t>(rfkey_swizzle_tiles_), d_modulus_.get(), static_cast<uint32_t>(group), idx0,
                        scratch.d_evalacc_indexPos, index_stride);
                } else if (use_evalacc_soa) {
                    if (evalacc_smem_enabled) {
                        if (index_nmajor) {
                            kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem_NMajor<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, d_monic_polys_.get(),
                                N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0,
                                scratch.d_evalacc_indexPos, index_stride);
                        } else {
                            kernel_EvalAccCore_Binary_Grouped_Batch_SoA_Smem<<<evalaccGrid, evalaccBlock, evalacc_smem, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, d_monic_polys_.get(),
                                N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0,
                                scratch.d_evalacc_indexPos, index_stride);
                        }
                    } else {
                        if (index_nmajor) {
                            kernel_EvalAccCore_Binary_Grouped_Batch_SoA_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, d_monic_polys_.get(),
                                N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0,
                                scratch.d_evalacc_indexPos, index_stride);
                        } else {
                            kernel_EvalAccCore_Binary_Grouped_Batch_SoA<<<evalaccGrid, evalaccBlock, 0, s>>>(
                                scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0_soa, ek0_shoup_soa, d_monic_polys_.get(),
                                N, evalacc_soa_tiles, d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0,
                                scratch.d_evalacc_indexPos, index_stride);
                        }
                    }
                } else {
                    if (index_nmajor) {
                        kernel_EvalAccCore_Binary_Grouped_Batch_NMajor<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, d_monic_polys_.get(), N,
                            d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, scratch.d_evalacc_indexPos, index_stride);
                    } else {
                        kernel_EvalAccCore_Binary_Grouped_Batch<<<evalaccGrid, evalaccBlock, 0, s>>>(
                            scratch.d_bootstrap_acc, scratch.d_evalacc_dct, ek0, ek0_shoup, d_monic_polys_.get(), N,
                            d_modulus_.get(), digitsG2, static_cast<uint32_t>(group), idx0, scratch.d_evalacc_indexPos, index_stride);
                    }
                }
            }
        }
        finalize_bootstrap_acc();
        finish_evalacc_detail_profile();
        return;
    }

    const uint32_t graph_multibit_mode = use_pbs_multibit4 ? 4u : (use_pbs_multibit3 ? 3u : (use_multibit2 ? 2u : 1u));
    const bool disableBatchGraph = (std::getenv("CIRBTS_DISABLE_EVALACC_BATCH_CUDA_GRAPH") != nullptr) ||
                                   use_evalacc_fuse_ms || profile_evalacc_detail;
    const bool canUseBatchGraph = !disableBatchGraph && scratch.evalacc_batch_graph_exec && scratch.evalacc_batch_graph_failed &&
                                  scratch.evalacc_batch_graph_ntt_tpb && scratch.evalacc_batch_graph_block_x &&
                                  scratch.evalacc_batch_graph_batch && scratch.evalacc_batch_graph_multibit &&
                                  scratch.evalacc_batch_graph_soa && scratch.evalacc_batch_graph_smem;
    if (canUseBatchGraph && !(*scratch.evalacc_batch_graph_failed)) {
        if (*scratch.evalacc_batch_graph_exec &&
            (*scratch.evalacc_batch_graph_ntt_tpb != fusedNttTpb || *scratch.evalacc_batch_graph_block_x != evalaccBlockX ||
             *scratch.evalacc_batch_graph_batch != batch ||
             *scratch.evalacc_batch_graph_multibit != graph_multibit_mode ||
             *scratch.evalacc_batch_graph_soa != use_evalacc_soa ||
             *scratch.evalacc_batch_graph_smem != evalacc_smem_enabled)) {
            cudaGraphExecDestroy(*scratch.evalacc_batch_graph_exec);
            *scratch.evalacc_batch_graph_exec = nullptr;
        }
        if (!*scratch.evalacc_batch_graph_exec) {
            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
            cudaGraph_t graph{};
            PHANTOM_CHECK_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            if (use_pbs_multibit4) {
                run_evalacc_mb4();
            } else if (use_pbs_multibit3) {
                run_evalacc_mb3();
            } else {
                for (uint32_t lweIndex = 0; lweIndex < n; ) {
                    inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                                   d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                                   d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

                    kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                        scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                        d_twiddles_shoup_.get(), d_modulus_.get());

                    if (use_multibit2 && (lweIndex + 1 < n)) {
                        launch_evalacc_pair(lweIndex, lweIndex + 1);
                        lweIndex += 2;
                    } else {
                        launch_evalacc_single(lweIndex);
                        ++lweIndex;
                    }
                }
            }
            PHANTOM_CHECK_CUDA(cudaStreamEndCapture(s, &graph));
            cudaGraphExec_t exec{};
            const auto inst = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
            cudaGraphDestroy(graph);
            if (inst != cudaSuccess) {
                *scratch.evalacc_batch_graph_failed = true;
            }
            else {
                *scratch.evalacc_batch_graph_exec = exec;
                *scratch.evalacc_batch_graph_ntt_tpb = fusedNttTpb;
                *scratch.evalacc_batch_graph_block_x = evalaccBlockX;
                *scratch.evalacc_batch_graph_batch = batch;
                *scratch.evalacc_batch_graph_multibit = graph_multibit_mode;
                *scratch.evalacc_batch_graph_soa = use_evalacc_soa;
                *scratch.evalacc_batch_graph_smem = evalacc_smem_enabled;
            }
        }

        if (*scratch.evalacc_batch_graph_exec) {
            PHANTOM_CHECK_CUDA(cudaGraphLaunch(*scratch.evalacc_batch_graph_exec, s));
        }
        else {
            if (use_pbs_multibit4) {
                run_evalacc_mb4();
            } else if (use_pbs_multibit3) {
                run_evalacc_mb3();
            } else {
                for (uint32_t lweIndex = 0; lweIndex < n; ) {
                    inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                                   d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                                   d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

                    kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                        scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                        d_twiddles_shoup_.get(), d_modulus_.get());

                    if (use_multibit2 && (lweIndex + 1 < n)) {
                        launch_evalacc_pair(lweIndex, lweIndex + 1);
                        lweIndex += 2;
                    } else {
                        launch_evalacc_single(lweIndex);
                        ++lweIndex;
                    }
                }
            }
        }
    }
    else {
        if (use_pbs_multibit4) {
            run_evalacc_mb4();
        } else if (use_pbs_multibit3) {
            run_evalacc_mb3();
        } else {
            for (uint32_t lweIndex = 0; lweIndex < n; ) {
                inwt_1d_opt_outofplace_batched(scratch.d_evalacc_ct, scratch.d_bootstrap_acc, d_itwiddles_.get(),
                                               d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                                               d_n_inv_mod_q_shoup_.get(), N, /*batch=*/static_cast<size_t>(2) * batch, s);

                kernel_SignedDigitDecompose2_FusedFNWT_Batch<<<static_cast<size_t>(digitsG2) * batch, fusedNttTpb, evalaccNttShmem, s>>>(
                    scratch.d_evalacc_dct, scratch.d_evalacc_ct, Qint, baseG, digitsGA, N, batch, d_twiddles_.get(),
                    d_twiddles_shoup_.get(), d_modulus_.get());

                if (use_multibit2 && (lweIndex + 1 < n)) {
                    launch_evalacc_pair(lweIndex, lweIndex + 1);
                    lweIndex += 2;
                } else {
                    launch_evalacc_single(lweIndex);
                    ++lweIndex;
                }
            }
        }
    }

    finalize_bootstrap_acc();
    finish_evalacc_detail_profile();
}
