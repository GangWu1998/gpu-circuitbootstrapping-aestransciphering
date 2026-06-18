/*
 * =============================================================================
 * File: cirbts-init.cu
 * Purpose: GPU resource initialization and teardown for CirBTS, including
 *          key uploads, NTT table setup, scratch sizing, and pinned buffers.
 * Key parameters:
 *   - max_batch_size_: controls batched scratch allocation sizes.
 *   - ring_dim_/num_luts_: shape NTT tables and LUT-related buffers.
 *   - pinned H2D staging buffers/events: enable async transfers.
 * Key points:
 *   - Centralizes allocations, stream/event creation, and device setup.
 *   - No algorithm changes; focuses on memory and pipeline readiness.
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
#include <cctype>
#include <vector>
#include <unordered_set>
#include <iostream>
#include <sstream>
#include <utility>
#include <cuda_runtime.h>
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
uint32_t parse_env_u32(const char* name) {
    if (const char* v = std::getenv(name); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0') {
            return static_cast<uint32_t>(parsed);
        }
    }
    return 0;
}

uint32_t get_lmk_auto_window() {
    return parse_env_u32("CIRBTS_LMK_AUTOKEYS_WINDOW");
}

uint32_t powmod_u32(uint32_t base, uint32_t exp, uint32_t mod) {
    uint64_t res = 1;
    uint64_t b   = base;
    while (exp) {
        if (exp & 1U) {
            res = (res * b) % mod;
        }
        b = (b * b) % mod;
        exp >>= 1U;
    }
    return static_cast<uint32_t>(res);
}

std::vector<uint32_t> build_lmk_auto_pows(uint32_t N, uint32_t numAutoKeys, uint32_t window) {
    constexpr uint32_t gen = 5;
    const uint32_t M = 2 * N;
    const uint32_t Nh = N / 2;

    std::vector<uint32_t> pows;
    pows.reserve(1 + numAutoKeys);
    std::unordered_set<uint32_t> seen;

    const uint32_t special = M - gen;
    pows.push_back(special);
    seen.insert(special);

    auto push_pow = [&](uint32_t k) {
        const uint32_t pow = powmod_u32(gen, k, M);
        if (seen.insert(pow).second) {
            pows.push_back(pow);
        }
    };

    if (window > 1) {
        const uint32_t max_k = (Nh > 0) ? (Nh - 1) : 0;
        for (uint32_t k = 1; k <= window; ++k) {
            push_pow(k);
        }
        for (uint32_t k = window; k <= max_k; k += window) {
            push_pow(k);
        }
    }
    else {
        for (uint32_t k = 1; k <= numAutoKeys; ++k) {
            push_pow(k);
        }
    }

    return pows;
}

size_t parse_env_bytes(const char* env) {
    if (!env || !*env) {
        return 0;
    }
    char* end = nullptr;
    double value = std::strtod(env, &end);
    if (end == env) {
        return 0;
    }
    size_t scale = 1;
    if (end && *end != '\0') {
        const char suffix = static_cast<char>(std::toupper(*end));
        if (suffix == 'K') {
            scale = 1024u;
        } else if (suffix == 'M') {
            scale = 1024u * 1024u;
        } else if (suffix == 'G') {
            scale = 1024u * 1024u * 1024u;
        }
    }
    const double scaled = value * static_cast<double>(scale);
    if (scaled <= 0.0) {
        return 0;
    }
    return static_cast<size_t>(scaled);
}

float parse_env_float(const char* env, float fallback) {
    if (!env || !*env) {
        return fallback;
    }
    char* end = nullptr;
    const float value = std::strtof(env, &end);
    if (end == env) {
        return fallback;
    }
    return value;
}
}  // namespace

GPUCirBTSContext::~GPUCirBTSContext() {
    if (cggi_evalacc_graph_exec_) {
        cudaGraphExecDestroy(cggi_evalacc_graph_exec_);
        cggi_evalacc_graph_exec_ = nullptr;
    }
    cggi_evalacc_graph_acc_ = nullptr;

    if (cggi_evalacc_mb2_graph_exec_) {
        cudaGraphExecDestroy(cggi_evalacc_mb2_graph_exec_);
        cggi_evalacc_mb2_graph_exec_ = nullptr;
    }
    cggi_evalacc_mb2_graph_acc_ = nullptr;

    if (split_fft_evalacc_graph_exec_) {
        cudaGraphExecDestroy(split_fft_evalacc_graph_exec_);
        split_fft_evalacc_graph_exec_ = nullptr;
    }
    split_fft_evalacc_graph_acc_ = nullptr;

    if (cggi_evalacc_batch_graph_exec_) {
        cudaGraphExecDestroy(cggi_evalacc_batch_graph_exec_);
        cggi_evalacc_batch_graph_exec_ = nullptr;
    }

    if (htss_graph_exec_) {
        cudaGraphExecDestroy(htss_graph_exec_);
        htss_graph_exec_ = nullptr;
    }

    if (h_monomial_inv_pinned_) {
        cudaFreeHost(h_monomial_inv_pinned_);
        h_monomial_inv_pinned_ = nullptr;
    }
    if (h_indexPos_pinned_) {
        cudaFreeHost(h_indexPos_pinned_);
        h_indexPos_pinned_ = nullptr;
    }
    if (h_rgsw_pinned_) {
        cudaFreeHost(h_rgsw_pinned_);
        h_rgsw_pinned_ = nullptr;
    }

    if (h2d_monomial_event_) {
        cudaEventDestroy(h2d_monomial_event_);
        h2d_monomial_event_ = nullptr;
    }
    if (h2d_index_event_) {
        cudaEventDestroy(h2d_index_event_);
        h2d_index_event_ = nullptr;
    }
}

void GPUCirBTSContext::gpu_init_cirbsk(CirBTSContext& cc) {
    /**
     * Main constructor for CirBTSCryptoParams
     *
     * @param lweparams a shared poiter to an instance of LWECryptoParams
     * @param rgswparams1 a shared poiter to an instance of RingGSWCryptoParams
     * @param rlweparams a shared poiter to an instance of RingLWECryptoParams
     * @param rgswparams2 a shared poiter to an instance of RingGSWCryptoParams
     * @param h_SSkey a shared poiter to an instance of RingGSWEvalKey(SSkey)
     * @param h_elements is SSkey inner 2D vector
     */
    const auto& LWE_params   = cc.GetParams()->GetLWEParams();
    const auto& RGSW_params1 = cc.GetParams()->GetRingGSWParams1();
    const auto& RLWE_params  = cc.GetParams()->GetRLWEParams();
    const auto& h_elements   = cc.GetSchemeSwitchingKey()->GetElements();
    const NativeInt key_modulus = RGSW_params1->GetQ().ConvertToInt();

    auto& s = stream_wrapper_.get_stream();
    const auto method = RGSW_params1->GetMethod();
    const auto& h_RFKey = cc.GetRefreshKey()->GetElements();

    backend_             = phantom::bitwise::ParseCirBTSBackend();
    use_split_fft_       = (backend_ == phantom::bitwise::CirBTSBackend::kSplitFFT);
    use_ntt32_rns_       = (backend_ == phantom::bitwise::CirBTSBackend::kNTT32RNS);
    bool swizzle_requested = false;
    if (const char* v = std::getenv("CIRBTS_USE_SWIZZLE"); v && *v) {
        swizzle_requested = (std::strcmp(v, "0") != 0);
    }
    split_fft_bits_      = 0;
    split_fft_limbs_     = 0;
    split_fft_fft_len_   = 0;
    split_fft_key_polys_ = 0;
    split_fft_key_stride_ = 0;
    split_fft_base_pows_.clear();
    split_fft_base_pows_shoup_.clear();
    d_RFkey_fft_limbs_.clear();
    d_monic_fft_ = {};
    split_fft_fpb_ = 1;
    use_swizzle_ = false;
    rfkey_swizzle_tiles_ = 0;
    d_RFkey_swizzle_ = {};
    d_RFkey_shoup_swizzle_ = {};
    ntt32_rns_limbs_ = 0;
    ntt32_rns_plan_.reset();
    d_RFkey_rns32_ = {};
    d_RFkey_rns32_shoup_ = {};
    d_HTkey_rns32_ = {};
    d_HTkey_rns32_shoup_ = {};
    d_SSkey_rns32_ = {};
    d_SSkey_rns32_shoup_ = {};
    d_monic_polys_rns32_ = {};
    d_monic_polys_rns32_shoup_ = {};
    rfkey_rns32_polys_ = 0;
    htkey_rns32_polys_ = 0;
    sskey_rns32_polys_ = 0;
    monic_rns32_polys_ = 0;
    d_HTkey_soa_ = {};
    d_HTkey_shoup_soa_ = {};
    htkey_soa_tiles_ = 0;
    d_SSkey_soa_ = {};
    d_SSkey_shoup_soa_ = {};
    sskey_soa_tiles_ = 0;
    l2_persist_enabled_ = false;
    l2_persist_failed_ = false;
    l2_persist_bytes_ = 0;
    l2_persist_hit_ratio_ = 0.6f;
    if (split_fft_evalacc_graph_exec_) {
        cudaGraphExecDestroy(split_fft_evalacc_graph_exec_);
    }
    split_fft_evalacc_graph_exec_  = nullptr;
    split_fft_evalacc_graph_failed_ = false;
    split_fft_evalacc_graph_acc_    = nullptr;
    split_fft_evalacc_graph_limbs_  = 0;
    fft_2048_fpb1_.reset();
    fft_2048_fpb2_.reset();
    fft_2048_fpb4_.reset();
    d_evalacc_dct_i64_ = {};
    d_evalacc_dct_fft_ = {};
    d_evalacc_acc_fft_ = {};
    d_evalacc_digits_s_ = {};
    d_evalacc_residual_s_ = {};
    d_evalacc_delta_ = {};
    d_evalacc_carry_flag_ = {};
    d_evalacc_acc_backup_ = {};
    evalacc_ntt_tpb_ = 0;
    evalacc_block_x_ = 0;
    evalacc_autotuned_ = false;

    const bool l2_requested = (std::getenv("CIRBTS_L2_PERSIST") != nullptr);
    const bool l2_verbose = (std::getenv("CIRBTS_L2_PERSIST_VERBOSE") != nullptr);
    const bool build_htss_soa = (std::getenv("CIRBTS_HTSS_SOA") != nullptr);
    if (l2_requested) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 10000)
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess) {
            int maxPersist = 0;
            int l2Size = 0;
            if (cudaDeviceGetAttribute(&maxPersist, cudaDevAttrMaxPersistingL2CacheSize, dev) == cudaSuccess &&
                cudaDeviceGetAttribute(&l2Size, cudaDevAttrL2CacheSize, dev) == cudaSuccess) {
                const size_t maxBytes = static_cast<size_t>(std::min(maxPersist, l2Size));
                size_t reqBytes = parse_env_bytes(std::getenv("CIRBTS_L2_PERSIST_BYTES"));
                if (reqBytes == 0) {
                    reqBytes = maxBytes;
                }
                reqBytes = std::min(reqBytes, maxBytes);
                if (reqBytes > 0) {
                    if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reqBytes) == cudaSuccess) {
                        l2_persist_bytes_ = reqBytes;
                        l2_persist_hit_ratio_ = parse_env_float(std::getenv("CIRBTS_L2_PERSIST_HIT_RATIO"), 0.6f);
                        l2_persist_enabled_ = true;
                        if (l2_verbose) {
                            std::cout << "[CIRBTS] L2 persist enabled (" << l2_persist_bytes_ << " bytes, hitRatio="
                                      << l2_persist_hit_ratio_ << ")." << std::endl;
                        }
                    } else if (l2_verbose) {
                        std::cout << "[CIRBTS] L2 persist request rejected by device." << std::endl;
                    }
                } else if (l2_verbose) {
                    std::cout << "[CIRBTS] L2 persist not supported (maxBytes=0)." << std::endl;
                }
            } else if (l2_verbose) {
                std::cout << "[CIRBTS] L2 persist attribute query failed." << std::endl;
            }
        }
#else
        if (l2_verbose) {
            std::cout << "[CIRBTS] L2 persist unsupported (CUDART_VERSION too old)." << std::endl;
        }
#endif
    }


    if (method == BINFHE_METHOD::GINX) {
#if defined(PHANTOM_ENABLE_CUFFTDX)
        use_split_fft_ = (backend_ == phantom::bitwise::CirBTSBackend::kSplitFFT);
#else
        use_split_fft_ = false;
#endif
        use_ntt32_rns_ = (backend_ == phantom::bitwise::CirBTSBackend::kNTT32RNS);
    }
    else {
        use_ntt32_rns_ = false;
    }
    use_swizzle_ = swizzle_requested && (method == BINFHE_METHOD::GINX) &&
                   (backend_ == phantom::bitwise::CirBTSBackend::kNTT);

    // Reset optional LMKCDEY data unless initialized below.
    d_LMK_evalkey_   = {};
    d_LMK_autokey_   = {};
    d_LMK_autoMaps_  = {};
    lmk_numAutoKeys_ = 0;
    lmk_auto_key_count_ = 0;
    lmk_auto_window_ = 0;
    d_lmk_ct_        = {};
    d_lmk_dct_       = {};
    d_lmk_perm_c0_   = {};
    d_lmk_perm_c1_   = {};
    d_lmk_dcta_      = {};
    d_lmk_ks_c0_     = {};
    d_lmk_ks_c1_     = {};
    lmk_digitsG_auto_   = 0;
    lmk_digitsG2_eval_  = 0;
    lmk_N_              = 0;
    rfkey_dim2_         = 1;
    rfkey_groups_       = 0;
    rfkey_grouped_      = false;

    if (method == BINFHE_METHOD::GINX) {
        // RFkey -> GPU (RingGSWACCKey) for CGGI2 (binary): [1][1][n][digitsGA*2][2][N]
        // Optional multi-bit (2-bit) EvalAcc adds a second dim2 block: [1][2][n][digitsGA*2][2][N].
        constexpr size_t RFkey_dim1          = 1;
        const size_t RFkey_dim2              = h_RFKey.empty() ? 0u : h_RFKey[0].size();
        const size_t RFkey_LWE_n             = LWE_params->Getn();
        const size_t RFkey_digitsG2          = RGSW_params1->GetDigitsGA() << 1;
        constexpr size_t RFkey_RGSW_col_size = 2;
        const size_t RFkey_RGSW_N            = RGSW_params1->GetN();
        uint32_t pbs_multibit = 0;
        if (const char* v = std::getenv("CIRBTS_PBS_MULTIBIT"); v && *v) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(v, &end, 10);
            if (end != v && end && *end == '\0') {
                pbs_multibit = static_cast<uint32_t>(parsed);
            }
        }
        const bool use_pbs_multibit2 = (pbs_multibit == 2u);
        const bool use_pbs_multibit3 = (pbs_multibit == 3u);
        const bool use_pbs_multibit4 = (pbs_multibit == 4u);

        if (h_RFKey.size() != RFkey_dim1) {
            OPENFHE_THROW(config_error, "GPUCirBTSContext RFkey has unexpected dim1 (expected 1)");
        }
        if (RFkey_dim2 != 1 && RFkey_dim2 != 2 && RFkey_dim2 != 4) {
            OPENFHE_THROW(config_error,
                          "GPUCirBTSContext RFkey has unexpected dim2 (expected 1, 2, or 4)");
        }
        if (h_RFKey[0][0].size() != RFkey_LWE_n) {
            OPENFHE_THROW(config_error, "GPUCirBTSContext RFkey has unexpected dim3 (n mismatch)");
        }
        if (!h_RFKey[0][0][0]) {
            OPENFHE_THROW(config_error, "GPUCirBTSContext RFkey contains null eval key");
        }
        if (h_RFKey[0][0][0]->GetElements().size() != RFkey_digitsG2) {
            OPENFHE_THROW(config_error,
                          "GPUCirBTSContext RFkey has unexpected digitsG2 (approx gadget decomposition mismatch)");
        }
        if (h_RFKey[0][0][0]->GetElements()[0].size() != RFkey_RGSW_col_size) {
            OPENFHE_THROW(config_error, "GPUCirBTSContext RFkey has unexpected RGSW column size");
        }
        if (h_RFKey[0][0][0]->GetElements()[0][0].GetValues().GetLength() != RFkey_RGSW_N) {
            OPENFHE_THROW(config_error, "GPUCirBTSContext RFkey has unexpected polynomial ring dimension");
        }

        if (use_pbs_multibit2 && RFkey_dim2 != 2) {
            OPENFHE_THROW(config_error,
                          "GPUCirBTSContext PBS multibit requires RFkey dim2=2 (set CIRBTS_PBS_MULTIBIT=2 during keygen)");
        }
        if (use_pbs_multibit3 && RFkey_dim2 != 4) {
            OPENFHE_THROW(config_error,
                          "GPUCirBTSContext PBS multibit=3 requires RFkey dim2=4 (set CIRBTS_PBS_MULTIBIT=3 during keygen)");
        }
        if (use_pbs_multibit4 && RFkey_dim2 != 4) {
            OPENFHE_THROW(config_error,
                          "GPUCirBTSContext PBS multibit=4 requires RFkey dim2=4 (set CIRBTS_PBS_MULTIBIT=4 during keygen)");
        }

        const size_t rfkey_groups = use_pbs_multibit2 ? ((RFkey_LWE_n + 1u) / 2u)
                                                      : (use_pbs_multibit3 ? ((RFkey_LWE_n + 2u) / 3u)
                                                                           : (use_pbs_multibit4 ? ((RFkey_LWE_n + 3u) / 4u)
                                                                                               : RFkey_LWE_n));
        const size_t rfkey_dim2_out = use_pbs_multibit2 ? 3u : (use_pbs_multibit3 ? 7u : (use_pbs_multibit4 ? 8u : RFkey_dim2));

        rfkey_dim2_     = static_cast<uint32_t>(rfkey_dim2_out);
        rfkey_groups_   = rfkey_groups;
        rfkey_grouped_  = use_pbs_multibit2 || use_pbs_multibit3 || use_pbs_multibit4;
        size_RFkey_     = RFkey_dim1 * rfkey_dim2_out * rfkey_groups;
        size_RFRGSWkey_ = RFkey_RGSW_col_size * RFkey_digitsG2 * RFkey_RGSW_N;
        d_RFkey_        = phantom::util::make_cuda_auto_ptr<NativeInt>(size_RFkey_ * size_RFRGSWkey_, s);
        d_RFkey_shoup_  = phantom::util::make_cuda_auto_ptr<NativeInt>(size_RFkey_ * size_RFRGSWkey_, s);

        auto copy_eval_key = [&](size_t dst_j, size_t dst_k, const std::shared_ptr<RingGSWEvalKeyImpl>& key) {
            if (!key) {
                OPENFHE_THROW(config_error, "GPUCirBTSContext RFkey contains null eval key");
            }
            if (key->GetElements().size() != RFkey_digitsG2 || key->GetElements()[0].size() != RFkey_RGSW_col_size ||
                key->GetElements()[0][0].GetValues().GetLength() != RFkey_RGSW_N) {
                OPENFHE_THROW(config_error, "GPUCirBTSContext RFkey has unexpected key shape");
            }
            std::vector<NativeInt> shoup_key(size_RFRGSWkey_);
            for (size_t l = 0; l < RFkey_digitsG2; ++l) {
                for (size_t m = 0; m < RFkey_RGSW_col_size; ++m) {
                    const size_t base =
                        dst_j * rfkey_groups * size_RFRGSWkey_ + dst_k * size_RFRGSWkey_ +
                        l * RFkey_RGSW_col_size * RFkey_RGSW_N + m * RFkey_RGSW_N;
                    const auto& poly = key->GetElements()[l][m].GetValues();
                    const auto* coeffs = &poly.at(0);
                    for (size_t coeff = 0; coeff < RFkey_RGSW_N; ++coeff) {
                        shoup_key[l * RFkey_RGSW_col_size * RFkey_RGSW_N + m * RFkey_RGSW_N + coeff] =
                            static_cast<NativeInt>(
                                phantom::arith::compute_shoup(
                                    static_cast<uint64_t>(coeffs[coeff].ConvertToInt<NativeInt>()),
                                    static_cast<uint64_t>(key_modulus)));
                    }
                    cudaMemcpyAsync(d_RFkey_.get() + base, coeffs, sizeof(NativeInt) * RFkey_RGSW_N,
                                    cudaMemcpyHostToDevice, s);
                }
            }
            cudaMemcpyAsync(d_RFkey_shoup_.get() + dst_j * rfkey_groups * size_RFRGSWkey_ + dst_k * size_RFRGSWkey_,
                            shoup_key.data(), shoup_key.size() * sizeof(NativeInt), cudaMemcpyHostToDevice, s);
            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
        };

        auto zero_eval_key = [&](size_t dst_j, size_t dst_k) {
            const size_t base = dst_j * rfkey_groups * size_RFRGSWkey_ + dst_k * size_RFRGSWkey_;
            PHANTOM_CHECK_CUDA(cudaMemsetAsync(d_RFkey_.get() + base, 0, size_RFRGSWkey_ * sizeof(NativeInt), s));
            PHANTOM_CHECK_CUDA(cudaMemsetAsync(d_RFkey_shoup_.get() + base, 0, size_RFRGSWkey_ * sizeof(NativeInt), s));
        };

        if (!use_pbs_multibit2 && !use_pbs_multibit3 && !use_pbs_multibit4) {
            for (size_t i = 0; i < RFkey_dim1; ++i) {
                for (size_t j = 0; j < RFkey_dim2; ++j) {
                    for (size_t k = 0; k < RFkey_LWE_n; ++k) {
                        copy_eval_key(j, k, h_RFKey[i][j][k]);
                    }
                }
            }
        } else if (use_pbs_multibit2) {
            for (size_t group = 0; group < rfkey_groups; ++group) {
                const size_t idx0 = group * 2u;
                const size_t idx1 = idx0 + 1u;
                copy_eval_key(0u, group, h_RFKey[0][0][idx0]);
                if (idx1 < RFkey_LWE_n) {
                    copy_eval_key(1u, group, h_RFKey[0][0][idx1]);
                    copy_eval_key(2u, group, h_RFKey[0][1][idx0]);
                } else {
                    zero_eval_key(1u, group);
                    zero_eval_key(2u, group);
                    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
                }
            }
        } else if (use_pbs_multibit3) {
            for (size_t group = 0; group < rfkey_groups; ++group) {
                const size_t idx0 = group * 3u;
                const size_t idx1 = idx0 + 1u;
                const size_t idx2 = idx0 + 2u;

                // singles: k0, k1, k2
                copy_eval_key(0u, group, h_RFKey[0][0][idx0]);
                if (idx1 < RFkey_LWE_n) {
                    copy_eval_key(1u, group, h_RFKey[0][0][idx1]);
                } else {
                    zero_eval_key(1u, group);
                }
                if (idx2 < RFkey_LWE_n) {
                    copy_eval_key(2u, group, h_RFKey[0][0][idx2]);
                } else {
                    zero_eval_key(2u, group);
                }

                // pairs/triple: k01, k02, k12, k012
                if (idx1 < RFkey_LWE_n) {
                    copy_eval_key(3u, group, h_RFKey[0][1][idx0]);
                    copy_eval_key(5u, group, h_RFKey[0][1][idx1]);
                } else {
                    zero_eval_key(3u, group);
                    zero_eval_key(5u, group);
                }
                if (idx2 < RFkey_LWE_n) {
                    copy_eval_key(4u, group, h_RFKey[0][2][idx0]);
                    copy_eval_key(6u, group, h_RFKey[0][3][idx0]);
                } else {
                    zero_eval_key(4u, group);
                    zero_eval_key(6u, group);
                }

                PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
            }
        } else {
            // MB4 staged path key map per group (size 4):
            // [0]=k0 [1]=k1 [2]=k2 [3]=k3 [4]=k01 [5]=k02 [6]=k12 [7]=k012
            for (size_t group = 0; group < rfkey_groups; ++group) {
                const size_t idx0 = group * 4u;
                const size_t idx1 = idx0 + 1u;
                const size_t idx2 = idx0 + 2u;
                const size_t idx3 = idx0 + 3u;

                if (idx0 < RFkey_LWE_n) {
                    copy_eval_key(0u, group, h_RFKey[0][0][idx0]);
                } else {
                    zero_eval_key(0u, group);
                }
                if (idx1 < RFkey_LWE_n) {
                    copy_eval_key(1u, group, h_RFKey[0][0][idx1]);
                } else {
                    zero_eval_key(1u, group);
                }
                if (idx2 < RFkey_LWE_n) {
                    copy_eval_key(2u, group, h_RFKey[0][0][idx2]);
                } else {
                    zero_eval_key(2u, group);
                }
                if (idx3 < RFkey_LWE_n) {
                    copy_eval_key(3u, group, h_RFKey[0][0][idx3]);
                } else {
                    zero_eval_key(3u, group);
                }

                if (idx1 < RFkey_LWE_n) {
                    copy_eval_key(4u, group, h_RFKey[0][1][idx0]);
                } else {
                    zero_eval_key(4u, group);
                }
                if (idx2 < RFkey_LWE_n) {
                    copy_eval_key(5u, group, h_RFKey[0][2][idx0]);
                    copy_eval_key(6u, group, h_RFKey[0][1][idx1]);
                    copy_eval_key(7u, group, h_RFKey[0][3][idx0]);
                } else {
                    zero_eval_key(5u, group);
                    zero_eval_key(6u, group);
                    zero_eval_key(7u, group);
                }

                PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
            }
        }

        if (use_swizzle_) {
            if ((RFkey_RGSW_N % 32u) != 0 || RFkey_digitsG2 != 2u) {
                use_swizzle_ = false;
                std::cerr << "[CIRBTS] RF key swizzle disabled (requires N divisible by 32 and digitsG2 == 2)." << std::endl;
            } else {
                rfkey_swizzle_tiles_ = static_cast<size_t>(RFkey_RGSW_N / 32u);
                const size_t total = size_RFkey_ * size_RFRGSWkey_;
                d_RFkey_swizzle_ = phantom::util::make_cuda_auto_ptr<NativeInt>(total, s);
                d_RFkey_shoup_swizzle_ = phantom::util::make_cuda_auto_ptr<NativeInt>(total, s);

                dim3 block(256);
                dim3 grid(static_cast<unsigned int>((total + block.x - 1) / block.x));
                kernel_SwizzleRFKey<<<grid, block, 0, s>>>(d_RFkey_swizzle_.get(), d_RFkey_.get(), RFkey_RGSW_N,
                                                           RFkey_digitsG2, static_cast<uint32_t>(rfkey_swizzle_tiles_),
                                                           size_RFkey_);
                kernel_SwizzleRFKey<<<grid, block, 0, s>>>(d_RFkey_shoup_swizzle_.get(), d_RFkey_shoup_.get(), RFkey_RGSW_N,
                                                           RFkey_digitsG2, static_cast<uint32_t>(rfkey_swizzle_tiles_),
                                                           size_RFkey_);
                PHANTOM_CHECK_CUDA(cudaPeekAtLastError());
            }
        }

        const char* soa_env = std::getenv("CIRBTS_EVALACC_SOA");
        const bool build_soa = (soa_env == nullptr) ? true : (soa_env[0] != '0');
        if (build_soa) {
            if ((RFkey_RGSW_N % 32u) != 0) {
                std::cerr << "[CIRBTS] RF key SoA disabled (requires N divisible by 32)." << std::endl;
            } else {
                rfkey_soa_tiles_ = static_cast<size_t>(RFkey_RGSW_N / 32u);
                const size_t total = size_RFkey_ * size_RFRGSWkey_;
                d_RFkey_soa_ = phantom::util::make_cuda_auto_ptr<NativeInt>(total, s);
                d_RFkey_shoup_soa_ = phantom::util::make_cuda_auto_ptr<NativeInt>(total, s);

                dim3 block(256);
                dim3 grid(static_cast<unsigned int>((total + block.x - 1) / block.x));
                kernel_SwizzleRFKey<<<grid, block, 0, s>>>(d_RFkey_soa_.get(), d_RFkey_.get(), RFkey_RGSW_N,
                                                           RFkey_digitsG2, static_cast<uint32_t>(rfkey_soa_tiles_),
                                                           size_RFkey_);
                kernel_SwizzleRFKey<<<grid, block, 0, s>>>(d_RFkey_shoup_soa_.get(), d_RFkey_shoup_.get(), RFkey_RGSW_N,
                                                           RFkey_digitsG2, static_cast<uint32_t>(rfkey_soa_tiles_),
                                                           size_RFkey_);
                PHANTOM_CHECK_CUDA(cudaPeekAtLastError());
            }
        }
    }
    else if (method == BINFHE_METHOD::LMKCDEY) {
        // RFkey for LMKCDEY: [1][2][n], where
        //   [0] contains RGSW(X^{s_i}) keys (digitsG2, 2 cols)
        //   [1] contains automorphism keys indexed by i in [0..numAutoKeys] (digitsG, 2 cols)
        const size_t lwe_n = LWE_params->Getn();
        const uint32_t N = RGSW_params1->GetN();
        const uint32_t digitsG = RGSW_params1->GetDigitsG();
        if (digitsG < 2) {
            OPENFHE_THROW(config_error, "GPUCirBTSContext LMKCDEY requires digitsG >= 2");
        }
        // OpenFHE LMKCDEY uses an approximate gadget decomposition that ignores the first digit.
        const uint32_t digitsG_auto  = digitsG - 1;
        const uint32_t digitsG2_eval = digitsG_auto << 1;
        lmk_numAutoKeys_ = RGSW_params1->GetNumAutoKeys();
        const uint32_t auto_window = get_lmk_auto_window();
        const auto autoPows = build_lmk_auto_pows(N, lmk_numAutoKeys_, auto_window);
        lmk_auto_key_count_ = static_cast<uint32_t>(autoPows.size());
        lmk_auto_window_ = auto_window;

        if (h_RFKey.size() != 1 || h_RFKey[0].size() != 2) {
            OPENFHE_THROW(config_error, "GPUCirBTSContext LMKCDEY RFkey has unexpected dim1/dim2");
        }
        if (h_RFKey[0][0].size() != lwe_n || h_RFKey[0][1].size() != lwe_n) {
            OPENFHE_THROW(config_error, "GPUCirBTSContext LMKCDEY RFkey has unexpected dim3 (n mismatch)");
        }
        if (lmk_auto_key_count_ > lwe_n) {
            OPENFHE_THROW(config_error, "GPUCirBTSContext LMKCDEY auto key count exceeds RFkey dim3");
        }

        const size_t evalKeyStride = static_cast<size_t>(digitsG2_eval) * 2 * N;
        const size_t autoKeyStride = static_cast<size_t>(digitsG_auto) * 2 * N;
        d_LMK_evalkey_  = phantom::util::make_cuda_auto_ptr<NativeInt>(lwe_n * evalKeyStride, s);
        d_LMK_autokey_  = phantom::util::make_cuda_auto_ptr<NativeInt>(static_cast<size_t>(lmk_auto_key_count_) * autoKeyStride, s);
        d_LMK_autoMaps_ = phantom::util::make_cuda_auto_ptr<uint32_t>(static_cast<size_t>(lmk_auto_key_count_) * N, s);

        // Copy eval keys: RGSW(X^{s_i})
        for (size_t i = 0; i < lwe_n; ++i) {
            const auto& key = h_RFKey[0][0][i];
            if (!key) {
                OPENFHE_THROW(config_error, "GPUCirBTSContext LMKCDEY RFkey[0][0] contains null eval key");
            }
            if (key->GetElements().size() != digitsG2_eval || key->GetElements()[0].size() != 2 ||
                key->GetElements()[0][0].GetValues().GetLength() != N) {
                OPENFHE_THROW(config_error, "GPUCirBTSContext LMKCDEY RFkey[0][0] has unexpected shape");
            }
            for (uint32_t d = 0; d < digitsG2_eval; ++d) {
                for (uint32_t col = 0; col < 2; ++col) {
                    cudaMemcpyAsync(d_LMK_evalkey_.get() + i * evalKeyStride + d * (2 * N) + col * N,
                                    &key->GetElements()[d][col].GetValues().at(0), sizeof(NativeInt) * N,
                                    cudaMemcpyHostToDevice, s);
                }
            }
        }

        // Copy automorphism keys: indices [0..auto_key_count)
        for (uint32_t i = 0; i < lmk_auto_key_count_; ++i) {
            const auto& key = h_RFKey[0][1][i];
            if (!key) {
                OPENFHE_THROW(config_error, "GPUCirBTSContext LMKCDEY RFkey[0][1] contains null automorphism key");
            }
            if (key->GetElements().size() != digitsG_auto || key->GetElements()[0].size() != 2 ||
                key->GetElements()[0][0].GetValues().GetLength() != N) {
                OPENFHE_THROW(config_error, "GPUCirBTSContext LMKCDEY RFkey[0][1] has unexpected shape");
            }
            for (uint32_t d = 0; d < digitsG_auto; ++d) {
                for (uint32_t col = 0; col < 2; ++col) {
                    cudaMemcpyAsync(d_LMK_autokey_.get() + static_cast<size_t>(i) * autoKeyStride + d * (2 * N) +
                                                      col * N,
                                    &key->GetElements()[d][col].GetValues().at(0), sizeof(NativeInt) * N,
                                    cudaMemcpyHostToDevice, s);
                }
            }
        }

        // Precompute automorphism maps (evaluation format) for the configured powers.
        std::vector<uint32_t> hostMaps(static_cast<size_t>(lmk_auto_key_count_) * N);
        for (uint32_t i = 0; i < lmk_auto_key_count_; ++i) {
            std::vector<uint32_t> vec(N);
            PrecomputeAutoMap(N, autoPows[i], &vec);
            std::copy(vec.begin(), vec.end(), hostMaps.begin() + static_cast<size_t>(i) * N);
        }
        cudaMemcpyAsync(d_LMK_autoMaps_.get(), hostMaps.data(), hostMaps.size() * sizeof(uint32_t), cudaMemcpyHostToDevice,
                        s);
        // hostMaps is a temporary buffer; ensure the async copy completes before it goes out of scope.
        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));

        // LMKCDEY EvalAcc scratch (reused per call).
        lmk_digitsG_auto_  = digitsG_auto;
        lmk_digitsG2_eval_ = digitsG2_eval;
        lmk_N_             = N;
        d_lmk_ct_        = phantom::util::make_cuda_auto_ptr<NativeInt>(static_cast<size_t>(2) * N, s);
        d_lmk_dct_       = phantom::util::make_cuda_auto_ptr<NativeInt>(static_cast<size_t>(digitsG2_eval) * N, s);
        d_lmk_perm_c0_   = phantom::util::make_cuda_auto_ptr<NativeInt>(N, s);
        d_lmk_perm_c1_   = phantom::util::make_cuda_auto_ptr<NativeInt>(N, s);
        d_lmk_dcta_      = phantom::util::make_cuda_auto_ptr<NativeInt>(static_cast<size_t>(digitsG_auto) * N, s);
        d_lmk_ks_c0_     = phantom::util::make_cuda_auto_ptr<NativeInt>(N, s);
        d_lmk_ks_c1_     = phantom::util::make_cuda_auto_ptr<NativeInt>(N, s);
    }
    else {
        std::ostringstream oss;
        oss << "GPUCirBTSContext: unsupported method=" << method;
        OPENFHE_THROW(config_error, oss.str());
    }

    // HTkey -> GPU (HomTraceKey)   
    constexpr size_t HTkey_dim1          = 1;
    constexpr size_t HTkey_dim2          = 1;
    const size_t HTkey_LWE_n             = static_cast<size_t>(log2(RLWE_params->GetN())) - RLWE_params->GetTraceShift();
    const size_t HTkey_digitsG2          = RLWE_params->GetDigitsHTA();
    constexpr size_t HTkey_RGSW_col_size = 2;
    const size_t HTkey_RGSW_N            = RLWE_params->GetN();
    const auto& h_HTKey                  = cc.GetHomTraceKey()->GetElements();

    if (h_HTKey.size() != HTkey_dim1 || h_HTKey[0].size() != HTkey_dim2 || h_HTKey[0][0].size() != HTkey_LWE_n ||
        !h_HTKey[0][0][0] || h_HTKey[0][0][0]->GetElements().size() != HTkey_digitsG2 ||
        h_HTKey[0][0][0]->GetElements()[0].size() != HTkey_RGSW_col_size ||
        h_HTKey[0][0][0]->GetElements()[0][0].GetValues().GetLength() != HTkey_RGSW_N) {
        OPENFHE_THROW(config_error, "GPUCirBTSContext HTkey has unexpected shape");
    }

    size_HTkey_     = HTkey_dim1 * HTkey_dim2 * HTkey_LWE_n;
    size_HTRGSWkey_ = HTkey_digitsG2 * HTkey_RGSW_col_size * HTkey_RGSW_N;
    d_HTkey_        = phantom::util::make_cuda_auto_ptr<NativeInt>(size_HTkey_ * size_HTRGSWkey_, s);
    d_HTkey_shoup_  = phantom::util::make_cuda_auto_ptr<NativeInt>(size_HTkey_ * size_HTRGSWkey_, s);

    for (size_t i = 0; i < HTkey_dim1; ++i)
    {
        for (size_t j = 0; j < HTkey_dim2; ++j)
        {
            for (size_t k = 0; k < HTkey_LWE_n; ++k)
            {
                std::vector<NativeInt> shoup_key(size_HTRGSWkey_);
                for (size_t l = 0; l < HTkey_digitsG2; ++l)
                {
                    for (size_t m = 0; m < HTkey_RGSW_col_size; ++m)
                    {

                        const size_t base = i * HTkey_dim2 * HTkey_LWE_n * size_HTRGSWkey_ +
                                            j * HTkey_LWE_n * size_HTRGSWkey_ + k * size_HTRGSWkey_ +
                                            l * HTkey_RGSW_col_size * HTkey_RGSW_N + m * HTkey_RGSW_N;
                        const auto& poly = h_HTKey[i][j][k]->GetElements()[l][m].GetValues();
                        const auto* coeffs = &poly.at(0);
                        for (size_t coeff = 0; coeff < HTkey_RGSW_N; ++coeff) {
                            shoup_key[l * HTkey_RGSW_col_size * HTkey_RGSW_N + m * HTkey_RGSW_N + coeff] =
                                static_cast<NativeInt>(
                                    phantom::arith::compute_shoup(
                                        static_cast<uint64_t>(coeffs[coeff].ConvertToInt<NativeInt>()),
                                                                  static_cast<uint64_t>(key_modulus)));
                        }

                        cudaMemcpyAsync(d_HTkey_.get() + base, coeffs, sizeof(NativeInt) * HTkey_RGSW_N, cudaMemcpyHostToDevice, s);
                    }
                }
                cudaMemcpyAsync(d_HTkey_shoup_.get() + i * HTkey_dim2 * HTkey_LWE_n * size_HTRGSWkey_ +
                                        j * HTkey_LWE_n * size_HTRGSWkey_ + k * size_HTRGSWkey_,
                                shoup_key.data(), shoup_key.size() * sizeof(NativeInt), cudaMemcpyHostToDevice, s);
                PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
            }
        }
    }

    if (build_htss_soa) {
        if ((HTkey_RGSW_N % 32u) != 0) {
            std::cerr << "[CIRBTS] HT/SS SoA disabled (requires N divisible by 32)." << std::endl;
        } else {
            htkey_soa_tiles_ = static_cast<size_t>(HTkey_RGSW_N / 32u);
            const size_t total = size_HTkey_ * size_HTRGSWkey_;
            d_HTkey_soa_ = phantom::util::make_cuda_auto_ptr<NativeInt>(total, s);
            d_HTkey_shoup_soa_ = phantom::util::make_cuda_auto_ptr<NativeInt>(total, s);

            dim3 block(256);
            dim3 grid(static_cast<unsigned int>((total + block.x - 1) / block.x));
            kernel_SwizzleRFKey<<<grid, block, 0, s>>>(d_HTkey_soa_.get(), d_HTkey_.get(), HTkey_RGSW_N,
                                                       static_cast<uint32_t>(HTkey_digitsG2),
                                                       static_cast<uint32_t>(htkey_soa_tiles_), size_HTkey_);
            kernel_SwizzleRFKey<<<grid, block, 0, s>>>(d_HTkey_shoup_soa_.get(), d_HTkey_shoup_.get(), HTkey_RGSW_N,
                                                       static_cast<uint32_t>(HTkey_digitsG2),
                                                       static_cast<uint32_t>(htkey_soa_tiles_), size_HTkey_);
            PHANTOM_CHECK_CUDA(cudaPeekAtLastError());
        }
    }


    // NTT conversion moved to gpu_init_ntt once twiddle tables are ready
    // SSkey -> GPU ()
    const size_t SSkey_rowSize = h_elements.size();
    const size_t SSkey_colSize = h_elements[0].size();
    const size_t N             = h_elements[0][0].GetValues().GetLength();
    size_SSkey_                = SSkey_rowSize * SSkey_colSize;
    d_SSkey_                   = phantom::util::make_cuda_auto_ptr<NativeInt>(N * size_SSkey_, s);
    d_SSkey_shoup_             = phantom::util::make_cuda_auto_ptr<NativeInt>(N * size_SSkey_, s);

    std::vector<NativeInt> shoup_row(static_cast<size_t>(SSkey_colSize) * N);
    for (size_t i = 0; i < SSkey_rowSize; ++i) {
        for (size_t j = 0; j < SSkey_colSize; ++j) {
            const auto& poly = h_elements[i][j].GetValues();
            const auto* coeffs = &poly.at(0);
            const size_t row_offset = j * N;
            for (size_t coeff = 0; coeff < N; ++coeff) {
                shoup_row[row_offset + coeff] = static_cast<NativeInt>(
                    phantom::arith::compute_shoup(
                        static_cast<uint64_t>(coeffs[coeff].ConvertToInt<NativeInt>()),
                                                  static_cast<uint64_t>(key_modulus)));
            }
            cudaMemcpyAsync(d_SSkey_.get() + i * SSkey_colSize * N + j * N, coeffs, sizeof(NativeInt) * N,
                            cudaMemcpyHostToDevice, s);
        }
        cudaMemcpyAsync(d_SSkey_shoup_.get() + i * SSkey_colSize * N, shoup_row.data(),
                        shoup_row.size() * sizeof(NativeInt), cudaMemcpyHostToDevice, s);
        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
    }

    if (build_htss_soa) {
        if ((N % 32u) != 0) {
            std::cerr << "[CIRBTS] HT/SS SoA disabled (requires N divisible by 32)." << std::endl;
        } else {
            const uint32_t digitsSS = RLWE_params->GetDigitsSSA();
            if (size_SSkey_ != static_cast<size_t>(digitsSS) * 2u) {
                std::cerr << "[CIRBTS] HT/SS SoA disabled (unexpected SS key shape)." << std::endl;
            } else {
            sskey_soa_tiles_ = static_cast<size_t>(N / 32u);
            const size_t total = size_SSkey_ * N;
            d_SSkey_soa_ = phantom::util::make_cuda_auto_ptr<NativeInt>(total, s);
            d_SSkey_shoup_soa_ = phantom::util::make_cuda_auto_ptr<NativeInt>(total, s);

            dim3 block(256);
            dim3 grid(static_cast<unsigned int>((total + block.x - 1) / block.x));
            kernel_SwizzleRFKey<<<grid, block, 0, s>>>(d_SSkey_soa_.get(), d_SSkey_.get(), N, digitsSS,
                                                       static_cast<uint32_t>(sskey_soa_tiles_), /*lwe_n=*/1);
            kernel_SwizzleRFKey<<<grid, block, 0, s>>>(d_SSkey_shoup_soa_.get(), d_SSkey_shoup_.get(), N, digitsSS,
                                                       static_cast<uint32_t>(sskey_soa_tiles_), /*lwe_n=*/1);
            PHANTOM_CHECK_CUDA(cudaPeekAtLastError());
            }
        }
    }
}

void GPUCirBTSContext::gpu_init_ntt(CirBTSContext& cc) {
    const auto& params       = cc.GetParams();
    const auto& rgsw_params1 = params->GetRingGSWParams1();

    ring_dim_     = rgsw_params1->GetN();
    log_ring_dim_ = static_cast<uint32_t>(phantom::arith::get_power_of_two(ring_dim_));
    num_luts_     = params->GetDigitsCC();
    method_       = rgsw_params1->GetMethod();

    const auto& rlwe_params = params->GetRLWEParams();
    const uint32_t digitsHT = rlwe_params->GetDigitsHTA();
    const uint32_t digitsSS = rlwe_params->GetDigitsSSA();
    digits_ht_             = digitsHT;
    digits_ss_             = digitsSS;
    lwe_n_                 = params->GetLWEParams()->Getn();
    digits_ga_             = (method_ == BINFHE_METHOD::GINX) ? rgsw_params1->GetDigitsGA() : 0u;

    const size_t batch      = std::max<uint32_t>(1u, max_batch_size_);
    const size_t total_luts = static_cast<size_t>(num_luts_) * batch;

    size_t maxBatch = 2 * batch;  // at least [batch][c0||c1]
    maxBatch        = std::max(maxBatch, total_luts);
    maxBatch        = std::max(maxBatch, total_luts * digitsHT);
    maxBatch        = std::max(maxBatch, total_luts * digitsSS);

    const auto method = rgsw_params1->GetMethod();
    if (method == BINFHE_METHOD::GINX) {
        maxBatch = std::max(maxBatch, (static_cast<size_t>(rgsw_params1->GetDigitsGA()) << 1) * batch);
    }
    else if (method == BINFHE_METHOD::LMKCDEY) {
        const uint32_t digitsG = rgsw_params1->GetDigitsG();
        const uint32_t digitsG_auto = (digitsG > 0) ? (digitsG - 1) : 0;
        maxBatch = std::max(maxBatch, (static_cast<size_t>(digitsG_auto) << 1) * batch);
        maxBatch = std::max(maxBatch, static_cast<size_t>(digitsG_auto) * batch);
    }

    ntt_batch_size_ = maxBatch;

    auto& s              = stream_wrapper_.get_stream();
    const NativeInt Q    = rgsw_params1->GetQ().ConvertToInt();
    const auto h_modulus = phantom::arith::Modulus(Q);

    // Build a single-modulus 1D NTT table shared across batched polynomials.
    d_modulus_ = phantom::util::make_cuda_auto_ptr<DModulus>(1, s);
    const DModulus host_modulus(h_modulus.value(), h_modulus.const_ratio()[0], h_modulus.const_ratio()[1]);
    cudaMemcpyAsync(d_modulus_.get(), &host_modulus, sizeof(DModulus), cudaMemcpyHostToDevice, s);

    const size_t N = ring_dim_;
    d_twiddles_          = phantom::util::make_cuda_auto_ptr<uint64_t>(N, s);
    d_twiddles_shoup_    = phantom::util::make_cuda_auto_ptr<uint64_t>(N, s);
    d_itwiddles_         = phantom::util::make_cuda_auto_ptr<uint64_t>(N, s);
    d_itwiddles_shoup_   = phantom::util::make_cuda_auto_ptr<uint64_t>(N, s);
    d_n_inv_mod_q_       = phantom::util::make_cuda_auto_ptr<uint64_t>(1, s);
    d_n_inv_mod_q_shoup_ = phantom::util::make_cuda_auto_ptr<uint64_t>(1, s);
    d_monomial_inv_      = phantom::util::make_cuda_auto_ptr<uint64_t>(N * batch, s);
    d_bootstrap_acc_     = phantom::util::make_cuda_auto_ptr<uint64_t>(2 * N * batch, s);

    d_lut_ = phantom::util::make_cuda_auto_ptr<NativeInt>(N, s);
    cudaMemcpyAsync(d_lut_.get(), &params->GetLUT().GetValues()[0], sizeof(NativeInt) * N, cudaMemcpyHostToDevice, s);

    // CGGI (GINX) EvalAcc scratch buffers.
    d_evalacc_ct_       = {};
    d_evalacc_dct_      = {};
    d_evalacc_indexPos_ = {};
    if (method == BINFHE_METHOD::GINX) {
        const size_t lwe_n      = params->GetLWEParams()->Getn();
        const uint32_t digitsGA = rgsw_params1->GetDigitsGA();
        const uint32_t digitsG2 = digitsGA << 1;
        d_evalacc_ct_           = phantom::util::make_cuda_auto_ptr<NativeInt>(2 * N * batch, s);
        d_evalacc_dct_          = phantom::util::make_cuda_auto_ptr<NativeInt>(static_cast<size_t>(digitsG2) * N * batch, s);
        d_evalacc_indexPos_     = phantom::util::make_cuda_auto_ptr<uint32_t>(lwe_n * batch, s);
        d_evalacc_b_idx_        = phantom::util::make_cuda_auto_ptr<uint32_t>(batch, s);
        d_evalacc_digits_s_     = phantom::util::make_cuda_auto_ptr<int64_t>(static_cast<size_t>(digitsG2) * N, s);
        d_evalacc_residual_s_   = phantom::util::make_cuda_auto_ptr<int64_t>(2 * N, s);
        d_evalacc_delta_        = phantom::util::make_cuda_auto_ptr<NativeInt>(2 * N, s);
        d_evalacc_carry_flag_   = phantom::util::make_cuda_auto_ptr<uint32_t>(1, s);
        d_evalacc_acc_backup_   = phantom::util::make_cuda_auto_ptr<NativeInt>(2 * N, s);
    }

    const auto h_ntt_table = phantom::arith::NTT(static_cast<int>(log_ring_dim_), h_modulus);

    cudaMemcpyAsync(d_twiddles_.get(), h_ntt_table.get_from_root_powers().data(), N * sizeof(NativeInt), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_twiddles_shoup_.get(), h_ntt_table.get_from_root_powers_shoup().data(), N * sizeof(NativeInt),
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_itwiddles_.get(), h_ntt_table.get_from_inv_root_powers().data(), N * sizeof(NativeInt),
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_itwiddles_shoup_.get(), h_ntt_table.get_from_inv_root_powers_shoup().data(), N * sizeof(NativeInt),
                    cudaMemcpyHostToDevice, s);

    const NativeInt host_n_inv_mod_q = h_ntt_table.inv_degree_modulo();
    const NativeInt host_n_inv_mod_q_shoup = h_ntt_table.inv_degree_modulo_shoup();
    cudaMemcpyAsync(d_n_inv_mod_q_.get(), &host_n_inv_mod_q, sizeof(NativeInt), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_n_inv_mod_q_shoup_.get(), &host_n_inv_mod_q_shoup, sizeof(NativeInt), cudaMemcpyHostToDevice, s);

    // Cache AG powers used in ModAddGpowScaled.
    d_gpow_ = phantom::util::make_cuda_auto_ptr<NativeInt>(num_luts_, s);
    cudaMemcpyAsync(d_gpow_.get(), params->GetRingGSWParams2()->GetAGPower().data(), sizeof(NativeInt) * num_luts_,
                    cudaMemcpyHostToDevice, s);

    // Cache monomials used in MV-RLWE generation: X^{-i} for i in [1..numLUT-1].
    d_monomials_ = {};
    std::vector<NativeInt> host_monomials;
    if (num_luts_ > 1) {
        const uint32_t traceShift = rlwe_params->GetTraceShift();
        const uint32_t step = (traceShift == 0) ? 1u : (1u << traceShift);
        host_monomials.resize(static_cast<size_t>(num_luts_ - 1) * N);
        for (uint32_t i = 1; i < num_luts_; ++i) {
            const auto mono = params->GetMonomial(i * step);
            std::memcpy(host_monomials.data() + static_cast<size_t>(i - 1) * N, &mono.GetValues().at(0), N * sizeof(NativeInt));
        }
        d_monomials_ = phantom::util::make_cuda_auto_ptr<NativeInt>(host_monomials.size(), s);
        cudaMemcpyAsync(d_monomials_.get(), host_monomials.data(), host_monomials.size() * sizeof(NativeInt), cudaMemcpyHostToDevice, s);
    }

    // Cache automorphism maps for homtrace (coefficient-domain), indexed by k in [0..logN-1].
    d_auto_maps_ = phantom::util::make_cuda_auto_ptr<uint32_t>(static_cast<size_t>(log_ring_dim_) * N, s);
    {
        const uint32_t m_mask = static_cast<uint32_t>((N << 1) - 1);
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x, log_ring_dim_);
        kernel_GenerateCoefficientAutoMaps<<<grid, block, 0, s>>>(d_auto_maps_.get(), static_cast<uint32_t>(N), log_ring_dim_, m_mask,
                                                                  log_ring_dim_);
    }

    // Cache monic polys for CGGI EvalAcc (GINX):
    // - OpenFHE monomials: (X^i - 1) for i in [0..N-1] and (-X^i - 1) for i in [N..2N-1].
    // - Store the full [2N][N] table to eliminate branchy reconstruction in EvalAcc.
    d_monic_polys_ = {};
    d_monomial_inv_table_ = {};
    if (method == BINFHE_METHOD::GINX) {
        const size_t monic_rows = static_cast<size_t>(2) * N;
        std::vector<NativeInt> host_monic(monic_rows * N);
        for (size_t i = 0; i < N; ++i) {
            const auto& mono = rgsw_params1->GetMonomial(i).GetValues();
            std::memcpy(host_monic.data() + i * N, &mono.at(0), N * sizeof(NativeInt));
        }
        const NativeInt q_minus_two = Q - 2;
        for (size_t i = 0; i < N; ++i) {
            const size_t src_base = i * N;
            const size_t dst_base = (i + N) * N;
            for (size_t coeff = 0; coeff < N; ++coeff) {
                const NativeInt v = host_monic[src_base + coeff];
                host_monic[dst_base + coeff] = (q_minus_two >= v) ? (q_minus_two - v) : (q_minus_two + Q - v);
            }
        }
        d_monic_polys_ = phantom::util::make_cuda_auto_ptr<NativeInt>(host_monic.size(), s);
        cudaMemcpyAsync(d_monic_polys_.get(), host_monic.data(), host_monic.size() * sizeof(NativeInt), cudaMemcpyHostToDevice, s);

        std::vector<NativeInt> host_monomial_inv(monic_rows * N);
        for (size_t i = 0; i < monic_rows; ++i) {
            const auto& mono = params->GetMonomial(static_cast<uint32_t>(i)).GetValues();
            std::memcpy(host_monomial_inv.data() + i * N, &mono.at(0), N * sizeof(NativeInt));
        }
        d_monomial_inv_table_ = phantom::util::make_cuda_auto_ptr<NativeInt>(host_monomial_inv.size(), s);
        cudaMemcpyAsync(d_monomial_inv_table_.get(), host_monomial_inv.data(),
                        host_monomial_inv.size() * sizeof(NativeInt), cudaMemcpyHostToDevice, s);
    }

    if (method == BINFHE_METHOD::GINX && use_ntt32_rns_) {
        if (!d_RFkey_.get()) {
            OPENFHE_THROW(config_error, "ntt32_rns backend requires RF key to be initialized on GPU");
        }
        ntt32_rns_plan_ = std::make_unique<phantom::cirbts::experimental::NTT32RNSPlan>(
            N, phantom::cirbts::experimental::NTT32RNSPlan::default_moduli(), s);
        ntt32_rns_limbs_ = static_cast<uint32_t>(ntt32_rns_plan_->limb_count());

        const bool verbose = (std::getenv("CIRBTS_NTT32_RNS_VERBOSE") != nullptr);
        auto build_rns32_eval_layout = [&](phantom::util::cuda_auto_ptr<uint32_t>& out,
                                           phantom::util::cuda_auto_ptr<uint32_t>& out_shoup,
                                           size_t& out_polys,
                                           const NativeInt* in_eval_q,
                                           size_t elems,
                                           const char* label) {
            if (!in_eval_q || elems == 0) {
                out = {};
                out_shoup = {};
                out_polys = 0;
                return;
            }
            if ((elems % N) != 0) {
                std::ostringstream oss;
                oss << "ntt32_rns layout conversion for " << label << " received non-polynomial-aligned input";
                OPENFHE_THROW(config_error, oss.str());
            }
            out_polys = elems / N;
            auto coeff_q = phantom::util::make_cuda_auto_ptr<NativeInt>(elems, s);
            cudaMemcpyAsync(coeff_q.get(), in_eval_q, elems * sizeof(NativeInt), cudaMemcpyDeviceToDevice, s);
            inwt_1d_opt_batched(coeff_q.get(), d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(),
                                d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N, out_polys, s);

            const size_t rns_elems = static_cast<size_t>(ntt32_rns_limbs_) * elems;
            out = phantom::util::make_cuda_auto_ptr<uint32_t>(rns_elems, s);
            out_shoup = phantom::util::make_cuda_auto_ptr<uint32_t>(rns_elems, s);
            ntt32_rns_plan_->native_coeffs_to_rns_eval(out.get(), out_shoup.get(), coeff_q.get(), out_polys, s);

            if (verbose) {
                const double mib = static_cast<double>(rns_elems * sizeof(uint32_t)) / (1024.0 * 1024.0);
                std::cout << "[CIRBTS][ntt32_rns] " << label << " layout: polys=" << out_polys
                          << " limbs=" << ntt32_rns_limbs_ << " value=" << mib
                          << " MiB shoup=" << mib << " MiB" << std::endl;
            }
        };

        build_rns32_eval_layout(d_RFkey_rns32_, d_RFkey_rns32_shoup_, rfkey_rns32_polys_,
                                d_RFkey_.get(), size_RFkey_ * size_RFRGSWkey_, "RFKey");
        build_rns32_eval_layout(d_HTkey_rns32_, d_HTkey_rns32_shoup_, htkey_rns32_polys_,
                                d_HTkey_.get(), size_HTkey_ * size_HTRGSWkey_, "HTKey");
        build_rns32_eval_layout(d_SSkey_rns32_, d_SSkey_rns32_shoup_, sskey_rns32_polys_,
                                d_SSkey_.get(), size_SSkey_ * N, "SSKey");
        build_rns32_eval_layout(d_monic_polys_rns32_, d_monic_polys_rns32_shoup_, monic_rns32_polys_,
                                d_monic_polys_.get(), static_cast<size_t>(2) * N * N, "monic");

        std::cout << "[CIRBTS] backend=ntt32_rns initialized experimental RNS32 layouts; "
                     "CBS execution still uses the production 64-bit kernels until RNS external-product kernels are enabled."
                  << std::endl;
    }

    if (method == BINFHE_METHOD::GINX && use_split_fft_) {
#if defined(PHANTOM_ENABLE_CUFFTDX)
        if (!d_RFkey_.get()) {
            OPENFHE_THROW(config_error, "split FFT requires RF key to be initialized on GPU");
        }

        const uint32_t digitsGA = digits_ga_;
        const uint32_t digitsG2 = digitsGA << 1;
        if (digitsG2 == 0) {
            OPENFHE_THROW(config_error, "split FFT requires digitsG2 > 0");
        }

        const uint32_t baseG = rgsw_params1->GetBaseG();
        const uint32_t gBits = (baseG > 0) ? static_cast<uint32_t>(__builtin_ctz(baseG)) : 0u;
        const uint32_t logN = log_ring_dim_;
        const uint32_t logDigits = (digitsG2 <= 1) ? 0u : static_cast<uint32_t>(32 - __builtin_clz(digitsG2 - 1));
        const uint32_t logQ = static_cast<uint32_t>(64 - __builtin_clzll(static_cast<unsigned long long>(Q)));
        constexpr uint32_t safeBits = 50;

        uint32_t limb_bits = 0;
        if (const char* v = std::getenv("CIRBTS_SPLIT_FFT_BITS"); v && *v) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(v, &end, 10);
            if (end != v && end && *end == '\0') {
                limb_bits = static_cast<uint32_t>(parsed);
            }
        }
        if (limb_bits == 0) {
            const int32_t avail = static_cast<int32_t>(safeBits) - static_cast<int32_t>(gBits + logN + logDigits);
            limb_bits = (avail > 8) ? static_cast<uint32_t>(avail) : 8u;
        }
        if (limb_bits == 0 || limb_bits >= 64u) {
            OPENFHE_THROW(config_error, "CIRBTS_SPLIT_FFT_BITS must be in [1, 63]");
        }

        split_fft_bits_      = limb_bits;
        split_fft_limbs_     = static_cast<uint32_t>((logQ + limb_bits - 1) / limb_bits);
        split_fft_fft_len_   = N / 2;
        split_fft_key_polys_ = size_RFkey_ * digitsG2 * 2;
        split_fft_key_stride_ = digitsG2 * 2 * split_fft_fft_len_;

        split_fft_base_pows_.clear();
        split_fft_base_pows_shoup_.clear();
        split_fft_base_pows_.reserve(split_fft_limbs_);
        split_fft_base_pows_shoup_.reserve(split_fft_limbs_);
        {
            const NativeInt q_native = rgsw_params1->GetQ().ConvertToInt();
            NativeInteger base = NativeInteger(1) << limb_bits;
            NativeInteger pow  = 1;
            for (uint32_t i = 0; i < split_fft_limbs_; ++i) {
                const NativeInt pow_int = pow.ConvertToInt();
                split_fft_base_pows_.push_back(pow_int);
                split_fft_base_pows_shoup_.push_back(phantom::arith::compute_shoup(static_cast<uint64_t>(pow_int),
                                                                                  static_cast<uint64_t>(q_native)));
                pow = pow.ModMulFast(base, rgsw_params1->GetQ());
            }
        }

        if (N != 2048) {
            OPENFHE_THROW(config_error, "split FFT currently supports N=2048 only");
        }
        {
            uint32_t fpb = 1;
            if (const char* v = std::getenv("CIRBTS_FFT_FPB"); v && *v) {
                char* end = nullptr;
                const unsigned long parsed = std::strtoul(v, &end, 10);
                if (end != v && end && *end == '\0') {
                    fpb = static_cast<uint32_t>(parsed);
                }
            }
            if (fpb != 1 && fpb != 2 && fpb != 4) {
                OPENFHE_THROW(config_error, "CIRBTS_FFT_FPB must be 1, 2, or 4");
            }

            auto supports_fpb = [](uint32_t cand_fpb) -> bool {
                cudaDeviceProp prop{};
                int dev = 0;
                if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
                    return false;
                }

                auto ok = [&](size_t shared_bytes, dim3 block_dim) -> bool {
                    const uint32_t threads = block_dim.x * block_dim.y * block_dim.z;
                    if (threads == 0 || threads > static_cast<uint32_t>(prop.maxThreadsPerBlock)) {
                        return false;
                    }
                    if (shared_bytes > static_cast<size_t>(prop.sharedMemPerBlock)) {
                        return false;
                    }
                    return true;
                };

                if (cand_fpb == 4) {
                    using FFT = phantom::bitwise::cuFFTDxWrapperCirBTS<double, 2048, 4, 16, PHANTOM_CUFFTDX_SM>;
                    const size_t fwd_shared = FFT::Forward_FFT::shared_memory_size;
                    const size_t inv_shared = FFT::Inverse_FFT::shared_memory_size;
                    return ok(std::max(fwd_shared, inv_shared), FFT::Forward_FFT::block_dim);
                } else if (cand_fpb == 2) {
                    using FFT = phantom::bitwise::cuFFTDxWrapperCirBTS<double, 2048, 2, 16, PHANTOM_CUFFTDX_SM>;
                    const size_t fwd_shared = FFT::Forward_FFT::shared_memory_size;
                    const size_t inv_shared = FFT::Inverse_FFT::shared_memory_size;
                    return ok(std::max(fwd_shared, inv_shared), FFT::Forward_FFT::block_dim);
                } else {
                    using FFT = phantom::bitwise::cuFFTDxWrapperCirBTS<double, 2048, 1, 16, PHANTOM_CUFFTDX_SM>;
                    const size_t fwd_shared = FFT::Forward_FFT::shared_memory_size;
                    const size_t inv_shared = FFT::Inverse_FFT::shared_memory_size;
                    return ok(std::max(fwd_shared, inv_shared), FFT::Forward_FFT::block_dim);
                }
            };

            uint32_t chosen_fpb = fpb;
            if (!supports_fpb(chosen_fpb)) {
                const uint32_t fallback = (chosen_fpb == 4) ? 2 : 1;
                if (supports_fpb(fallback)) {
                    chosen_fpb = fallback;
                } else {
                    chosen_fpb = 1;
                }
                std::cerr << "[CIRBTS] CIRBTS_FFT_FPB=" << fpb
                          << " is not supported on this GPU; using FPB=" << chosen_fpb << " instead."
                          << std::endl;
            }

            split_fft_fpb_ = chosen_fpb;
            if (chosen_fpb == 1) {
                fft_2048_fpb1_ = std::make_shared<phantom::bitwise::cuFFTDxWrapperCirBTS<double, 2048, 1, 16, PHANTOM_CUFFTDX_SM>>();
            } else if (chosen_fpb == 2) {
                fft_2048_fpb2_ = std::make_shared<phantom::bitwise::cuFFTDxWrapperCirBTS<double, 2048, 2, 16, PHANTOM_CUFFTDX_SM>>();
            } else {
                fft_2048_fpb4_ = std::make_shared<phantom::bitwise::cuFFTDxWrapperCirBTS<double, 2048, 4, 16, PHANTOM_CUFFTDX_SM>>();
            }
        }

        const size_t dct_elems = static_cast<size_t>(digitsG2) * N * batch;
        d_evalacc_dct_i64_ = phantom::util::make_cuda_auto_ptr<int64_t>(dct_elems, s);
        d_evalacc_dct_fft_ = phantom::util::make_cuda_auto_ptr<SplitFFTComplex>(
            static_cast<size_t>(digitsG2) * split_fft_fft_len_ * batch, s);
        d_evalacc_acc_fft_ = phantom::util::make_cuda_auto_ptr<SplitFFTComplex>(
            static_cast<size_t>(2) * split_fft_fft_len_, s);

        const size_t monic_rows = static_cast<size_t>(2) * N;
        d_monic_fft_ = phantom::util::make_cuda_auto_ptr<SplitFFTComplex>(monic_rows * split_fft_fft_len_, s);
        auto d_monic_i64 = phantom::util::make_cuda_auto_ptr<int64_t>(monic_rows * N, s);
        {
            std::vector<int64_t> host_monic_i64(monic_rows * N, 0);
            for (size_t i = 0; i < N; ++i) {
                const size_t row = i * N;
                host_monic_i64[row] -= 1;
                host_monic_i64[row + i] += 1;
            }
            for (size_t i = 0; i < N; ++i) {
                const size_t row = (static_cast<size_t>(N) + i) * N;
                host_monic_i64[row] -= 1;
                host_monic_i64[row + i] -= 1;
            }
            cudaMemcpyAsync(d_monic_i64.get(), host_monic_i64.data(), host_monic_i64.size() * sizeof(int64_t),
                            cudaMemcpyHostToDevice, s);
        }
        auto fft_i2c_forward = [&](SplitFFTComplex* out, const int64_t* in, size_t batch_size) {
            if (split_fft_fpb_ == 1) {
                fft_2048_fpb1_->i2c_forward(out, in, batch_size, s);
            } else if (split_fft_fpb_ == 2) {
                fft_2048_fpb2_->i2c_forward(out, in, batch_size, s);
            } else {
                fft_2048_fpb4_->i2c_forward(out, in, batch_size, s);
            }
        };
        fft_i2c_forward(d_monic_fft_.get(), d_monic_i64.get(), monic_rows);

        const size_t key_coeffs = size_RFkey_ * size_RFRGSWkey_;
        const size_t key_polys  = split_fft_key_polys_;
        auto d_RFkey_coeff = phantom::util::make_cuda_auto_ptr<NativeInt>(key_coeffs, s);
        cudaMemcpyAsync(d_RFkey_coeff.get(), d_RFkey_.get(), key_coeffs * sizeof(NativeInt),
                        cudaMemcpyDeviceToDevice, s);

        inwt_1d_opt_batched(d_RFkey_coeff.get(), d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(),
                            d_n_inv_mod_q_.get(), d_n_inv_mod_q_shoup_.get(), N, key_polys, s);

        auto d_key_limb = phantom::util::make_cuda_auto_ptr<int64_t>(key_coeffs, s);
        d_RFkey_fft_limbs_.clear();
        d_RFkey_fft_limbs_.reserve(split_fft_limbs_);

        dim3 block(256);
        dim3 grid(static_cast<unsigned int>((key_coeffs + block.x - 1) / block.x));

        for (uint32_t limb = 0; limb < split_fft_limbs_; ++limb) {
            auto limb_fft = phantom::util::make_cuda_auto_ptr<SplitFFTComplex>(key_polys * split_fft_fft_len_, s);
            d_RFkey_fft_limbs_.push_back(std::move(limb_fft));

            kernel_ExtractKeyLimb<<<grid, block, 0, s>>>(
                d_key_limb.get(), d_RFkey_coeff.get(), key_coeffs, split_fft_bits_, limb);
            fft_i2c_forward(d_RFkey_fft_limbs_[limb].get(), d_key_limb.get(), key_polys);
        }
#else
        OPENFHE_THROW(config_error, "split FFT requested but PHANTOM_ENABLE_CUFFTDX is disabled at build time");
#endif
    }

    // HT keys are generated in evaluation (NTT) form already on CPU; no extra NTT needed here.

    // Scratch buffers for circuit bootstrapping (reused across calls).
    scratch_lut_size_   = static_cast<size_t>(num_luts_) * N * batch;
    scratch_digits_max_ = std::max(digitsHT, digitsSS);

    d_ht_c0_      = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size_, s);
    d_ht_c1_      = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size_, s);
    d_perm_c0_    = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size_, s);
    d_ht_next_c0_ = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size_, s);
    d_ht_next_c1_ = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size_, s);
    d_ks_c0_      = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size_, s);
    d_ks_c1_      = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size_, s);
    d_digits_     = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size_ * scratch_digits_max_, s);
    d_ss_c0_      = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size_, s);
    d_ss_c1_      = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size_, s);
    d_rgsw_out_   = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size_ * 4, s);

    // Ensure temporary host buffers remain alive until all async copies complete.
    cudaStreamSynchronize(s);
}

GPUCirBTSContext::BatchScratchView GPUCirBTSContext::make_batch_scratch_view() const {
    BatchScratchView view{};
    view.stream                   = stream_wrapper_.get_stream();
    view.copy_stream              = copy_stream_wrapper_.get_stream();
    view.d_monomial_inv           = d_monomial_inv_.get();
    view.d_bootstrap_acc          = d_bootstrap_acc_.get();
    view.d_evalacc_ct             = d_evalacc_ct_.get();
    view.d_evalacc_dct            = d_evalacc_dct_.get();
    view.d_evalacc_indexPos       = d_evalacc_indexPos_.get();
    view.d_evalacc_b_idx          = d_evalacc_b_idx_.get();
    view.h_monomial_inv_pinned    = h_monomial_inv_pinned_;
    view.h_indexPos_pinned        = h_indexPos_pinned_;
    view.h2d_monomial_event       = &h2d_monomial_event_;
    view.h2d_index_event          = &h2d_index_event_;
    view.d_ht_c0                  = d_ht_c0_.get();
    view.d_ht_c1                  = d_ht_c1_.get();
    view.d_perm_c0                = d_perm_c0_.get();
    view.d_ht_next_c0             = d_ht_next_c0_.get();
    view.d_ht_next_c1             = d_ht_next_c1_.get();
    view.d_ks_c0                  = d_ks_c0_.get();
    view.d_ks_c1                  = d_ks_c1_.get();
    view.d_digits                 = d_digits_.get();
    view.d_ss_c0                  = d_ss_c0_.get();
    view.d_ss_c1                  = d_ss_c1_.get();
    view.d_rgsw_out               = d_rgsw_out_.get();
    view.scratch_lut_size         = scratch_lut_size_;
    view.scratch_digits_max       = scratch_digits_max_;
    view.evalacc_batch_graph_exec = &cggi_evalacc_batch_graph_exec_;
    view.evalacc_batch_graph_failed = &cggi_evalacc_batch_graph_failed_;
    view.evalacc_batch_graph_ntt_tpb = &cggi_evalacc_batch_graph_ntt_tpb_;
    view.evalacc_batch_graph_block_x = &cggi_evalacc_batch_graph_block_x_;
    view.evalacc_batch_graph_batch   = &cggi_evalacc_batch_graph_batch_;
    view.evalacc_batch_graph_multibit = &cggi_evalacc_batch_graph_multibit_;
    view.evalacc_batch_graph_soa = &cggi_evalacc_batch_graph_soa_;
    view.evalacc_batch_graph_smem = &cggi_evalacc_batch_graph_smem_;
    view.htss_graph_exec          = &htss_graph_exec_;
    view.htss_graph_failed        = &htss_graph_failed_;
    view.htss_graph_block_x       = &htss_graph_block_x_;
    view.htss_graph_num_luts      = &htss_graph_num_luts_;
    view.htss_graph_fuse_inwt_decomp = &htss_graph_fuse_inwt_decomp_;
    view.htss_graph_fuse_decomp_fnwt = &htss_graph_fuse_decomp_fnwt_;
    view.htss_graph_pipeline      = &htss_graph_pipeline_;
    view.htss_graph_smem          = &htss_graph_smem_;
    return view;
}

GPUCirBTSContext::BatchScratchView GPUCirBTSContext::make_batch_scratch_view(BatchWorkspace& workspace) const {
    BatchScratchView view{};
    view.stream             = workspace.stream.get_stream();
    view.copy_stream        = workspace.copy_stream.get_stream();
    view.d_monomial_inv     = workspace.d_monomial_inv.get();
    view.d_bootstrap_acc    = workspace.d_bootstrap_acc.get();
    view.d_evalacc_ct       = workspace.d_evalacc_ct.get();
    view.d_evalacc_dct      = workspace.d_evalacc_dct.get();
    view.d_evalacc_indexPos = workspace.d_evalacc_indexPos.get();
    view.d_evalacc_b_idx    = workspace.d_evalacc_b_idx.get();
    view.h_monomial_inv_pinned = workspace.h_monomial_inv_pinned;
    view.h_indexPos_pinned     = workspace.h_indexPos_pinned;
    view.h2d_monomial_event    = &workspace.h2d_monomial_event;
    view.h2d_index_event       = &workspace.h2d_index_event;
    view.d_ht_c0            = workspace.d_ht_c0.get();
    view.d_ht_c1            = workspace.d_ht_c1.get();
    view.d_perm_c0          = workspace.d_perm_c0.get();
    view.d_ht_next_c0       = workspace.d_ht_next_c0.get();
    view.d_ht_next_c1       = workspace.d_ht_next_c1.get();
    view.d_ks_c0            = workspace.d_ks_c0.get();
    view.d_ks_c1            = workspace.d_ks_c1.get();
    view.d_digits           = workspace.d_digits.get();
    view.d_ss_c0            = workspace.d_ss_c0.get();
    view.d_ss_c1            = workspace.d_ss_c1.get();
    view.d_rgsw_out         = workspace.d_rgsw_out.get();
    view.scratch_lut_size   = workspace.scratch_lut_size;
    view.scratch_digits_max = workspace.scratch_digits_max;
    // Disable CUDA graphs for extra workspaces to avoid pointer aliasing.
    view.evalacc_batch_graph_exec = nullptr;
    view.evalacc_batch_graph_failed = nullptr;
    view.evalacc_batch_graph_ntt_tpb = nullptr;
    view.evalacc_batch_graph_block_x = nullptr;
    view.evalacc_batch_graph_batch = nullptr;
    view.evalacc_batch_graph_multibit = nullptr;
    view.evalacc_batch_graph_soa = nullptr;
    view.evalacc_batch_graph_smem = nullptr;
    view.htss_graph_exec = nullptr;
    view.htss_graph_failed = nullptr;
    view.htss_graph_block_x = nullptr;
    view.htss_graph_num_luts = nullptr;
    view.htss_graph_fuse_inwt_decomp = nullptr;
    view.htss_graph_fuse_decomp_fnwt = nullptr;
    view.htss_graph_pipeline = nullptr;
    view.htss_graph_smem = nullptr;
    return view;
}

void GPUCirBTSContext::ensure_h2d_events() const {
    if (!h2d_monomial_event_) {
        PHANTOM_CHECK_CUDA(cudaEventCreateWithFlags(&h2d_monomial_event_, cudaEventDisableTiming));
    }
    if (!h2d_index_event_) {
        PHANTOM_CHECK_CUDA(cudaEventCreateWithFlags(&h2d_index_event_, cudaEventDisableTiming));
    }
}

GPUCirBTSContext::NativeInt* GPUCirBTSContext::ensure_pinned_monomial_inv(size_t elems) const {
    if (h_monomial_inv_pinned_ && h_monomial_inv_pinned_capacity_ >= elems) {
        return h_monomial_inv_pinned_;
    }
    if (h_monomial_inv_pinned_) {
        PHANTOM_CHECK_CUDA(cudaFreeHost(h_monomial_inv_pinned_));
        h_monomial_inv_pinned_ = nullptr;
    }
    PHANTOM_CHECK_CUDA(cudaMallocHost(&h_monomial_inv_pinned_, elems * sizeof(NativeInt)));
    h_monomial_inv_pinned_capacity_ = elems;
    return h_monomial_inv_pinned_;
}

uint32_t* GPUCirBTSContext::ensure_pinned_index_pos(size_t elems) const {
    if (h_indexPos_pinned_ && h_indexPos_pinned_capacity_ >= elems) {
        return h_indexPos_pinned_;
    }
    if (h_indexPos_pinned_) {
        PHANTOM_CHECK_CUDA(cudaFreeHost(h_indexPos_pinned_));
        h_indexPos_pinned_ = nullptr;
    }
    PHANTOM_CHECK_CUDA(cudaMallocHost(&h_indexPos_pinned_, elems * sizeof(uint32_t)));
    h_indexPos_pinned_capacity_ = elems;
    return h_indexPos_pinned_;
}

void GPUCirBTSContext::ensure_batch_workspaces(uint32_t count) const {
    if (count == 0) {
        return;
    }
    if (extra_workspaces_.size() >= count) {
        return;
    }
    extra_workspaces_.reserve(count);
    const uint32_t batch = std::max<uint32_t>(1u, max_batch_size_);
    const size_t N       = ring_dim_;
    const uint32_t digitsG2 = (digits_ga_ > 0) ? (digits_ga_ << 1) : 0u;
    const size_t scratch_lut_size = static_cast<size_t>(num_luts_) * N * batch;
    const uint32_t scratch_digits_max = std::max(digits_ht_, digits_ss_);

    for (size_t i = extra_workspaces_.size(); i < count; ++i) {
        auto workspace = std::make_unique<BatchWorkspace>();
        workspace->max_batch_size = batch;
        workspace->scratch_lut_size = scratch_lut_size;
        workspace->scratch_digits_max = scratch_digits_max;
        auto& s = workspace->stream.get_stream();

        workspace->d_monomial_inv  = phantom::util::make_cuda_auto_ptr<NativeInt>(N * batch, s);
        workspace->d_bootstrap_acc = phantom::util::make_cuda_auto_ptr<NativeInt>(2 * N * batch, s);

        if (method_ == BINFHE_METHOD::GINX && digitsG2 > 0) {
            workspace->d_evalacc_ct   = phantom::util::make_cuda_auto_ptr<NativeInt>(2 * N * batch, s);
            workspace->d_evalacc_dct  = phantom::util::make_cuda_auto_ptr<NativeInt>(static_cast<size_t>(digitsG2) * N * batch, s);
            workspace->d_evalacc_indexPos =
                phantom::util::make_cuda_auto_ptr<uint32_t>(static_cast<size_t>(lwe_n_) * batch, s);
            workspace->d_evalacc_b_idx =
                phantom::util::make_cuda_auto_ptr<uint32_t>(batch, s);
        }

        workspace->d_ht_c0      = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size, s);
        workspace->d_ht_c1      = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size, s);
        workspace->d_perm_c0    = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size, s);
        workspace->d_ht_next_c0 = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size, s);
        workspace->d_ht_next_c1 = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size, s);
        workspace->d_ks_c0      = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size, s);
        workspace->d_ks_c1      = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size, s);
        workspace->d_digits     = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size * scratch_digits_max, s);
        workspace->d_ss_c0      = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size, s);
        workspace->d_ss_c1      = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size, s);
        workspace->d_rgsw_out   = phantom::util::make_cuda_auto_ptr<NativeInt>(scratch_lut_size * 4, s);

        extra_workspaces_.push_back(std::move(workspace));
    }
}
