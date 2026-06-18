/*
 * =============================================================================
 * File: cirbts-lmkcdey.cu
 * Purpose: LMKCDEY-specific EvalAcc implementation for the GPU CirBTS path.
 * Key parameters:
 *   - LMKCDEY eval keys and auto maps must be initialized on GPU.
 *   - N, Q, and decomposition parameters from CirBTSCryptoParams.
 * Key points:
 *   - Only used when method=LMKCDEY; separate from the GINX path.
 *   - Assumes LMKCDEY keys/maps are already resident on device.
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
#include <unordered_set>
#include <unordered_map>
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
}  // namespace

void GPUCirBTSContext::LMKCDEY_EvalAcc(const std::shared_ptr<lbcrypto::CirBTSCryptoParams>& params,
                                      const NativeVector& a,
                                      const phantom::util::cuda_auto_ptr<NativeInt>& d_acc) const {
    const auto& rgswParams1 = params->GetRingGSWParams1();
    if (rgswParams1->GetMethod() != BINFHE_METHOD::LMKCDEY) {
        std::ostringstream oss;
        oss << "LMKCDEY_EvalAcc requires method=LMKCDEY; got method=" << rgswParams1->GetMethod();
        OPENFHE_THROW(config_error, oss.str());
    }
    if (!d_LMK_evalkey_.get() || !d_LMK_autokey_.get() || !d_LMK_autoMaps_.get()) {
        OPENFHE_THROW(config_error, "LMKCDEY_EvalAcc: LMKCDEY keys/maps are not initialized on GPU");
    }

    // assume a is all-odd ciphertext (using round-to-odd technique)
    const size_t n       = a.GetLength();
    const uint32_t N     = rgswParams1->GetN();
    const uint32_t Nh    = N / 2;
    const uint32_t M     = 2 * N;
    const uint32_t baseG = rgswParams1->GetBaseG();
    const uint32_t digitsG = rgswParams1->GetDigitsG();
    if (digitsG < 2) {
        OPENFHE_THROW(config_error, "LMKCDEY_EvalAcc: digitsG < 2");
    }
    // OpenFHE LMKCDEY uses an approximate gadget decomposition that ignores the first digit.
    const uint32_t digitsG_auto  = digitsG - 1;
    const uint32_t digitsG2_eval = digitsG_auto << 1;
    const uint32_t numAutoKeys = rgswParams1->GetNumAutoKeys();
    const uint32_t auto_window = get_lmk_auto_window();
    const auto autoPows = build_lmk_auto_pows(N, numAutoKeys, auto_window);
    if (static_cast<uint32_t>(autoPows.size()) != lmk_auto_key_count_) {
        OPENFHE_THROW(config_error, "LMKCDEY_EvalAcc: auto key count mismatch with initialized GPU context");
    }
    if (auto_window != lmk_auto_window_) {
        OPENFHE_THROW(config_error, "LMKCDEY_EvalAcc: auto window mismatch with initialized GPU context");
    }
    if (!d_lmk_ct_.get() || !d_lmk_dct_.get() || !d_lmk_perm_c0_.get() || !d_lmk_perm_c1_.get() ||
        !d_lmk_dcta_.get() || !d_lmk_ks_c0_.get() || !d_lmk_ks_c1_.get()) {
        OPENFHE_THROW(config_error, "LMKCDEY_EvalAcc: scratch buffers are not initialized on GPU");
    }
    if (lmk_digitsG_auto_ != digitsG_auto || lmk_digitsG2_eval_ != digitsG2_eval || lmk_N_ != N) {
        OPENFHE_THROW(config_error, "LMKCDEY_EvalAcc: scratch buffer dimensions mismatch");
    }

    const NativeInt Q = rgswParams1->GetQ().ConvertToInt();
    auto& s = stream_wrapper_.get_stream();

    // Build a dense bucket map (CSR) on the host:
    // bucket 0..Nh-2 : index = -(Nh-1) .. -1
    // bucket Nh-1    : index = 0
    // bucket Nh..2Nh-2: index = 1 .. (Nh-1)
    // bucket 2Nh-1   : index = M  (special bucket for -1)
    const uint32_t bucketCount = 2 * Nh;
    std::vector<uint32_t> bucketIds(n);
    std::vector<uint32_t> counts(bucketCount, 0);

    const auto& logGen = rgswParams1->GetLogGen();
    const uint32_t mask = M - 1;  // M is a power of two
    for (size_t i = 0; i < n; ++i) {
        const uint32_t ai = a[i].ConvertToInt<uint32_t>() & mask;
        const uint32_t aIOdd = ((0u - ai) & mask) | 1u;
        const int32_t index = logGen[aIOdd];

        uint32_t bucket = 0;
        if (index == static_cast<int32_t>(M)) {
            bucket = bucketCount - 1;
        }
        else {
            const int32_t shifted = index + static_cast<int32_t>(Nh - 1);
            if (shifted < 0 || shifted >= static_cast<int32_t>(bucketCount - 1)) {
                OPENFHE_THROW(config_error, "LMKCDEY_EvalAcc: logGen index out of expected range");
            }
            bucket = static_cast<uint32_t>(shifted);
        }

        bucketIds[i] = bucket;
        ++counts[bucket];
    }

    std::vector<uint32_t> offsets(bucketCount + 1, 0);
    for (uint32_t b = 0; b < bucketCount; ++b) {
        offsets[b + 1] = offsets[b] + counts[b];
    }
    std::vector<uint32_t> cursor = offsets;
    std::vector<uint32_t> items(n);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t b = bucketIds[i];
        items[cursor[b]++] = i;
    }

    std::unordered_map<uint32_t, uint32_t> pow_index;
    pow_index.reserve(autoPows.size() * 2);
    for (uint32_t i = 0; i < autoPows.size(); ++i) {
        pow_index.emplace(autoPows[i], i);
    }

    const size_t evalKeyStride = static_cast<size_t>(digitsG2_eval) * 2 * N;
    const size_t autoKeyStride = static_cast<size_t>(digitsG_auto) * 2 * N;

    uint32_t blockSize = 256;
    if (const char* v = std::getenv("CIRBTS_LMK_BLOCK_X"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0' && parsed > 0) {
            blockSize = static_cast<uint32_t>(std::min<unsigned long>(parsed, 1024ul));
        }
    }
    const int numBlocks = static_cast<int>((N + blockSize - 1) / blockSize);
    dim3 block(blockSize);
    dim3 grid((N + block.x - 1) / block.x, 1);
    const bool fuse_decomp_fnwt = (std::getenv("CIRBTS_LMK_FUSE_DECOMP_FNWT") != nullptr);
    uint32_t fusedNttTpb = static_cast<uint32_t>(std::min<size_t>(N / 2, 1024));
    if (const char* v = std::getenv("CIRBTS_LMK_FUSED_NTT_TPB"); v && *v) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 10);
        if (end != v && end && *end == '\0' && parsed > 0) {
            fusedNttTpb = static_cast<uint32_t>(std::min<unsigned long>(parsed, fusedNttTpb));
            fusedNttTpb = (fusedNttTpb / 32) * 32;
            if (fusedNttTpb == 0) {
                fusedNttTpb = 32;
            }
        }
    }
    const size_t nttShmem = N * sizeof(NativeInt);

    // Reusable device buffers (avoid per-call cudaMallocAsync/free inside the loop).
    auto d_ct      = d_lmk_ct_.get();
    auto d_dct     = d_lmk_dct_.get();
    auto d_perm_c0 = d_lmk_perm_c0_.get();
    auto d_perm_c1 = d_lmk_perm_c1_.get();
    auto d_dcta    = d_lmk_dcta_.get();
    auto d_ks_c0   = d_lmk_ks_c0_.get();
    auto d_ks_c1   = d_lmk_ks_c1_.get();

    auto AddToAccLMKCDEY_GPU = [&](uint32_t lweIndex) {
        const NativeInt* d_key = d_LMK_evalkey_.get() + static_cast<size_t>(lweIndex) * evalKeyStride;

        cudaMemcpyAsync(d_ct, d_acc.get(), 2 * N * sizeof(NativeInt), cudaMemcpyDeviceToDevice, s);

        // 2-polynomial inverse NTT (batched).
        inwt_1d_opt_batched(d_ct, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                            d_n_inv_mod_q_shoup_.get(), N, /*batch=*/2, s);

        if (fuse_decomp_fnwt) {
            kernel_SignedDigitDecomposeLMKCDEY_Ciphertext_FusedFNWT<<<digitsG2_eval, fusedNttTpb, nttShmem, s>>>(
                d_dct, d_ct, Q, baseG, digitsG2_eval, N, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());
        }
        else {
            kernel_SignedDigitDecomposeLMKCDEY_Ciphertext<<<numBlocks, blockSize, 0, s>>>(d_dct, d_ct, Q, baseG, digitsG2_eval, N);
            fnwt_1d_opt_batched(d_dct, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N, digitsG2_eval, s);
        }

        kernel_MultAdd<<<grid, block, 0, s>>>(d_acc.get(), d_acc.get() + N, d_dct, d_key, d_modulus_.get(), digitsG2_eval, N, 1);
    };

    auto AutomorphismLMKCDEY_GPU = [&](uint32_t autoIndex) {
        const uint32_t* d_map = d_LMK_autoMaps_.get() + static_cast<size_t>(autoIndex) * N;
        const NativeInt* d_key = d_LMK_autokey_.get() + static_cast<size_t>(autoIndex) * autoKeyStride;

        kernel_Permute_Only<<<grid, block, 0, s>>>(d_perm_c1, d_acc.get() + N, d_map, N, 1, Q);
        kernel_Permute_Only<<<grid, block, 0, s>>>(d_perm_c0, d_acc.get(), d_map, N, 1, Q);

        // acc1 = permuted(acc1)
        cudaMemcpyAsync(d_acc.get() + N, d_perm_c1, N * sizeof(NativeInt), cudaMemcpyDeviceToDevice, s);

        // Decompose permuted(acc0)
        inwt_1d_opt_batched(d_perm_c0, d_itwiddles_.get(), d_itwiddles_shoup_.get(), d_modulus_.get(), d_n_inv_mod_q_.get(),
                            d_n_inv_mod_q_shoup_.get(), N, 1, s);

        if (fuse_decomp_fnwt) {
            kernel_SignedDigitDecomposeLMKCDEY_Poly_FusedFNWT<<<digitsG_auto, fusedNttTpb, nttShmem, s>>>(
                d_dcta, d_perm_c0, Q, baseG, digitsG_auto, N, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get());
        }
        else {
            kernel_SignedDigitDecomposeLMKCDEY_Poly<<<numBlocks, blockSize, 0, s>>>(d_dcta, d_perm_c0, Q, baseG, digitsG_auto, N);
            fnwt_1d_opt_batched(d_dcta, d_twiddles_.get(), d_twiddles_shoup_.get(), d_modulus_.get(), N, digitsG_auto, s);
        }

        kernel_MultAdd<<<grid, block, 0, s>>>(d_ks_c0, d_ks_c1, d_dcta, d_key, d_modulus_.get(), digitsG_auto, N, 1);

        // acc0 = ks(c0)
        cudaMemcpyAsync(d_acc.get(), d_ks_c0, N * sizeof(NativeInt), cudaMemcpyDeviceToDevice, s);
        // acc1 = acc1 + ks(c0)
        {
            kernel_AddInPlace<<<numBlocks, blockSize, 0, s>>>(d_acc.get() + N, d_ks_c1, d_modulus_.get(), N);
        }
    };

    constexpr uint32_t gen = 5;
    const bool use_window = (auto_window > 1);
    auto key_index_for_step = [&](uint32_t step) -> uint32_t {
        const uint32_t pow = powmod_u32(gen, step, M);
        auto it = pow_index.find(pow);
        if (it == pow_index.end()) {
            OPENFHE_THROW(config_error, "LMKCDEY_EvalAcc: missing automorphism key for requested step");
        }
        return it->second;
    };
    auto apply_skip = [&](uint32_t skip) {
        if (skip == 0) {
            return;
        }
        if (!use_window) {
            AutomorphismLMKCDEY_GPU(key_index_for_step(skip));
            return;
        }
        const uint32_t q = skip / auto_window;
        const uint32_t r = skip % auto_window;
        if (q) {
            AutomorphismLMKCDEY_GPU(key_index_for_step(q * auto_window));
        }
        if (r) {
            AutomorphismLMKCDEY_GPU(key_index_for_step(r));
        }
    };

    // Initial step: acc1 = AutomorphismTransform(M - gen), without key switching.
    {
        const uint32_t* d_map0 = d_LMK_autoMaps_.get();  // index 0 is k = M - gen
        kernel_Permute_Only<<<grid, block, 0, s>>>(d_perm_c1, d_acc.get() + N, d_map0, N, 1, Q);
        cudaMemcpyAsync(d_acc.get() + N, d_perm_c1, N * sizeof(NativeInt), cudaMemcpyDeviceToDevice, s);
    }

    // for a_j = -5^i  (bucket order: 0 .. Nh-2)
    uint32_t nSkips = 0;
    for (uint32_t i = Nh - 1; i > 0; --i) {
        const uint32_t bucket = (Nh - 1) - i;
        if (offsets[bucket] != offsets[bucket + 1]) {
            if (nSkips != 0) {
                if (use_window) {
                    apply_skip(nSkips);
                } else {
                    AutomorphismLMKCDEY_GPU(nSkips);
                }
                nSkips = 0;
            }
            for (uint32_t pos = offsets[bucket]; pos < offsets[bucket + 1]; ++pos) {
                AddToAccLMKCDEY_GPU(items[pos]);
            }
        }
        ++nSkips;
        if (use_window) {
            if (i == 1) {
                apply_skip(nSkips);
                nSkips = 0;
            }
        } else {
            if (nSkips == numAutoKeys || i == 1) {
                AutomorphismLMKCDEY_GPU(nSkips);
                nSkips = 0;
            }
        }
    }

    // for -1 (index = M)
    {
        const uint32_t bucket = bucketCount - 1;
        for (uint32_t pos = offsets[bucket]; pos < offsets[bucket + 1]; ++pos) {
            AddToAccLMKCDEY_GPU(items[pos]);
        }
    }

    // Automorphism by k = M - gen (index 0)
    AutomorphismLMKCDEY_GPU(0);

    // for a_j = 5^i (bucket order: 2Nh-2 .. Nh)
    nSkips = 0;
    for (uint32_t i = Nh - 1; i > 0; --i) {
        const uint32_t bucket = (Nh - 1) + i;
        if (offsets[bucket] != offsets[bucket + 1]) {
            if (nSkips != 0) {
                if (use_window) {
                    apply_skip(nSkips);
                } else {
                    AutomorphismLMKCDEY_GPU(nSkips);
                }
                nSkips = 0;
            }
            for (uint32_t pos = offsets[bucket]; pos < offsets[bucket + 1]; ++pos) {
                AddToAccLMKCDEY_GPU(items[pos]);
            }
        }
        ++nSkips;
        if (use_window) {
            if (i == 1) {
                apply_skip(nSkips);
                nSkips = 0;
            }
        } else {
            if (nSkips == numAutoKeys || i == 1) {
                AutomorphismLMKCDEY_GPU(nSkips);
                nSkips = 0;
            }
        }
    }

    // for 0 (index = 0)
    {
        const uint32_t bucket = Nh - 1;
        for (uint32_t pos = offsets[bucket]; pos < offsets[bucket + 1]; ++pos) {
            AddToAccLMKCDEY_GPU(items[pos]);
        }
    }
}
