// AES-128 CTR transciphering example.

#include "openfhe.h"
#include "phantom.h"

#include "cirbts/cirbts.cuh"
#include "cirbts/kernel.cuh"
#include "cirbtscontext.h"
#include "lwe-pke.h"
#include "rlwe-ske.h"
#include "ntt.cuh"

#include "../aes_gpu_ks.cuh"

#include <algorithm>
#include <cmath>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace lbcrypto;

namespace {

#include "common.cuh"
#include "aes128.cuh"
#include "rlwe_tables.cuh"
#include "gpu_lut.cuh"
#include "extract.cuh"
#include "key_material.cuh"


}  // namespace

int main(int argc, char** argv) {
    try {
        std::string backend = "GPU";
        uint32_t seed = 1;
        uint32_t cbs_batch = 1;
        uint32_t cmux_gpu = 1;
        if (argc >= 2) {
            const std::string arg1 = argv[1];
            if (arg1 == "GPU" || arg1 == "gpu" || arg1 == "CPU" || arg1 == "cpu") {
                backend = arg1;
                if (argc >= 3) {
                    (void)ParseU32(argv[2], &seed);
                }
                if (argc >= 4) {
                    (void)ParseU32(argv[3], &cbs_batch);
                }
                if (argc >= 5) {
                    (void)ParseU32(argv[4], &cmux_gpu);
                }
            } else {
                // Shorthand: only specify seed and batch (backend defaults to GPU, cmux_gpu defaults to 1).
                (void)ParseU32(argv[1], &seed);
                if (argc >= 3) {
                    (void)ParseU32(argv[2], &cbs_batch);
                }
                if (argc >= 4) {
                    (void)ParseU32(argv[3], &cmux_gpu);
                }
            }
        }

        const bool useGPU = (backend == "GPU" || backend == "gpu");
        const bool useBatchCBS = useGPU && (cbs_batch != 0);
        const bool useGpuCMUX = useGPU && (cmux_gpu != 0);
        auto set_default_env = [](const char* key, const char* value) {
            if (std::getenv(key) == nullptr) {
                setenv(key, value, 1);
                return true;
            }
            return false;
        };
        auto env_flag = [](const char* key, bool default_value) {
            const char* v = std::getenv(key);
            if (!v || !*v) {
                return default_value;
            }
            return std::string(v) != "0";
        };
        if (useGPU && env_flag("AES_DEFAULT_BENCH", true)) {
            const bool bench_explicitly_disabled = (std::getenv("AES_BENCH") != nullptr) && !env_flag("AES_BENCH", true);
            if (!bench_explicitly_disabled) {
                bool bench_default_applied = false;
                bench_default_applied |= set_default_env("AES_CTR", "1");
                bench_default_applied |= set_default_env("AES_BENCH", "1");
                bench_default_applied |= set_default_env("AES_WARMUP", "5");
                bench_default_applied |= set_default_env("AES_REPEAT", "10");
                if (bench_default_applied) {
                    std::cout << "[AES] Applied AES CTR benchmark defaults (override with AES_* env or AES_DEFAULT_BENCH=0)." << std::endl;
                }
            }
        }
        if (useGPU && env_flag("AES_FAST_DEFAULTS", true)) {
            bool fast_default_applied = false;
            fast_default_applied |= set_default_env("CIRBTS_LUT_GRAPH", "1");
            fast_default_applied |= set_default_env("AES_GPU_FULL", "1");
            fast_default_applied |= set_default_env("AES_SUPER_BATCH", "1");
            fast_default_applied |= set_default_env("AES_GPU_INIT_STATE", "1");
            fast_default_applied |= set_default_env("AES_EXTRACT_SUPER_BATCH", "1");
            fast_default_applied |= set_default_env("AES_LUT_STAGE0_PRECOMP", "1");
            fast_default_applied |= set_default_env("AES_LUT_BATCH", "16");
            fast_default_applied |= set_default_env("AES_LUT_FUSED_ADD", "1");
            fast_default_applied |= set_default_env("AES_LUT_FUSED_KEY_ADD", "1");
            fast_default_applied |= set_default_env("CIRBTS_AES_FUSED_INWT_DCT", "1");
            fast_default_applied |= set_default_env("CIRBTS_FUSED_NTT_TPB", "384");
            if (fast_default_applied) {
                std::cout << "[AES] Applied AES GPU fast defaults (override with AES_* env or AES_FAST_DEFAULTS=0)." << std::endl;
            }
        }
        if (useBatchCBS) {
            set_default_env("CIRBTS_BATCH_CAMPAIGN", "1");
            set_default_env("CIRBTS_BATCH_CAMPAIGN_TARGET", "128");
            set_default_env("CIRBTS_BATCH_INDEX_NMAJOR", "1");

            bool mb3_default_applied = false;
            mb3_default_applied |= set_default_env("CIRBTS_PBS_MULTIBIT", "3");
            mb3_default_applied |= set_default_env("CIRBTS_MB3_EXPERIMENTAL", "1");
            mb3_default_applied |= set_default_env("CIRBTS_EVALACC_SOA", "1");
            mb3_default_applied |= set_default_env("CIRBTS_EVALACC_SMEM", "0");
            mb3_default_applied |= set_default_env("CIRBTS_BATCH_INDEX_NMAJOR", "1");
            mb3_default_applied |= set_default_env("CIRBTS_EVALACC_BATCH_BLOCK_X", "128");
            mb3_default_applied |= set_default_env("CIRBTS_EVALACC_BLOCK_X", "160");
            if (mb3_default_applied) {
                std::cout << "[AES] Applied default MB3 batch-CBS tuning (can override via CIRBTS_* env)." << std::endl;
            }
        }
        // Default to the on-GPU path to keep CBS->LUT->extract on device unless explicitly disabled.
        bool gpuResidentRequested = useGPU;
        if (const char* v = std::getenv("AES_GPU_RESIDENT"); v && *v) {
            gpuResidentRequested = (std::string(v) != "0");
        }
        const bool gpuFullRequested = useGPU && (std::getenv("AES_GPU_FULL") != nullptr);
        bool timing_sync = (std::getenv("AES_TIMING_SYNC") != nullptr);
        const bool round_graph_requested = useGPU && (std::getenv("AES_ROUND_GRAPH") != nullptr);
        const bool aes_super_batch_requested = (std::getenv("AES_SUPER_BATCH") != nullptr);
        const bool aes_gpu_init_state_requested = []() {
            const char* v = std::getenv("AES_GPU_INIT_STATE");
            return !(v && *v && std::string(v) == "0");
        }();
        const bool lutStage0Precompute = []() {
            const char* v = std::getenv("AES_LUT_STAGE0_PRECOMP");
            return v && *v && std::string(v) != "0";
        }();
        const bool lutFusedFinalAdd = []() {
            const char* v = std::getenv("AES_LUT_FUSED_ADD");
            return v && *v && std::string(v) != "0";
        }();
        const bool lutFusedKeyAdd = []() {
            const char* v = std::getenv("AES_LUT_FUSED_KEY_ADD");
            return v && *v && std::string(v) != "0";
        }();
        const bool extractSuperBatchRequested = []() {
            const char* v = std::getenv("AES_EXTRACT_SUPER_BATCH");
            return !(v && *v && std::string(v) == "0");
        }();
        uint32_t block_count = 128;
        if (const char* v = std::getenv("AES_BLOCKS")) {
            block_count = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
            if (block_count == 0) {
                block_count = 1;
            }
        }
        // CBS EvalAcc/HTSS kernels use grid.y = digitsCC * (AES blocks * 128 TLWE bits).
        // With digitsCC=4, 128 AES blocks would require grid.y=65536, just beyond CUDA's 65535 y-limit.
        uint32_t gpu_full_blocks_capacity = aes_super_batch_requested ? std::min<uint32_t>(block_count, 80u) : 1u;
        if (const char* v = std::getenv("AES_SUPER_BATCH_BLOCKS"); v && *v) {
            gpu_full_blocks_capacity = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
            if (gpu_full_blocks_capacity == 0) {
                gpu_full_blocks_capacity = 1;
            }
        }
        const uint32_t gpu_cbs_max_batch = std::max<uint32_t>(128u, gpu_full_blocks_capacity * 128u);

        // CirBTS context (GINX, 128-bit). Default to CMUX_1_EQ for robust chaining.
        CirBTSContext cc;
        CirBTS_PARAMSET paramset = STD128_CircuitBootstrap_CMUX_1_EQ;
        if (const char* v = std::getenv("CIRBTS_PARAMSET"); v && *v) {
            const std::string paramArg(v);
            if (paramArg == "1" || paramArg == "CMUX_1" || paramArg == "STD128_CircuitBootstrap_CMUX_1") {
                paramset = STD128_CircuitBootstrap_CMUX_1;
            } else if (paramArg == "1_MB2" || paramArg == "CMUX_1_MB2" || paramArg == "STD128_CircuitBootstrap_CMUX_1_MB2") {
                paramset = STD128_CircuitBootstrap_CMUX_1_MB2;
            } else if (paramArg == "1_EQ" || paramArg == "CMUX_1_EQ" || paramArg == "STD128_CircuitBootstrap_CMUX_1_EQ") {
                paramset = STD128_CircuitBootstrap_CMUX_1_EQ;
            } else if (paramArg == "1_SUB1" || paramArg == "CMUX_1_SUB1" || paramArg == "STD128_CircuitBootstrap_CMUX_1_SUB1") {
                paramset = STD128_CircuitBootstrap_CMUX_1_SUB1;
            } else if (paramArg == "2" || paramArg == "CMUX_2" || paramArg == "STD128_CircuitBootstrap_CMUX_2") {
                paramset = STD128_CircuitBootstrap_CMUX_2;
            } else if (paramArg == "3" || paramArg == "CMUX_3" || paramArg == "STD128_CircuitBootstrap_CMUX_3") {
                paramset = STD128_CircuitBootstrap_CMUX_3;
            } else if (paramArg == "4" || paramArg == "CMUX_4" || paramArg == "STD128_CircuitBootstrap_CMUX_4") {
                paramset = STD128_CircuitBootstrap_CMUX_4;
            } else if (paramArg == "5" || paramArg == "CMUX_5" || paramArg == "STD128_CircuitBootstrap_CMUX_5") {
                paramset = STD128_CircuitBootstrap_CMUX_5;
            } else {
                std::cerr << "Unsupported CIRBTS_PARAMSET: " << paramArg << std::endl;
                return 1;
            }
        }
        if (useBatchCBS && paramset == STD128_CircuitBootstrap_CMUX_1_EQ) {
            bool ep_default_applied = false;
            ep_default_applied |= set_default_env("CIRBTS_BASE_EP", "33554432");
            ep_default_applied |= set_default_env("CIRBTS_DIGITS_EP", "1");
            if (ep_default_applied) {
                std::cout << "[AES] Applied MB3-safe CMUX_1_EQ EP gadget (BaseEP=2^25, DigitsEP=1)." << std::endl;
            }
        }
        cc.GenerateCirBTSContext(paramset, GINX);

        auto params = cc.GetParams();
        auto rlweParams = params->GetRLWEParams();
        const uint32_t basecc = params->GetRingGSWParams2()->GetBaseG();
        const uint32_t digitscc = params->GetDigitsCC();
        const uint32_t N = static_cast<uint32_t>(rlweParams->GetN());
        const uint32_t lwe_n = params->GetLWEParams()->Getn();

        // Keys (client-side).
        auto sk = cc.KeyGen();        // level-0 LWE secret
        auto sk2 = cc.RLWEKeyGen();   // level-2 RLWE secret
        cc.CirBTKeyGen(sk, sk2);
        const auto& cirbt_key = cc.GetCirBTSKey();

        // Extraction keys (TLWE_N -> TLWE_n).
        LWEEncryptionScheme lweScheme;
        // Default GPU-friendly KS params unless overridden.
        NativeInteger qKS = NativeInteger(16384);
        uint32_t baseKS = 128u;
        if (const char* env_qks = std::getenv("AES_EXTRACT_QKS")) {
            qKS = NativeInteger(std::strtoull(env_qks, nullptr, 10));
        }
        if (const char* env_bks = std::getenv("AES_EXTRACT_BASEKS")) {
            baseKS = static_cast<uint32_t>(std::strtoul(env_bks, nullptr, 10));
        }
        auto lweParamsKS = std::make_shared<LWECryptoParams>(params->GetLWEParams()->Getn(), params->GetLWEParams()->GetN(), params->GetLWEParams()->Getq(),
                                                             params->GetLWEParams()->GetQ(), qKS, params->GetLWEParams()->GetDgg().GetStd(), baseKS,
                                                             params->GetLWEParams()->GetKeyDist());
        auto skN = DeriveLWEKeyFromRLWESecret(lweParamsKS, sk2);
        auto ksk_rn_to_n = lweScheme.KeySwitchGen(lweParamsKS, sk, skN);

        // GPU offload for the TLWE_N -> TLWE_n KeySwitch used during RLWE->TLWE extraction.
        // Enable with AES_EXTRACT_GPU_KS=1 (only supports uint16-packed qKS/baseKS with digitCount<=2).
        // If AES_EXTRACT_GPU_MODSWITCH=1 is set, the ModSwitch to q is also done on GPU and only raw LWE arrays are copied back.
        // Default to enabling GPU KS + GPU ModSwitch on GPU unless explicitly disabled.
        bool useGpuExtractKS = useGPU;
        if (const char* v = std::getenv("AES_EXTRACT_GPU_KS"); v && *v) {
            useGpuExtractKS = (std::string(v) != "0");
        }
        bool useGpuExtractModswitch = useGPU;
        if (const char* v = std::getenv("AES_EXTRACT_GPU_MODSWITCH"); v && *v) {
            useGpuExtractModswitch = (std::string(v) != "0");
        }
        aes_example::GpuLweKeySwitchKeyU16 gpu_ksk;
        std::unique_ptr<aes_example::GpuLweKeySwitchWorkspaceU16> gpu_ks_ws;
        bool hasGpuExtractKS = false;
        bool gpu_ksk_ready = false;
        if (useGpuExtractKS) {
            std::string why;
            if (!aes_example::GpuLweKeySwitchKeyU16::TryInit(&gpu_ksk, lweParamsKS, ksk_rn_to_n, &why)) {
                std::cerr << "[AES] AES_EXTRACT_GPU_KS requested but disabled: " << why << std::endl;
            } else {
                gpu_ksk_ready = true;
            }
        }

        const bool use_ctr = env_flag("AES_CTR", false);
        std::array<uint8_t, 16> ctr_base{};
        if (const char* hex = std::getenv("AES_CTR_COUNTER_HEX"); hex && *hex) {
            if (!ParseHex128(hex, ctr_base)) {
                OPENFHE_THROW(config_error, "AES_CTR_COUNTER_HEX must be 32 hex chars");
            }
        } else if (const char* dec = std::getenv("AES_CTR_COUNTER"); dec && *dec) {
            const uint64_t v = static_cast<uint64_t>(std::strtoull(dec, nullptr, 10));
            StoreBE64(v, ctr_base, 8);
        }
        if (use_ctr) {
            std::cout << "[AES] CTR mode enabled (AES-ENC on counter, XOR with public ciphertext)." << std::endl;
        }

        // Build decryption tables (Td0..3, Td4_*).
        const auto decTables = BuildAesDecTables();
        std::cout << "Building 8x256 RLWE tables (Td0..3 + Td4_0..3)..." << std::endl;
        auto td0_ct = BuildTableRLWE(rlweParams, sk2, decTables.td0);
        auto td1_ct = BuildTableRLWE(rlweParams, sk2, decTables.td1);
        auto td2_ct = BuildTableRLWE(rlweParams, sk2, decTables.td2);
        auto td3_ct = BuildTableRLWE(rlweParams, sk2, decTables.td3);
        auto td4_0_ct = BuildTableRLWE(rlweParams, sk2, decTables.td4_0);
        auto td4_1_ct = BuildTableRLWE(rlweParams, sk2, decTables.td4_1);
        auto td4_2_ct = BuildTableRLWE(rlweParams, sk2, decTables.td4_2);
        auto td4_3_ct = BuildTableRLWE(rlweParams, sk2, decTables.td4_3);

        std::vector<RLWECiphertext> te0_ct, te1_ct, te2_ct, te3_ct, te4_0_ct, te4_1_ct, te4_2_ct, te4_3_ct;
        if (use_ctr) {
            const auto encTables = BuildAesEncTables();
            std::cout << "Building 8x256 RLWE tables (Te0..3 + Te4_0..3)..." << std::endl;
            te0_ct = BuildTableRLWE(rlweParams, sk2, encTables.te0);
            te1_ct = BuildTableRLWE(rlweParams, sk2, encTables.te1);
            te2_ct = BuildTableRLWE(rlweParams, sk2, encTables.te2);
            te3_ct = BuildTableRLWE(rlweParams, sk2, encTables.te3);
            te4_0_ct = BuildTableRLWE(rlweParams, sk2, encTables.te4_0);
            te4_1_ct = BuildTableRLWE(rlweParams, sk2, encTables.te4_1);
            te4_2_ct = BuildTableRLWE(rlweParams, sk2, encTables.te4_2);
            te4_3_ct = BuildTableRLWE(rlweParams, sk2, encTables.te4_3);
        }

        // GPU context + CMUX workspace + upload tables if needed.
        std::unique_ptr<GPUCirBTSContext> gpu_cc;
        Lut8x32GpuWorkspace gpu_ws{};
        std::array<DeviceRLWETable, 8> d_tables_dec{};
        std::array<DeviceRLWETable, 8> d_tables_enc{};
        if (useGPU) {
            gpu_cc = std::make_unique<GPUCirBTSContext>(cc, gpu_cbs_max_batch);
            if (round_graph_requested || gpuFullRequested) {
                gpu_ws.SetStream(gpu_cc->stream());
            }
        }
        if (useGpuCMUX) {
            gpu_ws.N = N;
            gpu_ws.digitscc = digitscc;

            d_tables_dec[0] = UploadTableRLWEToGPU(rlweParams, td0_ct, gpu_ws.stream());
            d_tables_dec[1] = UploadTableRLWEToGPU(rlweParams, td1_ct, gpu_ws.stream());
            d_tables_dec[2] = UploadTableRLWEToGPU(rlweParams, td2_ct, gpu_ws.stream());
            d_tables_dec[3] = UploadTableRLWEToGPU(rlweParams, td3_ct, gpu_ws.stream());
            d_tables_dec[4] = UploadTableRLWEToGPU(rlweParams, td4_0_ct, gpu_ws.stream());
            d_tables_dec[5] = UploadTableRLWEToGPU(rlweParams, td4_1_ct, gpu_ws.stream());
            d_tables_dec[6] = UploadTableRLWEToGPU(rlweParams, td4_2_ct, gpu_ws.stream());
            d_tables_dec[7] = UploadTableRLWEToGPU(rlweParams, td4_3_ct, gpu_ws.stream());
            if (use_ctr) {
                d_tables_enc[0] = UploadTableRLWEToGPU(rlweParams, te0_ct, gpu_ws.stream());
                d_tables_enc[1] = UploadTableRLWEToGPU(rlweParams, te1_ct, gpu_ws.stream());
                d_tables_enc[2] = UploadTableRLWEToGPU(rlweParams, te2_ct, gpu_ws.stream());
                d_tables_enc[3] = UploadTableRLWEToGPU(rlweParams, te3_ct, gpu_ws.stream());
                d_tables_enc[4] = UploadTableRLWEToGPU(rlweParams, te4_0_ct, gpu_ws.stream());
                d_tables_enc[5] = UploadTableRLWEToGPU(rlweParams, te4_1_ct, gpu_ws.stream());
                d_tables_enc[6] = UploadTableRLWEToGPU(rlweParams, te4_2_ct, gpu_ws.stream());
                d_tables_enc[7] = UploadTableRLWEToGPU(rlweParams, te4_3_ct, gpu_ws.stream());
            }

            const uint32_t rows = digitscc * 2;
            gpu_ws.ctrl_per_bit_stride = static_cast<size_t>(rows) * 2 * N;
            uint32_t lut_batch_max = 8;
            if (const char* v = std::getenv("AES_LUT_BATCH"); v && *v) {
                lut_batch_max = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
            }
            if (lut_batch_max == 0) {
                lut_batch_max = 1;
            }
            if (lut_batch_max > 16) {
                lut_batch_max = 16;
            }
            gpu_ws.lut_batch_max = lut_batch_max;
            gpu_ws.ctrl_batch_stride = gpu_ws.ctrl_per_bit_stride * lut_batch_max;
            gpu_ws.d_ctrl_gsw = phantom::util::make_cuda_auto_ptr<BasicInteger>(gpu_ws.ctrl_batch_stride * 8, gpu_ws.stream());
            gpu_ws.d_ctrl_gsw_shoup = phantom::util::make_cuda_auto_ptr<BasicInteger>(gpu_ws.ctrl_batch_stride * 8, gpu_ws.stream());

            const size_t stage_elems = static_cast<size_t>(lut_batch_max) * 128 * N;
            gpu_ws.d_stage0_c0 = phantom::util::make_cuda_auto_ptr<BasicInteger>(stage_elems, gpu_ws.stream());
            gpu_ws.d_stage0_c1 = phantom::util::make_cuda_auto_ptr<BasicInteger>(stage_elems, gpu_ws.stream());
            gpu_ws.d_stage1_c0 = phantom::util::make_cuda_auto_ptr<BasicInteger>(stage_elems, gpu_ws.stream());
            gpu_ws.d_stage1_c1 = phantom::util::make_cuda_auto_ptr<BasicInteger>(stage_elems, gpu_ws.stream());
            gpu_ws.d_delta = phantom::util::make_cuda_auto_ptr<BasicInteger>(static_cast<size_t>(lut_batch_max) * 256 * N, gpu_ws.stream());
            gpu_ws.d_digits =
                phantom::util::make_cuda_auto_ptr<BasicInteger>(static_cast<size_t>(digitscc) * lut_batch_max * 256 * N, gpu_ws.stream());
            gpu_ws.d_lut_indices = phantom::util::make_cuda_auto_ptr<uint32_t>(lut_batch_max, gpu_ws.stream());
            gpu_ws.d_lut_word_map = phantom::util::make_cuda_auto_ptr<uint32_t>(lut_batch_max, gpu_ws.stream());

            const size_t ctrl_all_shoup_elems = gpu_ws.ctrl_per_bit_stride * static_cast<size_t>(gpu_cbs_max_batch);
            size_t ctrl_all_shoup_cache_max = (1ull << 26);  // 512 MiB for uint64 coefficients.
            if (const char* v = std::getenv("AES_CTRL_SHOUP_CACHE_MAX_ELEMS"); v && *v) {
                ctrl_all_shoup_cache_max = static_cast<size_t>(std::strtoull(v, nullptr, 10));
            }
            if (ctrl_all_shoup_elems <= ctrl_all_shoup_cache_max) {
                gpu_ws.d_ctrl_all_shoup = phantom::util::make_cuda_auto_ptr<BasicInteger>(ctrl_all_shoup_elems, gpu_ws.stream());
            } else {
                std::cout << "[AES] Full control Shoup cache disabled (" << (ctrl_all_shoup_elems * sizeof(BasicInteger) / (1024 * 1024))
                          << " MiB > limit); computing per LUT chunk." << std::endl;
            }

            if (lutStage0Precompute) {
                auto precompute_table_set = [&](std::array<DeviceRLWETable, 8>& tables) {
                    for (auto& table : tables) {
                        PrecomputeLUTStage0Digits(*gpu_cc, rlweParams, gpu_ws, table, basecc);
                    }
                };
                precompute_table_set(use_ctr ? d_tables_enc : d_tables_dec);
                const uint64_t stage0_digit_polys = 256ull;
                const uint64_t stage0_static_polys = UseLUTFourNodeStage0() ? static_cast<uint64_t>(LUTStaticDeltaPolyCount()) : 0ull;
                const uint64_t stage1_corr_polys =
                    (UseLUTFourNodeStage0() && UseLUTAllStaticDelta() && UseLUTStage1StaticCorrection())
                        ? static_cast<uint64_t>(LUTStage1CorrectionPolyCount())
                        : 0ull;
                const double mib = static_cast<double>(8ull * (digitscc * (stage0_digit_polys + stage1_corr_polys) + stage0_static_polys) *
                                                       static_cast<unsigned long long>(N) * sizeof(BasicInteger)) /
                                   (1024.0 * 1024.0);
                std::cout << "[AES] AES_LUT_STAGE0_PRECOMP enabled (" << mib << " MiB cached for active LUT tables)." << std::endl;
                if (stage1_corr_polys != 0ull) {
                    std::cout << "[AES] AES_LUT_STAGE1_STATIC_CORRECTION enabled (precomputed digit-space finite differences)." << std::endl;
                }
            }
        }

        if (gpu_ksk_ready) {
            if (useGpuCMUX) {
                gpu_ks_ws = std::make_unique<aes_example::GpuLweKeySwitchWorkspaceU16>(gpu_cbs_max_batch, gpu_ksk.N, gpu_ksk.n, gpu_ws.stream());
            } else {
                gpu_ks_ws = std::make_unique<aes_example::GpuLweKeySwitchWorkspaceU16>(gpu_cbs_max_batch, gpu_ksk.N, gpu_ksk.n);
            }

            std::vector<uint32_t> h_info(gpu_cbs_max_batch);
            for (uint32_t t = 0; t < gpu_cbs_max_batch; ++t) {
                const uint32_t block_id = t >> 7;
                const uint32_t local_t = t & 127u;
                const uint32_t byteIndex = local_t / 8u;
                const uint32_t b = local_t % 8u;
                const uint32_t col = byteIndex / 4u;
                const uint32_t row = byteIndex % 4u;
                const uint32_t startBit = (3u - row) * 8u;
                const uint32_t bitIdx = startBit + b;
                h_info[t] = ((block_id * 4u + col) << 16) | (bitIdx & 0xFFFFu);
            }
            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(gpu_ks_ws->d_extract_info, h_info.data(), h_info.size() * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                               gpu_ks_ws->stream));
            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(gpu_ks_ws->stream));

            hasGpuExtractKS = true;
            std::cout << "[AES] GPU extract KeySwitch enabled (uint16 packed)." << std::endl;
        }

        const bool useGpuResident = gpuResidentRequested && useBatchCBS && useGpuCMUX && hasGpuExtractKS;
        bool useGpuResidentFull = false;
        if (gpuFullRequested) {
            if (!useGpuResident) {
                std::cerr << "[AES] AES_GPU_FULL requested but disabled (requires GPU + cbs_batch=1 + cmux_gpu=1 + AES_EXTRACT_GPU_KS=1)." << std::endl;
            } else {
                const uint64_t q_lwe_u64 = params->GetLWEParams()->Getq().ConvertToInt();
                if (q_lwe_u64 == 0) {
                    std::cerr << "[AES] AES_GPU_FULL disabled: q must be non-zero for device chaining." << std::endl;
                } else {
                    useGpuResidentFull = true;
                }
            }
        }
        if (gpuResidentRequested && !useGpuResident) {
            std::cerr << "[AES] AES_GPU_RESIDENT requested but disabled (requires GPU + cbs_batch=1 + cmux_gpu=1 + AES_EXTRACT_GPU_KS=1)." << std::endl;
        }

        const bool use_round_graph = useGpuResidentFull && round_graph_requested;
        if (use_round_graph) {
            gpu_ws.disable_graph = true;
            timing_sync = false;
        }
        const bool use_single_stream = useGpuResidentFull;

        CudaEventGuard cbs_ready_event;
        CudaEventGuard lwe_ready_event;
        if (useGpuResident && !use_single_stream) {
            PHANTOM_CHECK_CUDA(cudaEventCreateWithFlags(&cbs_ready_event.ev, cudaEventDisableTiming));
        }
        if (useGpuResidentFull && !use_single_stream) {
            PHANTOM_CHECK_CUDA(cudaEventCreateWithFlags(&lwe_ready_event.ev, cudaEventDisableTiming));
        }

        phantom::util::cuda_auto_ptr<BasicInteger> d_state_words_c0;
        phantom::util::cuda_auto_ptr<BasicInteger> d_state_words_c1;
        phantom::util::cuda_auto_ptr<BasicInteger> d_state_lwe_a;
        phantom::util::cuda_auto_ptr<BasicInteger> d_state_lwe_b;
        phantom::util::cuda_auto_ptr<BasicInteger> d_rk0_lwe_a_dev;
        phantom::util::cuda_auto_ptr<BasicInteger> d_rk0_lwe_b_dev;
        phantom::util::cuda_auto_ptr<uint8_t> d_public_input_bits;
        std::array<DeviceRLWEWords4, 10> d_rk_imc_words{};
        DeviceRLWEWords4 d_rk0_words_dev{};
        std::array<DeviceRLWEWords4, 11> d_rk_enc_words{};
        if (useGpuResident) {
            const cudaStream_t s = gpu_ws.stream();
            d_state_words_c0 = phantom::util::make_cuda_auto_ptr<BasicInteger>(static_cast<size_t>(gpu_full_blocks_capacity) * 4 * N, s);
            d_state_words_c1 = phantom::util::make_cuda_auto_ptr<BasicInteger>(static_cast<size_t>(gpu_full_blocks_capacity) * 4 * N, s);
            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
            if (useGpuResidentFull || useGpuExtractModswitch) {
                d_state_lwe_a = phantom::util::make_cuda_auto_ptr<BasicInteger>(static_cast<size_t>(gpu_full_blocks_capacity) * 128 * lwe_n, s);
                d_state_lwe_b = phantom::util::make_cuda_auto_ptr<BasicInteger>(static_cast<size_t>(gpu_full_blocks_capacity) * 128, s);
                if (useGpuResidentFull) {
                    std::cout << "[AES] AES_GPU_FULL enabled (CBS->LUT->extract chained on GPU)." << std::endl;
                    if (aes_super_batch_requested && use_ctr) {
                        std::cout << "[AES] AES_SUPER_BATCH capacity=" << gpu_full_blocks_capacity
                                  << " AES blocks (CBS batch capacity=" << gpu_cbs_max_batch << " TLWE bits)." << std::endl;
                        if (extractSuperBatchRequested) {
                            std::cout << "[AES] AES_EXTRACT_SUPER_BATCH enabled (batched multi-block sample-extract + KS)." << std::endl;
                        }
                        if (aes_gpu_init_state_requested) {
                            std::cout << "[AES] AES_GPU_INIT_STATE enabled (public input + rk0 composed on GPU)." << std::endl;
                        }
                    }
                } else if (useGpuExtractModswitch) {
                    std::cout << "[AES] AES_GPU_RESIDENT GPU-ModSwitch enabled (CBS->LUT->extract on GPU, ModSwitch on GPU)." << std::endl;
                }
            }
            if (useGpuResidentFull && aes_super_batch_requested && aes_gpu_init_state_requested) {
                d_rk0_lwe_a_dev = phantom::util::make_cuda_auto_ptr<BasicInteger>(static_cast<size_t>(128) * lwe_n, s);
                d_rk0_lwe_b_dev = phantom::util::make_cuda_auto_ptr<BasicInteger>(static_cast<size_t>(128), s);
                d_public_input_bits = phantom::util::make_cuda_auto_ptr<uint8_t>(static_cast<size_t>(gpu_full_blocks_capacity) * 128, s);
            }
            if (!useGpuResidentFull) {
                std::cout << "[AES] AES_GPU_RESIDENT enabled (CBS->LUT->extract on GPU)." << std::endl;
            }
        }

        uint32_t random_tests = 0;
        if (const char* v = std::getenv("AES_RANDOM_TESTS")) {
            random_tests = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
        }
        const bool run_nist = (std::getenv("AES_NIST") != nullptr);
        bool bench_enabled = env_flag("AES_BENCH", false);
        const bool bench_async = env_flag("AES_BENCH_ASYNC", false);
        uint32_t warmup = 0;
        uint32_t repeat = 0;
        const bool warmup_env_set = (std::getenv("AES_WARMUP") != nullptr);
        const bool repeat_env_set = (std::getenv("AES_REPEAT") != nullptr);
        if (const char* v = std::getenv("AES_WARMUP")) {
            warmup = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
        }
        if (const char* v = std::getenv("AES_REPEAT")) {
            repeat = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
        }
        if (bench_enabled && !warmup_env_set) {
            warmup = 3;
        }
        if (bench_enabled && !repeat_env_set) {
            repeat = 10;
        }
        if (warmup > 0 || repeat > 0) {
            bench_enabled = true;
        }
        const bool sweep_batches = (std::getenv("AES_SWEEP_BATCH") != nullptr);
        const std::vector<uint32_t> batch_list = sweep_batches ? std::vector<uint32_t>{1, 8, 32, 128} : std::vector<uint32_t>{block_count};

        if (bench_enabled && !bench_async) {
            timing_sync = true;
        }

        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> byteDist(0u, 255u);
        auto random_block = [&]() {
            std::array<uint8_t, 16> pt{};
            for (uint32_t i = 0; i < 16; ++i) {
                pt[i] = static_cast<uint8_t>(byteDist(rng));
            }
            return pt;
        };

        std::array<uint8_t, 16> base_key{};
        for (uint32_t i = 0; i < 16; ++i) {
            base_key[i] = static_cast<uint8_t>(byteDist(rng));
        }
        EncryptedAesKeyMaterial keymat = BuildEncryptedKeyMaterial(cc, rlweParams, sk, sk2, base_key);

        auto upload_gpu_keys = [&](const EncryptedAesKeyMaterial& km) {
            if (!useGpuResident) {
                return;
            }
            const cudaStream_t s = gpu_ws.stream();
            for (uint32_t round = 1; round <= 9; ++round) {
                d_rk_imc_words[round] = UploadWords4RLWEToGPU(rlweParams, km.rk_imc_words[round], s);
            }
            d_rk0_words_dev = UploadWords4RLWEToGPU(rlweParams, km.rk0_words, s);
            if (use_ctr) {
                for (uint32_t round = 0; round <= 10; ++round) {
                    d_rk_enc_words[round] = UploadWords4RLWEToGPU(rlweParams, km.rk_words[round], s);
                }
            }
            if (d_rk0_lwe_a_dev.get() && d_rk0_lwe_b_dev.get()) {
                std::vector<BasicInteger> h_a(static_cast<size_t>(128) * lwe_n);
                std::vector<BasicInteger> h_b(128);
                for (uint32_t idx = 0; idx < 128; ++idx) {
                    const auto& ct = km.rk0_bits[idx];
                    const auto& a = ct->GetA();
                    if (a.GetLength() != lwe_n) {
                        OPENFHE_THROW(config_error, "upload_gpu_keys: unexpected rk0 LWE dimension");
                    }
                    for (uint32_t k = 0; k < lwe_n; ++k) {
                        h_a[static_cast<size_t>(idx) * lwe_n + k] = a[k].ConvertToInt<BasicInteger>();
                    }
                    h_b[idx] = ct->GetB().ConvertToInt<BasicInteger>();
                }
                PHANTOM_CHECK_CUDA(cudaMemcpyAsync(d_rk0_lwe_a_dev.get(), h_a.data(), h_a.size() * sizeof(BasicInteger), cudaMemcpyHostToDevice, s));
                PHANTOM_CHECK_CUDA(cudaMemcpyAsync(d_rk0_lwe_b_dev.get(), h_b.data(), h_b.size() * sizeof(BasicInteger), cudaMemcpyHostToDevice, s));
            }
            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(s));
        };
        upload_gpu_keys(keymat);

        auto RunTranscipherBlock = [&](const std::array<uint8_t, 16>& plaintext, const EncryptedAesKeyMaterial& km, TimingSample* timing_out,
                                       bool verify, uint64_t block_idx) -> bool {
            std::array<uint8_t, 16> counter_block{};
            std::array<uint8_t, 16> ciphertext{};
            if (use_ctr) {
                counter_block = MakeCtrBlock(ctr_base, block_idx);
                const auto keystream = Aes128EncryptPlain(counter_block, km.roundKeys);
                for (uint32_t i = 0; i < 16; ++i) {
                    ciphertext[i] = static_cast<uint8_t>(plaintext[i] ^ keystream[i]);
                }
            } else {
                ciphertext = Aes128EncryptPlain(plaintext, km.roundKeys);
            }

            std::vector<LWECiphertext> state_bits;
            state_bits.reserve(16 * 8);
            const auto enc_start = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < 16; ++i) {
                const uint8_t x = use_ctr ? counter_block[i] : ciphertext[i];
                for (uint32_t b = 0; b < 8; ++b) {
                    const uint32_t bit = (x >> b) & 1u;
                    state_bits.emplace_back(cc.Encrypt(sk, static_cast<LWEPlaintext>(bit), /*p=*/2));
                }
            }
            const auto enc_end = std::chrono::steady_clock::now();

            const auto ark10_start = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < 16 * 8; ++i) {
                if (use_ctr) {
                    lweScheme.EvalAddEq(state_bits[i], km.rk0_bits[i]);
                } else {
                    lweScheme.EvalAddEq(state_bits[i], km.rk10_bits[i]);
                }
            }
            const auto ark10_end = std::chrono::steady_clock::now();

            double total_cbs_ms = 0.0;
            double total_lut_ms = 0.0;
            double total_extract_ms = 0.0;
            g_transition_profile.Reset(false);

            if (useGpuResidentFull) {
                if (!gpu_cc || !gpu_ks_ws || !d_state_words_c0.get() || !d_state_words_c1.get() || !d_state_lwe_a.get() || !d_state_lwe_b.get()) {
                    OPENFHE_THROW(config_error, "AES_GPU_FULL: missing GPU workspaces");
                }

                auto run_round_gpu_full = [&](uint32_t round, bool final_round) {
                    const BasicInteger Q = rlweParams->GetQ().ConvertToInt<BasicInteger>();

                    const auto cbs_start = std::chrono::steady_clock::now();
                    const auto view = gpu_cc->gpu_CircuitBootstrappingBatchToDevice(params, cirbt_key, state_bits);
                    if (timing_sync) {
                        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(view.stream));
                        const auto cbs_end = std::chrono::steady_clock::now();
                        total_cbs_ms += DurationMs(cbs_start, cbs_end);
                    } else if (!use_single_stream) {
                        if (!cbs_ready_event.ev) {
                            OPENFHE_THROW(config_error, "AES_GPU_FULL: missing CBS event");
                        }
                        PHANTOM_CHECK_CUDA(cudaEventRecord(cbs_ready_event.ev, view.stream));
                        PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(gpu_ws.stream(), cbs_ready_event.ev, 0));
                    }

                    if (view.batch != 128 || view.N != N || view.per_ciphertext_elems != gpu_ws.ctrl_per_bit_stride) {
                        OPENFHE_THROW(config_error, "AES_GPU_FULL: unexpected CBS batch view shape");
                    }

                    const BasicInteger* d_all = reinterpret_cast<const BasicInteger*>(view.d_rgsw);
                    const size_t stride = view.per_ciphertext_elems;
                    const dim3 add_block(256);
                    const dim3 add_grid((N + add_block.x - 1) / add_block.x);

                    const auto lut_start = std::chrono::steady_clock::now();
                    PHANTOM_CHECK_CUDA(cudaMemsetAsync(d_state_words_c0.get(), 0, static_cast<size_t>(4) * N * sizeof(BasicInteger), gpu_ws.stream()));
                    PHANTOM_CHECK_CUDA(cudaMemsetAsync(d_state_words_c1.get(), 0, static_cast<size_t>(4) * N * sizeof(BasicInteger), gpu_ws.stream()));

                    const bool use_lut_batch = (gpu_ws.lut_batch_max >= 4);
                    if (use_lut_batch) {
                        const uint32_t lut_batch = 4;
                        const dim3 add_grid_batch((N + add_block.x - 1) / add_block.x, lut_batch);
                        for (uint32_t row = 0; row < 4; ++row) {
                            const uint32_t tableId = final_round ? (4 + row) : row;  // 0..3 => Td0..3, 4..7 => Td4_0..3
                            uint32_t idx[4] = {AesIndex(row, 0), AesIndex(row, 1), AesIndex(row, 2), AesIndex(row, 3)};
                            uint32_t word_map[4];
                            if (use_ctr) {
                                word_map[0] = static_cast<uint32_t>((0 + 4u - row) & 3u);
                                word_map[1] = static_cast<uint32_t>((1 + 4u - row) & 3u);
                                word_map[2] = static_cast<uint32_t>((2 + 4u - row) & 3u);
                                word_map[3] = static_cast<uint32_t>((3 + 4u - row) & 3u);
                            } else {
                                word_map[0] = static_cast<uint32_t>((0 + row) & 3u);
                                word_map[1] = static_cast<uint32_t>((1 + row) & 3u);
                                word_map[2] = static_cast<uint32_t>((2 + row) & 3u);
                                word_map[3] = static_cast<uint32_t>((3 + row) & 3u);
                            }
                            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(gpu_ws.d_lut_word_map.get(), word_map, sizeof(word_map), cudaMemcpyHostToDevice,
                                                               gpu_ws.stream()));
                            const auto lut_view = EvalLUT8x32_RLWE_GPU_DeviceCtrl_Batch(
                                *gpu_cc, rlweParams, gpu_ws, d_all, stride, idx, lut_batch,
                                use_ctr ? d_tables_enc[tableId] : d_tables_dec[tableId], basecc);
                            kernel_RLWE_AddInplace_Batch<<<add_grid_batch, add_block, 0, gpu_ws.stream()>>>(
                                d_state_words_c0.get(), d_state_words_c1.get(), lut_view.c0, lut_view.c1, gpu_ws.d_lut_word_map.get(), Q, N, lut_batch);
                            PHANTOM_CHECK_CUDA_LAST();
                        }
                    } else {
                        for (uint32_t col = 0; col < 4; ++col) {
                            for (uint32_t row = 0; row < 4; ++row) {
                                const uint32_t i = AesIndex(row, col);
                                const BasicInteger* d_ctrl = d_all + static_cast<size_t>(i * 8) * stride;
                                const uint32_t tableId = final_round ? (4 + row) : row;  // 0..3 => Td0..3, 4..7 => Td4_0..3
                                const auto lut_ct = EvalLUT8x32_RLWE_GPU_DeviceCtrl(
                                    *gpu_cc, rlweParams, gpu_ws, d_ctrl, use_ctr ? d_tables_enc[tableId] : d_tables_dec[tableId], basecc);
                                const uint32_t word = use_ctr ? static_cast<uint32_t>((col + 4u - row) & 3u) : static_cast<uint32_t>((col + row) & 3u);
                                kernel_RLWE_AddInplace<<<add_grid, add_block, 0, gpu_ws.stream()>>>(
                                    d_state_words_c0.get() + static_cast<size_t>(word) * N,
                                    d_state_words_c1.get() + static_cast<size_t>(word) * N,
                                    lut_ct.c0, lut_ct.c1, Q, N);
                                PHANTOM_CHECK_CUDA_LAST();
                            }
                        }
                    }

                    const dim3 key_grid((N + add_block.x - 1) / add_block.x, 4);
                    if (use_ctr) {
                        const uint32_t rk_round = final_round ? 10u : round;
                        kernel_RLWE_AddWords4Inplace<<<key_grid, add_block, 0, gpu_ws.stream()>>>(
                            d_state_words_c0.get(), d_state_words_c1.get(), d_rk_enc_words[rk_round].d_c0.get(), d_rk_enc_words[rk_round].d_c1.get(), Q, N);
                    } else {
                        if (final_round) {
                            kernel_RLWE_AddWords4Inplace<<<key_grid, add_block, 0, gpu_ws.stream()>>>(
                                d_state_words_c0.get(), d_state_words_c1.get(), d_rk0_words_dev.d_c0.get(), d_rk0_words_dev.d_c1.get(), Q, N);
                        } else {
                            kernel_RLWE_AddWords4Inplace<<<key_grid, add_block, 0, gpu_ws.stream()>>>(
                                d_state_words_c0.get(), d_state_words_c1.get(), d_rk_imc_words[round].d_c0.get(), d_rk_imc_words[round].d_c1.get(), Q, N);
                        }
                    }
                    PHANTOM_CHECK_CUDA_LAST();
                    if (timing_sync) {
                        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(gpu_ws.stream()));
                        const auto lut_end = std::chrono::steady_clock::now();
                        total_lut_ms += DurationMs(lut_start, lut_end);
                    }

                    const auto ex_start = std::chrono::steady_clock::now();
                    if (use_round_graph) {
                        PHANTOM_CHECK_CUDA(cudaEventRecord(lwe_ready_event.ev, gpu_ws.stream()));
                        PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(gpu_cc->stream(), lwe_ready_event.ev, 0));
                    }

                    (void)ExtractStateBytesToLWEDevice(*gpu_cc, params, gpu_ksk, *gpu_ks_ws, d_state_words_c0.get(), d_state_words_c1.get(),
                                                       d_state_lwe_a.get(), d_state_lwe_b.get(), lwe_n);
                    if (timing_sync) {
                        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(gpu_ws.stream()));
                        const auto ex_end = std::chrono::steady_clock::now();
                        total_extract_ms += DurationMs(ex_start, ex_end);
                    }
                };

                if (use_ctr) {
                    for (uint32_t round = 1; round <= 9; ++round) {
                        run_round_gpu_full(round, /*final_round=*/false);

                        auto next_bits = DownloadLWECiphertextsFromGPU(d_state_lwe_a.get(), d_state_lwe_b.get(), lwe_n, 128,
                                                                       params->GetLWEParams()->Getq(), gpu_ws.stream());
                        state_bits.swap(next_bits);
                    }
                    run_round_gpu_full(/*round=*/10, /*final_round=*/true);
                } else {
                    for (uint32_t round = 9; round >= 1; --round) {
                        run_round_gpu_full(round, /*final_round=*/false);

                        auto next_bits = DownloadLWECiphertextsFromGPU(d_state_lwe_a.get(), d_state_lwe_b.get(), lwe_n, 128,
                                                                       params->GetLWEParams()->Getq(), gpu_ws.stream());
                        state_bits.swap(next_bits);
                        if (round == 1) {
                            break;
                        }
                    }
                    run_round_gpu_full(/*round=*/0, /*final_round=*/true);
                }

            } else if (useGpuResident) {
                if (!gpu_cc || !gpu_ks_ws || !d_state_words_c0.get() || !d_state_words_c1.get()) {
                    OPENFHE_THROW(config_error, "AES_GPU_RESIDENT: missing GPU workspaces");
                }

                auto run_round_gpu_resident = [&](uint32_t round) {
                    const auto cbs_start = std::chrono::steady_clock::now();
                    const auto view = gpu_cc->gpu_CircuitBootstrappingBatchToDevice(params, cirbt_key, state_bits);
                    if (timing_sync) {
                        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(view.stream));
                        const auto cbs_end = std::chrono::steady_clock::now();
                        total_cbs_ms += DurationMs(cbs_start, cbs_end);
                    } else if (!use_single_stream) {
                        if (!cbs_ready_event.ev) {
                            OPENFHE_THROW(config_error, "AES_GPU_RESIDENT: missing CBS event");
                        }
                        PHANTOM_CHECK_CUDA(cudaEventRecord(cbs_ready_event.ev, view.stream));
                        PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(gpu_ws.stream(), cbs_ready_event.ev, 0));
                    }

                    if (view.batch != 128 || view.N != N || view.per_ciphertext_elems != gpu_ws.ctrl_per_bit_stride) {
                        OPENFHE_THROW(config_error, "AES_GPU_RESIDENT: unexpected CBS batch view shape");
                    }

                    const BasicInteger Q = rlweParams->GetQ().ConvertToInt<BasicInteger>();
                    const auto lut_start = std::chrono::steady_clock::now();
                    PHANTOM_CHECK_CUDA(cudaMemsetAsync(d_state_words_c0.get(), 0, static_cast<size_t>(4) * N * sizeof(BasicInteger), gpu_ws.stream()));
                    PHANTOM_CHECK_CUDA(cudaMemsetAsync(d_state_words_c1.get(), 0, static_cast<size_t>(4) * N * sizeof(BasicInteger), gpu_ws.stream()));

                    const BasicInteger* d_all = reinterpret_cast<const BasicInteger*>(view.d_rgsw);
                    const size_t stride = view.per_ciphertext_elems;
                    const dim3 add_block(256);
                    const dim3 add_grid((N + add_block.x - 1) / add_block.x);

                    const bool use_lut_batch = (gpu_ws.lut_batch_max >= 4);
                    if (use_lut_batch) {
                        const uint32_t lut_batch = 4;
                        const dim3 add_grid_batch((N + add_block.x - 1) / add_block.x, lut_batch);
                        for (uint32_t row = 0; row < 4; ++row) {
                            const uint32_t tableId = row;  // 0..3 => Td0..3
                            uint32_t idx[4] = {AesIndex(row, 0), AesIndex(row, 1), AesIndex(row, 2), AesIndex(row, 3)};
                            uint32_t word_map[4];
                            if (use_ctr) {
                                word_map[0] = static_cast<uint32_t>((0 + 4u - row) & 3u);
                                word_map[1] = static_cast<uint32_t>((1 + 4u - row) & 3u);
                                word_map[2] = static_cast<uint32_t>((2 + 4u - row) & 3u);
                                word_map[3] = static_cast<uint32_t>((3 + 4u - row) & 3u);
                            } else {
                                word_map[0] = static_cast<uint32_t>((0 + row) & 3u);
                                word_map[1] = static_cast<uint32_t>((1 + row) & 3u);
                                word_map[2] = static_cast<uint32_t>((2 + row) & 3u);
                                word_map[3] = static_cast<uint32_t>((3 + row) & 3u);
                            }
                            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(gpu_ws.d_lut_word_map.get(), word_map, sizeof(word_map), cudaMemcpyHostToDevice,
                                                               gpu_ws.stream()));
                            const auto lut_view = EvalLUT8x32_RLWE_GPU_DeviceCtrl_Batch(
                                *gpu_cc, rlweParams, gpu_ws, d_all, stride, idx, lut_batch,
                                use_ctr ? d_tables_enc[tableId] : d_tables_dec[tableId], basecc);
                            kernel_RLWE_AddInplace_Batch<<<add_grid_batch, add_block, 0, gpu_ws.stream()>>>(
                                d_state_words_c0.get(), d_state_words_c1.get(), lut_view.c0, lut_view.c1, gpu_ws.d_lut_word_map.get(), Q, N, lut_batch);
                            PHANTOM_CHECK_CUDA_LAST();
                        }
                    } else {
                        for (uint32_t col = 0; col < 4; ++col) {
                            for (uint32_t row = 0; row < 4; ++row) {
                                const uint32_t i = AesIndex(row, col);
                                const BasicInteger* d_ctrl = d_all + static_cast<size_t>(i * 8) * stride;
                                const uint32_t tableId = row;  // 0..3 => Td0..3
                                const auto lut_ct = EvalLUT8x32_RLWE_GPU_DeviceCtrl(
                                    *gpu_cc, rlweParams, gpu_ws, d_ctrl, use_ctr ? d_tables_enc[tableId] : d_tables_dec[tableId], basecc);
                                const uint32_t word = use_ctr ? static_cast<uint32_t>((col + 4u - row) & 3u) : static_cast<uint32_t>((col + row) & 3u);
                                kernel_RLWE_AddInplace<<<add_grid, add_block, 0, gpu_ws.stream()>>>(
                                    d_state_words_c0.get() + static_cast<size_t>(word) * N,
                                    d_state_words_c1.get() + static_cast<size_t>(word) * N,
                                    lut_ct.c0, lut_ct.c1, Q, N);
                                PHANTOM_CHECK_CUDA_LAST();
                            }
                        }
                    }

                    const dim3 key_grid((N + add_block.x - 1) / add_block.x, 4);
                    if (use_ctr) {
                        kernel_RLWE_AddWords4Inplace<<<key_grid, add_block, 0, gpu_ws.stream()>>>(
                            d_state_words_c0.get(), d_state_words_c1.get(), d_rk_enc_words[round].d_c0.get(), d_rk_enc_words[round].d_c1.get(), Q, N);
                    } else {
                        kernel_RLWE_AddWords4Inplace<<<key_grid, add_block, 0, gpu_ws.stream()>>>(
                            d_state_words_c0.get(), d_state_words_c1.get(), d_rk_imc_words[round].d_c0.get(), d_rk_imc_words[round].d_c1.get(), Q, N);
                    }
                    PHANTOM_CHECK_CUDA_LAST();
                    if (timing_sync) {
                        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(gpu_ws.stream()));
                        const auto lut_end = std::chrono::steady_clock::now();
                        total_lut_ms += DurationMs(lut_start, lut_end);
                    }

                    const auto ex_start = std::chrono::steady_clock::now();
                    std::vector<LWECiphertext> next_bits;
                    if (useGpuExtractModswitch) {
                        if (!d_state_lwe_a.get() || !d_state_lwe_b.get()) {
                            OPENFHE_THROW(config_error, "AES_GPU_RESIDENT: missing device LWE buffers for GPU ModSwitch");
                        }
                        (void)ExtractStateBytesToLWEDevice(*gpu_cc, params, gpu_ksk, *gpu_ks_ws, d_state_words_c0.get(), d_state_words_c1.get(),
                                                           d_state_lwe_a.get(), d_state_lwe_b.get(), lwe_n);
                        if (timing_sync) {
                            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(gpu_ws.stream()));
                        }
                        next_bits = DownloadLWECiphertextsFromGPU(d_state_lwe_a.get(), d_state_lwe_b.get(), lwe_n, 128,
                                                                  params->GetLWEParams()->Getq(), gpu_ws.stream());
                    } else {
                        next_bits = ExtractStateBytesToLWETwitter_DeviceWords(*gpu_cc, params, lweParamsKS, lweScheme, gpu_ksk, *gpu_ks_ws,
                                                                             d_state_words_c0.get(), d_state_words_c1.get());
                    }
                    const auto ex_end = std::chrono::steady_clock::now();
                    if (timing_sync) {
                        total_extract_ms += DurationMs(ex_start, ex_end);
                    }
                    state_bits.swap(next_bits);
                };

                if (use_ctr) {
                    for (uint32_t round = 1; round <= 9; ++round) {
                        run_round_gpu_resident(round);
                        if (round == 9) {
                            break;
                        }
                    }
                } else {
                    for (uint32_t round = 9; round >= 1; --round) {
                        run_round_gpu_resident(round);
                        if (round == 1) {
                            break;
                        }
                    }
                }
            } else {
                auto run_round_host = [&](uint32_t round) {
                    if (g_transition_profile.enabled) {
                        g_transition_profile.rounds += 1;
                    }
                    std::vector<RGSWCiphertext> all_ctrl;
                    all_ctrl.reserve(16 * 8);
                    const auto cbs_start = std::chrono::steady_clock::now();
                    if (useGPU) {
                        if (useBatchCBS) {
                            if (g_transition_profile.enabled) {
                                g_transition_profile.cbs_groups += 1;
                                g_transition_profile.cbs_requests += state_bits.size();
                                g_transition_profile.host_device_boundaries += 2;  // LWE H2D + RGSW D2H.
                            }
                            all_ctrl = gpu_cc->gpu_CircuitBootstrappingBatch(params, cirbt_key, state_bits);
                        } else {
                            for (uint32_t i = 0; i < 16; ++i) {
                                std::vector<LWECiphertext> byteBits(8);
                                for (uint32_t b = 0; b < 8; ++b) {
                                    byteBits[b] = state_bits[i * 8 + b];
                                }
                                if (g_transition_profile.enabled) {
                                    g_transition_profile.cbs_groups += 1;
                                    g_transition_profile.cbs_requests += byteBits.size();
                                    g_transition_profile.host_device_boundaries += 2;  // byte LWE H2D + byte RGSW D2H.
                                }
                                auto byteCtrl = gpu_cc->gpu_CircuitBootstrappingBatch(params, cirbt_key, byteBits);
                                all_ctrl.insert(all_ctrl.end(), byteCtrl.begin(), byteCtrl.end());
                            }
                        }
                    } else {
                        for (uint32_t i = 0; i < 16 * 8; ++i) {
                            all_ctrl.emplace_back(cc.CircuitBootstrapping(state_bits[i]));
                        }
                    }
                    const auto cbs_end = std::chrono::steady_clock::now();
                    total_cbs_ms += DurationMs(cbs_start, cbs_end);
                    if (all_ctrl.size() != 16 * 8) {
                        OPENFHE_THROW(config_error, "CircuitBootstrappingBatch returned unexpected size");
                    }

                    std::array<std::array<RLWECiphertext, 4>, 4> lutRowsCols{};
                    const auto lut_start = std::chrono::steady_clock::now();
                    for (uint32_t col = 0; col < 4; ++col) {
                        for (uint32_t row = 0; row < 4; ++row) {
                            const uint32_t i = AesIndex(row, col);
                            std::array<RGSWCiphertext, 8> ctrl{};
                            for (uint32_t b = 0; b < 8; ++b) {
                                ctrl[b] = all_ctrl[i * 8 + b];
                            }
                            const uint32_t tableId = row;  // 0..3 => Td/Te
                            if (useGpuCMUX) {
                                lutRowsCols[row][col] = EvalLUT8x32_RLWE_GPU(
                                    *gpu_cc, rlweParams, gpu_ws, ctrl, use_ctr ? d_tables_enc[tableId] : d_tables_dec[tableId], basecc);
                            } else {
                                const std::vector<RLWECiphertext>* table = nullptr;
                                if (use_ctr) {
                                    if (tableId == 0)
                                        table = &te0_ct;
                                    else if (tableId == 1)
                                        table = &te1_ct;
                                    else if (tableId == 2)
                                        table = &te2_ct;
                                    else
                                        table = &te3_ct;
                                } else {
                                    if (tableId == 0)
                                        table = &td0_ct;
                                    else if (tableId == 1)
                                        table = &td1_ct;
                                    else if (tableId == 2)
                                        table = &td2_ct;
                                    else
                                        table = &td3_ct;
                                }
                                lutRowsCols[row][col] = EvalLUT8x32_RLWE_CPU(rlweParams, ctrl, *table, basecc, digitscc);
                            }
                        }
                    }

                    std::array<RLWECiphertext, 4> outWords{};
                    for (uint32_t col = 0; col < 4; ++col) {
                        RLWECiphertext outNoKey;
                        if (use_ctr) {
                            outNoKey = XorRLWE4(lutRowsCols[0][col], lutRowsCols[1][(col + 1) & 3u], lutRowsCols[2][(col + 2) & 3u],
                                                lutRowsCols[3][(col + 3) & 3u]);
                            outWords[col] = XorRLWE(outNoKey, km.rk_words[round][col]);
                        } else {
                            outNoKey = XorRLWE4(lutRowsCols[0][col], lutRowsCols[1][(col + 3) & 3u], lutRowsCols[2][(col + 2) & 3u],
                                                lutRowsCols[3][(col + 1) & 3u]);
                            outWords[col] = XorRLWE(outNoKey, km.rk_imc_words[round][col]);
                        }
                    }
                    const auto lut_end = std::chrono::steady_clock::now();
                    total_lut_ms += DurationMs(lut_start, lut_end);

                    const auto ex_start = std::chrono::steady_clock::now();
                    state_bits = ExtractStateBytesToLWETwitter(params, lweParamsKS, lweScheme, ksk_rn_to_n, outWords, hasGpuExtractKS ? &gpu_ksk : nullptr,
                                                              hasGpuExtractKS ? gpu_ks_ws.get() : nullptr);
                    const auto ex_end = std::chrono::steady_clock::now();
                    total_extract_ms += DurationMs(ex_start, ex_end);
                    if (g_transition_profile.enabled) {
                        g_transition_profile.gpu_idle_gaps += 1;
                    }
                };

                if (use_ctr) {
                    for (uint32_t round = 1; round <= 9; ++round) {
                        run_round_host(round);
                        if (round == 9) {
                            break;
                        }
                    }
                } else {
                    for (uint32_t round = 9; round >= 1; --round) {
                        run_round_host(round);
                        if (round == 1) {
                            break;
                        }
                    }
                }
            }

            // Final round: Td4_* (dec) or Te4_* (enc/CTR) + AddRoundKey0 (dec) or AddRoundKey10 (enc/CTR).
            std::vector<LWECiphertext> out_bits;
            if (useGpuResidentFull) {
                out_bits = DownloadLWECiphertextsFromGPU(d_state_lwe_a.get(), d_state_lwe_b.get(), lwe_n, 128,
                                                         params->GetLWEParams()->Getq(), gpu_ws.stream());
            } else if (useGpuResident) {
                if (!gpu_cc || !gpu_ks_ws || !d_state_words_c0.get() || !d_state_words_c1.get()) {
                    OPENFHE_THROW(config_error, "AES_GPU_RESIDENT: missing GPU workspaces (final)");
                }
                const auto cbs_start = std::chrono::steady_clock::now();
                const auto view = gpu_cc->gpu_CircuitBootstrappingBatchToDevice(params, cirbt_key, state_bits);
                if (timing_sync) {
                    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(view.stream));
                    const auto cbs_end = std::chrono::steady_clock::now();
                    total_cbs_ms += DurationMs(cbs_start, cbs_end);
                } else if (!use_single_stream) {
                    if (!cbs_ready_event.ev) {
                        OPENFHE_THROW(config_error, "AES_GPU_RESIDENT: missing CBS event (final)");
                    }
                    PHANTOM_CHECK_CUDA(cudaEventRecord(cbs_ready_event.ev, view.stream));
                    PHANTOM_CHECK_CUDA(cudaStreamWaitEvent(gpu_ws.stream(), cbs_ready_event.ev, 0));
                }

                if (view.batch != 128 || view.N != N || view.per_ciphertext_elems != gpu_ws.ctrl_per_bit_stride) {
                    OPENFHE_THROW(config_error, "AES_GPU_RESIDENT: unexpected CBS batch view shape (final)");
                }

                const BasicInteger Q = rlweParams->GetQ().ConvertToInt<BasicInteger>();
                const auto lut_start = std::chrono::steady_clock::now();
                PHANTOM_CHECK_CUDA(cudaMemsetAsync(d_state_words_c0.get(), 0, static_cast<size_t>(4) * N * sizeof(BasicInteger), gpu_ws.stream()));
                PHANTOM_CHECK_CUDA(cudaMemsetAsync(d_state_words_c1.get(), 0, static_cast<size_t>(4) * N * sizeof(BasicInteger), gpu_ws.stream()));

                const BasicInteger* d_all = reinterpret_cast<const BasicInteger*>(view.d_rgsw);
                const size_t stride = view.per_ciphertext_elems;
                const dim3 add_block(256);
                const dim3 add_grid((N + add_block.x - 1) / add_block.x);

                const bool use_lut_batch = (gpu_ws.lut_batch_max >= 4);
                if (use_lut_batch) {
                    const uint32_t lut_batch = 4;
                    const dim3 add_grid_batch((N + add_block.x - 1) / add_block.x, lut_batch);
                    for (uint32_t row = 0; row < 4; ++row) {
                        const uint32_t tableId = 4 + row;  // 4..7 => Td4_0..3
                        uint32_t idx[4] = {AesIndex(row, 0), AesIndex(row, 1), AesIndex(row, 2), AesIndex(row, 3)};
                        uint32_t word_map[4];
                        if (use_ctr) {
                            word_map[0] = static_cast<uint32_t>((0 + 4u - row) & 3u);
                            word_map[1] = static_cast<uint32_t>((1 + 4u - row) & 3u);
                            word_map[2] = static_cast<uint32_t>((2 + 4u - row) & 3u);
                            word_map[3] = static_cast<uint32_t>((3 + 4u - row) & 3u);
                        } else {
                            word_map[0] = static_cast<uint32_t>((0 + row) & 3u);
                            word_map[1] = static_cast<uint32_t>((1 + row) & 3u);
                            word_map[2] = static_cast<uint32_t>((2 + row) & 3u);
                            word_map[3] = static_cast<uint32_t>((3 + row) & 3u);
                        }
                        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(gpu_ws.d_lut_word_map.get(), word_map, sizeof(word_map), cudaMemcpyHostToDevice,
                                                           gpu_ws.stream()));
                        const auto lut_view = EvalLUT8x32_RLWE_GPU_DeviceCtrl_Batch(
                            *gpu_cc, rlweParams, gpu_ws, d_all, stride, idx, lut_batch,
                            use_ctr ? d_tables_enc[tableId] : d_tables_dec[tableId], basecc);
                        kernel_RLWE_AddInplace_Batch<<<add_grid_batch, add_block, 0, gpu_ws.stream()>>>(
                            d_state_words_c0.get(), d_state_words_c1.get(), lut_view.c0, lut_view.c1, gpu_ws.d_lut_word_map.get(), Q, N, lut_batch);
                        PHANTOM_CHECK_CUDA_LAST();
                    }
                } else {
                    for (uint32_t col = 0; col < 4; ++col) {
                        for (uint32_t row = 0; row < 4; ++row) {
                            const uint32_t i = AesIndex(row, col);
                            const BasicInteger* d_ctrl = d_all + static_cast<size_t>(i * 8) * stride;
                            const uint32_t tableId = 4 + row;  // 4..7 => Td4_0..3
                            const auto lut_ct = EvalLUT8x32_RLWE_GPU_DeviceCtrl(
                                *gpu_cc, rlweParams, gpu_ws, d_ctrl, use_ctr ? d_tables_enc[tableId] : d_tables_dec[tableId], basecc);
                            const uint32_t word = use_ctr ? static_cast<uint32_t>((col + 4u - row) & 3u) : static_cast<uint32_t>((col + row) & 3u);
                            kernel_RLWE_AddInplace<<<add_grid, add_block, 0, gpu_ws.stream()>>>(
                                d_state_words_c0.get() + static_cast<size_t>(word) * N,
                                d_state_words_c1.get() + static_cast<size_t>(word) * N,
                                lut_ct.c0, lut_ct.c1, Q, N);
                            PHANTOM_CHECK_CUDA_LAST();
                        }
                    }
                }

                const dim3 key_grid((N + add_block.x - 1) / add_block.x, 4);
                if (use_ctr) {
                    kernel_RLWE_AddWords4Inplace<<<key_grid, add_block, 0, gpu_ws.stream()>>>(
                        d_state_words_c0.get(), d_state_words_c1.get(), d_rk_enc_words[10].d_c0.get(), d_rk_enc_words[10].d_c1.get(), Q, N);
                } else {
                    kernel_RLWE_AddWords4Inplace<<<key_grid, add_block, 0, gpu_ws.stream()>>>(
                        d_state_words_c0.get(), d_state_words_c1.get(), d_rk0_words_dev.d_c0.get(), d_rk0_words_dev.d_c1.get(), Q, N);
                }
                PHANTOM_CHECK_CUDA_LAST();
                if (timing_sync) {
                    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(gpu_ws.stream()));
                    const auto lut_end = std::chrono::steady_clock::now();
                    total_lut_ms += DurationMs(lut_start, lut_end);
                }

                const auto ex_start = std::chrono::steady_clock::now();
                if (useGpuExtractModswitch) {
                    if (!d_state_lwe_a.get() || !d_state_lwe_b.get()) {
                        OPENFHE_THROW(config_error, "AES_GPU_RESIDENT: missing device LWE buffers for GPU ModSwitch (final)");
                    }
                    (void)ExtractStateBytesToLWEDevice(*gpu_cc, params, gpu_ksk, *gpu_ks_ws, d_state_words_c0.get(), d_state_words_c1.get(),
                                                       d_state_lwe_a.get(), d_state_lwe_b.get(), lwe_n);
                    if (timing_sync) {
                        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(gpu_ws.stream()));
                    }
                    out_bits = DownloadLWECiphertextsFromGPU(d_state_lwe_a.get(), d_state_lwe_b.get(), lwe_n, 128,
                                                             params->GetLWEParams()->Getq(), gpu_ws.stream());
                } else {
                    out_bits = ExtractStateBytesToLWETwitter_DeviceWords(*gpu_cc, params, lweParamsKS, lweScheme, gpu_ksk, *gpu_ks_ws,
                                                                        d_state_words_c0.get(), d_state_words_c1.get());
                }
                const auto ex_end = std::chrono::steady_clock::now();
                if (timing_sync) {
                    total_extract_ms += DurationMs(ex_start, ex_end);
                }
            } else {
                if (g_transition_profile.enabled) {
                    g_transition_profile.rounds += 1;
                }
                std::vector<RGSWCiphertext> all_ctrl;
                all_ctrl.reserve(16 * 8);
                const auto cbs_start = std::chrono::steady_clock::now();
                if (useGPU) {
                    if (useBatchCBS) {
                        if (g_transition_profile.enabled) {
                            g_transition_profile.cbs_groups += 1;
                            g_transition_profile.cbs_requests += state_bits.size();
                            g_transition_profile.host_device_boundaries += 2;  // LWE H2D + RGSW D2H.
                        }
                        all_ctrl = gpu_cc->gpu_CircuitBootstrappingBatch(params, cirbt_key, state_bits);
                    } else {
                        for (uint32_t i = 0; i < 16; ++i) {
                            std::vector<LWECiphertext> byteBits(8);
                            for (uint32_t b = 0; b < 8; ++b) {
                                byteBits[b] = state_bits[i * 8 + b];
                            }
                            if (g_transition_profile.enabled) {
                                g_transition_profile.cbs_groups += 1;
                                g_transition_profile.cbs_requests += byteBits.size();
                                g_transition_profile.host_device_boundaries += 2;  // byte LWE H2D + byte RGSW D2H.
                            }
                            auto byteCtrl = gpu_cc->gpu_CircuitBootstrappingBatch(params, cirbt_key, byteBits);
                            all_ctrl.insert(all_ctrl.end(), byteCtrl.begin(), byteCtrl.end());
                        }
                    }
                } else {
                    for (uint32_t i = 0; i < 16 * 8; ++i) {
                        all_ctrl.emplace_back(cc.CircuitBootstrapping(state_bits[i]));
                    }
                }
                const auto cbs_end = std::chrono::steady_clock::now();
                total_cbs_ms += DurationMs(cbs_start, cbs_end);

                std::array<std::array<RLWECiphertext, 4>, 4> lutRowsCols{};
                const auto lut_start = std::chrono::steady_clock::now();
                for (uint32_t col = 0; col < 4; ++col) {
                    for (uint32_t row = 0; row < 4; ++row) {
                        const uint32_t i = AesIndex(row, col);
                        std::array<RGSWCiphertext, 8> ctrl{};
                        for (uint32_t b = 0; b < 8; ++b) {
                            ctrl[b] = all_ctrl[i * 8 + b];
                        }
                        const uint32_t tableId = 4 + row;  // 4..7 => Td4_* / Te4_*
                        if (useGpuCMUX) {
                            lutRowsCols[row][col] = EvalLUT8x32_RLWE_GPU(
                                *gpu_cc, rlweParams, gpu_ws, ctrl, use_ctr ? d_tables_enc[tableId] : d_tables_dec[tableId], basecc);
                        } else {
                            const std::vector<RLWECiphertext>* table = nullptr;
                            if (use_ctr) {
                                if (row == 0)
                                    table = &te4_0_ct;
                                else if (row == 1)
                                    table = &te4_1_ct;
                                else if (row == 2)
                                    table = &te4_2_ct;
                                else
                                    table = &te4_3_ct;
                            } else {
                                if (row == 0)
                                    table = &td4_0_ct;
                                else if (row == 1)
                                    table = &td4_1_ct;
                                else if (row == 2)
                                    table = &td4_2_ct;
                                else
                                    table = &td4_3_ct;
                            }
                            lutRowsCols[row][col] = EvalLUT8x32_RLWE_CPU(rlweParams, ctrl, *table, basecc, digitscc);
                        }
                    }
                }

                std::array<RLWECiphertext, 4> outWords{};
                for (uint32_t col = 0; col < 4; ++col) {
                    RLWECiphertext outNoKey;
                    if (use_ctr) {
                        outNoKey = XorRLWE4(lutRowsCols[0][col], lutRowsCols[1][(col + 1) & 3u], lutRowsCols[2][(col + 2) & 3u],
                                             lutRowsCols[3][(col + 3) & 3u]);
                        outWords[col] = XorRLWE(outNoKey, km.rk_words[10][col]);
                    } else {
                        outNoKey = XorRLWE4(lutRowsCols[0][col], lutRowsCols[1][(col + 3) & 3u], lutRowsCols[2][(col + 2) & 3u],
                                             lutRowsCols[3][(col + 1) & 3u]);
                        outWords[col] = XorRLWE(outNoKey, km.rk0_words[col]);
                    }
                }
                const auto lut_end = std::chrono::steady_clock::now();
                total_lut_ms += DurationMs(lut_start, lut_end);

                const auto ex_start = std::chrono::steady_clock::now();
                out_bits = ExtractStateBytesToLWETwitter(params, lweParamsKS, lweScheme, ksk_rn_to_n, outWords, hasGpuExtractKS ? &gpu_ksk : nullptr,
                                                         hasGpuExtractKS ? gpu_ks_ws.get() : nullptr);
                const auto ex_end = std::chrono::steady_clock::now();
                total_extract_ms += DurationMs(ex_start, ex_end);
                if (g_transition_profile.enabled) {
                    g_transition_profile.gpu_idle_gaps += 1;
                }
            }

            if (use_ctr) {
                // In CTR, out_bits are the keystream bits; XOR with public ciphertext to obtain plaintext bits.
                if (!out_bits.empty()) {
                    const NativeInteger q = out_bits[0]->GetModulus();
                    const NativeInteger half = q >> 1;
                    for (uint32_t i = 0; i < 16; ++i) {
                        const uint8_t c = ciphertext[i];
                        for (uint32_t b = 0; b < 8; ++b) {
                            if ((c >> b) & 1u) {
                                lweScheme.EvalAddConstEq(out_bits[i * 8 + b], half);
                            }
                        }
                    }
                }
            }

            const auto dec_start = std::chrono::steady_clock::now();
            std::array<uint8_t, 16> got{};
            for (uint32_t i = 0; i < 16; ++i) {
                uint8_t v = 0;
                for (uint32_t b = 0; b < 8; ++b) {
                    LWEPlaintext bit = 0;
                    cc.Decrypt(sk, out_bits[i * 8 + b], &bit, /*p=*/2);
                    v |= static_cast<uint8_t>((bit & 1u) << b);
                }
                got[i] = v;
            }
            const auto dec_end = std::chrono::steady_clock::now();

            bool ok = true;
            if (verify) {
                for (uint32_t i = 0; i < 16; ++i) {
                    if (got[i] != plaintext[i]) {
                        ok = false;
                        break;
                    }
                }
            }

            if (timing_out) {
                timing_out->encrypt_ms = DurationMs(enc_start, enc_end);
                timing_out->addkey_ms = DurationMs(ark10_start, ark10_end);
                timing_out->cbs_ms = total_cbs_ms;
                timing_out->lut_ms = total_lut_ms;
                timing_out->extract_ms = total_extract_ms;
                timing_out->decrypt_ms = DurationMs(dec_start, dec_end);
            }

            return ok;
        };

        std::function<bool(const std::vector<std::array<uint8_t, 16>>&, const EncryptedAesKeyMaterial&, TimingSample*, bool, uint64_t, uint32_t*)>
            RunTranscipherBatch;
        RunTranscipherBatch = [&](const std::vector<std::array<uint8_t, 16>>& plaintexts, const EncryptedAesKeyMaterial& km,
                                  TimingSample* timing_out, bool verify, uint64_t block_idx_base, uint32_t* fail_blocks) -> bool {
            const uint32_t blocks = static_cast<uint32_t>(plaintexts.size());
            if (blocks == 0) {
                return true;
            }
            if (aes_super_batch_requested && useGpuResidentFull && use_ctr && blocks > gpu_full_blocks_capacity) {
                bool ok_all = true;
                TimingSample acc{};
                for (uint32_t off = 0; off < blocks; off += gpu_full_blocks_capacity) {
                    const uint32_t chunk = std::min<uint32_t>(gpu_full_blocks_capacity, blocks - off);
                    std::vector<std::array<uint8_t, 16>> sub(plaintexts.begin() + off, plaintexts.begin() + off + chunk);
                    TimingSample t{};
                    const bool ok = RunTranscipherBatch(sub, km, timing_out ? &t : nullptr, verify, block_idx_base + off, fail_blocks);
                    ok_all = ok_all && ok;
                    acc.encrypt_ms += t.encrypt_ms;
                    acc.addkey_ms += t.addkey_ms;
                    acc.cbs_ms += t.cbs_ms;
                    acc.lut_ms += t.lut_ms;
                    acc.extract_ms += t.extract_ms;
                    acc.decrypt_ms += t.decrypt_ms;
                }
                if (timing_out) {
                    *timing_out = acc;
                }
                return ok_all;
            }
            if (!aes_super_batch_requested || !useGpuResidentFull || !use_ctr || blocks == 1) {
                bool ok_all = true;
                TimingSample acc{};
                for (uint32_t b = 0; b < blocks; ++b) {
                    TimingSample t{};
                    const bool ok = RunTranscipherBlock(plaintexts[b], km, timing_out ? &t : nullptr, verify, block_idx_base + b);
                    ok_all = ok_all && ok;
                    if (verify && !ok && fail_blocks) {
                        ++(*fail_blocks);
                    }
                    if (timing_out) {
                        acc.encrypt_ms += t.encrypt_ms;
                        acc.addkey_ms += t.addkey_ms;
                        acc.cbs_ms += t.cbs_ms;
                        acc.lut_ms += t.lut_ms;
                        acc.extract_ms += t.extract_ms;
                        acc.decrypt_ms += t.decrypt_ms;
                    }
                }
                if (timing_out) {
                    *timing_out = acc;
                }
                return ok_all;
            }
            if (blocks > gpu_full_blocks_capacity) {
                OPENFHE_THROW(config_error, "AES_SUPER_BATCH: batch exceeds GPU_FULL workspace capacity");
            }
            if (!gpu_cc || !gpu_ks_ws || !d_state_words_c0.get() || !d_state_words_c1.get() ||
                !d_state_lwe_a.get() || !d_state_lwe_b.get()) {
                OPENFHE_THROW(config_error, "AES_SUPER_BATCH: missing GPU workspaces");
            }
            constexpr bool super_batch_device_chain = true;

            const uint32_t total_bits = blocks * 128u;
            std::vector<std::array<uint8_t, 16>> counter_blocks(blocks);
            std::vector<std::array<uint8_t, 16>> ciphertexts(blocks);
            std::vector<uint8_t> h_public_bits(static_cast<size_t>(total_bits));
            std::vector<LWECiphertext> state_bits;
            const bool useGpuInitState = aes_gpu_init_state_requested &&
                                         d_rk0_lwe_a_dev.get() && d_rk0_lwe_b_dev.get() && d_public_input_bits.get();
            if (!useGpuInitState) {
                state_bits.reserve(total_bits);
            }

            const auto enc_start = std::chrono::steady_clock::now();
            const NativeInteger lwe_half = params->GetLWEParams()->Getq() >> 1;
            for (uint32_t sidx = 0; sidx < blocks; ++sidx) {
                counter_blocks[sidx] = MakeCtrBlock(ctr_base, block_idx_base + sidx);
                const auto keystream = Aes128EncryptPlain(counter_blocks[sidx], km.roundKeys);
                for (uint32_t i = 0; i < 16; ++i) {
                    ciphertexts[sidx][i] = static_cast<uint8_t>(plaintexts[sidx][i] ^ keystream[i]);
                    const uint8_t x = counter_blocks[sidx][i];
                    for (uint32_t b = 0; b < 8; ++b) {
                        const uint32_t bit = (x >> b) & 1u;
                        const uint32_t idx = i * 8u + b;
                        h_public_bits[static_cast<size_t>(sidx) * 128u + idx] = static_cast<uint8_t>(bit);
                        if (!useGpuInitState) {
                            auto ct = std::make_shared<LWECiphertextImpl>(*km.rk0_bits[idx]);
                            ct->SetptModulus(2);
                            if (bit) {
                                lweScheme.EvalAddConstEq(ct, lwe_half);
                            }
                            state_bits.emplace_back(std::move(ct));
                        }
                    }
                }
            }
            if (useGpuInitState) {
                PHANTOM_CHECK_CUDA(cudaMemcpyAsync(d_public_input_bits.get(), h_public_bits.data(),
                                                   h_public_bits.size() * sizeof(uint8_t),
                                                   cudaMemcpyHostToDevice, gpu_ws.stream()));
                const BasicInteger q_lwe_u = params->GetLWEParams()->Getq().ConvertToInt<BasicInteger>();
                const BasicInteger half_u = lwe_half.ConvertToInt<BasicInteger>();
                const uint32_t out_len = lwe_n + 1u;
                const uint32_t total_init = total_bits * out_len;
                const dim3 init_block(256);
                const dim3 init_grid((total_init + init_block.x - 1) / init_block.x);
                kernel_InitLWEStateFromRoundKey<<<init_grid, init_block, 0, gpu_ws.stream()>>>(
                    d_state_lwe_a.get(), d_state_lwe_b.get(), d_rk0_lwe_a_dev.get(), d_rk0_lwe_b_dev.get(),
                    d_public_input_bits.get(), half_u, q_lwe_u, lwe_n, total_bits);
                PHANTOM_CHECK_CUDA_LAST();
                if (timing_sync) {
                    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(gpu_ws.stream()));
                }
            }
            const auto enc_end = std::chrono::steady_clock::now();
            const auto ark_start = std::chrono::steady_clock::now();
            const auto ark_end = std::chrono::steady_clock::now();

            double total_cbs_ms = 0.0;
            double total_lut_ms = 0.0;
            double total_extract_ms = 0.0;

            auto run_round_batch = [&](uint32_t round, bool final_round, bool input_on_device) {
                const auto cbs_start = std::chrono::steady_clock::now();
                const auto view = input_on_device
                                      ? gpu_cc->gpu_CircuitBootstrappingBatchToDeviceFromDeviceLWE(
                                            params, cirbt_key, d_state_lwe_a.get(), d_state_lwe_b.get(), total_bits, lwe_n)
                                      : gpu_cc->gpu_CircuitBootstrappingBatchToDevice(params, cirbt_key, state_bits);
                if (timing_sync) {
                    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(view.stream));
                    const auto cbs_end = std::chrono::steady_clock::now();
                    total_cbs_ms += DurationMs(cbs_start, cbs_end);
                }

                if (view.batch != total_bits || view.N != N || view.per_ciphertext_elems != gpu_ws.ctrl_per_bit_stride) {
                    OPENFHE_THROW(config_error, "AES_SUPER_BATCH: unexpected CBS batch view shape");
                }

                const BasicInteger* d_all = reinterpret_cast<const BasicInteger*>(view.d_rgsw);
                const size_t stride = view.per_ciphertext_elems;
                const dim3 add_block(256);
                const auto lut_start = std::chrono::steady_clock::now();
                const size_t total_ctrl_elems = static_cast<size_t>(total_bits) * stride;
                const BasicInteger* d_all_shoup = nullptr;
                if (gpu_ws.d_ctrl_all_shoup.get()) {
                    const dim3 shoup_grid((total_ctrl_elems + add_block.x - 1) / add_block.x);
                    kernel_ComputeShoupU64<<<shoup_grid, add_block, 0, gpu_ws.stream()>>>(
                        reinterpret_cast<uint64_t*>(gpu_ws.d_ctrl_all_shoup.get()),
                        reinterpret_cast<const uint64_t*>(d_all), static_cast<uint64_t>(rlweParams->GetQ().ConvertToInt<BasicInteger>()),
                        total_ctrl_elems);
                    PHANTOM_CHECK_CUDA_LAST();
                    d_all_shoup = gpu_ws.d_ctrl_all_shoup.get();
                }

                const BasicInteger Q = rlweParams->GetQ().ConvertToInt<BasicInteger>();
                PHANTOM_CHECK_CUDA(cudaMemsetAsync(d_state_words_c0.get(), 0,
                                                   static_cast<size_t>(blocks) * 4 * N * sizeof(BasicInteger), gpu_ws.stream()));
                PHANTOM_CHECK_CUDA(cudaMemsetAsync(d_state_words_c1.get(), 0,
                                                   static_cast<size_t>(blocks) * 4 * N * sizeof(BasicInteger), gpu_ws.stream()));

                const uint32_t lut_batch_cap = std::max<uint32_t>(1u, gpu_ws.lut_batch_max);
                const uint32_t rk_round = final_round ? 10u : round;
                std::vector<uint32_t> h_indices(lut_batch_cap);
                std::vector<uint32_t> h_word_map(lut_batch_cap);
                for (uint32_t row = 0; row < 4; ++row) {
                    const uint32_t tableId = final_round ? (4 + row) : row;
                    const bool fuse_key_this_row = lutFusedFinalAdd && lutFusedKeyAdd && row == 3u;
                    const BasicInteger* fused_key_c0 = fuse_key_this_row ? d_rk_enc_words[rk_round].d_c0.get() : nullptr;
                    const BasicInteger* fused_key_c1 = fuse_key_this_row ? d_rk_enc_words[rk_round].d_c1.get() : nullptr;
                    for (uint32_t start = 0; start < blocks * 4u; start += lut_batch_cap) {
                        const uint32_t chunk = std::min<uint32_t>(lut_batch_cap, blocks * 4u - start);
                        for (uint32_t j = 0; j < chunk; ++j) {
                            const uint32_t item = start + j;
                            const uint32_t block_id = item >> 2;
                            const uint32_t col = item & 3u;
                            h_indices[j] = block_id * 16u + AesIndex(row, col);
                            h_word_map[j] = block_id * 4u + static_cast<uint32_t>((col + 4u - row) & 3u);
                        }
                        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(gpu_ws.d_lut_word_map.get(), h_word_map.data(),
                                                           static_cast<size_t>(chunk) * sizeof(uint32_t),
                                                           cudaMemcpyHostToDevice, gpu_ws.stream()));
                        if (lutFusedFinalAdd) {
                            (void)EvalLUT8x32_RLWE_GPU_DeviceCtrl_Batch(
                                *gpu_cc, rlweParams, gpu_ws, d_all, stride, h_indices.data(), chunk,
                                d_tables_enc[tableId], basecc, d_all_shoup,
                                d_state_words_c0.get(), d_state_words_c1.get(), gpu_ws.d_lut_word_map.get(),
                                fused_key_c0, fused_key_c1);
                        } else {
                            const auto lut_view = EvalLUT8x32_RLWE_GPU_DeviceCtrl_Batch(
                                *gpu_cc, rlweParams, gpu_ws, d_all, stride, h_indices.data(), chunk,
                                d_tables_enc[tableId], basecc, d_all_shoup);
                            const dim3 add_grid_batch((N + add_block.x - 1) / add_block.x, chunk);
                            kernel_RLWE_AddInplace_Batch<<<add_grid_batch, add_block, 0, gpu_ws.stream()>>>(
                                d_state_words_c0.get(), d_state_words_c1.get(), lut_view.c0, lut_view.c1,
                                gpu_ws.d_lut_word_map.get(), Q, N, chunk);
                            PHANTOM_CHECK_CUDA_LAST();
                        }
                    }
                }

                if (!(lutFusedFinalAdd && lutFusedKeyAdd)) {
                    const dim3 key_grid((N + add_block.x - 1) / add_block.x, blocks * 4u);
                    kernel_RLWE_AddWords4Inplace_Blocks<<<key_grid, add_block, 0, gpu_ws.stream()>>>(
                        d_state_words_c0.get(), d_state_words_c1.get(),
                        d_rk_enc_words[rk_round].d_c0.get(), d_rk_enc_words[rk_round].d_c1.get(),
                        Q, N, blocks);
                    PHANTOM_CHECK_CUDA_LAST();
                }
                if (timing_sync) {
                    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(gpu_ws.stream()));
                    const auto lut_end = std::chrono::steady_clock::now();
                    total_lut_ms += DurationMs(lut_start, lut_end);
                }

                const auto ex_start = std::chrono::steady_clock::now();
                if (extractSuperBatchRequested) {
                    (void)ExtractStateBytesToLWEDeviceBatch(*gpu_cc, params, gpu_ksk, *gpu_ks_ws,
                                                            d_state_words_c0.get(), d_state_words_c1.get(),
                                                            d_state_lwe_a.get(), d_state_lwe_b.get(), lwe_n, blocks);
                } else {
                    for (uint32_t sidx = 0; sidx < blocks; ++sidx) {
                        BasicInteger* words_c0 = d_state_words_c0.get() + static_cast<size_t>(sidx) * 4 * N;
                        BasicInteger* words_c1 = d_state_words_c1.get() + static_cast<size_t>(sidx) * 4 * N;
                        BasicInteger* out_a = d_state_lwe_a.get() + static_cast<size_t>(sidx) * 128 * lwe_n;
                        BasicInteger* out_b = d_state_lwe_b.get() + static_cast<size_t>(sidx) * 128;
                        (void)ExtractStateBytesToLWEDevice(*gpu_cc, params, gpu_ksk, *gpu_ks_ws,
                                                           words_c0, words_c1, out_a, out_b, lwe_n);
                    }
                }
                if (timing_sync) {
                    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(gpu_ws.stream()));
                    const auto ex_end = std::chrono::steady_clock::now();
                    total_extract_ms += DurationMs(ex_start, ex_end);
                }
                if (!final_round && !super_batch_device_chain) {
                    state_bits = DownloadLWECiphertextsFromGPU(d_state_lwe_a.get(), d_state_lwe_b.get(), lwe_n, total_bits,
                                                               params->GetLWEParams()->Getq(), gpu_ws.stream());
                }
            };

            bool input_on_device = useGpuInitState;
            for (uint32_t round = 1; round <= 9; ++round) {
                run_round_batch(round, /*final_round=*/false, input_on_device);
                input_on_device = super_batch_device_chain;
            }
            run_round_batch(/*round=*/10, /*final_round=*/true, input_on_device);

            std::vector<LWECiphertext> out_bits =
                DownloadLWECiphertextsFromGPU(d_state_lwe_a.get(), d_state_lwe_b.get(), lwe_n, total_bits,
                                              params->GetLWEParams()->Getq(), gpu_ws.stream());

            const auto dec_start = std::chrono::steady_clock::now();
            bool ok_all = true;
            for (uint32_t sidx = 0; sidx < blocks; ++sidx) {
                const NativeInteger q = out_bits[static_cast<size_t>(sidx) * 128]->GetModulus();
                const NativeInteger half = q >> 1;
                for (uint32_t i = 0; i < 16; ++i) {
                    const uint8_t c = ciphertexts[sidx][i];
                    for (uint32_t b = 0; b < 8; ++b) {
                        if ((c >> b) & 1u) {
                            lweScheme.EvalAddConstEq(out_bits[static_cast<size_t>(sidx) * 128 + i * 8 + b], half);
                        }
                    }
                }

                std::array<uint8_t, 16> got{};
                for (uint32_t i = 0; i < 16; ++i) {
                    uint8_t v = 0;
                    for (uint32_t b = 0; b < 8; ++b) {
                        LWEPlaintext bit = 0;
                        cc.Decrypt(sk, out_bits[static_cast<size_t>(sidx) * 128 + i * 8 + b], &bit, /*p=*/2);
                        v |= static_cast<uint8_t>((bit & 1u) << b);
                    }
                    got[i] = v;
                }

                bool ok = true;
                if (verify) {
                    for (uint32_t i = 0; i < 16; ++i) {
                        if (got[i] != plaintexts[sidx][i]) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) {
                        if (fail_blocks) {
                            ++(*fail_blocks);
                        }
                    }
                }
                ok_all = ok_all && ok;
            }
            const auto dec_end = std::chrono::steady_clock::now();

            if (timing_out) {
                timing_out->encrypt_ms = DurationMs(enc_start, enc_end);
                timing_out->addkey_ms = DurationMs(ark_start, ark_end);
                timing_out->cbs_ms = total_cbs_ms;
                timing_out->lut_ms = total_lut_ms;
                timing_out->extract_ms = total_extract_ms;
                timing_out->decrypt_ms = DurationMs(dec_start, dec_end);
            }
            return ok_all;
        };

        if (run_nist) {
            uint32_t nist_fail = 0;
            uint32_t nist_total = 0;
            for (const auto& tv : kNistAes128) {
                EncryptedAesKeyMaterial km = BuildEncryptedKeyMaterial(cc, rlweParams, sk, sk2, tv.key);
                upload_gpu_keys(km);
                const auto ct = Aes128EncryptPlain(tv.pt, km.roundKeys);
                if (ct != tv.ct) {
                    std::cerr << "[NIST] AES implementation mismatch for test vector" << std::endl;
                }
                const bool ok = RunTranscipherBlock(tv.pt, km, nullptr, /*verify=*/true, /*block_idx=*/0);
                ++nist_total;
                if (!ok) {
                    ++nist_fail;
                }
            }
            std::cout << "[NIST] total=" << nist_total << " fail=" << nist_fail << "" << std::endl;
            keymat = BuildEncryptedKeyMaterial(cc, rlweParams, sk, sk2, base_key);
            upload_gpu_keys(keymat);
        }

        if (random_tests > 0) {
            uint32_t rand_fail = 0;
            for (uint32_t i = 0; i < random_tests; ++i) {
                const auto pt = random_block();
                const bool ok = RunTranscipherBlock(pt, keymat, nullptr, /*verify=*/true, static_cast<uint64_t>(i));
                if (!ok) {
                    ++rand_fail;
                }
            }
            std::cout << "[RANDOM] total=" << random_tests << " fail=" << rand_fail << "" << std::endl;
        }

        if (!run_nist && random_tests == 0 && !bench_enabled) {
            const auto pt = random_block();
            TimingSample timing{};
            const bool ok = RunTranscipherBlock(pt, keymat, &timing, /*verify=*/true, /*block_idx=*/0);
            if (!ok) {
                std::cerr << "Error: AES-128 transcipher mismatch." << std::endl;
                return 2;
            }
            std::cout << "[TC] AES-128 transciphering OK." << std::endl;
            std::cout << "[TIMING] mode=" << (useGPU ? "GPU" : "CPU") << " cbs_batch=" << (useBatchCBS ? 1 : 0) << " cmux_gpu="
                      << (useGpuCMUX ? 1 : 0) << " seed=" << seed << std::endl;
            if (useGpuResident && !timing_sync) {
                std::cout << "[TIMING] encrypt_ms=" << timing.encrypt_ms << " addkey10_ms=" << timing.addkey_ms
                          << " cbs_ms(total)=n/a lut_ms(total)=n/a extract_ms(total)=n/a (set AES_TIMING_SYNC=1 for GPU timings)" << std::endl;
            } else {
                std::cout << "[TIMING] encrypt_ms=" << timing.encrypt_ms << " addkey10_ms=" << timing.addkey_ms << " cbs_ms(total)="
                          << timing.cbs_ms << " lut_ms(total)=" << timing.lut_ms << " extract_ms(total)=" << timing.extract_ms << std::endl;
            }
            std::cout << "OK: AES-128 transciphering produced TLWE ciphertexts of plaintext block." << std::endl;
            return 0;
        }

        if (bench_enabled) {
            for (uint32_t batch : batch_list) {
                if (warmup > 0) {
                    for (uint32_t w = 0; w < warmup; ++w) {
                        if (aes_super_batch_requested && useGpuResidentFull && use_ctr) {
                            std::vector<std::array<uint8_t, 16>> pts(batch);
                            for (uint32_t b = 0; b < batch; ++b) {
                                pts[b] = random_block();
                            }
                            (void)RunTranscipherBatch(pts, keymat, nullptr, false, static_cast<uint64_t>(w) * batch, nullptr);
                        } else {
                            for (uint32_t b = 0; b < batch; ++b) {
                                const auto pt = random_block();
                                (void)RunTranscipherBlock(pt, keymat, nullptr, false, static_cast<uint64_t>(w) * batch + b);
                            }
                        }
                    }
                }
                std::vector<double> compute_samples;
                std::vector<double> e2e_samples;
                std::vector<double> wall_samples;
                std::vector<double> enc_samples;
                std::vector<double> addkey_samples;
                std::vector<double> cbs_samples;
                std::vector<double> lut_samples;
                std::vector<double> extract_samples;
                std::vector<double> homdec_samples;
                if (!repeat_env_set && repeat == 0) {
                    repeat = 10;
                }
                for (uint32_t r = 0; r < repeat; ++r) {
                    TimingSample acc{};
                    const auto wall_start = std::chrono::steady_clock::now();
                    if (aes_super_batch_requested && useGpuResidentFull && use_ctr) {
                        std::vector<std::array<uint8_t, 16>> pts(batch);
                        for (uint32_t b = 0; b < batch; ++b) {
                            pts[b] = random_block();
                        }
                        (void)RunTranscipherBatch(pts, keymat, &acc, false, static_cast<uint64_t>(r) * batch, nullptr);
                    } else {
                        for (uint32_t b = 0; b < batch; ++b) {
                            const auto pt = random_block();
                            TimingSample t{};
                            (void)RunTranscipherBlock(pt, keymat, &t, false, static_cast<uint64_t>(r) * batch + b);
                            acc.encrypt_ms += t.encrypt_ms;
                            acc.addkey_ms += t.addkey_ms;
                            acc.cbs_ms += t.cbs_ms;
                            acc.lut_ms += t.lut_ms;
                            acc.extract_ms += t.extract_ms;
                            acc.decrypt_ms += t.decrypt_ms;
                        }
                    }
                    const auto wall_end = std::chrono::steady_clock::now();
                    const double denom = static_cast<double>(batch);
                    const double wall_ms = DurationMs(wall_start, wall_end);
                    wall_samples.push_back(wall_ms / denom);
                    compute_samples.push_back(bench_async ? (wall_ms / denom) : (acc.ComputeMs() / denom));
                    e2e_samples.push_back(bench_async ? (wall_ms / denom) : (acc.E2eMs() / denom));
                    enc_samples.push_back(acc.encrypt_ms / denom);
                    addkey_samples.push_back(acc.addkey_ms / denom);
                    cbs_samples.push_back(acc.cbs_ms / denom);
                    lut_samples.push_back(acc.lut_ms / denom);
                    extract_samples.push_back(acc.extract_ms / denom);
                    homdec_samples.push_back((acc.addkey_ms + acc.cbs_ms + acc.lut_ms + acc.extract_ms) / denom);
                }
                std::cout << "[BENCH] batch=" << batch << " mode=" << (useGPU ? "GPU" : "CPU") << " cbs_batch=" << (useBatchCBS ? 1 : 0)
                          << " cmux_gpu=" << (useGpuCMUX ? 1 : 0) << "" << std::endl;
                if (bench_async) {
                    PrintLatencyStats("wall_ms", wall_samples);
                }
                PrintLatencyStats("compute_only_ms", compute_samples);
                PrintLatencyStats("e2e_ms", e2e_samples);
                if (timing_sync || !useGpuResident) {
                    std::cout << "[BENCH_TIMING] per-block breakdown_ms" << std::endl;
                    PrintLatencyStats("encrypt_ms", enc_samples);
                    PrintLatencyStats("addkey10_ms", addkey_samples);
                    PrintLatencyStats("cbs_ms(total)", cbs_samples);
                    PrintLatencyStats("lut_ms(total)", lut_samples);
                    PrintLatencyStats("extract_ms(total)", extract_samples);
                    PrintLatencyStats("homdec_ms(total)", homdec_samples);
                } else {
                    std::cout << "[BENCH_TIMING] per-block breakdown_ms unavailable (set AES_TIMING_SYNC=1 for GPU timings)" << std::endl;
                }
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
