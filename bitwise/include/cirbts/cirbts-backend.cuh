/*
 * =============================================================================
 * File: cirbts-backend.cuh
 * Purpose: Runtime backend selection for CirBTS GPU pipeline.
 * Key parameters:
 *   - CIRBTS_BACKEND: ntt|split_fft|ntt32_rns
 *   - CIRBTS_USE_SPLIT_FFT: legacy flag for split-FFT EvalAcc
 * Key points:
 *   - Defaults preserve existing behavior (NTT unless split-FFT is enabled).
 * =============================================================================
 */
#pragma once

#include <cstdint>

namespace phantom::bitwise {

enum class CirBTSBackend : uint8_t {
    kNTT = 0,
    kSplitFFT,
    kNTT32RNS,
};

CirBTSBackend ParseCirBTSBackend();
const char* CirBTSBackendName(CirBTSBackend backend);

}  // namespace phantom::bitwise
