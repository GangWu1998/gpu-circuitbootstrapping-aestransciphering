/*
 * =============================================================================
 * File: cirbts-backend.cu
 * Purpose: Parse CIRBTS_BACKEND and keep legacy behavior intact.
 * =============================================================================
 */
#include "cirbts/cirbts-backend.cuh"

#include <cstdlib>
#include <cstring>

namespace phantom::bitwise {

static CirBTSBackend parse_backend_from_env() {
    if (const char* v = std::getenv("CIRBTS_BACKEND"); v && *v) {
        if (std::strcmp(v, "ntt") == 0) {
            return CirBTSBackend::kNTT;
        }
        if (std::strcmp(v, "split_fft") == 0) {
            return CirBTSBackend::kSplitFFT;
        }
        if (std::strcmp(v, "ntt32_rns") == 0 || std::strcmp(v, "rns32_ntt") == 0) {
            return CirBTSBackend::kNTT32RNS;
        }
    }
    return CirBTSBackend::kNTT;
}

CirBTSBackend ParseCirBTSBackend() {
    const CirBTSBackend backend = parse_backend_from_env();
    if (backend != CirBTSBackend::kNTT) {
        return backend;
    }
    if (const char* env = std::getenv("CIRBTS_USE_SPLIT_FFT"); env && *env && std::strcmp(env, "0") != 0) {
        return CirBTSBackend::kSplitFFT;
    }
    return CirBTSBackend::kNTT;
}

const char* CirBTSBackendName(CirBTSBackend backend) {
    switch (backend) {
        case CirBTSBackend::kNTT:
            return "ntt";
        case CirBTSBackend::kSplitFFT:
            return "split_fft";
        case CirBTSBackend::kNTT32RNS:
            return "ntt32_rns";
    }
    return "unknown";
}

}  // namespace phantom::bitwise
