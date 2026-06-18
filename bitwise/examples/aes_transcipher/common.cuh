// Common timing, parsing, and CUDA guard helpers for AES transciphering.

bool ParseU32(std::string_view s, uint32_t* out) {
    if (!out) {
        return false;
    }
    char* end = nullptr;
    const auto v = std::strtoul(std::string(s).c_str(), &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    *out = static_cast<uint32_t>(v);
    return true;
}

double DurationMs(const std::chrono::steady_clock::time_point& start, const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
}

struct TimingSample {
    double encrypt_ms = 0.0;
    double addkey_ms = 0.0;
    double cbs_ms = 0.0;
    double lut_ms = 0.0;
    double extract_ms = 0.0;
    double decrypt_ms = 0.0;

    double ComputeMs() const {
        return cbs_ms + lut_ms + extract_ms;
    }
    double E2eMs() const {
        return encrypt_ms + addkey_ms + ComputeMs() + decrypt_ms;
    }
};

struct TransitionProfileStats {
    bool enabled = false;
    double lut_gpu_ms = 0.0;
    double d2h_ms = 0.0;
    double se_cpu_ms = 0.0;
    double ks_cpu_ms = 0.0;
    double pack_host_ms = 0.0;
    double h2d_ms = 0.0;
    uint64_t rounds = 0;
    uint64_t lut_calls = 0;
    uint64_t d2h_events = 0;
    uint64_t h2d_events = 0;
    uint64_t cbs_groups = 0;
    uint64_t cbs_requests = 0;
    uint64_t host_device_boundaries = 0;
    uint64_t gpu_idle_gaps = 0;

    void Reset(bool on) {
        *this = TransitionProfileStats{};
        enabled = on;
    }

};

TransitionProfileStats g_transition_profile;

double Percentile(std::vector<double> v, double p) {
    if (v.empty()) {
        return 0.0;
    }
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    std::sort(v.begin(), v.end());
    const size_t idx = static_cast<size_t>(std::round(p * static_cast<double>(v.size() - 1)));
    return v[idx];
}

void PrintLatencyStats(const std::string& name, const std::vector<double>& samples) {
    if (samples.empty()) {
        std::cout << "[BENCH] " << name << ": no samples" << std::endl;
        return;
    }
    const double p50 = Percentile(samples, 0.50);
    const double p90 = Percentile(samples, 0.90);
    const double p99 = Percentile(samples, 0.99);
    std::cout << "[BENCH] " << name << " p50=" << p50 << "ms p90=" << p90 << "ms p99=" << p99 << "ms samples=" << samples.size()
              << std::endl;
}

struct CudaEventGuard {
    cudaEvent_t ev{nullptr};
    ~CudaEventGuard() {
        if (ev) {
            cudaEventDestroy(ev);
        }
    }
};
