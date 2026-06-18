// GPU LUT workspace, kernels, and 8-to-32 RLWE LUT evaluation.


struct DeviceRLWETable {
    phantom::util::cuda_auto_ptr<BasicInteger> d_table_c0;
    phantom::util::cuda_auto_ptr<BasicInteger> d_table_c1;
    phantom::util::cuda_auto_ptr<BasicInteger> d_stage0_digits;
    phantom::util::cuda_auto_ptr<BasicInteger> d_stage0_fournode_digits;
    phantom::util::cuda_auto_ptr<BasicInteger> d_stage0_fournode_static_delta;
    phantom::util::cuda_auto_ptr<BasicInteger> d_stage1_basecorr_digits;
};

struct DeviceRLWEWords4 {
    phantom::util::cuda_auto_ptr<BasicInteger> d_c0;  // [4][N]
    phantom::util::cuda_auto_ptr<BasicInteger> d_c1;  // [4][N]
};

struct DeviceRLWECiphertextView {
    const BasicInteger* c0{};
    const BasicInteger* c1{};
};

struct DeviceRLWECiphertextBatchView {
    const BasicInteger* c0{};
    const BasicInteger* c1{};
    uint32_t batch{};
};

struct DeviceLWEBatchView {
    BasicInteger* d_a{};
    BasicInteger* d_b{};
    uint32_t n{};
    uint32_t batch{};
    uint32_t stride{};
};

inline bool UseLUTTwoStageFusion() {
    const char* v = std::getenv("AES_LUT_2STAGE_FUSION");
    return v && *v && *v != '0';
}

inline bool UseLUTSyncDelta() {
    const char* v = std::getenv("AES_LUT_SYNC_DELTA");
    return v && *v && *v != '0';
}

inline bool UseLUTFourNodeStage0() {
    const char* v = std::getenv("AES_LUT_FOUR_NODE_STAGE0");
    return v && *v && *v != '0';
}

inline bool UseLUTNextDigitDiff() {
    const char* v = std::getenv("AES_LUT_NEXT_DIGIT_DIFF");
    return v && *v && *v != '0';
}

inline bool UseLUTAllStaticDelta() {
    const char* v = std::getenv("AES_LUT_ALL_STATIC_DELTA");
    return v && *v && *v != '0';
}

inline uint32_t LUTStaticDeltaPolyCount() {
    return UseLUTAllStaticDelta() ? 254u : 128u;
}

inline bool UseLUTStage1StaticCorrection() {
    const char* v = std::getenv("AES_LUT_STAGE1_STATIC_CORRECTION");
    return v && *v && *v != '0';
}

inline uint32_t LUTStage1CorrectionPolyCount() {
    return 64u;
}

DeviceRLWETable UploadTableRLWEToGPU(const std::shared_ptr<RLWECryptoParams>& rlweParams, const std::vector<RLWECiphertext>& table_ct,
                                     cudaStream_t stream) {
    const uint32_t N = static_cast<uint32_t>(rlweParams->GetN());
    if (table_ct.size() != 256) {
        OPENFHE_THROW(config_error, "UploadTableRLWEToGPU: expected 256 entries");
    }

    std::vector<BasicInteger> h_c0(static_cast<size_t>(256) * N);
    std::vector<BasicInteger> h_c1(static_cast<size_t>(256) * N);
    for (uint32_t x = 0; x < 256; ++x) {
        const auto& elems = table_ct[x]->GetElements();
        if (elems.size() != 2) {
            OPENFHE_THROW(config_error, "UploadTableRLWEToGPU: expected RLWE with 2 polys");
        }
        const auto& c0 = elems[0];
        const auto& c1 = elems[1];
        if (c0.GetLength() != N || c1.GetLength() != N) {
            OPENFHE_THROW(config_error, "UploadTableRLWEToGPU: unexpected ring dimension");
        }
        for (uint32_t i = 0; i < N; ++i) {
            h_c0[static_cast<size_t>(x) * N + i] = c0[i].ConvertToInt<BasicInteger>();
            h_c1[static_cast<size_t>(x) * N + i] = c1[i].ConvertToInt<BasicInteger>();
        }
    }

    DeviceRLWETable out{};
    out.d_table_c0 = phantom::util::make_cuda_auto_ptr<BasicInteger>(h_c0.size(), stream);
    out.d_table_c1 = phantom::util::make_cuda_auto_ptr<BasicInteger>(h_c1.size(), stream);
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(out.d_table_c0.get(), h_c0.data(), h_c0.size() * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream));
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(out.d_table_c1.get(), h_c1.data(), h_c1.size() * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream));
    return out;
}

DeviceRLWEWords4 UploadWords4RLWEToGPU(const std::shared_ptr<RLWECryptoParams>& rlweParams, const std::array<RLWECiphertext, 4>& words,
                                      cudaStream_t stream) {
    const uint32_t N = static_cast<uint32_t>(rlweParams->GetN());
    std::vector<BasicInteger> h_c0(static_cast<size_t>(4) * N);
    std::vector<BasicInteger> h_c1(static_cast<size_t>(4) * N);
    for (uint32_t w = 0; w < 4; ++w) {
        const auto& elems = words[w]->GetElements();
        if (elems.size() != 2) {
            OPENFHE_THROW(config_error, "UploadWords4RLWEToGPU: expected RLWE with 2 polys");
        }
        const auto& c0 = elems[0];
        const auto& c1 = elems[1];
        if (c0.GetLength() != N || c1.GetLength() != N) {
            OPENFHE_THROW(config_error, "UploadWords4RLWEToGPU: unexpected ring dimension");
        }
        for (uint32_t i = 0; i < N; ++i) {
            h_c0[static_cast<size_t>(w) * N + i] = c0[i].ConvertToInt<BasicInteger>();
            h_c1[static_cast<size_t>(w) * N + i] = c1[i].ConvertToInt<BasicInteger>();
        }
    }

    DeviceRLWEWords4 out{};
    out.d_c0 = phantom::util::make_cuda_auto_ptr<BasicInteger>(h_c0.size(), stream);
    out.d_c1 = phantom::util::make_cuda_auto_ptr<BasicInteger>(h_c1.size(), stream);
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(out.d_c0.get(), h_c0.data(), h_c0.size() * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream));
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(out.d_c1.get(), h_c1.data(), h_c1.size() * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream));
    return out;
}

void UploadLWECiphertextsToGPU(const std::vector<LWECiphertext>& cts, BasicInteger* d_a, BasicInteger* d_b, uint32_t n,
                               cudaStream_t stream) {
    const size_t batch = cts.size();
    if (batch == 0) {
        return;
    }
    std::vector<BasicInteger> h_a(batch * static_cast<size_t>(n));
    std::vector<BasicInteger> h_b(batch);
    for (size_t i = 0; i < batch; ++i) {
        const auto& ct = cts[i];
        const auto& a = ct->GetA();
        if (a.GetLength() != n) {
            OPENFHE_THROW(config_error, "UploadLWECiphertextsToGPU: unexpected LWE dimension");
        }
        for (uint32_t k = 0; k < n; ++k) {
            h_a[i * static_cast<size_t>(n) + k] = a[k].ConvertToInt<BasicInteger>();
        }
        h_b[i] = ct->GetB().ConvertToInt<BasicInteger>();
    }
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(d_a, h_a.data(), h_a.size() * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream));
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(d_b, h_b.data(), h_b.size() * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream));
}

std::vector<LWECiphertext> DownloadLWECiphertextsFromGPU(const BasicInteger* d_a, const BasicInteger* d_b, uint32_t n, uint32_t batch,
                                                         const NativeInteger& q, cudaStream_t stream) {
    std::vector<LWECiphertext> out(batch);
    if (batch == 0) {
        return out;
    }
    std::vector<BasicInteger> h_a(batch * static_cast<size_t>(n));
    std::vector<BasicInteger> h_b(batch);
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(h_a.data(), d_a, h_a.size() * sizeof(BasicInteger), cudaMemcpyDeviceToHost, stream));
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(h_b.data(), d_b, h_b.size() * sizeof(BasicInteger), cudaMemcpyDeviceToHost, stream));
    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(stream));

    for (uint32_t i = 0; i < batch; ++i) {
        NativeVector a(n, q);
        for (uint32_t k = 0; k < n; ++k) {
            a[k] = NativeInteger(h_a[static_cast<size_t>(i) * n + k]);
        }
        auto ct = std::make_shared<LWECiphertextImpl>(std::move(a), NativeInteger(h_b[i]));
        ct->SetptModulus(2);
        out[i] = std::move(ct);
    }
    return out;
}

struct Lut8x32GpuWorkspace {
    phantom::util::cuda_stream_wrapper stream_wrapper{};
    cudaStream_t external_stream{};
    bool disable_graph{false};

    struct LutGraphEntry {
        const BasicInteger* table_c0{};
        const BasicInteger* table_c1{};
        cudaGraphExec_t exec{};
    };
    struct LutBatchGraphEntry {
        const BasicInteger* table_c0{};
        const BasicInteger* table_c1{};
        uint32_t batch{};
        bool final_add{};
        const BasicInteger* add_out_c0{};
        const BasicInteger* add_out_c1{};
        const BasicInteger* key_c0{};
        const BasicInteger* key_c1{};
        cudaGraphExec_t exec{};
    };

    std::array<LutGraphEntry, 8> lut_graph_cache{};
    uint32_t lut_graph_count{0};
    bool lut_graph_failed{false};
    std::array<LutBatchGraphEntry, 64> lut_batch_graph_cache{};
    uint32_t lut_batch_graph_count{0};
    bool lut_batch_graph_failed{false};

    size_t ctrl_per_bit_stride{};
    phantom::util::cuda_auto_ptr<BasicInteger> d_ctrl_gsw;
    phantom::util::cuda_auto_ptr<BasicInteger> d_ctrl_gsw_shoup;

    phantom::util::cuda_auto_ptr<BasicInteger> d_stage0_c0;
    phantom::util::cuda_auto_ptr<BasicInteger> d_stage0_c1;
    phantom::util::cuda_auto_ptr<BasicInteger> d_stage1_c0;
    phantom::util::cuda_auto_ptr<BasicInteger> d_stage1_c1;

    phantom::util::cuda_auto_ptr<BasicInteger> d_delta;
    phantom::util::cuda_auto_ptr<BasicInteger> d_digits;

    uint32_t lut_batch_max{1};
    size_t ctrl_batch_stride{};
    phantom::util::cuda_auto_ptr<uint32_t> d_lut_indices;
    phantom::util::cuda_auto_ptr<uint32_t> d_lut_word_map;
    phantom::util::cuda_auto_ptr<BasicInteger> d_ctrl_all_shoup;

    uint32_t digitscc{};
    uint32_t N{};

    ~Lut8x32GpuWorkspace() {
        for (auto& e : lut_graph_cache) {
            if (e.exec) {
                cudaGraphExecDestroy(e.exec);
                e.exec = nullptr;
            }
        }
        for (auto& e : lut_batch_graph_cache) {
            if (e.exec) {
                cudaGraphExecDestroy(e.exec);
                e.exec = nullptr;
            }
        }
    }

    [[nodiscard]] cudaStream_t stream() const {
        return external_stream ? external_stream : stream_wrapper.get_stream();
    }

    void SetStream(cudaStream_t s) {
        external_stream = s;
    }
};

void UploadRGSWControlsToGPU(Lut8x32GpuWorkspace& ws, const std::array<RGSWCiphertext, 8>& ctrl_bits_lsb_to_msb, BasicInteger Q) {
    const uint32_t rows = ws.digitscc * 2;
    const uint32_t N = ws.N;
    const size_t perBitElems = static_cast<size_t>(rows) * 2 * N;
    const size_t totalElems = perBitElems * 8;

    std::vector<BasicInteger> h_gsw(totalElems);
    std::vector<BasicInteger> h_gsw_shoup(totalElems);

    for (uint32_t b = 0; b < 8; ++b) {
        const auto& gsw = ctrl_bits_lsb_to_msb[b];
        const auto& elems = gsw->GetElements();
        if (elems.size() != rows) {
            OPENFHE_THROW(config_error, "UploadRGSWControlsToGPU: unexpected RGSW row count");
        }
        for (uint32_t r = 0; r < rows; ++r) {
            if (elems[r].size() != 2) {
                OPENFHE_THROW(config_error, "UploadRGSWControlsToGPU: expected 2 columns per row");
            }
            for (uint32_t c = 0; c < 2; ++c) {
                const auto& poly = elems[r][c];
                if (poly.GetLength() != N) {
                    OPENFHE_THROW(config_error, "UploadRGSWControlsToGPU: unexpected RGSW ring dimension");
                }
                const size_t base = static_cast<size_t>(b) * perBitElems + (static_cast<size_t>(r) * 2 + c) * N;
                std::memcpy(h_gsw.data() + base, &poly.GetValues()[0], N * sizeof(BasicInteger));
                for (uint32_t i = 0; i < N; ++i) {
                    const BasicInteger v = h_gsw[base + i];
                    h_gsw_shoup[base + i] = static_cast<BasicInteger>(phantom::arith::compute_shoup(v, Q));
                }
            }
        }
    }

    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(ws.d_ctrl_gsw.get(), h_gsw.data(), h_gsw.size() * sizeof(BasicInteger), cudaMemcpyHostToDevice, ws.stream()));
    PHANTOM_CHECK_CUDA(
        cudaMemcpyAsync(ws.d_ctrl_gsw_shoup.get(), h_gsw_shoup.data(), h_gsw_shoup.size() * sizeof(BasicInteger), cudaMemcpyHostToDevice, ws.stream()));
    if (g_transition_profile.enabled) {
        g_transition_profile.host_device_boundaries += 1;
    }
}

__device__ __forceinline__ uint64_t aes_compute_shoup_u64(uint64_t operand, uint64_t modulus) {
    unsigned __int128 tmp = static_cast<unsigned __int128>(operand) << 64;
    return static_cast<uint64_t>(tmp / modulus);
}

__global__ void kernel_ComputeShoupU64(uint64_t* __restrict__ out, const uint64_t* __restrict__ in, uint64_t modulus, size_t n) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    out[tid] = aes_compute_shoup_u64(in[tid], modulus);
}

void UploadRGSWControlsToGPU_Device(Lut8x32GpuWorkspace& ws, const BasicInteger* d_ctrl_bits_lsb_to_msb, BasicInteger Q) {
    const size_t totalElems = ws.ctrl_per_bit_stride * 8;
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(ws.d_ctrl_gsw.get(), d_ctrl_bits_lsb_to_msb, totalElems * sizeof(BasicInteger),
                                       cudaMemcpyDeviceToDevice, ws.stream()));

    const dim3 block(256);
    const dim3 grid((totalElems + block.x - 1) / block.x);
    kernel_ComputeShoupU64<<<grid, block, 0, ws.stream()>>>(reinterpret_cast<uint64_t*>(ws.d_ctrl_gsw_shoup.get()),
                                                           reinterpret_cast<const uint64_t*>(ws.d_ctrl_gsw.get()),
                                                           static_cast<uint64_t>(Q), totalElems);
    PHANTOM_CHECK_CUDA_LAST();
}

__global__ void kernel_GatherCtrlBits(BasicInteger* __restrict__ out, const BasicInteger* __restrict__ in, const uint32_t* __restrict__ idx,
                                      size_t stride, uint32_t batch) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(batch) * 8 * stride;
    if (tid >= total) {
        return;
    }
    const size_t t = tid / stride;
    const size_t k = tid - t * stride;
    const uint32_t bit = static_cast<uint32_t>(t / batch);
    const uint32_t b = static_cast<uint32_t>(t - static_cast<size_t>(bit) * batch);
    const uint32_t byte = idx[b];
    const size_t src = (static_cast<size_t>(byte) * 8 + bit) * stride + k;
    out[tid] = in[src];
}

void UploadRGSWControlsToGPU_Device_Batch(Lut8x32GpuWorkspace& ws, const BasicInteger* d_ctrl_all, const BasicInteger* d_ctrl_all_shoup,
                                          const uint32_t* h_indices, uint32_t batch, BasicInteger Q, size_t stride) {
    if (batch == 0 || batch > ws.lut_batch_max) {
        OPENFHE_THROW(config_error, "UploadRGSWControlsToGPU_Device_Batch: batch exceeds workspace capacity");
    }
    if (!ws.d_lut_indices.get()) {
        OPENFHE_THROW(config_error, "UploadRGSWControlsToGPU_Device_Batch: index buffer not initialized");
    }
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(ws.d_lut_indices.get(), h_indices, static_cast<size_t>(batch) * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                       ws.stream()));

    const size_t totalElems = static_cast<size_t>(batch) * 8 * stride;
    const dim3 block(256);
    const dim3 grid((totalElems + block.x - 1) / block.x);
    kernel_GatherCtrlBits<<<grid, block, 0, ws.stream()>>>(ws.d_ctrl_gsw.get(), d_ctrl_all, ws.d_lut_indices.get(), stride, batch);
    PHANTOM_CHECK_CUDA_LAST();

    if (d_ctrl_all_shoup) {
        kernel_GatherCtrlBits<<<grid, block, 0, ws.stream()>>>(ws.d_ctrl_gsw_shoup.get(), d_ctrl_all_shoup, ws.d_lut_indices.get(), stride, batch);
        PHANTOM_CHECK_CUDA_LAST();
    } else {
        kernel_ComputeShoupU64<<<grid, block, 0, ws.stream()>>>(reinterpret_cast<uint64_t*>(ws.d_ctrl_gsw_shoup.get()),
                                                                 reinterpret_cast<const uint64_t*>(ws.d_ctrl_gsw.get()),
                                                                 static_cast<uint64_t>(Q), totalElems);
        PHANTOM_CHECK_CUDA_LAST();
    }
}

__device__ __forceinline__ uint64_t aes_addmod_u64(uint64_t a, uint64_t b, uint64_t mod) {
    unsigned __int128 sum = static_cast<unsigned __int128>(a) + b;
    sum -= (sum >= mod) ? mod : 0;
    return static_cast<uint64_t>(sum);
}

__global__ void kernel_RLWE_AddInplace(BasicInteger* __restrict__ out_c0, BasicInteger* __restrict__ out_c1,
                                       const BasicInteger* __restrict__ in_c0, const BasicInteger* __restrict__ in_c1,
                                       BasicInteger Q, uint32_t N) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= N) {
        return;
    }
    out_c0[coeff] = aes_addmod_u64(out_c0[coeff], in_c0[coeff], Q);
    out_c1[coeff] = aes_addmod_u64(out_c1[coeff], in_c1[coeff], Q);
}

__global__ void kernel_RLWE_AddInplace_Batch(BasicInteger* __restrict__ out_c0, BasicInteger* __restrict__ out_c1,
                                            const BasicInteger* __restrict__ in_c0, const BasicInteger* __restrict__ in_c1,
                                            const uint32_t* __restrict__ word_map, BasicInteger Q, uint32_t N, uint32_t batch) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t b = static_cast<uint32_t>(blockIdx.y);
    if (coeff >= N || b >= batch) {
        return;
    }
    const uint32_t word = word_map[b];
    const size_t out_base = static_cast<size_t>(word) * N + coeff;
    const size_t in_base = static_cast<size_t>(b) * N + coeff;
    out_c0[out_base] = aes_addmod_u64(out_c0[out_base], in_c0[in_base], Q);
    out_c1[out_base] = aes_addmod_u64(out_c1[out_base], in_c1[in_base], Q);
}

__global__ void kernel_RLWE_AddWords4Inplace(BasicInteger* __restrict__ out_c0, BasicInteger* __restrict__ out_c1,
                                            const BasicInteger* __restrict__ in_c0, const BasicInteger* __restrict__ in_c1,
                                            BasicInteger Q, uint32_t N) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t word = static_cast<uint32_t>(blockIdx.y);
    if (word >= 4 || coeff >= N) {
        return;
    }
    const size_t base = static_cast<size_t>(word) * N + coeff;
    out_c0[base] = aes_addmod_u64(out_c0[base], in_c0[base], Q);
    out_c1[base] = aes_addmod_u64(out_c1[base], in_c1[base], Q);
}

__global__ void kernel_RLWE_AddWords4Inplace_Blocks(BasicInteger* __restrict__ out_c0, BasicInteger* __restrict__ out_c1,
                                                   const BasicInteger* __restrict__ in_c0, const BasicInteger* __restrict__ in_c1,
                                                   BasicInteger Q, uint32_t N, uint32_t blocks) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t word_global = static_cast<uint32_t>(blockIdx.y);
    if (coeff >= N || word_global >= blocks * 4u) {
        return;
    }
    const uint32_t word = word_global & 3u;
    const size_t out_base = static_cast<size_t>(word_global) * N + coeff;
    const size_t key_base = static_cast<size_t>(word) * N + coeff;
    out_c0[out_base] = aes_addmod_u64(out_c0[out_base], in_c0[key_base], Q);
    out_c1[out_base] = aes_addmod_u64(out_c1[out_base], in_c1[key_base], Q);
}

__global__ void kernel_InitLWEStateFromRoundKey(BasicInteger* __restrict__ out_a, BasicInteger* __restrict__ out_b,
                                                const BasicInteger* __restrict__ rk_a, const BasicInteger* __restrict__ rk_b,
                                                const uint8_t* __restrict__ public_bits,
                                                BasicInteger half, BasicInteger q, uint32_t n, uint32_t total_bits) {
    const uint32_t out_len = n + 1u;
    const uint32_t tid = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t total = total_bits * out_len;
    if (tid >= total) {
        return;
    }
    const uint32_t t = tid / out_len;
    const uint32_t k = tid - t * out_len;
    const uint32_t bit_idx = t & 127u;
    if (k == n) {
        BasicInteger v = rk_b[bit_idx];
        if (public_bits[t]) {
            v = aes_addmod_u64(v, half, q);
        }
        out_b[t] = v;
    } else {
        out_a[static_cast<size_t>(t) * n + k] = rk_a[static_cast<size_t>(bit_idx) * n + k];
    }
}

__global__ void kernel_U16ToLWE(BasicInteger* __restrict__ out_a, BasicInteger* __restrict__ out_b,
                                const uint16_t* __restrict__ in, uint32_t n, uint32_t batch,
                                uint64_t scale, uint32_t stride) {
    const uint32_t out_len = n + 1u;
    const uint32_t idx = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t total = batch * out_len;
    if (idx >= total) {
        return;
    }
    const uint32_t t = idx / out_len;
    const uint32_t k = idx - t * out_len;
    const uint64_t v = static_cast<uint64_t>(in[idx]);
    const BasicInteger val = static_cast<BasicInteger>(v * scale);
    if (k == n) {
        out_b[t] = val;
    } else {
        out_a[static_cast<size_t>(t) * stride + k] = val;
    }
}

__device__ __forceinline__ uint64_t u128_div_u32_round(uint64_t hi, uint64_t lo, uint32_t d) {
    const uint64_t add = static_cast<uint64_t>(d >> 1);
    const uint64_t lo_prev = lo;
    lo += add;
    if (lo < lo_prev) {
        hi += 1u;
    }

    uint32_t w0 = static_cast<uint32_t>(lo);
    uint32_t w1 = static_cast<uint32_t>(lo >> 32);
    uint32_t w2 = static_cast<uint32_t>(hi);
    uint32_t w3 = static_cast<uint32_t>(hi >> 32);

    uint64_t r = 0;
    uint32_t qwords[4] = {0, 0, 0, 0};
    for (int i = 3; i >= 0; --i) {
        const uint64_t cur = (r << 32) | static_cast<uint64_t>(i == 3 ? w3 : (i == 2 ? w2 : (i == 1 ? w1 : w0)));
        const uint32_t q = static_cast<uint32_t>(cur / d);
        r = cur - static_cast<uint64_t>(q) * d;
        qwords[i] = q;
    }
    return (static_cast<uint64_t>(qwords[1]) << 32) | qwords[0];
}

__global__ void kernel_U16ToLWE_ModSwitch(BasicInteger* __restrict__ out_a, BasicInteger* __restrict__ out_b,
                                          const uint16_t* __restrict__ in, uint32_t n, uint32_t batch,
                                          uint32_t qks, uint64_t q, uint32_t stride) {
    const uint32_t out_len = n + 1u;
    const uint32_t idx = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t total = batch * out_len;
    if (idx >= total) {
        return;
    }
    const uint32_t t = idx / out_len;
    const uint32_t k = idx - t * out_len;
    const uint64_t v = static_cast<uint64_t>(in[idx]);
    const uint64_t lo = v * q;
    const uint64_t hi = __umul64hi(v, q);
    const uint64_t scaled = u128_div_u32_round(hi, lo, qks);
    const BasicInteger val = static_cast<BasicInteger>(scaled);
    if (k == n) {
        out_b[t] = val;
    } else {
        out_a[static_cast<size_t>(t) * stride + k] = val;
    }
}

__global__ void kernel_LUT_CMUX_Delta(BasicInteger* d_delta, const BasicInteger* d_in_c0, const BasicInteger* d_in_c1, const DModulus* modulus,
                                      uint32_t N, uint32_t inCount) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t pair = static_cast<uint32_t>(blockIdx.y);
    const uint32_t outCount = inCount >> 1;
    if (coeff >= N || pair >= outCount) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const uint32_t ct0 = pair << 1;
    const uint32_t ct1 = ct0 + 1;
    const size_t base0 = static_cast<size_t>(ct0) * N + coeff;
    const size_t base1 = static_cast<size_t>(ct1) * N + coeff;

    d_delta[base0] = phantom::arith::sub_uint64_uint64_mod(d_in_c0[base1], d_in_c0[base0], Q);
    d_delta[base1] = phantom::arith::sub_uint64_uint64_mod(d_in_c1[base1], d_in_c1[base0], Q);
}

__global__ void kernel_LUT_CMUX_ExternalProduct(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_digits, const BasicInteger* d_gsw,
                                               const BasicInteger* d_gsw_shoup, const BasicInteger* d_in_c0, const BasicInteger* d_in_c1,
                                               const DModulus* modulus, uint32_t base_digits, uint32_t N, uint32_t inCount) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t pair = static_cast<uint32_t>(blockIdx.y);
    const uint32_t outCount = inCount >> 1;
    if (coeff >= N || pair >= outCount) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const size_t numLUT = static_cast<size_t>(outCount) * 2;
    const size_t digit_stride = numLUT * N;

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;

    const uint32_t lut_base = pair * 2;
    for (uint32_t d = 0; d < base_digits; ++d) {
        const size_t plane = static_cast<size_t>(d) * digit_stride;
        const size_t lut0 = static_cast<size_t>(lut_base + 0) * N + coeff;
        const size_t lut1 = static_cast<size_t>(lut_base + 1) * N + coeff;

        const BasicInteger digit0 = d_digits[plane + lut0];
        const BasicInteger digit1 = d_digits[plane + lut1];

        const uint32_t row0 = (d << 1) | 0u;
        const uint32_t row1 = (d << 1) | 1u;
        const size_t key0_0 = (static_cast<size_t>(row0) * 2 + 0) * N + coeff;
        const size_t key0_1 = key0_0 + N;
        const size_t key1_0 = (static_cast<size_t>(row1) * 2 + 0) * N + coeff;
        const size_t key1_1 = key1_0 + N;

        const BasicInteger g00 = d_gsw[key0_0];
        const BasicInteger g01 = d_gsw[key0_1];
        const BasicInteger g10 = d_gsw[key1_0];
        const BasicInteger g11 = d_gsw[key1_1];

        const BasicInteger g00s = d_gsw_shoup[key0_0];
        const BasicInteger g01s = d_gsw_shoup[key0_1];
        const BasicInteger g10s = d_gsw_shoup[key1_0];
        const BasicInteger g11s = d_gsw_shoup[key1_1];

        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit0, g00, g00s, Q), Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit0, g01, g01s, Q), Q);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit1, g10, g10s, Q), Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit1, g11, g11s, Q), Q);
    }

    const uint32_t ct0 = pair << 1;
    const size_t in0 = static_cast<size_t>(ct0) * N + coeff;
    const size_t out = static_cast<size_t>(pair) * N + coeff;

    d_out_c0[out] = phantom::arith::add_uint64_uint64_mod(d_in_c0[in0], sum0, Q);
    d_out_c1[out] = phantom::arith::add_uint64_uint64_mod(d_in_c1[in0], sum1, Q);
}

__global__ void kernel_LUT_CMUX_Delta_Batch(BasicInteger* d_delta, const BasicInteger* d_in_c0, const BasicInteger* d_in_c1, const DModulus* modulus,
                                            uint32_t N, uint32_t inCount, uint32_t batch) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t outCount = inCount >> 1;
    const uint32_t pair_global = static_cast<uint32_t>(blockIdx.y);
    if (coeff >= N || pair_global >= outCount * batch) {
        return;
    }

    const uint32_t batch_idx = pair_global / outCount;
    const uint32_t pair = pair_global - batch_idx * outCount;
    const BasicInteger Q = modulus[0].value();
    const uint32_t ct0 = pair << 1;
    const uint32_t ct1 = ct0 + 1;
    const size_t base0 = (static_cast<size_t>(batch_idx) * inCount + ct0) * N + coeff;
    const size_t base1 = (static_cast<size_t>(batch_idx) * inCount + ct1) * N + coeff;

    d_delta[base0] = phantom::arith::sub_uint64_uint64_mod(d_in_c0[base1], d_in_c0[base0], Q);
    d_delta[base1] = phantom::arith::sub_uint64_uint64_mod(d_in_c1[base1], d_in_c1[base0], Q);
}

__global__ void kernel_LUT_CMUX_Delta_TableBatch(BasicInteger* d_delta, const BasicInteger* d_table_c0, const BasicInteger* d_table_c1,
                                                 const DModulus* modulus, uint32_t N, uint32_t inCount, uint32_t batch) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t outCount = inCount >> 1;
    const uint32_t pair_global = static_cast<uint32_t>(blockIdx.y);
    if (coeff >= N || pair_global >= outCount * batch) {
        return;
    }

    const uint32_t batch_idx = pair_global / outCount;
    const uint32_t pair = pair_global - batch_idx * outCount;
    const BasicInteger Q = modulus[0].value();
    const uint32_t ct0 = pair << 1;
    const uint32_t ct1 = ct0 + 1;
    const size_t tbl0 = static_cast<size_t>(ct0) * N + coeff;
    const size_t tbl1 = static_cast<size_t>(ct1) * N + coeff;
    const size_t base0 = (static_cast<size_t>(batch_idx) * inCount + ct0) * N + coeff;
    const size_t base1 = (static_cast<size_t>(batch_idx) * inCount + ct1) * N + coeff;

    d_delta[base0] = phantom::arith::sub_uint64_uint64_mod(d_table_c0[tbl1], d_table_c0[tbl0], Q);
    d_delta[base1] = phantom::arith::sub_uint64_uint64_mod(d_table_c1[tbl1], d_table_c1[tbl0], Q);
}

__global__ void kernel_LUT_CMUX_PackStage0FourNodeDigits(BasicInteger* d_fournode_digits, const BasicInteger* d_stage0_digits,
                                                         const DModulus* modulus, uint32_t digitscc, uint32_t N) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t poly = static_cast<uint32_t>(blockIdx.y);
    constexpr uint32_t kStage0InCount = 256;
    constexpr uint32_t kFourNodeInCount = 256;
    if (coeff >= N || poly >= digitscc * kFourNodeInCount) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const uint32_t digit = poly / kFourNodeInCount;
    const uint32_t slot = poly - digit * kFourNodeInCount;
    const uint32_t group = slot >> 2;
    const uint32_t term = (slot >> 1) & 1u;
    const uint32_t component = slot & 1u;

    const size_t first_slot = static_cast<size_t>(digit) * kStage0InCount + group * 4u + component;
    const size_t out = static_cast<size_t>(poly) * N + coeff;
    if (term == 0u) {
        d_fournode_digits[out] = d_stage0_digits[first_slot * N + coeff];
    } else {
        const size_t second_slot = first_slot + 2u;
        d_fournode_digits[out] =
            phantom::arith::sub_uint64_uint64_mod(d_stage0_digits[second_slot * N + coeff], d_stage0_digits[first_slot * N + coeff], Q);
    }
}

__global__ void kernel_LUT_CMUX_PackStage1BaseCorrectionDigits(BasicInteger* d_corr_digits,
                                                               const BasicInteger* d_stage0_digits,
                                                               const DModulus* modulus, uint32_t digitscc,
                                                               uint32_t N) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t poly = static_cast<uint32_t>(blockIdx.y);
    constexpr uint32_t kStage0InCount = 256;
    constexpr uint32_t kStage1CorrCount = 64;
    if (coeff >= N || poly >= digitscc * kStage1CorrCount) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const uint32_t digit = poly / kStage1CorrCount;
    const uint32_t slot = poly - digit * kStage1CorrCount;
    const uint32_t group = slot >> 1;
    const uint32_t component = slot & 1u;

    const size_t first_pair = static_cast<size_t>(digit) * kStage0InCount + group * 8u + component;
    const size_t second_pair = first_pair + 4u;
    const size_t out = static_cast<size_t>(poly) * N + coeff;
    d_corr_digits[out] =
        phantom::arith::sub_uint64_uint64_mod(d_stage0_digits[second_pair * N + coeff],
                                              d_stage0_digits[first_pair * N + coeff], Q);
}

__global__ void kernel_LUT_CMUX_PackStageStaticDelta(BasicInteger* d_static_delta, const BasicInteger* d_table_c0,
                                                     const BasicInteger* d_table_c1, const DModulus* modulus, uint32_t N,
                                                     uint32_t levels) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t poly = static_cast<uint32_t>(blockIdx.y);
    if (coeff >= N || levels == 0u) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const uint32_t component = poly & 1u;
    uint32_t group_linear = poly >> 1;
    uint32_t level = 0;
    uint32_t groups = 64;
    while (level + 1u < levels && group_linear >= groups) {
        group_linear -= groups;
        ++level;
        groups >>= 1;
    }
    if (level >= levels || group_linear >= groups) {
        return;
    }

    const uint32_t stride = 4u << level;
    const uint32_t half = 2u << level;
    const uint32_t t0 = group_linear * stride;
    const uint32_t t1 = t0 + half;
    const size_t i0 = static_cast<size_t>(t0) * N + coeff;
    const size_t i1 = static_cast<size_t>(t1) * N + coeff;
    const size_t out = static_cast<size_t>(poly) * N + coeff;

    if (component == 0u) {
        d_static_delta[out] = phantom::arith::sub_uint64_uint64_mod(d_table_c0[i1], d_table_c0[i0], Q);
    } else {
        d_static_delta[out] = phantom::arith::sub_uint64_uint64_mod(d_table_c1[i1], d_table_c1[i0], Q);
    }
}

__global__ void kernel_LUT_CMUX_ExternalProduct_Batch(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_digits, const BasicInteger* d_gsw,
                                                     const BasicInteger* d_gsw_shoup, const BasicInteger* d_in_c0, const BasicInteger* d_in_c1,
                                                     const DModulus* modulus, uint32_t base_digits, uint32_t N, uint32_t inCount, uint32_t batch,
                                                     size_t ctrl_stride) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t outCount = inCount >> 1;
    const uint32_t pair_global = static_cast<uint32_t>(blockIdx.y);
    if (coeff >= N || pair_global >= outCount * batch) {
        return;
    }

    const uint32_t batch_idx = pair_global / outCount;
    const uint32_t pair = pair_global - batch_idx * outCount;
    const BasicInteger Q = modulus[0].value();
    const size_t numLUT = static_cast<size_t>(inCount) * batch;
    const size_t digit_stride = numLUT * N;

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;

    const uint32_t lut_base = batch_idx * inCount + pair * 2;
    for (uint32_t d = 0; d < base_digits; ++d) {
        const size_t plane = static_cast<size_t>(d) * digit_stride;
        const size_t lut0 = static_cast<size_t>(lut_base + 0) * N + coeff;
        const size_t lut1 = static_cast<size_t>(lut_base + 1) * N + coeff;

        const BasicInteger digit0 = d_digits[plane + lut0];
        const BasicInteger digit1 = d_digits[plane + lut1];

        const uint32_t row0 = (d << 1) | 0u;
        const uint32_t row1 = (d << 1) | 1u;
        const size_t key0_0 = (static_cast<size_t>(row0) * 2 + 0) * N + coeff;
        const size_t key0_1 = key0_0 + N;
        const size_t key1_0 = (static_cast<size_t>(row1) * 2 + 0) * N + coeff;
        const size_t key1_1 = key1_0 + N;

        const size_t gsw_base = static_cast<size_t>(batch_idx) * ctrl_stride;
        const BasicInteger g00 = d_gsw[gsw_base + key0_0];
        const BasicInteger g01 = d_gsw[gsw_base + key0_1];
        const BasicInteger g10 = d_gsw[gsw_base + key1_0];
        const BasicInteger g11 = d_gsw[gsw_base + key1_1];

        const BasicInteger g00s = d_gsw_shoup[gsw_base + key0_0];
        const BasicInteger g01s = d_gsw_shoup[gsw_base + key0_1];
        const BasicInteger g10s = d_gsw_shoup[gsw_base + key1_0];
        const BasicInteger g11s = d_gsw_shoup[gsw_base + key1_1];

        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit0, g00, g00s, Q), Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit0, g01, g01s, Q), Q);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit1, g10, g10s, Q), Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit1, g11, g11s, Q), Q);
    }

    const uint32_t ct0 = pair << 1;
    const size_t in0 = (static_cast<size_t>(batch_idx) * inCount + ct0) * N + coeff;
    const size_t out = (static_cast<size_t>(batch_idx) * outCount + pair) * N + coeff;

    d_out_c0[out] = phantom::arith::add_uint64_uint64_mod(d_in_c0[in0], sum0, Q);
    d_out_c1[out] = phantom::arith::add_uint64_uint64_mod(d_in_c1[in0], sum1, Q);
}

__global__ void kernel_LUT_CMUX_ExternalProduct_TableBatch(BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_digits,
                                                           const BasicInteger* d_gsw, const BasicInteger* d_gsw_shoup, const BasicInteger* d_table_c0,
                                                           const BasicInteger* d_table_c1, const DModulus* modulus, uint32_t base_digits, uint32_t N,
                                                           uint32_t inCount, uint32_t batch, size_t ctrl_stride) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t outCount = inCount >> 1;
    const uint32_t pair_global = static_cast<uint32_t>(blockIdx.y);
    if (coeff >= N || pair_global >= outCount * batch) {
        return;
    }

    const uint32_t batch_idx = pair_global / outCount;
    const uint32_t pair = pair_global - batch_idx * outCount;
    const BasicInteger Q = modulus[0].value();
    const size_t numLUT = static_cast<size_t>(inCount) * batch;
    const size_t digit_stride = numLUT * N;

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;

    const uint32_t lut_base = batch_idx * inCount + pair * 2;
    for (uint32_t d = 0; d < base_digits; ++d) {
        const size_t plane = static_cast<size_t>(d) * digit_stride;
        const size_t lut0 = static_cast<size_t>(lut_base + 0) * N + coeff;
        const size_t lut1 = static_cast<size_t>(lut_base + 1) * N + coeff;

        const BasicInteger digit0 = d_digits[plane + lut0];
        const BasicInteger digit1 = d_digits[plane + lut1];

        const uint32_t row0 = (d << 1) | 0u;
        const uint32_t row1 = (d << 1) | 1u;
        const size_t key0_0 = (static_cast<size_t>(row0) * 2 + 0) * N + coeff;
        const size_t key0_1 = key0_0 + N;
        const size_t key1_0 = (static_cast<size_t>(row1) * 2 + 0) * N + coeff;
        const size_t key1_1 = key1_0 + N;

        const size_t gsw_base = static_cast<size_t>(batch_idx) * ctrl_stride;
        const BasicInteger g00 = d_gsw[gsw_base + key0_0];
        const BasicInteger g01 = d_gsw[gsw_base + key0_1];
        const BasicInteger g10 = d_gsw[gsw_base + key1_0];
        const BasicInteger g11 = d_gsw[gsw_base + key1_1];

        const BasicInteger g00s = d_gsw_shoup[gsw_base + key0_0];
        const BasicInteger g01s = d_gsw_shoup[gsw_base + key0_1];
        const BasicInteger g10s = d_gsw_shoup[gsw_base + key1_0];
        const BasicInteger g11s = d_gsw_shoup[gsw_base + key1_1];

        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit0, g00, g00s, Q), Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit0, g01, g01s, Q), Q);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit1, g10, g10s, Q), Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit1, g11, g11s, Q), Q);
    }

    const uint32_t ct0 = pair << 1;
    const size_t in0 = static_cast<size_t>(ct0) * N + coeff;
    const size_t out = (static_cast<size_t>(batch_idx) * outCount + pair) * N + coeff;

    d_out_c0[out] = phantom::arith::add_uint64_uint64_mod(d_table_c0[in0], sum0, Q);
    d_out_c1[out] = phantom::arith::add_uint64_uint64_mod(d_table_c1[in0], sum1, Q);
}

__global__ void kernel_LUT_CMUX_ExternalProduct_TableBatch_PrecompStage0(
    BasicInteger* d_out_c0, BasicInteger* d_out_c1, const BasicInteger* d_stage0_digits, const BasicInteger* d_gsw,
    const BasicInteger* d_gsw_shoup, const BasicInteger* d_table_c0, const BasicInteger* d_table_c1, const DModulus* modulus,
    uint32_t base_digits, uint32_t N, uint32_t inCount, uint32_t batch, size_t ctrl_stride) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t outCount = inCount >> 1;
    const uint32_t pair_global = static_cast<uint32_t>(blockIdx.y);
    if (coeff >= N || pair_global >= outCount * batch) {
        return;
    }

    const uint32_t batch_idx = pair_global / outCount;
    const uint32_t pair = pair_global - batch_idx * outCount;
    const BasicInteger Q = modulus[0].value();
    const size_t digit_stride = static_cast<size_t>(inCount) * N;

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;

    const uint32_t lut_base = pair * 2;
    for (uint32_t d = 0; d < base_digits; ++d) {
        const size_t plane = static_cast<size_t>(d) * digit_stride;
        const size_t lut0 = static_cast<size_t>(lut_base + 0) * N + coeff;
        const size_t lut1 = static_cast<size_t>(lut_base + 1) * N + coeff;

        const BasicInteger digit0 = d_stage0_digits[plane + lut0];
        const BasicInteger digit1 = d_stage0_digits[plane + lut1];

        const uint32_t row0 = (d << 1) | 0u;
        const uint32_t row1 = (d << 1) | 1u;
        const size_t key0_0 = (static_cast<size_t>(row0) * 2 + 0) * N + coeff;
        const size_t key0_1 = key0_0 + N;
        const size_t key1_0 = (static_cast<size_t>(row1) * 2 + 0) * N + coeff;
        const size_t key1_1 = key1_0 + N;

        const size_t gsw_base = static_cast<size_t>(batch_idx) * ctrl_stride;
        const BasicInteger g00 = d_gsw[gsw_base + key0_0];
        const BasicInteger g01 = d_gsw[gsw_base + key0_1];
        const BasicInteger g10 = d_gsw[gsw_base + key1_0];
        const BasicInteger g11 = d_gsw[gsw_base + key1_1];

        const BasicInteger g00s = d_gsw_shoup[gsw_base + key0_0];
        const BasicInteger g01s = d_gsw_shoup[gsw_base + key0_1];
        const BasicInteger g10s = d_gsw_shoup[gsw_base + key1_0];
        const BasicInteger g11s = d_gsw_shoup[gsw_base + key1_1];

        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit0, g00, g00s, Q), Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit0, g01, g01s, Q), Q);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit1, g10, g10s, Q), Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit1, g11, g11s, Q), Q);
    }

    const uint32_t ct0 = pair << 1;
    const size_t in0 = static_cast<size_t>(ct0) * N + coeff;
    const size_t out = (static_cast<size_t>(batch_idx) * outCount + pair) * N + coeff;

    d_out_c0[out] = phantom::arith::add_uint64_uint64_mod(d_table_c0[in0], sum0, Q);
    d_out_c1[out] = phantom::arith::add_uint64_uint64_mod(d_table_c1[in0], sum1, Q);
}

__global__ void kernel_LUT_CMUX_Stage0Stage1Delta_Precomp_Batch(
    BasicInteger* __restrict__ d_stage0_c0, BasicInteger* __restrict__ d_stage0_c1,
    BasicInteger* __restrict__ d_stage1_delta, const BasicInteger* __restrict__ d_stage0_digits,
    const BasicInteger* __restrict__ d_gsw0, const BasicInteger* __restrict__ d_gsw0_shoup,
    const BasicInteger* __restrict__ d_table_c0, const BasicInteger* __restrict__ d_table_c1,
    const DModulus* __restrict__ modulus, uint32_t base_digits, uint32_t N, uint32_t batch, size_t ctrl_stride) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t pair_global = static_cast<uint32_t>(blockIdx.y);
    constexpr uint32_t kStage1OutCount = 64;
    if (coeff >= N || pair_global >= kStage1OutCount * batch) {
        return;
    }

    const uint32_t batch_idx = pair_global / kStage1OutCount;
    const uint32_t pair = pair_global - batch_idx * kStage1OutCount;
    const BasicInteger Q = modulus[0].value();
    constexpr uint32_t kStage0InCount = 256;
    constexpr uint32_t kStage1InCount = 128;
    const size_t digit_stride = static_cast<size_t>(kStage0InCount) * N;
    const size_t gsw_base = static_cast<size_t>(batch_idx) * ctrl_stride;

    BasicInteger out0_c0 = 0;
    BasicInteger out0_c1 = 0;
    BasicInteger out1_c0 = 0;
    BasicInteger out1_c1 = 0;

    for (uint32_t child = 0; child < 2; ++child) {
        const uint32_t stage0_pair = pair * 2u + child;
        const uint32_t lut_base = stage0_pair * 2u;
        BasicInteger sum0 = 0;
        BasicInteger sum1 = 0;

        for (uint32_t d = 0; d < base_digits; ++d) {
            const size_t plane = static_cast<size_t>(d) * digit_stride;
            const size_t lut0 = static_cast<size_t>(lut_base + 0u) * N + coeff;
            const size_t lut1 = static_cast<size_t>(lut_base + 1u) * N + coeff;

            const BasicInteger digit0 = d_stage0_digits[plane + lut0];
            const BasicInteger digit1 = d_stage0_digits[plane + lut1];

            const uint32_t row0 = (d << 1) | 0u;
            const uint32_t row1 = (d << 1) | 1u;
            const size_t key0_0 = (static_cast<size_t>(row0) * 2u + 0u) * N + coeff;
            const size_t key0_1 = key0_0 + N;
            const size_t key1_0 = (static_cast<size_t>(row1) * 2u + 0u) * N + coeff;
            const size_t key1_1 = key1_0 + N;

            const BasicInteger g00 = d_gsw0[gsw_base + key0_0];
            const BasicInteger g01 = d_gsw0[gsw_base + key0_1];
            const BasicInteger g10 = d_gsw0[gsw_base + key1_0];
            const BasicInteger g11 = d_gsw0[gsw_base + key1_1];

            const BasicInteger g00s = d_gsw0_shoup[gsw_base + key0_0];
            const BasicInteger g01s = d_gsw0_shoup[gsw_base + key0_1];
            const BasicInteger g10s = d_gsw0_shoup[gsw_base + key1_0];
            const BasicInteger g11s = d_gsw0_shoup[gsw_base + key1_1];

            sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit0, g00, g00s, Q), Q);
            sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit0, g01, g01s, Q), Q);
            sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit1, g10, g10s, Q), Q);
            sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit1, g11, g11s, Q), Q);
        }

        const size_t in0 = static_cast<size_t>(lut_base) * N + coeff;
        BasicInteger v0 = phantom::arith::add_uint64_uint64_mod(d_table_c0[in0], sum0, Q);
        BasicInteger v1 = phantom::arith::add_uint64_uint64_mod(d_table_c1[in0], sum1, Q);
        if (child == 0) {
            out0_c0 = v0;
            out0_c1 = v1;
        } else {
            out1_c0 = v0;
            out1_c1 = v1;
        }
    }

    const size_t stage0_base = (static_cast<size_t>(batch_idx) * kStage1InCount + pair * 2u) * N + coeff;
    d_stage0_c0[stage0_base] = out0_c0;
    d_stage0_c1[stage0_base] = out0_c1;

    const size_t delta_c0 = (static_cast<size_t>(batch_idx) * kStage1InCount + pair * 2u) * N + coeff;
    const size_t delta_c1 = delta_c0 + N;
    d_stage1_delta[delta_c0] = phantom::arith::sub_uint64_uint64_mod(out1_c0, out0_c0, Q);
    d_stage1_delta[delta_c1] = phantom::arith::sub_uint64_uint64_mod(out1_c1, out0_c1, Q);
}

__global__ void kernel_LUT_CMUX_Stage0Stage1Delta_FourNode_Precomp_Batch(
    BasicInteger* __restrict__ d_stage0_c0, BasicInteger* __restrict__ d_stage0_c1,
    BasicInteger* __restrict__ d_stage1_delta, const BasicInteger* __restrict__ d_stage0_fournode_digits,
    const BasicInteger* __restrict__ d_stage0_fournode_static_delta,
    const BasicInteger* __restrict__ d_gsw0, const BasicInteger* __restrict__ d_gsw0_shoup,
    const BasicInteger* __restrict__ d_table_c0, const BasicInteger* __restrict__ d_table_c1,
    const DModulus* __restrict__ modulus, uint32_t base_digits, uint32_t N, uint32_t batch, size_t ctrl_stride) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t pair_global = static_cast<uint32_t>(blockIdx.y);
    constexpr uint32_t kStage1OutCount = 64;
    if (coeff >= N || pair_global >= kStage1OutCount * batch) {
        return;
    }

    const uint32_t batch_idx = pair_global / kStage1OutCount;
    const uint32_t pair = pair_global - batch_idx * kStage1OutCount;
    const BasicInteger Q = modulus[0].value();
    constexpr uint32_t kFourNodeInCount = 256;
    constexpr uint32_t kStage1InCount = 128;
    const size_t digit_stride = static_cast<size_t>(kFourNodeInCount) * N;
    const size_t gsw_base = static_cast<size_t>(batch_idx) * ctrl_stride;

    BasicInteger base_sum0 = 0;
    BasicInteger base_sum1 = 0;
    BasicInteger delta_sum0 = 0;
    BasicInteger delta_sum1 = 0;

    for (uint32_t d = 0; d < base_digits; ++d) {
        const size_t plane = static_cast<size_t>(d) * digit_stride;

        const size_t group_base = static_cast<size_t>(pair * 4u) * N + coeff;
        const size_t first_digit0 = group_base + 0u * N;
        const size_t first_digit1 = group_base + 1u * N;
        const size_t second_digit0 = group_base + 2u * N;
        const size_t second_digit1 = group_base + 3u * N;

        const BasicInteger bdigit0 = d_stage0_fournode_digits[plane + first_digit0];
        const BasicInteger bdigit1 = d_stage0_fournode_digits[plane + first_digit1];
        const BasicInteger ddigit0 = d_stage0_fournode_digits[plane + second_digit0];
        const BasicInteger ddigit1 = d_stage0_fournode_digits[plane + second_digit1];

        const uint32_t row0 = (d << 1) | 0u;
        const uint32_t row1 = (d << 1) | 1u;
        const size_t key0_0 = (static_cast<size_t>(row0) * 2u + 0u) * N + coeff;
        const size_t key0_1 = key0_0 + N;
        const size_t key1_0 = (static_cast<size_t>(row1) * 2u + 0u) * N + coeff;
        const size_t key1_1 = key1_0 + N;

        const BasicInteger g00 = d_gsw0[gsw_base + key0_0];
        const BasicInteger g01 = d_gsw0[gsw_base + key0_1];
        const BasicInteger g10 = d_gsw0[gsw_base + key1_0];
        const BasicInteger g11 = d_gsw0[gsw_base + key1_1];

        const BasicInteger g00s = d_gsw0_shoup[gsw_base + key0_0];
        const BasicInteger g01s = d_gsw0_shoup[gsw_base + key0_1];
        const BasicInteger g10s = d_gsw0_shoup[gsw_base + key1_0];
        const BasicInteger g11s = d_gsw0_shoup[gsw_base + key1_1];

        base_sum0 = phantom::arith::add_uint64_uint64_mod(base_sum0, phantom::arith::multiply_and_reduce_shoup(bdigit0, g00, g00s, Q), Q);
        base_sum1 = phantom::arith::add_uint64_uint64_mod(base_sum1, phantom::arith::multiply_and_reduce_shoup(bdigit0, g01, g01s, Q), Q);
        base_sum0 = phantom::arith::add_uint64_uint64_mod(base_sum0, phantom::arith::multiply_and_reduce_shoup(bdigit1, g10, g10s, Q), Q);
        base_sum1 = phantom::arith::add_uint64_uint64_mod(base_sum1, phantom::arith::multiply_and_reduce_shoup(bdigit1, g11, g11s, Q), Q);

        delta_sum0 = phantom::arith::add_uint64_uint64_mod(delta_sum0, phantom::arith::multiply_and_reduce_shoup(ddigit0, g00, g00s, Q), Q);
        delta_sum1 = phantom::arith::add_uint64_uint64_mod(delta_sum1, phantom::arith::multiply_and_reduce_shoup(ddigit0, g01, g01s, Q), Q);
        delta_sum0 = phantom::arith::add_uint64_uint64_mod(delta_sum0, phantom::arith::multiply_and_reduce_shoup(ddigit1, g10, g10s, Q), Q);
        delta_sum1 = phantom::arith::add_uint64_uint64_mod(delta_sum1, phantom::arith::multiply_and_reduce_shoup(ddigit1, g11, g11s, Q), Q);
    }

    const uint32_t a = pair * 4u;
    const size_t ai = static_cast<size_t>(a) * N + coeff;

    const BasicInteger base0 = phantom::arith::add_uint64_uint64_mod(d_table_c0[ai], base_sum0, Q);
    const BasicInteger base1 = phantom::arith::add_uint64_uint64_mod(d_table_c1[ai], base_sum1, Q);
    const size_t static_delta = static_cast<size_t>(pair * 2u) * N + coeff;
    const BasicInteger ca0 = d_stage0_fournode_static_delta[static_delta];
    const BasicInteger ca1 = d_stage0_fournode_static_delta[static_delta + N];

    const size_t stage0_base = (static_cast<size_t>(batch_idx) * kStage1InCount + pair * 2u) * N + coeff;
    d_stage0_c0[stage0_base] = base0;
    d_stage0_c1[stage0_base] = base1;

    d_stage1_delta[stage0_base] = phantom::arith::add_uint64_uint64_mod(ca0, delta_sum0, Q);
    d_stage1_delta[stage0_base + N] = phantom::arith::add_uint64_uint64_mod(ca1, delta_sum1, Q);
}

__global__ void kernel_LUT_CMUX_ExternalProduct_NextDelta_Batch(
    BasicInteger* __restrict__ d_next_base_c0, BasicInteger* __restrict__ d_next_base_c1,
    BasicInteger* __restrict__ d_next_delta, const BasicInteger* __restrict__ d_digits,
    const BasicInteger* __restrict__ d_gsw, const BasicInteger* __restrict__ d_gsw_shoup,
    const BasicInteger* __restrict__ d_in_c0, const BasicInteger* __restrict__ d_in_c1,
    const DModulus* __restrict__ modulus, uint32_t base_digits, uint32_t N, uint32_t inCount, uint32_t batch,
    size_t ctrl_stride) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t pair_global = static_cast<uint32_t>(blockIdx.y);
    const uint32_t outCount = inCount >> 1;
    const uint32_t nextPairCount = outCount >> 1;
    if (coeff >= N || pair_global >= nextPairCount * batch) {
        return;
    }

    const uint32_t batch_idx = pair_global / nextPairCount;
    const uint32_t next_pair = pair_global - batch_idx * nextPairCount;
    const BasicInteger Q = modulus[0].value();
    const size_t numLUT = static_cast<size_t>(inCount) * batch;
    const size_t digit_stride = numLUT * N;
    const size_t gsw_base = static_cast<size_t>(batch_idx) * ctrl_stride;

    BasicInteger out0_c0 = 0;
    BasicInteger out0_c1 = 0;
    BasicInteger out1_c0 = 0;
    BasicInteger out1_c1 = 0;

    for (uint32_t child = 0; child < 2; ++child) {
        const uint32_t stage_pair = next_pair * 2u + child;
        const uint32_t lut_base_u32 = stage_pair * 2u;
        const size_t lut_base = static_cast<size_t>(batch_idx) * inCount + lut_base_u32;

        BasicInteger sum0 = 0;
        BasicInteger sum1 = 0;
        for (uint32_t d = 0; d < base_digits; ++d) {
            const size_t plane = static_cast<size_t>(d) * digit_stride;
            const size_t lut0 = (lut_base + 0u) * N + coeff;
            const size_t lut1 = (lut_base + 1u) * N + coeff;

            const BasicInteger digit0 = d_digits[plane + lut0];
            const BasicInteger digit1 = d_digits[plane + lut1];

            const uint32_t row0 = (d << 1) | 0u;
            const uint32_t row1 = (d << 1) | 1u;
            const size_t key0_0 = (static_cast<size_t>(row0) * 2u + 0u) * N + coeff;
            const size_t key0_1 = key0_0 + N;
            const size_t key1_0 = (static_cast<size_t>(row1) * 2u + 0u) * N + coeff;
            const size_t key1_1 = key1_0 + N;

            const BasicInteger g00 = d_gsw[gsw_base + key0_0];
            const BasicInteger g01 = d_gsw[gsw_base + key0_1];
            const BasicInteger g10 = d_gsw[gsw_base + key1_0];
            const BasicInteger g11 = d_gsw[gsw_base + key1_1];

            const BasicInteger g00s = d_gsw_shoup[gsw_base + key0_0];
            const BasicInteger g01s = d_gsw_shoup[gsw_base + key0_1];
            const BasicInteger g10s = d_gsw_shoup[gsw_base + key1_0];
            const BasicInteger g11s = d_gsw_shoup[gsw_base + key1_1];

            sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit0, g00, g00s, Q), Q);
            sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit0, g01, g01s, Q), Q);
            sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit1, g10, g10s, Q), Q);
            sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit1, g11, g11s, Q), Q);
        }

        const size_t in0 = lut_base * N + coeff;
        BasicInteger v0 = phantom::arith::add_uint64_uint64_mod(d_in_c0[in0], sum0, Q);
        BasicInteger v1 = phantom::arith::add_uint64_uint64_mod(d_in_c1[in0], sum1, Q);
        if (child == 0) {
            out0_c0 = v0;
            out0_c1 = v1;
        } else {
            out1_c0 = v0;
            out1_c1 = v1;
        }
    }

    const size_t base_out = (static_cast<size_t>(batch_idx) * outCount + next_pair * 2u) * N + coeff;
    d_next_base_c0[base_out] = out0_c0;
    d_next_base_c1[base_out] = out0_c1;

    d_next_delta[base_out] = phantom::arith::sub_uint64_uint64_mod(out1_c0, out0_c0, Q);
    d_next_delta[base_out + N] = phantom::arith::sub_uint64_uint64_mod(out1_c1, out0_c1, Q);
}

__global__ void kernel_LUT_CMUX_ExternalProduct_NextDelta_DigitDiff_Batch(
    BasicInteger* __restrict__ d_next_base_c0, BasicInteger* __restrict__ d_next_base_c1,
    BasicInteger* __restrict__ d_next_delta, const BasicInteger* __restrict__ d_digits,
    const BasicInteger* __restrict__ d_gsw, const BasicInteger* __restrict__ d_gsw_shoup,
    const BasicInteger* __restrict__ d_in_c0, const BasicInteger* __restrict__ d_in_c1,
    const BasicInteger* __restrict__ d_static_delta, const BasicInteger* __restrict__ d_stage1_basecorr_digits,
    const BasicInteger* __restrict__ d_basecorr_gsw, const BasicInteger* __restrict__ d_basecorr_gsw_shoup,
    const DModulus* __restrict__ modulus, uint32_t base_digits, uint32_t N, uint32_t inCount, uint32_t batch,
    size_t ctrl_stride, uint32_t static_level) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t pair_global = static_cast<uint32_t>(blockIdx.y);
    const uint32_t outCount = inCount >> 1;
    const uint32_t nextPairCount = outCount >> 1;
    if (coeff >= N || pair_global >= nextPairCount * batch) {
        return;
    }

    const uint32_t batch_idx = pair_global / nextPairCount;
    const uint32_t next_pair = pair_global - batch_idx * nextPairCount;
    const BasicInteger Q = modulus[0].value();
    const size_t numLUT = static_cast<size_t>(inCount) * batch;
    const size_t digit_stride = numLUT * N;
    const size_t gsw_base = static_cast<size_t>(batch_idx) * ctrl_stride;
    const uint32_t base_lut_u32 = next_pair * 4u;
    const size_t base_lut = static_cast<size_t>(batch_idx) * inCount + base_lut_u32;
    const size_t sibling_lut = base_lut + 2u;

    BasicInteger base_sum0 = 0;
    BasicInteger base_sum1 = 0;
    BasicInteger delta_sum0 = 0;
    BasicInteger delta_sum1 = 0;
    BasicInteger corr_sum0 = 0;
    BasicInteger corr_sum1 = 0;
    const bool use_stage1_basecorr =
        d_static_delta && d_stage1_basecorr_digits && d_basecorr_gsw && d_basecorr_gsw_shoup && static_level == 1u;

    for (uint32_t d = 0; d < base_digits; ++d) {
        const size_t plane = static_cast<size_t>(d) * digit_stride;
        const size_t b0 = (base_lut + 0u) * N + coeff;
        const size_t b1 = (base_lut + 1u) * N + coeff;
        const size_t s0 = (sibling_lut + 0u) * N + coeff;
        const size_t s1 = (sibling_lut + 1u) * N + coeff;

        const BasicInteger bdigit0 = d_digits[plane + b0];
        const BasicInteger bdigit1 = d_digits[plane + b1];
        const BasicInteger ddigit0 = phantom::arith::sub_uint64_uint64_mod(d_digits[plane + s0], bdigit0, Q);
        const BasicInteger ddigit1 = phantom::arith::sub_uint64_uint64_mod(d_digits[plane + s1], bdigit1, Q);

        const uint32_t row0 = (d << 1) | 0u;
        const uint32_t row1 = (d << 1) | 1u;
        const size_t key0_0 = (static_cast<size_t>(row0) * 2u + 0u) * N + coeff;
        const size_t key0_1 = key0_0 + N;
        const size_t key1_0 = (static_cast<size_t>(row1) * 2u + 0u) * N + coeff;
        const size_t key1_1 = key1_0 + N;

        const BasicInteger g00 = d_gsw[gsw_base + key0_0];
        const BasicInteger g01 = d_gsw[gsw_base + key0_1];
        const BasicInteger g10 = d_gsw[gsw_base + key1_0];
        const BasicInteger g11 = d_gsw[gsw_base + key1_1];

        const BasicInteger g00s = d_gsw_shoup[gsw_base + key0_0];
        const BasicInteger g01s = d_gsw_shoup[gsw_base + key0_1];
        const BasicInteger g10s = d_gsw_shoup[gsw_base + key1_0];
        const BasicInteger g11s = d_gsw_shoup[gsw_base + key1_1];

        base_sum0 = phantom::arith::add_uint64_uint64_mod(base_sum0, phantom::arith::multiply_and_reduce_shoup(bdigit0, g00, g00s, Q), Q);
        base_sum1 = phantom::arith::add_uint64_uint64_mod(base_sum1, phantom::arith::multiply_and_reduce_shoup(bdigit0, g01, g01s, Q), Q);
        base_sum0 = phantom::arith::add_uint64_uint64_mod(base_sum0, phantom::arith::multiply_and_reduce_shoup(bdigit1, g10, g10s, Q), Q);
        base_sum1 = phantom::arith::add_uint64_uint64_mod(base_sum1, phantom::arith::multiply_and_reduce_shoup(bdigit1, g11, g11s, Q), Q);

        delta_sum0 = phantom::arith::add_uint64_uint64_mod(delta_sum0, phantom::arith::multiply_and_reduce_shoup(ddigit0, g00, g00s, Q), Q);
        delta_sum1 = phantom::arith::add_uint64_uint64_mod(delta_sum1, phantom::arith::multiply_and_reduce_shoup(ddigit0, g01, g01s, Q), Q);
        delta_sum0 = phantom::arith::add_uint64_uint64_mod(delta_sum0, phantom::arith::multiply_and_reduce_shoup(ddigit1, g10, g10s, Q), Q);
        delta_sum1 = phantom::arith::add_uint64_uint64_mod(delta_sum1, phantom::arith::multiply_and_reduce_shoup(ddigit1, g11, g11s, Q), Q);

        if (use_stage1_basecorr) {
            constexpr uint32_t kStage1CorrCount = 64;
            const size_t corr_plane = static_cast<size_t>(d) * kStage1CorrCount * N;
            const size_t corr_base = (static_cast<size_t>(next_pair) * 2u) * N + coeff;
            const BasicInteger cdigit0 = d_stage1_basecorr_digits[corr_plane + corr_base];
            const BasicInteger cdigit1 = d_stage1_basecorr_digits[corr_plane + corr_base + N];

            const BasicInteger c00 = d_basecorr_gsw[gsw_base + key0_0];
            const BasicInteger c01 = d_basecorr_gsw[gsw_base + key0_1];
            const BasicInteger c10 = d_basecorr_gsw[gsw_base + key1_0];
            const BasicInteger c11 = d_basecorr_gsw[gsw_base + key1_1];

            const BasicInteger c00s = d_basecorr_gsw_shoup[gsw_base + key0_0];
            const BasicInteger c01s = d_basecorr_gsw_shoup[gsw_base + key0_1];
            const BasicInteger c10s = d_basecorr_gsw_shoup[gsw_base + key1_0];
            const BasicInteger c11s = d_basecorr_gsw_shoup[gsw_base + key1_1];

            corr_sum0 = phantom::arith::add_uint64_uint64_mod(corr_sum0, phantom::arith::multiply_and_reduce_shoup(cdigit0, c00, c00s, Q), Q);
            corr_sum1 = phantom::arith::add_uint64_uint64_mod(corr_sum1, phantom::arith::multiply_and_reduce_shoup(cdigit0, c01, c01s, Q), Q);
            corr_sum0 = phantom::arith::add_uint64_uint64_mod(corr_sum0, phantom::arith::multiply_and_reduce_shoup(cdigit1, c10, c10s, Q), Q);
            corr_sum1 = phantom::arith::add_uint64_uint64_mod(corr_sum1, phantom::arith::multiply_and_reduce_shoup(cdigit1, c11, c11s, Q), Q);
        }
    }

    const size_t base_in = base_lut * N + coeff;
    const size_t sibling_in = sibling_lut * N + coeff;
    const size_t base_out = (static_cast<size_t>(batch_idx) * outCount + next_pair * 2u) * N + coeff;

    const BasicInteger base0 = phantom::arith::add_uint64_uint64_mod(d_in_c0[base_in], base_sum0, Q);
    const BasicInteger base1 = phantom::arith::add_uint64_uint64_mod(d_in_c1[base_in], base_sum1, Q);
    BasicInteger static_delta0 = 0;
    BasicInteger static_delta1 = 0;
    if (use_stage1_basecorr) {
        uint32_t static_poly_offset = 0;
        for (uint32_t l = 0; l < static_level; ++l) {
            static_poly_offset += 2u * (64u >> l);
        }
        const size_t static_base = static_cast<size_t>(static_poly_offset + next_pair * 2u) * N + coeff;
        static_delta0 = phantom::arith::add_uint64_uint64_mod(d_static_delta[static_base], corr_sum0, Q);
        static_delta1 = phantom::arith::add_uint64_uint64_mod(d_static_delta[static_base + N], corr_sum1, Q);
    } else if (d_static_delta && static_level > 0u && static_level < 7u) {
        uint32_t static_poly_offset = 0;
        for (uint32_t l = 0; l < static_level; ++l) {
            static_poly_offset += 2u * (64u >> l);
        }
        const size_t static_base = static_cast<size_t>(static_poly_offset + next_pair * 2u) * N + coeff;
        static_delta0 = d_static_delta[static_base];
        static_delta1 = d_static_delta[static_base + N];
    } else {
        static_delta0 = phantom::arith::sub_uint64_uint64_mod(d_in_c0[sibling_in], d_in_c0[base_in], Q);
        static_delta1 = phantom::arith::sub_uint64_uint64_mod(d_in_c1[sibling_in], d_in_c1[base_in], Q);
    }

    d_next_base_c0[base_out] = base0;
    d_next_base_c1[base_out] = base1;
    d_next_delta[base_out] = phantom::arith::add_uint64_uint64_mod(static_delta0, delta_sum0, Q);
    d_next_delta[base_out + N] = phantom::arith::add_uint64_uint64_mod(static_delta1, delta_sum1, Q);
}

__global__ void kernel_LUT_CMUX_ExternalProduct_FinalAdd_Batch(
    BasicInteger* __restrict__ d_acc_c0, BasicInteger* __restrict__ d_acc_c1,
    const BasicInteger* __restrict__ d_digits, const BasicInteger* __restrict__ d_gsw,
    const BasicInteger* __restrict__ d_gsw_shoup, const BasicInteger* __restrict__ d_in_c0,
    const BasicInteger* __restrict__ d_in_c1, const uint32_t* __restrict__ word_map,
    const BasicInteger* __restrict__ d_key_c0, const BasicInteger* __restrict__ d_key_c1,
    const DModulus* __restrict__ modulus, uint32_t base_digits, uint32_t N, uint32_t batch,
    size_t ctrl_stride) {
    const uint32_t coeff = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t batch_idx = static_cast<uint32_t>(blockIdx.y);
    if (coeff >= N || batch_idx >= batch) {
        return;
    }

    const BasicInteger Q = modulus[0].value();
    const size_t numLUT = static_cast<size_t>(2) * batch;
    const size_t digit_stride = numLUT * N;
    const uint32_t lut_base = batch_idx * 2u;

    BasicInteger sum0 = 0;
    BasicInteger sum1 = 0;
    for (uint32_t d = 0; d < base_digits; ++d) {
        const size_t plane = static_cast<size_t>(d) * digit_stride;
        const size_t lut0 = (static_cast<size_t>(lut_base) + 0u) * N + coeff;
        const size_t lut1 = (static_cast<size_t>(lut_base) + 1u) * N + coeff;

        const BasicInteger digit0 = d_digits[plane + lut0];
        const BasicInteger digit1 = d_digits[plane + lut1];

        const uint32_t row0 = (d << 1) | 0u;
        const uint32_t row1 = (d << 1) | 1u;
        const size_t key0_0 = (static_cast<size_t>(row0) * 2 + 0) * N + coeff;
        const size_t key0_1 = key0_0 + N;
        const size_t key1_0 = (static_cast<size_t>(row1) * 2 + 0) * N + coeff;
        const size_t key1_1 = key1_0 + N;

        const size_t gsw_base = static_cast<size_t>(batch_idx) * ctrl_stride;
        const BasicInteger g00 = d_gsw[gsw_base + key0_0];
        const BasicInteger g01 = d_gsw[gsw_base + key0_1];
        const BasicInteger g10 = d_gsw[gsw_base + key1_0];
        const BasicInteger g11 = d_gsw[gsw_base + key1_1];

        const BasicInteger g00s = d_gsw_shoup[gsw_base + key0_0];
        const BasicInteger g01s = d_gsw_shoup[gsw_base + key0_1];
        const BasicInteger g10s = d_gsw_shoup[gsw_base + key1_0];
        const BasicInteger g11s = d_gsw_shoup[gsw_base + key1_1];

        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit0, g00, g00s, Q), Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit0, g01, g01s, Q), Q);
        sum0 = phantom::arith::add_uint64_uint64_mod(sum0, phantom::arith::multiply_and_reduce_shoup(digit1, g10, g10s, Q), Q);
        sum1 = phantom::arith::add_uint64_uint64_mod(sum1, phantom::arith::multiply_and_reduce_shoup(digit1, g11, g11s, Q), Q);
    }

    const size_t in0 = static_cast<size_t>(lut_base) * N + coeff;
    const uint32_t word = word_map[batch_idx];
    const size_t out = static_cast<size_t>(word) * N + coeff;
    BasicInteger v0 = phantom::arith::add_uint64_uint64_mod(d_in_c0[in0], sum0, Q);
    BasicInteger v1 = phantom::arith::add_uint64_uint64_mod(d_in_c1[in0], sum1, Q);
    if (d_key_c0 && d_key_c1) {
        const size_t key_base = static_cast<size_t>(word & 3u) * N + coeff;
        v0 = phantom::arith::add_uint64_uint64_mod(v0, d_key_c0[key_base], Q);
        v1 = phantom::arith::add_uint64_uint64_mod(v1, d_key_c1[key_base], Q);
    }
    d_acc_c0[out] = phantom::arith::add_uint64_uint64_mod(d_acc_c0[out], v0, Q);
    d_acc_c1[out] = phantom::arith::add_uint64_uint64_mod(d_acc_c1[out], v1, Q);
}

void PrecomputeLUTStage0Digits(const GPUCirBTSContext& gpu_cc, const std::shared_ptr<RLWECryptoParams>& rlweParams,
                               Lut8x32GpuWorkspace& ws, DeviceRLWETable& table, uint32_t basecc) {
    const bool need_four_node = UseLUTFourNodeStage0();
    const bool need_stage1_basecorr = need_four_node && UseLUTAllStaticDelta() && UseLUTStage1StaticCorrection();
    if ((need_four_node && table.d_stage0_fournode_digits.get() && table.d_stage0_fournode_static_delta.get() &&
         (!need_stage1_basecorr || table.d_stage1_basecorr_digits.get())) ||
        (!need_four_node && table.d_stage0_digits.get())) {
        return;
    }
    const uint32_t N = ws.N;
    const uint32_t digitscc = ws.digitscc;
    const BasicInteger Q = rlweParams->GetQ().ConvertToInt<BasicInteger>();
    constexpr uint32_t inCount = 256;
    constexpr uint32_t outCount = inCount >> 1;
    const dim3 block(256);
    const dim3 grid_delta((N + block.x - 1) / block.x, outCount);
    const size_t nttShmem = static_cast<size_t>(N) * sizeof(BasicInteger);

    auto d_delta = phantom::util::make_cuda_auto_ptr<BasicInteger>(static_cast<size_t>(inCount) * N, ws.stream());
    auto d_tmp_digits = phantom::util::make_cuda_auto_ptr<BasicInteger>(static_cast<size_t>(digitscc) * inCount * N, ws.stream());

    kernel_LUT_CMUX_Delta<<<grid_delta, block, 0, ws.stream()>>>(
        d_delta.get(), table.d_table_c0.get(), table.d_table_c1.get(), gpu_cc.d_modulus(), N, inCount);
    PHANTOM_CHECK_CUDA_LAST();
    inwt_1d_opt_batched(d_delta.get(), gpu_cc.d_itwiddles(), gpu_cc.d_itwiddles_shoup(), gpu_cc.d_modulus(),
                        gpu_cc.d_n_inv_mod_q(), gpu_cc.d_n_inv_mod_q_shoup(), N, inCount, ws.stream());

    const size_t ss_batch = static_cast<size_t>(digitscc) * inCount;
    kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, nttShmem, ws.stream()>>>(
        d_tmp_digits.get(), d_delta.get(), Q, basecc, digitscc, N, inCount,
        gpu_cc.d_twiddles(), gpu_cc.d_twiddles_shoup(), gpu_cc.d_modulus());
    PHANTOM_CHECK_CUDA_LAST();

    if (need_four_node) {
        table.d_stage0_fournode_digits =
            phantom::util::make_cuda_auto_ptr<BasicInteger>(static_cast<size_t>(digitscc) * inCount * N, ws.stream());
        const uint32_t static_levels = UseLUTAllStaticDelta() ? 7u : 1u;
        const uint32_t static_polys = LUTStaticDeltaPolyCount();
        table.d_stage0_fournode_static_delta =
            phantom::util::make_cuda_auto_ptr<BasicInteger>(static_cast<size_t>(static_polys) * N, ws.stream());
        const dim3 grid_pack((N + block.x - 1) / block.x, digitscc * inCount);
        kernel_LUT_CMUX_PackStage0FourNodeDigits<<<grid_pack, block, 0, ws.stream()>>>(
            table.d_stage0_fournode_digits.get(), d_tmp_digits.get(), gpu_cc.d_modulus(), digitscc, N);
        PHANTOM_CHECK_CUDA_LAST();
        const dim3 grid_static((N + block.x - 1) / block.x, static_polys);
        kernel_LUT_CMUX_PackStageStaticDelta<<<grid_static, block, 0, ws.stream()>>>(
            table.d_stage0_fournode_static_delta.get(), table.d_table_c0.get(), table.d_table_c1.get(), gpu_cc.d_modulus(), N,
            static_levels);
        PHANTOM_CHECK_CUDA_LAST();
        if (need_stage1_basecorr) {
            const uint32_t corr_polys = LUTStage1CorrectionPolyCount();
            table.d_stage1_basecorr_digits =
                phantom::util::make_cuda_auto_ptr<BasicInteger>(static_cast<size_t>(digitscc) * corr_polys * N, ws.stream());
            const dim3 grid_corr((N + block.x - 1) / block.x, digitscc * corr_polys);
            kernel_LUT_CMUX_PackStage1BaseCorrectionDigits<<<grid_corr, block, 0, ws.stream()>>>(
                table.d_stage1_basecorr_digits.get(), d_tmp_digits.get(), gpu_cc.d_modulus(), digitscc, N);
            PHANTOM_CHECK_CUDA_LAST();
        }
    } else {
        table.d_stage0_digits = std::move(d_tmp_digits);
    }
    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(ws.stream()));
}

RLWECiphertext EvalLUT8x32_RLWE_GPU(const GPUCirBTSContext& gpu_cc, const std::shared_ptr<RLWECryptoParams>& rlweParams, Lut8x32GpuWorkspace& ws,
                                   const std::array<RGSWCiphertext, 8>& ctrl_bits_lsb_to_msb, const DeviceRLWETable& table, uint32_t basecc) {
    const uint32_t N = ws.N;
    const BasicInteger Q = rlweParams->GetQ().ConvertToInt<BasicInteger>();

    UploadRGSWControlsToGPU(ws, ctrl_bits_lsb_to_msb, Q);
    CudaEventGuard lut_gpu_start;
    CudaEventGuard lut_gpu_end;
    CudaEventGuard d2h_start;
    CudaEventGuard d2h_end;
    if (g_transition_profile.enabled) {
        PHANTOM_CHECK_CUDA(cudaEventCreate(&lut_gpu_start.ev));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&lut_gpu_end.ev));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&d2h_start.ev));
        PHANTOM_CHECK_CUDA(cudaEventCreate(&d2h_end.ev));
        PHANTOM_CHECK_CUDA(cudaEventRecord(lut_gpu_start.ev, ws.stream()));
    }

    const size_t nttShmem = static_cast<size_t>(N) * sizeof(BasicInteger);

    const BasicInteger* d_tw = gpu_cc.d_twiddles();
    const BasicInteger* d_tw_sh = gpu_cc.d_twiddles_shoup();
    const BasicInteger* d_itw = gpu_cc.d_itwiddles();
    const BasicInteger* d_itw_sh = gpu_cc.d_itwiddles_shoup();
    const BasicInteger* d_ninv = gpu_cc.d_n_inv_mod_q();
    const BasicInteger* d_ninv_sh = gpu_cc.d_n_inv_mod_q_shoup();
    const DModulus* d_mod = gpu_cc.d_modulus();

    const dim3 block(256);
    uint32_t curCount = 256;
    const char* lut_graph_env = std::getenv("CIRBTS_LUT_GRAPH");
    const bool use_graph = (lut_graph_env == nullptr) ? true : (std::string(lut_graph_env) != "0");
    if (ws.disable_graph && use_graph) {
        // Respect explicit disable (e.g., when capturing a larger CUDA graph).
    }
    const bool use_graph_effective = use_graph && !ws.disable_graph;
    const BasicInteger* cur_c0 = nullptr;
    const BasicInteger* cur_c1 = nullptr;

    if (use_graph_effective && !ws.lut_graph_failed) {
        const BasicInteger* table_c0 = table.d_table_c0.get();
        const BasicInteger* table_c1 = table.d_table_c1.get();
        cudaGraphExec_t exec = nullptr;
        for (uint32_t i = 0; i < ws.lut_graph_count; ++i) {
            if (ws.lut_graph_cache[i].table_c0 == table_c0 && ws.lut_graph_cache[i].table_c1 == table_c1) {
                exec = ws.lut_graph_cache[i].exec;
                break;
            }
        }

        if (!exec && ws.lut_graph_count < ws.lut_graph_cache.size()) {
            const cudaError_t sync_st = cudaStreamSynchronize(ws.stream());
            if (sync_st != cudaSuccess) {
                ws.lut_graph_failed = true;
            } else {
                cudaGraph_t graph{};
                cudaError_t cap_st = cudaStreamBeginCapture(ws.stream(), cudaStreamCaptureModeThreadLocal);
                if (cap_st != cudaSuccess) {
                    ws.lut_graph_failed = true;
                } else {
                    auto run_stage = [&](uint32_t bit, uint32_t inCount, const BasicInteger* in0, const BasicInteger* in1,
                                         BasicInteger* out0, BasicInteger* out1) {
                        const uint32_t outCount = inCount >> 1;
                        const dim3 grid_delta((N + block.x - 1) / block.x, outCount);
                        kernel_LUT_CMUX_Delta<<<grid_delta, block, 0, ws.stream()>>>(ws.d_delta.get(), in0, in1, d_mod, N, inCount);
                        inwt_1d_opt_batched(ws.d_delta.get(), d_itw, d_itw_sh, d_mod, d_ninv, d_ninv_sh, N,
                                            static_cast<size_t>(outCount) * 2, ws.stream());
                        const uint32_t numLUT = outCount * 2;
                        const size_t ss_batch = static_cast<size_t>(ws.digitscc) * numLUT;
                        kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, nttShmem, ws.stream()>>>(
                            ws.d_digits.get(), ws.d_delta.get(), Q, basecc, ws.digitscc, N, numLUT, d_tw, d_tw_sh, d_mod);
                        const BasicInteger* d_gsw = ws.d_ctrl_gsw.get() + static_cast<size_t>(bit) * ws.ctrl_per_bit_stride;
                        const BasicInteger* d_gsw_shoup = ws.d_ctrl_gsw_shoup.get() + static_cast<size_t>(bit) * ws.ctrl_per_bit_stride;
                        kernel_LUT_CMUX_ExternalProduct<<<grid_delta, block, 0, ws.stream()>>>(
                            out0, out1, ws.d_digits.get(), d_gsw, d_gsw_shoup, in0, in1, d_mod, ws.digitscc, N, inCount);
                    };

                    run_stage(0, 256, table_c0, table_c1, ws.d_stage0_c0.get(), ws.d_stage0_c1.get());
                    run_stage(1, 128, ws.d_stage0_c0.get(), ws.d_stage0_c1.get(), ws.d_stage1_c0.get(), ws.d_stage1_c1.get());
                    run_stage(2, 64, ws.d_stage1_c0.get(), ws.d_stage1_c1.get(), ws.d_stage0_c0.get(), ws.d_stage0_c1.get());
                    run_stage(3, 32, ws.d_stage0_c0.get(), ws.d_stage0_c1.get(), ws.d_stage1_c0.get(), ws.d_stage1_c1.get());
                    run_stage(4, 16, ws.d_stage1_c0.get(), ws.d_stage1_c1.get(), ws.d_stage0_c0.get(), ws.d_stage0_c1.get());
                    run_stage(5, 8, ws.d_stage0_c0.get(), ws.d_stage0_c1.get(), ws.d_stage1_c0.get(), ws.d_stage1_c1.get());
                    run_stage(6, 4, ws.d_stage1_c0.get(), ws.d_stage1_c1.get(), ws.d_stage0_c0.get(), ws.d_stage0_c1.get());
                    run_stage(7, 2, ws.d_stage0_c0.get(), ws.d_stage0_c1.get(), ws.d_stage1_c0.get(), ws.d_stage1_c1.get());

                    cap_st = cudaStreamEndCapture(ws.stream(), &graph);
                    if (cap_st != cudaSuccess || !graph) {
                        ws.lut_graph_failed = true;
                        if (graph) {
                            cudaGraphDestroy(graph);
                        }
                    } else {
                        cudaGraphExec_t new_exec{};
                        const cudaError_t inst = cudaGraphInstantiate(&new_exec, graph, nullptr, nullptr, 0);
                        cudaGraphDestroy(graph);
                        if (inst != cudaSuccess) {
                            ws.lut_graph_failed = true;
                        } else {
                            auto& slot = ws.lut_graph_cache[ws.lut_graph_count++];
                            slot.table_c0 = table_c0;
                            slot.table_c1 = table_c1;
                            slot.exec = new_exec;
                            exec = new_exec;
                        }
                    }
                }
            }
        }

        if (exec) {
            PHANTOM_CHECK_CUDA(cudaGraphLaunch(exec, ws.stream()));
            cur_c0 = ws.d_stage1_c0.get();
            cur_c1 = ws.d_stage1_c1.get();
            curCount = 1;
        }
    }

    if (!cur_c0 || !cur_c1) {
        cur_c0 = table.d_table_c0.get();
        cur_c1 = table.d_table_c1.get();
        BasicInteger* next_c0 = ws.d_stage0_c0.get();
        BasicInteger* next_c1 = ws.d_stage0_c1.get();

        for (uint32_t bit = 0; bit < 8; ++bit) {
            const uint32_t outCount = curCount >> 1;
            const dim3 grid_delta((N + block.x - 1) / block.x, outCount);
            kernel_LUT_CMUX_Delta<<<grid_delta, block, 0, ws.stream()>>>(ws.d_delta.get(), cur_c0, cur_c1, d_mod, N, curCount);

            inwt_1d_opt_batched(ws.d_delta.get(), d_itw, d_itw_sh, d_mod, d_ninv, d_ninv_sh, N, static_cast<size_t>(outCount) * 2, ws.stream());

            const uint32_t numLUT = outCount * 2;
            const size_t ss_batch = static_cast<size_t>(ws.digitscc) * numLUT;
            kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, nttShmem, ws.stream()>>>(
                ws.d_digits.get(), ws.d_delta.get(), Q, basecc, ws.digitscc, N, numLUT, d_tw, d_tw_sh, d_mod);

            const BasicInteger* d_gsw = ws.d_ctrl_gsw.get() + static_cast<size_t>(bit) * ws.ctrl_per_bit_stride;
            const BasicInteger* d_gsw_shoup = ws.d_ctrl_gsw_shoup.get() + static_cast<size_t>(bit) * ws.ctrl_per_bit_stride;
            kernel_LUT_CMUX_ExternalProduct<<<grid_delta, block, 0, ws.stream()>>>(
                next_c0, next_c1, ws.d_digits.get(), d_gsw, d_gsw_shoup, cur_c0, cur_c1, d_mod, ws.digitscc, N, curCount);

            curCount = outCount;
            cur_c0 = next_c0;
            cur_c1 = next_c1;
            if (next_c0 == ws.d_stage0_c0.get()) {
                next_c0 = ws.d_stage1_c0.get();
                next_c1 = ws.d_stage1_c1.get();
            } else {
                next_c0 = ws.d_stage0_c0.get();
                next_c1 = ws.d_stage0_c1.get();
            }
        }
    }

    if (g_transition_profile.enabled) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(lut_gpu_end.ev, ws.stream()));
        PHANTOM_CHECK_CUDA(cudaEventSynchronize(lut_gpu_end.ev));
        float ms = 0.0f;
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms, lut_gpu_start.ev, lut_gpu_end.ev));
        g_transition_profile.lut_gpu_ms += static_cast<double>(ms);
        g_transition_profile.lut_calls += 1;
    }

    std::vector<BasicInteger> h_c0(N);
    std::vector<BasicInteger> h_c1(N);
    if (g_transition_profile.enabled) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(d2h_start.ev, ws.stream()));
    }
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(h_c0.data(), cur_c0, static_cast<size_t>(N) * sizeof(BasicInteger), cudaMemcpyDeviceToHost, ws.stream()));
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(h_c1.data(), cur_c1, static_cast<size_t>(N) * sizeof(BasicInteger), cudaMemcpyDeviceToHost, ws.stream()));
    if (g_transition_profile.enabled) {
        PHANTOM_CHECK_CUDA(cudaEventRecord(d2h_end.ev, ws.stream()));
        PHANTOM_CHECK_CUDA(cudaEventSynchronize(d2h_end.ev));
        float ms = 0.0f;
        PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms, d2h_start.ev, d2h_end.ev));
        g_transition_profile.d2h_ms += static_cast<double>(ms);
        g_transition_profile.d2h_events += 1;
        g_transition_profile.host_device_boundaries += 1;
    }
    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(ws.stream()));

    const auto pack0 = std::chrono::steady_clock::now();
    auto polyParams = rlweParams->GetPolyParams();
    NativePoly out_c0(polyParams, EVALUATION, true);
    NativePoly out_c1(polyParams, EVALUATION, true);
    for (uint32_t i = 0; i < N; ++i) {
        out_c0[i].SetValue(NativeInteger(h_c0[i]));
        out_c1[i].SetValue(NativeInteger(h_c1[i]));
    }
    std::vector<NativePoly> out_elems = {out_c0, out_c1};
    const auto pack1 = std::chrono::steady_clock::now();
    if (g_transition_profile.enabled) {
        g_transition_profile.pack_host_ms += DurationMs(pack0, pack1);
    }
    return std::make_shared<RLWECiphertextImpl>(std::move(out_elems));
}

// GPU LUT that consumes RGSW controls already on device and returns a device view of the final RLWE ciphertext.
// The returned pointers are valid until the next LUT call that reuses the same workspace staging buffers.
DeviceRLWECiphertextView EvalLUT8x32_RLWE_GPU_DeviceCtrl(const GPUCirBTSContext& gpu_cc, const std::shared_ptr<RLWECryptoParams>& rlweParams,
                                                         Lut8x32GpuWorkspace& ws, const BasicInteger* d_ctrl_bits_lsb_to_msb,
                                                         const DeviceRLWETable& table, uint32_t basecc) {
    const uint32_t digitscc = ws.digitscc;
    const uint32_t N = ws.N;
    const BasicInteger Q = rlweParams->GetQ().ConvertToInt<BasicInteger>();

    UploadRGSWControlsToGPU_Device(ws, d_ctrl_bits_lsb_to_msb, Q);

    const size_t nttShmem = static_cast<size_t>(N) * sizeof(BasicInteger);

    const BasicInteger* d_tw = gpu_cc.d_twiddles();
    const BasicInteger* d_tw_sh = gpu_cc.d_twiddles_shoup();
    const BasicInteger* d_itw = gpu_cc.d_itwiddles();
    const BasicInteger* d_itw_sh = gpu_cc.d_itwiddles_shoup();
    const BasicInteger* d_ninv = gpu_cc.d_n_inv_mod_q();
    const BasicInteger* d_ninv_sh = gpu_cc.d_n_inv_mod_q_shoup();
    const DModulus* d_mod = gpu_cc.d_modulus();

    const dim3 block(256);
    uint32_t curCount = 256;
    const char* lut_graph_env2 = std::getenv("CIRBTS_LUT_GRAPH");
    const bool use_graph2 = (lut_graph_env2 == nullptr) ? true : (std::string(lut_graph_env2) != "0");
    if (ws.disable_graph && use_graph2) {
        // Respect explicit disable (e.g., when capturing a larger CUDA graph).
    }
    const bool use_graph2_effective = use_graph2 && !ws.disable_graph;
    const BasicInteger* cur_c0 = nullptr;
    const BasicInteger* cur_c1 = nullptr;

    if (use_graph2_effective && !ws.lut_graph_failed) {
        const BasicInteger* table_c0 = table.d_table_c0.get();
        const BasicInteger* table_c1 = table.d_table_c1.get();
        cudaGraphExec_t exec = nullptr;
        for (uint32_t i = 0; i < ws.lut_graph_count; ++i) {
            if (ws.lut_graph_cache[i].table_c0 == table_c0 && ws.lut_graph_cache[i].table_c1 == table_c1) {
                exec = ws.lut_graph_cache[i].exec;
                break;
            }
        }

        if (!exec && ws.lut_graph_count < ws.lut_graph_cache.size()) {
            const cudaError_t sync_st = cudaStreamSynchronize(ws.stream());
            if (sync_st != cudaSuccess) {
                ws.lut_graph_failed = true;
            } else {
                cudaGraph_t graph{};
                cudaError_t cap_st = cudaStreamBeginCapture(ws.stream(), cudaStreamCaptureModeThreadLocal);
                if (cap_st != cudaSuccess) {
                    ws.lut_graph_failed = true;
                } else {
                    auto run_stage = [&](uint32_t bit, uint32_t inCount, const BasicInteger* in0, const BasicInteger* in1,
                                         BasicInteger* out0, BasicInteger* out1) {
                        const uint32_t outCount = inCount >> 1;
                        const dim3 grid_delta((N + block.x - 1) / block.x, outCount);
                        kernel_LUT_CMUX_Delta<<<grid_delta, block, 0, ws.stream()>>>(ws.d_delta.get(), in0, in1, d_mod, N, inCount);
                        inwt_1d_opt_batched(ws.d_delta.get(), d_itw, d_itw_sh, d_mod, d_ninv, d_ninv_sh, N,
                                            static_cast<size_t>(outCount) * 2, ws.stream());
                        const uint32_t numLUT = outCount * 2;
                        const size_t ss_batch = static_cast<size_t>(digitscc) * numLUT;
                        kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, nttShmem, ws.stream()>>>(
                            ws.d_digits.get(), ws.d_delta.get(), Q, basecc, digitscc, N, numLUT, d_tw, d_tw_sh, d_mod);
                        const BasicInteger* d_gsw = ws.d_ctrl_gsw.get() + static_cast<size_t>(bit) * ws.ctrl_per_bit_stride;
                        const BasicInteger* d_gsw_shoup = ws.d_ctrl_gsw_shoup.get() + static_cast<size_t>(bit) * ws.ctrl_per_bit_stride;
                        kernel_LUT_CMUX_ExternalProduct<<<grid_delta, block, 0, ws.stream()>>>(
                            out0, out1, ws.d_digits.get(), d_gsw, d_gsw_shoup, in0, in1, d_mod, digitscc, N, inCount);
                    };

                    run_stage(0, 256, table_c0, table_c1, ws.d_stage0_c0.get(), ws.d_stage0_c1.get());
                    run_stage(1, 128, ws.d_stage0_c0.get(), ws.d_stage0_c1.get(), ws.d_stage1_c0.get(), ws.d_stage1_c1.get());
                    run_stage(2, 64, ws.d_stage1_c0.get(), ws.d_stage1_c1.get(), ws.d_stage0_c0.get(), ws.d_stage0_c1.get());
                    run_stage(3, 32, ws.d_stage0_c0.get(), ws.d_stage0_c1.get(), ws.d_stage1_c0.get(), ws.d_stage1_c1.get());
                    run_stage(4, 16, ws.d_stage1_c0.get(), ws.d_stage1_c1.get(), ws.d_stage0_c0.get(), ws.d_stage0_c1.get());
                    run_stage(5, 8, ws.d_stage0_c0.get(), ws.d_stage0_c1.get(), ws.d_stage1_c0.get(), ws.d_stage1_c1.get());
                    run_stage(6, 4, ws.d_stage1_c0.get(), ws.d_stage1_c1.get(), ws.d_stage0_c0.get(), ws.d_stage0_c1.get());
                    run_stage(7, 2, ws.d_stage0_c0.get(), ws.d_stage0_c1.get(), ws.d_stage1_c0.get(), ws.d_stage1_c1.get());

                    cap_st = cudaStreamEndCapture(ws.stream(), &graph);
                    if (cap_st != cudaSuccess || !graph) {
                        ws.lut_graph_failed = true;
                        if (graph) {
                            cudaGraphDestroy(graph);
                        }
                    } else {
                        cudaGraphExec_t new_exec{};
                        const cudaError_t inst = cudaGraphInstantiate(&new_exec, graph, nullptr, nullptr, 0);
                        cudaGraphDestroy(graph);
                        if (inst != cudaSuccess) {
                            ws.lut_graph_failed = true;
                        } else {
                            auto& slot = ws.lut_graph_cache[ws.lut_graph_count++];
                            slot.table_c0 = table_c0;
                            slot.table_c1 = table_c1;
                            slot.exec = new_exec;
                            exec = new_exec;
                        }
                    }
                }
            }
        }

        if (exec) {
            PHANTOM_CHECK_CUDA(cudaGraphLaunch(exec, ws.stream()));
            cur_c0 = ws.d_stage1_c0.get();
            cur_c1 = ws.d_stage1_c1.get();
            curCount = 1;
        }
    }

    if (!cur_c0 || !cur_c1) {
        cur_c0 = table.d_table_c0.get();
        cur_c1 = table.d_table_c1.get();
        BasicInteger* next_c0 = ws.d_stage0_c0.get();
        BasicInteger* next_c1 = ws.d_stage0_c1.get();

        for (uint32_t bit = 0; bit < 8; ++bit) {
            const uint32_t outCount = curCount >> 1;
            const dim3 grid_delta((N + block.x - 1) / block.x, outCount);
            kernel_LUT_CMUX_Delta<<<grid_delta, block, 0, ws.stream()>>>(ws.d_delta.get(), cur_c0, cur_c1, d_mod, N, curCount);

            inwt_1d_opt_batched(ws.d_delta.get(), d_itw, d_itw_sh, d_mod, d_ninv, d_ninv_sh, N, static_cast<size_t>(outCount) * 2, ws.stream());

            const uint32_t numLUT = outCount * 2;
            const size_t ss_batch = static_cast<size_t>(digitscc) * numLUT;
            kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, nttShmem, ws.stream()>>>(
                ws.d_digits.get(), ws.d_delta.get(), Q, basecc, digitscc, N, numLUT, d_tw, d_tw_sh, d_mod);

            const BasicInteger* d_gsw = ws.d_ctrl_gsw.get() + static_cast<size_t>(bit) * ws.ctrl_per_bit_stride;
            const BasicInteger* d_gsw_shoup = ws.d_ctrl_gsw_shoup.get() + static_cast<size_t>(bit) * ws.ctrl_per_bit_stride;
            kernel_LUT_CMUX_ExternalProduct<<<grid_delta, block, 0, ws.stream()>>>(next_c0, next_c1, ws.d_digits.get(), d_gsw, d_gsw_shoup, cur_c0,
                                                                                  cur_c1, d_mod, digitscc, N, curCount);

            curCount = outCount;
            cur_c0 = next_c0;
            cur_c1 = next_c1;
            if (next_c0 == ws.d_stage0_c0.get()) {
                next_c0 = ws.d_stage1_c0.get();
                next_c1 = ws.d_stage1_c1.get();
            } else {
                next_c0 = ws.d_stage0_c0.get();
                next_c1 = ws.d_stage0_c1.get();
            }
        }
    }

    return {cur_c0, cur_c1};
}

DeviceRLWECiphertextBatchView EvalLUT8x32_RLWE_GPU_DeviceCtrl_Batch(const GPUCirBTSContext& gpu_cc, const std::shared_ptr<RLWECryptoParams>& rlweParams,
                                                                    Lut8x32GpuWorkspace& ws, const BasicInteger* d_ctrl_all, size_t ctrl_stride,
                                                                    const uint32_t* h_indices, uint32_t batch, const DeviceRLWETable& table,
                                                                    uint32_t basecc, const BasicInteger* d_ctrl_all_shoup = nullptr,
                                                                    BasicInteger* d_final_add_c0 = nullptr,
                                                                    BasicInteger* d_final_add_c1 = nullptr,
                                                                    const uint32_t* d_final_word_map = nullptr,
                                                                    const BasicInteger* d_final_key_c0 = nullptr,
                                                                    const BasicInteger* d_final_key_c1 = nullptr) {
    if (batch == 0) {
        return {};
    }
    if (batch > ws.lut_batch_max) {
        OPENFHE_THROW(config_error, "EvalLUT8x32_RLWE_GPU_DeviceCtrl_Batch: batch exceeds workspace capacity");
    }
    if (ctrl_stride != ws.ctrl_per_bit_stride) {
        OPENFHE_THROW(config_error, "EvalLUT8x32_RLWE_GPU_DeviceCtrl_Batch: control stride mismatch");
    }

    const uint32_t digitscc = ws.digitscc;
    const uint32_t N = ws.N;
    const BasicInteger Q = rlweParams->GetQ().ConvertToInt<BasicInteger>();

    UploadRGSWControlsToGPU_Device_Batch(ws, d_ctrl_all, d_ctrl_all_shoup, h_indices, batch, Q, ctrl_stride);

    const size_t nttShmem = static_cast<size_t>(N) * sizeof(BasicInteger);

    const BasicInteger* d_tw = gpu_cc.d_twiddles();
    const BasicInteger* d_tw_sh = gpu_cc.d_twiddles_shoup();
    const BasicInteger* d_itw = gpu_cc.d_itwiddles();
    const BasicInteger* d_itw_sh = gpu_cc.d_itwiddles_shoup();
    const BasicInteger* d_ninv = gpu_cc.d_n_inv_mod_q();
    const BasicInteger* d_ninv_sh = gpu_cc.d_n_inv_mod_q_shoup();
    const DModulus* d_mod = gpu_cc.d_modulus();
    const size_t ctrl_batch_stride = ws.ctrl_per_bit_stride * static_cast<size_t>(batch);
    const bool fuse_final_add = d_final_add_c0 && d_final_add_c1 && d_final_word_map;

    const dim3 block(256);
    uint32_t curCount = 256;

    const char* lut_graph_env = std::getenv("CIRBTS_LUT_GRAPH");
    const bool use_graph = (lut_graph_env == nullptr) ? true : (std::string(lut_graph_env) != "0");
    const bool use_graph_effective = use_graph && !ws.disable_graph;

    if (use_graph_effective && !ws.lut_batch_graph_failed) {
        const BasicInteger* table_c0 = table.d_table_c0.get();
        const BasicInteger* table_c1 = table.d_table_c1.get();
        const BasicInteger* table_stage0_digits = table.d_stage0_digits.get();
        const BasicInteger* table_stage0_fournode_digits = table.d_stage0_fournode_digits.get();
        const BasicInteger* table_stage0_fournode_static_delta = table.d_stage0_fournode_static_delta.get();
        const BasicInteger* table_stage1_basecorr_digits = table.d_stage1_basecorr_digits.get();
        const bool use_four_node_stage0 =
            UseLUTFourNodeStage0() && table_stage0_fournode_digits != nullptr && table_stage0_fournode_static_delta != nullptr;
        const bool use_sync_delta = use_four_node_stage0 || (UseLUTSyncDelta() && table_stage0_digits != nullptr);
        const bool use_next_digit_diff = UseLUTNextDigitDiff();
        const bool use_stage1_static_correction =
            use_four_node_stage0 && UseLUTAllStaticDelta() && UseLUTStage1StaticCorrection() && table_stage1_basecorr_digits != nullptr;
        const bool use_two_stage_fusion = !use_sync_delta && UseLUTTwoStageFusion() && table_stage0_digits != nullptr;
        cudaGraphExec_t exec = nullptr;
        for (uint32_t i = 0; i < ws.lut_batch_graph_count; ++i) {
            const auto& e = ws.lut_batch_graph_cache[i];
            if (e.table_c0 == table_c0 && e.table_c1 == table_c1 && e.batch == batch &&
                e.final_add == fuse_final_add &&
                (!fuse_final_add || (e.add_out_c0 == d_final_add_c0 && e.add_out_c1 == d_final_add_c1 &&
                                     e.key_c0 == d_final_key_c0 && e.key_c1 == d_final_key_c1))) {
                exec = e.exec;
                break;
            }
        }

        if (!exec && ws.lut_batch_graph_count < ws.lut_batch_graph_cache.size()) {
            const cudaError_t sync_st = cudaStreamSynchronize(ws.stream());
            if (sync_st != cudaSuccess) {
                ws.lut_batch_graph_failed = true;
            } else {
                cudaGraph_t graph{};
                cudaError_t cap_st = cudaStreamBeginCapture(ws.stream(), cudaStreamCaptureModeThreadLocal);
                if (cap_st != cudaSuccess) {
                    ws.lut_batch_graph_failed = true;
                } else {
                    uint32_t capCount = 256;
                    const BasicInteger* cap_cur_c0 = nullptr;
                    const BasicInteger* cap_cur_c1 = nullptr;
                    BasicInteger* cap_next_c0 = ws.d_stage0_c0.get();
                    BasicInteger* cap_next_c1 = ws.d_stage0_c1.get();

                    uint32_t start_bit = 0;
                    if (use_sync_delta) {
                        const dim3 grid_fused((N + block.x - 1) / block.x, 64u * batch);
                        const BasicInteger* d_gsw0 = ws.d_ctrl_gsw.get();
                        const BasicInteger* d_gsw0_shoup = ws.d_ctrl_gsw_shoup.get();
                        if (use_four_node_stage0) {
                            kernel_LUT_CMUX_Stage0Stage1Delta_FourNode_Precomp_Batch<<<grid_fused, block, 0, ws.stream()>>>(
                                cap_next_c0, cap_next_c1, ws.d_delta.get(), table_stage0_fournode_digits, table_stage0_fournode_static_delta,
                                d_gsw0, d_gsw0_shoup, table_c0, table_c1, d_mod, digitscc, N, batch, ws.ctrl_per_bit_stride);
                        } else {
                            kernel_LUT_CMUX_Stage0Stage1Delta_Precomp_Batch<<<grid_fused, block, 0, ws.stream()>>>(
                                cap_next_c0, cap_next_c1, ws.d_delta.get(), table_stage0_digits, d_gsw0, d_gsw0_shoup,
                                table_c0, table_c1, d_mod, digitscc, N, batch, ws.ctrl_per_bit_stride);
                        }

                        capCount = 128;
                        cap_cur_c0 = cap_next_c0;
                        cap_cur_c1 = cap_next_c1;
                        cap_next_c0 = ws.d_stage1_c0.get();
                        cap_next_c1 = ws.d_stage1_c1.get();
                        start_bit = 1;
                    } else if (use_two_stage_fusion) {
                        const dim3 grid_fused((N + block.x - 1) / block.x, 64u * batch);
                        const BasicInteger* d_gsw0 = ws.d_ctrl_gsw.get();
                        const BasicInteger* d_gsw0_shoup = ws.d_ctrl_gsw_shoup.get();
                        kernel_LUT_CMUX_Stage0Stage1Delta_Precomp_Batch<<<grid_fused, block, 0, ws.stream()>>>(
                            cap_next_c0, cap_next_c1, ws.d_delta.get(), table_stage0_digits, d_gsw0, d_gsw0_shoup,
                            table_c0, table_c1, d_mod, digitscc, N, batch, ws.ctrl_per_bit_stride);

                        inwt_1d_opt_batched(ws.d_delta.get(), d_itw, d_itw_sh, d_mod, d_ninv, d_ninv_sh, N,
                                            static_cast<size_t>(128) * batch, ws.stream());

                        const size_t ss_batch = static_cast<size_t>(digitscc) * 128u * batch;
                        kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, nttShmem, ws.stream()>>>(
                            ws.d_digits.get(), ws.d_delta.get(), Q, basecc, digitscc, N, 128u * batch, d_tw, d_tw_sh, d_mod);

                        const BasicInteger* d_gsw1 = ws.d_ctrl_gsw.get() + ctrl_batch_stride;
                        const BasicInteger* d_gsw1_shoup = ws.d_ctrl_gsw_shoup.get() + ctrl_batch_stride;
                        kernel_LUT_CMUX_ExternalProduct_Batch<<<grid_fused, block, 0, ws.stream()>>>(
                            ws.d_stage1_c0.get(), ws.d_stage1_c1.get(), ws.d_digits.get(), d_gsw1, d_gsw1_shoup,
                            cap_next_c0, cap_next_c1, d_mod, digitscc, N, 128u, batch, ws.ctrl_per_bit_stride);

                        capCount = 64;
                        cap_cur_c0 = ws.d_stage1_c0.get();
                        cap_cur_c1 = ws.d_stage1_c1.get();
                        cap_next_c0 = ws.d_stage0_c0.get();
                        cap_next_c1 = ws.d_stage0_c1.get();
                        start_bit = 2;
                    }

                    for (uint32_t bit = start_bit; bit < 8; ++bit) {
                        const uint32_t outCount = capCount >> 1;
                        const dim3 grid_delta((N + block.x - 1) / block.x, outCount * batch);
                        const bool use_stage0_precomp = (bit == 0 && table_stage0_digits != nullptr);

                        if (!(use_sync_delta && bit > 0) && !use_stage0_precomp) {
                            if (bit == 0) {
                                kernel_LUT_CMUX_Delta_TableBatch<<<grid_delta, block, 0, ws.stream()>>>(
                                    ws.d_delta.get(), table_c0, table_c1, d_mod, N, capCount, batch);
                            } else {
                                kernel_LUT_CMUX_Delta_Batch<<<grid_delta, block, 0, ws.stream()>>>(
                                    ws.d_delta.get(), cap_cur_c0, cap_cur_c1, d_mod, N, capCount, batch);
                            }

                            inwt_1d_opt_batched(ws.d_delta.get(), d_itw, d_itw_sh, d_mod, d_ninv, d_ninv_sh, N,
                                                static_cast<size_t>(outCount) * 2 * batch, ws.stream());

                            const uint32_t numLUTTotal = capCount * batch;
                            const size_t ss_batch = static_cast<size_t>(digitscc) * numLUTTotal;
                            kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, nttShmem, ws.stream()>>>(
                                ws.d_digits.get(), ws.d_delta.get(), Q, basecc, digitscc, N, numLUTTotal, d_tw, d_tw_sh, d_mod);
                        }
                        if (use_sync_delta && bit > 0) {
                            inwt_1d_opt_batched(ws.d_delta.get(), d_itw, d_itw_sh, d_mod, d_ninv, d_ninv_sh, N,
                                                static_cast<size_t>(capCount) * batch, ws.stream());

                            const uint32_t numLUTTotal = capCount * batch;
                            const size_t ss_batch = static_cast<size_t>(digitscc) * numLUTTotal;
                            kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, nttShmem, ws.stream()>>>(
                                ws.d_digits.get(), ws.d_delta.get(), Q, basecc, digitscc, N, numLUTTotal, d_tw, d_tw_sh, d_mod);
                        }

                        const BasicInteger* d_gsw = ws.d_ctrl_gsw.get() + static_cast<size_t>(bit) * ctrl_batch_stride;
                        const BasicInteger* d_gsw_shoup = ws.d_ctrl_gsw_shoup.get() + static_cast<size_t>(bit) * ctrl_batch_stride;

                        if (bit == 7 && fuse_final_add) {
                            kernel_LUT_CMUX_ExternalProduct_FinalAdd_Batch<<<grid_delta, block, 0, ws.stream()>>>(
                                d_final_add_c0, d_final_add_c1, ws.d_digits.get(), d_gsw, d_gsw_shoup, cap_cur_c0, cap_cur_c1,
                                d_final_word_map, d_final_key_c0, d_final_key_c1, d_mod, digitscc, N, batch, ws.ctrl_per_bit_stride);
                        } else if (bit == 0) {
                            if (table_stage0_digits) {
                            kernel_LUT_CMUX_ExternalProduct_TableBatch_PrecompStage0<<<grid_delta, block, 0, ws.stream()>>>(
                                    cap_next_c0, cap_next_c1, table_stage0_digits, d_gsw, d_gsw_shoup, table_c0, table_c1, d_mod, digitscc, N,
                                    capCount, batch, ws.ctrl_per_bit_stride);
                            } else {
                                kernel_LUT_CMUX_ExternalProduct_TableBatch<<<grid_delta, block, 0, ws.stream()>>>(
                                    cap_next_c0, cap_next_c1, ws.d_digits.get(), d_gsw, d_gsw_shoup, table_c0, table_c1, d_mod, digitscc, N,
                                    capCount, batch, ws.ctrl_per_bit_stride);
                            }
                        } else {
                            if (use_sync_delta && bit < 7) {
                                const dim3 grid_next_delta((N + block.x - 1) / block.x, (outCount >> 1) * batch);
                                if (use_next_digit_diff) {
                                    const bool use_stage1_corr_this_bit = use_stage1_static_correction && bit == 1u;
                                    const BasicInteger* static_delta =
                                        use_stage1_corr_this_bit ? table_stage0_fournode_static_delta : nullptr;
                                    const BasicInteger* stage1_corr_digits =
                                        use_stage1_corr_this_bit ? table_stage1_basecorr_digits : nullptr;
                                    const BasicInteger* stage1_corr_gsw = use_stage1_corr_this_bit ? ws.d_ctrl_gsw.get() : nullptr;
                                    const BasicInteger* stage1_corr_gsw_shoup =
                                        use_stage1_corr_this_bit ? ws.d_ctrl_gsw_shoup.get() : nullptr;
                                    kernel_LUT_CMUX_ExternalProduct_NextDelta_DigitDiff_Batch<<<grid_next_delta, block, 0, ws.stream()>>>(
                                        cap_next_c0, cap_next_c1, ws.d_delta.get(), ws.d_digits.get(), d_gsw, d_gsw_shoup,
                                        cap_cur_c0, cap_cur_c1, static_delta, stage1_corr_digits, stage1_corr_gsw,
                                        stage1_corr_gsw_shoup, d_mod, digitscc, N, capCount, batch, ws.ctrl_per_bit_stride, bit);
                                } else {
                                    kernel_LUT_CMUX_ExternalProduct_NextDelta_Batch<<<grid_next_delta, block, 0, ws.stream()>>>(
                                        cap_next_c0, cap_next_c1, ws.d_delta.get(), ws.d_digits.get(), d_gsw, d_gsw_shoup,
                                        cap_cur_c0, cap_cur_c1, d_mod, digitscc, N, capCount, batch, ws.ctrl_per_bit_stride);
                                }
                            } else {
                                kernel_LUT_CMUX_ExternalProduct_Batch<<<grid_delta, block, 0, ws.stream()>>>(
                                    cap_next_c0, cap_next_c1, ws.d_digits.get(), d_gsw, d_gsw_shoup, cap_cur_c0, cap_cur_c1, d_mod, digitscc, N,
                                    capCount, batch, ws.ctrl_per_bit_stride);
                            }
                        }

                        capCount = outCount;
                        cap_cur_c0 = cap_next_c0;
                        cap_cur_c1 = cap_next_c1;
                        if (cap_next_c0 == ws.d_stage0_c0.get()) {
                            cap_next_c0 = ws.d_stage1_c0.get();
                            cap_next_c1 = ws.d_stage1_c1.get();
                        } else {
                            cap_next_c0 = ws.d_stage0_c0.get();
                            cap_next_c1 = ws.d_stage0_c1.get();
                        }
                    }

                    cap_st = cudaStreamEndCapture(ws.stream(), &graph);
                    if (cap_st != cudaSuccess || !graph) {
                        ws.lut_batch_graph_failed = true;
                        if (graph) {
                            cudaGraphDestroy(graph);
                        }
                    } else {
                        cudaGraphExec_t new_exec{};
                        const cudaError_t inst = cudaGraphInstantiate(&new_exec, graph, nullptr, nullptr, 0);
                        cudaGraphDestroy(graph);
                        if (inst != cudaSuccess) {
                            ws.lut_batch_graph_failed = true;
                        } else {
                            auto& slot = ws.lut_batch_graph_cache[ws.lut_batch_graph_count++];
                            slot.table_c0 = table_c0;
                            slot.table_c1 = table_c1;
                            slot.batch = batch;
                            slot.final_add = fuse_final_add;
                            slot.add_out_c0 = d_final_add_c0;
                            slot.add_out_c1 = d_final_add_c1;
                            slot.key_c0 = d_final_key_c0;
                            slot.key_c1 = d_final_key_c1;
                            slot.exec = new_exec;
                            exec = new_exec;
                        }
                    }
                }
            }
        }

        if (exec) {
            PHANTOM_CHECK_CUDA(cudaGraphLaunch(exec, ws.stream()));
            if (fuse_final_add) {
                return {nullptr, nullptr, batch};
            }
            return {ws.d_stage1_c0.get(), ws.d_stage1_c1.get(), batch};
        }
    }

    const BasicInteger* cur_c0 = nullptr;
    const BasicInteger* cur_c1 = nullptr;
    BasicInteger* next_c0 = ws.d_stage0_c0.get();
    BasicInteger* next_c1 = ws.d_stage0_c1.get();

    const BasicInteger* table_stage0_digits = table.d_stage0_digits.get();
    const BasicInteger* table_stage0_fournode_digits = table.d_stage0_fournode_digits.get();
    const BasicInteger* table_stage0_fournode_static_delta = table.d_stage0_fournode_static_delta.get();
    const BasicInteger* table_stage1_basecorr_digits = table.d_stage1_basecorr_digits.get();
    const bool use_four_node_stage0 =
        UseLUTFourNodeStage0() && table_stage0_fournode_digits != nullptr && table_stage0_fournode_static_delta != nullptr;
    const bool use_sync_delta = use_four_node_stage0 || (UseLUTSyncDelta() && table_stage0_digits != nullptr);
    const bool use_next_digit_diff = UseLUTNextDigitDiff();
    const bool use_stage1_static_correction =
        use_four_node_stage0 && UseLUTAllStaticDelta() && UseLUTStage1StaticCorrection() && table_stage1_basecorr_digits != nullptr;
    const bool use_two_stage_fusion = !use_sync_delta && UseLUTTwoStageFusion() && table_stage0_digits != nullptr;
    uint32_t start_bit = 0;
    if (use_sync_delta) {
        const dim3 grid_fused((N + block.x - 1) / block.x, 64u * batch);
        const BasicInteger* d_gsw0 = ws.d_ctrl_gsw.get();
        const BasicInteger* d_gsw0_shoup = ws.d_ctrl_gsw_shoup.get();
        if (use_four_node_stage0) {
            kernel_LUT_CMUX_Stage0Stage1Delta_FourNode_Precomp_Batch<<<grid_fused, block, 0, ws.stream()>>>(
                next_c0, next_c1, ws.d_delta.get(), table_stage0_fournode_digits, table_stage0_fournode_static_delta,
                d_gsw0, d_gsw0_shoup, table.d_table_c0.get(), table.d_table_c1.get(), d_mod, digitscc, N, batch,
                ws.ctrl_per_bit_stride);
        } else {
            kernel_LUT_CMUX_Stage0Stage1Delta_Precomp_Batch<<<grid_fused, block, 0, ws.stream()>>>(
                next_c0, next_c1, ws.d_delta.get(), table_stage0_digits, d_gsw0, d_gsw0_shoup,
                table.d_table_c0.get(), table.d_table_c1.get(), d_mod, digitscc, N, batch, ws.ctrl_per_bit_stride);
        }
        PHANTOM_CHECK_CUDA_LAST();

        curCount = 128;
        cur_c0 = next_c0;
        cur_c1 = next_c1;
        next_c0 = ws.d_stage1_c0.get();
        next_c1 = ws.d_stage1_c1.get();
        start_bit = 1;
    } else if (use_two_stage_fusion) {
        const dim3 grid_fused((N + block.x - 1) / block.x, 64u * batch);
        const BasicInteger* d_gsw0 = ws.d_ctrl_gsw.get();
        const BasicInteger* d_gsw0_shoup = ws.d_ctrl_gsw_shoup.get();
        kernel_LUT_CMUX_Stage0Stage1Delta_Precomp_Batch<<<grid_fused, block, 0, ws.stream()>>>(
            next_c0, next_c1, ws.d_delta.get(), table_stage0_digits, d_gsw0, d_gsw0_shoup,
            table.d_table_c0.get(), table.d_table_c1.get(), d_mod, digitscc, N, batch, ws.ctrl_per_bit_stride);
        PHANTOM_CHECK_CUDA_LAST();

        inwt_1d_opt_batched(ws.d_delta.get(), d_itw, d_itw_sh, d_mod, d_ninv, d_ninv_sh, N,
                            static_cast<size_t>(128) * batch, ws.stream());

        const size_t ss_batch = static_cast<size_t>(digitscc) * 128u * batch;
        kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, nttShmem, ws.stream()>>>(
            ws.d_digits.get(), ws.d_delta.get(), Q, basecc, digitscc, N, 128u * batch, d_tw, d_tw_sh, d_mod);
        PHANTOM_CHECK_CUDA_LAST();

        const BasicInteger* d_gsw1 = ws.d_ctrl_gsw.get() + ctrl_batch_stride;
        const BasicInteger* d_gsw1_shoup = ws.d_ctrl_gsw_shoup.get() + ctrl_batch_stride;
        kernel_LUT_CMUX_ExternalProduct_Batch<<<grid_fused, block, 0, ws.stream()>>>(
            ws.d_stage1_c0.get(), ws.d_stage1_c1.get(), ws.d_digits.get(), d_gsw1, d_gsw1_shoup,
            next_c0, next_c1, d_mod, digitscc, N, 128u, batch, ws.ctrl_per_bit_stride);
        PHANTOM_CHECK_CUDA_LAST();

        curCount = 64;
        cur_c0 = ws.d_stage1_c0.get();
        cur_c1 = ws.d_stage1_c1.get();
        next_c0 = ws.d_stage0_c0.get();
        next_c1 = ws.d_stage0_c1.get();
        start_bit = 2;
    }

    for (uint32_t bit = start_bit; bit < 8; ++bit) {
        const uint32_t outCount = curCount >> 1;
        const dim3 grid_delta((N + block.x - 1) / block.x, outCount * batch);
        const bool use_stage0_precomp = (bit == 0 && table_stage0_digits != nullptr);

        if (!(use_sync_delta && bit > 0) && !use_stage0_precomp) {
            if (bit == 0) {
                kernel_LUT_CMUX_Delta_TableBatch<<<grid_delta, block, 0, ws.stream()>>>(
                    ws.d_delta.get(), table.d_table_c0.get(), table.d_table_c1.get(), d_mod, N, curCount, batch);
            } else {
                kernel_LUT_CMUX_Delta_Batch<<<grid_delta, block, 0, ws.stream()>>>(
                    ws.d_delta.get(), cur_c0, cur_c1, d_mod, N, curCount, batch);
            }

            inwt_1d_opt_batched(ws.d_delta.get(), d_itw, d_itw_sh, d_mod, d_ninv, d_ninv_sh, N, static_cast<size_t>(outCount) * 2 * batch,
                                ws.stream());

            const uint32_t numLUTTotal = curCount * batch;
            const size_t ss_batch = static_cast<size_t>(digitscc) * numLUTTotal;
            kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, nttShmem, ws.stream()>>>(
                ws.d_digits.get(), ws.d_delta.get(), Q, basecc, digitscc, N, numLUTTotal, d_tw, d_tw_sh, d_mod);
        }
        if (use_sync_delta && bit > 0) {
            inwt_1d_opt_batched(ws.d_delta.get(), d_itw, d_itw_sh, d_mod, d_ninv, d_ninv_sh, N,
                                static_cast<size_t>(curCount) * batch, ws.stream());

            const uint32_t numLUTTotal = curCount * batch;
            const size_t ss_batch = static_cast<size_t>(digitscc) * numLUTTotal;
            kernel_SignedDigitDecompose_FusedFNWT<<<ss_batch, N / 2, nttShmem, ws.stream()>>>(
                ws.d_digits.get(), ws.d_delta.get(), Q, basecc, digitscc, N, numLUTTotal, d_tw, d_tw_sh, d_mod);
            PHANTOM_CHECK_CUDA_LAST();
        }

        const BasicInteger* d_gsw = ws.d_ctrl_gsw.get() + static_cast<size_t>(bit) * ctrl_batch_stride;
        const BasicInteger* d_gsw_shoup = ws.d_ctrl_gsw_shoup.get() + static_cast<size_t>(bit) * ctrl_batch_stride;

        if (bit == 7 && fuse_final_add) {
            kernel_LUT_CMUX_ExternalProduct_FinalAdd_Batch<<<grid_delta, block, 0, ws.stream()>>>(
                d_final_add_c0, d_final_add_c1, ws.d_digits.get(), d_gsw, d_gsw_shoup, cur_c0, cur_c1,
                d_final_word_map, d_final_key_c0, d_final_key_c1, d_mod, digitscc, N, batch, ws.ctrl_per_bit_stride);
        } else if (bit == 0) {
            if (table_stage0_digits) {
                kernel_LUT_CMUX_ExternalProduct_TableBatch_PrecompStage0<<<grid_delta, block, 0, ws.stream()>>>(
                    next_c0, next_c1, table_stage0_digits, d_gsw, d_gsw_shoup, table.d_table_c0.get(), table.d_table_c1.get(), d_mod, digitscc,
                    N, curCount, batch, ws.ctrl_per_bit_stride);
            } else {
                kernel_LUT_CMUX_ExternalProduct_TableBatch<<<grid_delta, block, 0, ws.stream()>>>(
                    next_c0, next_c1, ws.d_digits.get(), d_gsw, d_gsw_shoup, table.d_table_c0.get(), table.d_table_c1.get(), d_mod, digitscc, N,
                curCount, batch, ws.ctrl_per_bit_stride);
            }
        } else {
            if (use_sync_delta && bit < 7) {
                const dim3 grid_next_delta((N + block.x - 1) / block.x, (outCount >> 1) * batch);
                if (use_next_digit_diff) {
                    const bool use_stage1_corr_this_bit = use_stage1_static_correction && bit == 1u;
                    const BasicInteger* static_delta =
                        use_stage1_corr_this_bit ? table_stage0_fournode_static_delta : nullptr;
                    const BasicInteger* stage1_corr_digits = use_stage1_corr_this_bit ? table_stage1_basecorr_digits : nullptr;
                    const BasicInteger* stage1_corr_gsw = use_stage1_corr_this_bit ? ws.d_ctrl_gsw.get() : nullptr;
                    const BasicInteger* stage1_corr_gsw_shoup =
                        use_stage1_corr_this_bit ? ws.d_ctrl_gsw_shoup.get() : nullptr;
                    kernel_LUT_CMUX_ExternalProduct_NextDelta_DigitDiff_Batch<<<grid_next_delta, block, 0, ws.stream()>>>(
                        next_c0, next_c1, ws.d_delta.get(), ws.d_digits.get(), d_gsw, d_gsw_shoup,
                        cur_c0, cur_c1, static_delta, stage1_corr_digits, stage1_corr_gsw, stage1_corr_gsw_shoup,
                        d_mod, digitscc, N, curCount, batch, ws.ctrl_per_bit_stride, bit);
                } else {
                    kernel_LUT_CMUX_ExternalProduct_NextDelta_Batch<<<grid_next_delta, block, 0, ws.stream()>>>(
                        next_c0, next_c1, ws.d_delta.get(), ws.d_digits.get(), d_gsw, d_gsw_shoup,
                        cur_c0, cur_c1, d_mod, digitscc, N, curCount, batch, ws.ctrl_per_bit_stride);
                }
            } else {
                kernel_LUT_CMUX_ExternalProduct_Batch<<<grid_delta, block, 0, ws.stream()>>>(
                    next_c0, next_c1, ws.d_digits.get(), d_gsw, d_gsw_shoup, cur_c0, cur_c1, d_mod, digitscc, N, curCount, batch,
                    ws.ctrl_per_bit_stride);
            }
        }
        PHANTOM_CHECK_CUDA_LAST();

        if (bit != 7 || !fuse_final_add) {
            curCount = outCount;
            cur_c0 = next_c0;
            cur_c1 = next_c1;
            if (next_c0 == ws.d_stage0_c0.get()) {
                next_c0 = ws.d_stage1_c0.get();
                next_c1 = ws.d_stage1_c1.get();
            } else {
                next_c0 = ws.d_stage0_c0.get();
                next_c1 = ws.d_stage0_c1.get();
            }
        }
    }

    if (fuse_final_add) {
        return {nullptr, nullptr, batch};
    }
    return {cur_c0, cur_c1, batch};
}
