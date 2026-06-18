// RLWE-to-LWE sample extraction and GPU-side extract helpers.

// ---------------------------
// RLWE -> LWE sample extraction (CPU) + key switch to level-0 LWE key.
// ---------------------------
LWECiphertext SampleExtractIndex(const std::shared_ptr<RLWECryptoParams>& rlweParams, const RLWECiphertext& ct_coeff, uint32_t idx,
                                 LWEPlaintextModulus p = 2) {
    const uint32_t N = static_cast<uint32_t>(rlweParams->GetN());
    const NativeInteger Q = rlweParams->GetQ();

    const auto& a = ct_coeff->GetElements()[0];
    const auto& b = ct_coeff->GetElements()[1];
    if (a.GetFormat() != COEFFICIENT || b.GetFormat() != COEFFICIENT) {
        OPENFHE_THROW(config_error, "SampleExtractIndex expects COEFFICIENT format RLWE");
    }
    if (idx >= N) {
        OPENFHE_THROW(config_error, "SampleExtractIndex: idx out of range");
    }

    const bool profile = g_transition_profile.enabled;
    const auto se0 = std::chrono::steady_clock::now();
    NativeVector avec(N, Q);
    for (uint32_t j = 0; j < N; ++j) {
        if (j <= idx) {
            avec[j] = a[idx - j];
        } else {
            const uint32_t k = N + idx - j;
            const auto& v = a[k];
            avec[j] = NativeInteger(0).ModSubFast(v, Q);
        }
    }
    const auto se1 = std::chrono::steady_clock::now();

    auto out = std::make_shared<LWECiphertextImpl>(std::move(avec), b[idx]);
    out->SetptModulus(p);
    const auto pack1 = std::chrono::steady_clock::now();
    if (profile) {
#ifdef _OPENMP
#pragma omp critical(aes_transition_profile)
#endif
        {
            g_transition_profile.se_cpu_ms += DurationMs(se0, se1);
            g_transition_profile.pack_host_ms += DurationMs(se1, pack1);
        }
    }
    return out;
}

LWECiphertext SwitchTLWENToTLWEn(const LWEEncryptionScheme& lweScheme, const std::shared_ptr<LWECryptoParams>& lweParamsKS,
                                 ConstLWESwitchingKey& ksk_rn_to_n, ConstLWECiphertext& tlweN) {
    const bool profile = g_transition_profile.enabled;
    const auto ks0 = std::chrono::steady_clock::now();
    LWECiphertext out;
    if (tlweN->GetModulus() == lweParamsKS->GetqKS()) {
        auto ctKS = lweScheme.KeySwitch(lweParamsKS, ksk_rn_to_n, tlweN);
        out = lweScheme.ModSwitch(lweParamsKS->Getq(), ctKS);
    } else {
        out = lweScheme.SwitchCTtoqn(lweParamsKS, ksk_rn_to_n, tlweN);
    }
    const auto ks1 = std::chrono::steady_clock::now();
    if (profile) {
#ifdef _OPENMP
#pragma omp critical(aes_transition_profile)
#endif
        {
            g_transition_profile.ks_cpu_ms += DurationMs(ks0, ks1);
        }
    }
    return out;
}

LWEPrivateKey DeriveLWEKeyFromRLWESecret(const std::shared_ptr<LWECryptoParams>& lweParams, const RLWEPrivateKey& sk2) {
    const NativeInteger qKS = lweParams->GetqKS();
    NativePoly s = sk2->GetElement();
    s.SetFormat(COEFFICIENT);
    NativeVector sv(s.GetValues());
    sv.SwitchModulus(qKS);
    return std::make_shared<LWEPrivateKeyImpl>(std::move(sv));
}

inline bool UseFusedExtractKS() {
    const char* v = std::getenv("AES_EXTRACT_FUSED_KS");
    return v && *v && *v != '0';
}

inline bool UsePreencodedExtractMap() {
    const char* v = std::getenv("AES_EXTRACT_PREENCODE_MAP");
    return v && *v && *v != '0';
}

inline bool UseDirectFusedExtractOutput() {
    const char* v = std::getenv("AES_EXTRACT_DIRECT_OUT");
    return !v || !*v || *v != '0';
}

inline bool RunFusedExtractKeySwitch(const aes_example::GpuLweKeySwitchKeyU16& gpu_ksk,
                                     aes_example::GpuLweKeySwitchWorkspaceU16& gpu_ws,
                                     const uint64_t* d_rlwe_a, const uint64_t* d_rlwe_b,
                                     const uint32_t* d_extract_info, size_t batch, uint64_t Q64, double q_over_Q) {
    if (UsePreencodedExtractMap() &&
        aes_example::KeySwitchBatchDeviceFromMappedExtractedRLWE(gpu_ksk, gpu_ws, d_rlwe_a, d_rlwe_b, d_extract_info, batch, Q64, q_over_Q)) {
        return true;
    }
    return aes_example::KeySwitchBatchDeviceFromExtractedRLWE(gpu_ksk, gpu_ws, d_rlwe_a, d_rlwe_b, d_extract_info, batch, Q64, q_over_Q);
}

inline bool RunFusedExtractKeySwitchToLWE(const aes_example::GpuLweKeySwitchKeyU16& gpu_ksk,
                                          aes_example::GpuLweKeySwitchWorkspaceU16& gpu_ws,
                                          const uint64_t* d_rlwe_a, const uint64_t* d_rlwe_b,
                                          const uint32_t* d_extract_info, BasicInteger* d_out_a,
                                          BasicInteger* d_out_b, size_t batch, uint64_t Q64, double q_over_Q,
                                          bool q_divisible, uint64_t scale, uint64_t q_lwe, uint32_t out_stride) {
    if (!UseDirectFusedExtractOutput()) {
        return false;
    }
    if (UsePreencodedExtractMap() &&
        aes_example::KeySwitchBatchDeviceFromMappedExtractedRLWEToLWE(
            gpu_ksk, gpu_ws, d_rlwe_a, d_rlwe_b, d_extract_info, d_out_a, d_out_b, batch, Q64, q_over_Q, q_divisible,
            scale, q_lwe, out_stride)) {
        return true;
    }
    return aes_example::KeySwitchBatchDeviceFromExtractedRLWEToLWE(
        gpu_ksk, gpu_ws, d_rlwe_a, d_rlwe_b, d_extract_info, d_out_a, d_out_b, batch, Q64, q_over_Q, q_divisible, scale,
        q_lwe, out_stride);
}

// GPU-side extraction:
//   (1) INWT 4 RLWE words (a,b) on GPU,
//   (2) sample-extract+pack 128 TLWE_N ciphertexts mod qKS (uint16), or fuse it into KS,
//   (3) key switch on GPU to TLWE_n mod qKS,
//   (4) copy back and wrap as OpenFHE LWECiphertext (then ModSwitch to q).
std::vector<LWECiphertext> ExtractStateBytesToLWETwitter_DeviceWords(const GPUCirBTSContext& gpu_cc,
                                                                     const std::shared_ptr<CirBTSCryptoParams>& params,
                                                                     const std::shared_ptr<LWECryptoParams>& lweParamsKS,
                                                                     const LWEEncryptionScheme& lweScheme,
                                                                     const aes_example::GpuLweKeySwitchKeyU16& gpu_ksk,
                                                                     aes_example::GpuLweKeySwitchWorkspaceU16& gpu_ws,
                                                                     BasicInteger* d_words_c0_eval, BasicInteger* d_words_c1_eval) {
    std::vector<LWECiphertext> out(128);

    const size_t batch = 128;
    const uint32_t N = gpu_ksk.N;
    const uint32_t n = gpu_ksk.n;
    if (N != gpu_ws.N || n != gpu_ws.n) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWETwitter_DeviceWords: GPU KS workspace mismatch");
    }
    if (gpu_ws.maxBatch < batch) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWETwitter_DeviceWords: GPU KS workspace maxBatch too small");
    }

    const NativeInteger qKS = lweParamsKS->GetqKS();
    const uint64_t Q64 = params->GetRLWEParams()->GetQ().ConvertToInt();
    const uint32_t qks_u32 = gpu_ksk.qKS;
    if ((Q64 == 0) || (qks_u32 == 0) || ((qks_u32 & (qks_u32 - 1u)) != 0) || (Q64 < qks_u32)) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWETwitter_DeviceWords: requires power-of-two qKS with Q>=qKS");
    }
    const double q_over_Q = static_cast<double>(qks_u32) / static_cast<double>(Q64);

    inwt_1d_opt_batched(d_words_c0_eval, gpu_cc.d_itwiddles(), gpu_cc.d_itwiddles_shoup(), gpu_cc.d_modulus(), gpu_cc.d_n_inv_mod_q(),
                        gpu_cc.d_n_inv_mod_q_shoup(), N, /*batch=*/4, gpu_ws.stream);
    inwt_1d_opt_batched(d_words_c1_eval, gpu_cc.d_itwiddles(), gpu_cc.d_itwiddles_shoup(), gpu_cc.d_modulus(), gpu_cc.d_n_inv_mod_q(),
                        gpu_cc.d_n_inv_mod_q_shoup(), N, /*batch=*/4, gpu_ws.stream);

    const dim3 block(256);
    const bool fused_extract_ks = UseFusedExtractKS();
    if (!fused_extract_ks) {
        const dim3 grid((N + block.x - 1) / block.x, static_cast<uint32_t>(batch));
        aes_example::kernel_SampleExtractPackU16<<<grid, block, 0, gpu_ws.stream>>>(
            gpu_ws.d_in_a, gpu_ws.d_in_b, reinterpret_cast<const uint64_t*>(d_words_c0_eval),
            reinterpret_cast<const uint64_t*>(d_words_c1_eval), gpu_ws.d_extract_info, static_cast<uint32_t>(batch), N, Q64,
            q_over_Q, gpu_ksk.qKS_mask);
        PHANTOM_CHECK_CUDA_LAST();
    }
    if (fused_extract_ks) {
        if (!RunFusedExtractKeySwitch(gpu_ksk, gpu_ws, reinterpret_cast<const uint64_t*>(d_words_c0_eval),
                                      reinterpret_cast<const uint64_t*>(d_words_c1_eval), gpu_ws.d_extract_info, batch, Q64, q_over_Q)) {
            OPENFHE_THROW(config_error, "fused extract+KeySwitch failed");
        }
    } else if (!aes_example::KeySwitchBatchDevice(gpu_ksk, gpu_ws, batch)) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWETwitter_DeviceWords: KeySwitchBatchDevice failed");
    }

    const uint32_t outLen = n + 1u;
    std::vector<uint16_t> h_out(batch * static_cast<size_t>(outLen));
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(h_out.data(), gpu_ws.d_out, static_cast<size_t>(batch) * outLen * sizeof(uint16_t), cudaMemcpyDeviceToHost,
                                       gpu_ws.stream));
    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(gpu_ws.stream));

    for (uint32_t idx = 0; idx < 128; ++idx) {
        NativeVector a(n, qKS);
        const size_t base = static_cast<size_t>(idx) * outLen;
        for (uint32_t k = 0; k < n; ++k) {
            a[k] = NativeInteger(h_out[base + k]);
        }
        NativeInteger b = NativeInteger(h_out[base + n]);
        auto ctKS = std::make_shared<LWECiphertextImpl>(std::move(a), b);
        out[idx] = lweScheme.ModSwitch(lweParamsKS->Getq(), ctKS);
    }
    return out;
}

bool ExtractStateBytesToLWEDevice(const GPUCirBTSContext& gpu_cc,
                                  const std::shared_ptr<CirBTSCryptoParams>& params,
                                  const aes_example::GpuLweKeySwitchKeyU16& gpu_ksk,
                                  aes_example::GpuLweKeySwitchWorkspaceU16& gpu_ws,
                                  BasicInteger* d_words_c0_eval, BasicInteger* d_words_c1_eval,
                                  BasicInteger* d_out_a, BasicInteger* d_out_b, uint32_t out_stride) {
    const size_t batch = 128;
    const uint32_t N = gpu_ksk.N;
    const uint32_t n = gpu_ksk.n;
    if (N != gpu_ws.N || n != gpu_ws.n) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDevice: GPU KS workspace mismatch");
    }
    if (gpu_ws.maxBatch < batch) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDevice: GPU KS workspace maxBatch too small");
    }
    if (out_stride < n) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDevice: out_stride < n");
    }

    const uint64_t Q64 = params->GetRLWEParams()->GetQ().ConvertToInt();
    const uint32_t qks_u32 = gpu_ksk.qKS;
    if ((Q64 == 0) || (qks_u32 == 0) || ((qks_u32 & (qks_u32 - 1u)) != 0) || (Q64 < qks_u32)) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDevice: requires power-of-two qKS with Q>=qKS");
    }
    const double q_over_Q = static_cast<double>(qks_u32) / static_cast<double>(Q64);

    const auto q_lwe = params->GetLWEParams()->Getq().ConvertToInt();
    const uint64_t q_lwe_u64 = static_cast<uint64_t>(q_lwe);
    if (q_lwe_u64 == 0) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDevice: q must be non-zero");
    }
    const bool q_divisible = (q_lwe_u64 % qks_u32) == 0;
    const uint64_t scale = q_divisible ? (q_lwe_u64 / qks_u32) : 0;

    inwt_1d_opt_batched(d_words_c0_eval, gpu_cc.d_itwiddles(), gpu_cc.d_itwiddles_shoup(), gpu_cc.d_modulus(), gpu_cc.d_n_inv_mod_q(),
                        gpu_cc.d_n_inv_mod_q_shoup(), N, /*batch=*/4, gpu_ws.stream);
    inwt_1d_opt_batched(d_words_c1_eval, gpu_cc.d_itwiddles(), gpu_cc.d_itwiddles_shoup(), gpu_cc.d_modulus(), gpu_cc.d_n_inv_mod_q(),
                        gpu_cc.d_n_inv_mod_q_shoup(), N, /*batch=*/4, gpu_ws.stream);

    const dim3 block(256);
    const bool fused_extract_ks = UseFusedExtractKS();
    bool direct_lwe_out = false;
    if (fused_extract_ks) {
        direct_lwe_out = RunFusedExtractKeySwitchToLWE(
            gpu_ksk, gpu_ws, reinterpret_cast<const uint64_t*>(d_words_c0_eval), reinterpret_cast<const uint64_t*>(d_words_c1_eval),
            gpu_ws.d_extract_info, d_out_a, d_out_b, batch, Q64, q_over_Q, q_divisible, scale, q_lwe_u64, out_stride);
        if (!direct_lwe_out && !RunFusedExtractKeySwitch(gpu_ksk, gpu_ws, reinterpret_cast<const uint64_t*>(d_words_c0_eval),
                                                         reinterpret_cast<const uint64_t*>(d_words_c1_eval), gpu_ws.d_extract_info, batch,
                                                         Q64, q_over_Q)) {
            OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDevice: fused extract+KeySwitch failed");
        }
    } else {
        const dim3 grid((N + block.x - 1) / block.x, static_cast<uint32_t>(batch));
        aes_example::kernel_SampleExtractPackU16<<<grid, block, 0, gpu_ws.stream>>>(
            gpu_ws.d_in_a, gpu_ws.d_in_b, reinterpret_cast<const uint64_t*>(d_words_c0_eval),
            reinterpret_cast<const uint64_t*>(d_words_c1_eval), gpu_ws.d_extract_info, static_cast<uint32_t>(batch), N, Q64,
            q_over_Q, gpu_ksk.qKS_mask);
        PHANTOM_CHECK_CUDA_LAST();

        if (!aes_example::KeySwitchBatchDevice(gpu_ksk, gpu_ws, batch)) {
            OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDevice: KeySwitchBatchDevice failed");
        }
    }
    if (direct_lwe_out) {
        return true;
    }

    const uint32_t out_len = n + 1u;
    const uint32_t total = static_cast<uint32_t>(batch) * out_len;
    const dim3 grid_out((total + block.x - 1) / block.x);
    if (q_divisible) {
        kernel_U16ToLWE<<<grid_out, block, 0, gpu_ws.stream>>>(
            d_out_a, d_out_b, gpu_ws.d_out, n, static_cast<uint32_t>(batch), scale, out_stride);
    } else {
        kernel_U16ToLWE_ModSwitch<<<grid_out, block, 0, gpu_ws.stream>>>(
            d_out_a, d_out_b, gpu_ws.d_out, n, static_cast<uint32_t>(batch), qks_u32, q_lwe_u64, out_stride);
    }
    PHANTOM_CHECK_CUDA_LAST();
    return true;
}

bool ExtractStateBytesToLWEDeviceBatch(const GPUCirBTSContext& gpu_cc,
                                       const std::shared_ptr<CirBTSCryptoParams>& params,
                                       const aes_example::GpuLweKeySwitchKeyU16& gpu_ksk,
                                       aes_example::GpuLweKeySwitchWorkspaceU16& gpu_ws,
                                       BasicInteger* d_words_c0_eval, BasicInteger* d_words_c1_eval,
                                       BasicInteger* d_out_a, BasicInteger* d_out_b,
                                       uint32_t out_stride, uint32_t blocks) {
    if (blocks == 0) {
        return true;
    }
    const size_t batch = static_cast<size_t>(blocks) * 128u;
    const uint32_t N = gpu_ksk.N;
    const uint32_t n = gpu_ksk.n;
    if (N != gpu_ws.N || n != gpu_ws.n) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDeviceBatch: GPU KS workspace mismatch");
    }
    if (gpu_ws.maxBatch < batch) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDeviceBatch: GPU KS workspace maxBatch too small");
    }
    if (out_stride < n) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDeviceBatch: out_stride < n");
    }

    const uint64_t Q64 = params->GetRLWEParams()->GetQ().ConvertToInt();
    const uint32_t qks_u32 = gpu_ksk.qKS;
    if ((Q64 == 0) || (qks_u32 == 0) || ((qks_u32 & (qks_u32 - 1u)) != 0) || (Q64 < qks_u32)) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDeviceBatch: requires power-of-two qKS with Q>=qKS");
    }
    const double q_over_Q = static_cast<double>(qks_u32) / static_cast<double>(Q64);

    const auto q_lwe = params->GetLWEParams()->Getq().ConvertToInt();
    const uint64_t q_lwe_u64 = static_cast<uint64_t>(q_lwe);
    if (q_lwe_u64 == 0) {
        OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDeviceBatch: q must be non-zero");
    }
    const bool q_divisible = (q_lwe_u64 % qks_u32) == 0;
    const uint64_t scale = q_divisible ? (q_lwe_u64 / qks_u32) : 0;

    inwt_1d_opt_batched(d_words_c0_eval, gpu_cc.d_itwiddles(), gpu_cc.d_itwiddles_shoup(), gpu_cc.d_modulus(), gpu_cc.d_n_inv_mod_q(),
                        gpu_cc.d_n_inv_mod_q_shoup(), N, static_cast<size_t>(blocks) * 4u, gpu_ws.stream);
    inwt_1d_opt_batched(d_words_c1_eval, gpu_cc.d_itwiddles(), gpu_cc.d_itwiddles_shoup(), gpu_cc.d_modulus(), gpu_cc.d_n_inv_mod_q(),
                        gpu_cc.d_n_inv_mod_q_shoup(), N, static_cast<size_t>(blocks) * 4u, gpu_ws.stream);

    const dim3 block(256);
    const bool fused_extract_ks = UseFusedExtractKS();
    bool direct_lwe_out = false;
    if (fused_extract_ks) {
        direct_lwe_out = RunFusedExtractKeySwitchToLWE(
            gpu_ksk, gpu_ws, reinterpret_cast<const uint64_t*>(d_words_c0_eval), reinterpret_cast<const uint64_t*>(d_words_c1_eval),
            gpu_ws.d_extract_info, d_out_a, d_out_b, batch, Q64, q_over_Q, q_divisible, scale, q_lwe_u64, out_stride);
        if (!direct_lwe_out && !RunFusedExtractKeySwitch(gpu_ksk, gpu_ws, reinterpret_cast<const uint64_t*>(d_words_c0_eval),
                                                         reinterpret_cast<const uint64_t*>(d_words_c1_eval), gpu_ws.d_extract_info, batch,
                                                         Q64, q_over_Q)) {
            OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDeviceBatch: fused extract+KeySwitch failed");
        }
    } else {
        const dim3 grid((N + block.x - 1) / block.x, static_cast<uint32_t>(batch));
        aes_example::kernel_SampleExtractPackU16<<<grid, block, 0, gpu_ws.stream>>>(
            gpu_ws.d_in_a, gpu_ws.d_in_b, reinterpret_cast<const uint64_t*>(d_words_c0_eval),
            reinterpret_cast<const uint64_t*>(d_words_c1_eval), gpu_ws.d_extract_info, static_cast<uint32_t>(batch), N, Q64,
            q_over_Q, gpu_ksk.qKS_mask);
        PHANTOM_CHECK_CUDA_LAST();

        if (!aes_example::KeySwitchBatchDevice(gpu_ksk, gpu_ws, batch)) {
            OPENFHE_THROW(config_error, "ExtractStateBytesToLWEDeviceBatch: KeySwitchBatchDevice failed");
        }
    }
    if (direct_lwe_out) {
        return true;
    }

    const uint32_t out_len = n + 1u;
    const uint32_t total = static_cast<uint32_t>(batch) * out_len;
    const dim3 grid_out((total + block.x - 1) / block.x);
    if (q_divisible) {
        kernel_U16ToLWE<<<grid_out, block, 0, gpu_ws.stream>>>(
            d_out_a, d_out_b, gpu_ws.d_out, n, static_cast<uint32_t>(batch), scale, out_stride);
    } else {
        kernel_U16ToLWE_ModSwitch<<<grid_out, block, 0, gpu_ws.stream>>>(
            d_out_a, d_out_b, gpu_ws.d_out, n, static_cast<uint32_t>(batch), qks_u32, q_lwe_u64, out_stride);
    }
    PHANTOM_CHECK_CUDA_LAST();
    return true;
}

std::vector<LWECiphertext> ExtractStateBytesToLWETwitter(const std::shared_ptr<CirBTSCryptoParams>& params, const std::shared_ptr<LWECryptoParams>& lweParamsKS,
                                                        const LWEEncryptionScheme& lweScheme, ConstLWESwitchingKey& ksk_rn_to_n,
                                                        const std::array<RLWECiphertext, 4>& stateWords,
                                                        const aes_example::GpuLweKeySwitchKeyU16* gpu_ksk = nullptr,
                                                        aes_example::GpuLweKeySwitchWorkspaceU16* gpu_ws = nullptr) {
    std::vector<LWECiphertext> out(16 * 8);
    auto rlweParams = params->GetRLWEParams();

    std::array<RLWECiphertext, 4> ct_coeff{};
    for (uint32_t col = 0; col < 4; ++col) {
        auto ct = std::make_shared<RLWECiphertextImpl>(stateWords[col]->GetElements());
        ct->SetFormat(COEFFICIENT);
        ct_coeff[col] = std::move(ct);
    }

#ifdef __CUDACC__
    if (gpu_ksk && gpu_ws) {
        const size_t batch = 128;
        const uint32_t N = gpu_ksk->N;
        const uint32_t n = gpu_ksk->n;
        const NativeInteger qKS = lweParamsKS->GetqKS();

        std::vector<uint16_t> h_out(batch * (static_cast<size_t>(n) + 1));

        const uint64_t Q64 = rlweParams->GetQ().ConvertToInt();
        const uint32_t qks_u32 = gpu_ksk->qKS;
        if ((Q64 == 0) || (qks_u32 == 0) || ((qks_u32 & (qks_u32 - 1u)) != 0) || (Q64 < qks_u32)) {
            std::cerr << "[AES] GPU extract pack disabled: requires power-of-two qKS with Q>=qKS." << std::endl;
        } else {
            const double q_over_Q = static_cast<double>(qks_u32) / static_cast<double>(Q64);
            std::array<uint32_t, 128> h_info{};
            for (uint32_t t = 0; t < 128; ++t) {
                const uint32_t byteIndex = t / 8u;
                const uint32_t b         = t % 8u;
                const uint32_t col       = byteIndex / 4u;
                const uint32_t row       = byteIndex % 4u;
                const uint32_t startBit  = (3u - row) * 8u;
                const uint32_t bitIdx    = startBit + b;
                h_info[t] = (col << 16) | (bitIdx & 0xFFFFu);
            }

            std::vector<uint64_t> h_rlwe_a(static_cast<size_t>(4) * N);
            std::vector<uint64_t> h_rlwe_b(static_cast<size_t>(4) * N);
            for (uint32_t col = 0; col < 4; ++col) {
                const auto& a = ct_coeff[col]->GetElements()[0].GetValues();
                const auto& b = ct_coeff[col]->GetElements()[1].GetValues();
                std::memcpy(h_rlwe_a.data() + static_cast<size_t>(col) * N, &a[0], N * sizeof(uint64_t));
                std::memcpy(h_rlwe_b.data() + static_cast<size_t>(col) * N, &b[0], N * sizeof(uint64_t));
            }

            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(gpu_ws->d_rlwe_a, h_rlwe_a.data(), h_rlwe_a.size() * sizeof(uint64_t),
                                               cudaMemcpyHostToDevice, gpu_ws->stream));
            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(gpu_ws->d_rlwe_b, h_rlwe_b.data(), h_rlwe_b.size() * sizeof(uint64_t),
                                               cudaMemcpyHostToDevice, gpu_ws->stream));
            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(gpu_ws->d_extract_info, h_info.data(), h_info.size() * sizeof(uint32_t),
                                               cudaMemcpyHostToDevice, gpu_ws->stream));

            const dim3 block(256);
            const bool fused_extract_ks = UseFusedExtractKS();
            if (!fused_extract_ks) {
                const dim3 grid((N + block.x - 1) / block.x, static_cast<uint32_t>(batch));
                aes_example::kernel_SampleExtractPackU16<<<grid, block, 0, gpu_ws->stream>>>(
                    gpu_ws->d_in_a, gpu_ws->d_in_b, gpu_ws->d_rlwe_a, gpu_ws->d_rlwe_b, gpu_ws->d_extract_info,
                    static_cast<uint32_t>(batch), N, Q64, q_over_Q, gpu_ksk->qKS_mask);
                PHANTOM_CHECK_CUDA_LAST();
            }

            const bool ks_ok = fused_extract_ks
                                   ? RunFusedExtractKeySwitch(*gpu_ksk, *gpu_ws, gpu_ws->d_rlwe_a, gpu_ws->d_rlwe_b,
                                                              gpu_ws->d_extract_info, batch, Q64, q_over_Q)
                                   : aes_example::KeySwitchBatchDevice(*gpu_ksk, *gpu_ws, batch);
            if (!ks_ok) {
                std::cerr << "[AES] GPU KeySwitchBatchDevice failed; falling back to CPU KeySwitch." << std::endl;
            } else {
                const uint32_t total = static_cast<uint32_t>(batch) * (n + 1u);
                PHANTOM_CHECK_CUDA(cudaMemcpyAsync(h_out.data(), gpu_ws->d_out, static_cast<size_t>(total) * sizeof(uint16_t),
                                                   cudaMemcpyDeviceToHost, gpu_ws->stream));
                PHANTOM_CHECK_CUDA(cudaStreamSynchronize(gpu_ws->stream));
                for (uint32_t idx = 0; idx < 128; ++idx) {
                    NativeVector a(n, qKS);
                    const size_t base = static_cast<size_t>(idx) * (static_cast<size_t>(n) + 1);
                    for (uint32_t k = 0; k < n; ++k) {
                        a[k] = NativeInteger(h_out[base + k]);
                    }
                    NativeInteger b = NativeInteger(h_out[base + n]);
                    auto ctKS = std::make_shared<LWECiphertextImpl>(std::move(a), b);
                    out[idx] = lweScheme.ModSwitch(lweParamsKS->Getq(), ctKS);
                }
                return out;
            }
        }
    }
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int idx = 0; idx < 128; ++idx) {
        const uint32_t byteIndex = static_cast<uint32_t>(idx / 8);
        const uint32_t b         = static_cast<uint32_t>(idx % 8);
        const uint32_t col       = byteIndex / 4;
        const uint32_t row       = byteIndex % 4;
        const uint32_t startBit  = (3u - row) * 8u;
        const uint32_t bitIdx    = startBit + b;

        auto tlweN             = SampleExtractIndex(rlweParams, ct_coeff[col], bitIdx, /*p=*/2);
        out[byteIndex * 8 + b] = SwitchTLWENToTLWEn(lweScheme, lweParamsKS, ksk_rn_to_n, tlweN);
    }
    return out;
}
