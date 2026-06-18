// RLWE word/table helpers and CPU reference LUT evaluation.

// RLWE helpers: encode/encrypt 32-bit word (bits) in the first 32 coefficients, p=2.
// ---------------------------
RLWECiphertext EncryptWord32RLWE(const std::shared_ptr<RLWECryptoParams>& rlweParams, const RLWEPrivateKey& sk2, uint32_t w,
                                 LWEPlaintextModulus p = 2) {
    RLWEEncryptionScheme rlwecontext;
    auto polyParams = rlweParams->GetPolyParams();
    const auto Q = polyParams->GetModulus();
    NativePoly m(polyParams, COEFFICIENT, true);
    for (uint32_t b = 0; b < 32; ++b) {
        m[b].SetValue(NativeInteger((w >> b) & 1u));
    }
    return rlwecontext.Encrypt(rlweParams, sk2, std::move(m), p, Q);
}

std::vector<RLWECiphertext> BuildTableRLWE(const std::shared_ptr<RLWECryptoParams>& rlweParams, const RLWEPrivateKey& sk2,
                                          const Table256& table) {
    std::vector<RLWECiphertext> out(256);
    for (uint32_t x = 0; x < 256; ++x) {
        out[x] = EncryptWord32RLWE(rlweParams, sk2, table[x], /*p=*/2);
    }
    return out;
}

RLWECiphertext XorRLWE(const RLWECiphertext& a, const RLWECiphertext& b) {
    std::vector<NativePoly> out = {a->GetElements()[0] + b->GetElements()[0], a->GetElements()[1] + b->GetElements()[1]};
    return std::make_shared<RLWECiphertextImpl>(std::move(out));
}

RLWECiphertext XorRLWE4(const RLWECiphertext& a, const RLWECiphertext& b, const RLWECiphertext& c, const RLWECiphertext& d) {
    std::vector<NativePoly> out = {a->GetElements()[0] + b->GetElements()[0] + c->GetElements()[0] + d->GetElements()[0],
                                   a->GetElements()[1] + b->GetElements()[1] + c->GetElements()[1] + d->GetElements()[1]};
    return std::make_shared<RLWECiphertextImpl>(std::move(out));
}

// Decrypt an RLWE word (first 32 coeffs) to uint32 (p=2).
uint32_t DecryptWord32(const std::shared_ptr<RLWECryptoParams>& rlweParams, const RLWEPrivateKey& sk2, const RLWECiphertext& ct) {
    RLWEEncryptionScheme rlwecontext;
    NativePoly m(rlweParams->GetPolyParams(), COEFFICIENT, false);
    rlwecontext.Decrypt(rlweParams, sk2, ct, &m, /*p=*/2);
    uint32_t w = 0;
    for (uint32_t b = 0; b < 32; ++b) {
        w |= (m[b].ConvertToInt<uint32_t>() & 1u) << b;
    }
    return w;
}

// ---------------------------
// CPU CMUX (reference) 8->32 LUT.
// ---------------------------
RLWECiphertext EvalExternalProduct(const std::shared_ptr<RLWECryptoParams>& rlweParams, const RGSWCiphertext& ct_gsw,
                                   const RLWECiphertext& ct_in, uint32_t basecc, uint32_t digitscc) {
    auto polyParams = rlweParams->GetPolyParams();
    RLWEEncryptionScheme rlwecontext;

    std::vector<NativePoly> digits(2 * digitscc, NativePoly(polyParams, COEFFICIENT, true));
    auto ct_coeff = std::make_shared<RLWECiphertextImpl>(ct_in->GetElements());
    ct_coeff->SetFormat(COEFFICIENT);
    rlwecontext.SignedDigitDecompose(rlweParams, ct_coeff, basecc, digitscc, digits);

    for (auto& d : digits) {
        d.SetFormat(EVALUATION);
    }

    std::vector<NativePoly> out(2);
    out[0] = NativePoly(polyParams, EVALUATION, true);
    out[1] = NativePoly(polyParams, EVALUATION, true);

    const uint32_t rows = digitscc * 2;
    const auto& gsw = ct_gsw->GetElements();
    for (uint32_t r = 0; r < rows; ++r) {
        out[0] += digits[r] * gsw[r][0];
        out[1] += digits[r] * gsw[r][1];
    }
    return std::make_shared<RLWECiphertextImpl>(std::move(out));
}

RLWECiphertext EvalCMUX(const std::shared_ptr<RLWECryptoParams>& rlweParams, const RGSWCiphertext& ct_gsw, const RLWECiphertext& ct0,
                        const RLWECiphertext& ct1, uint32_t basecc, uint32_t digitscc) {
    auto diff = std::make_shared<RLWECiphertextImpl>(ct1->GetElements());
    diff->SetFormat(EVALUATION);
    diff->GetElements()[0] -= ct0->GetElements()[0];
    diff->GetElements()[1] -= ct0->GetElements()[1];

    auto prod = EvalExternalProduct(rlweParams, ct_gsw, diff, basecc, digitscc);
    prod->SetFormat(EVALUATION);
    prod->GetElements()[0] += ct0->GetElements()[0];
    prod->GetElements()[1] += ct0->GetElements()[1];
    return prod;
}

RLWECiphertext EvalLUT8x32_RLWE_CPU(const std::shared_ptr<RLWECryptoParams>& rlweParams, const std::array<RGSWCiphertext, 8>& ctrl_bits_lsb_to_msb,
                                    const std::vector<RLWECiphertext>& table, uint32_t basecc, uint32_t digitscc) {
    if (table.size() != 256) {
        OPENFHE_THROW(config_error, "EvalLUT8x32_RLWE_CPU: expected table size 256");
    }
    std::vector<RLWECiphertext> cur = table;
    for (uint32_t bit = 0; bit < 8; ++bit) {
        std::vector<RLWECiphertext> next;
        next.reserve(cur.size() / 2);
        for (size_t i = 0; i < cur.size(); i += 2) {
            next.push_back(EvalCMUX(rlweParams, ctrl_bits_lsb_to_msb[bit], cur[i], cur[i + 1], basecc, digitscc));
        }
        cur = std::move(next);
    }
    return cur[0];
}
