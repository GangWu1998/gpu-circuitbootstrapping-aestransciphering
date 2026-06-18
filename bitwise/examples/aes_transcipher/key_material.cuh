// Encrypted AES key material construction.

struct EncryptedAesKeyMaterial {
    std::array<std::array<uint8_t, 16>, 11> roundKeys{};
    std::vector<LWECiphertext> rk10_bits;
    std::vector<LWECiphertext> rk0_bits;
    std::array<RLWECiphertext, 4> rk0_words{};
    std::array<std::array<RLWECiphertext, 4>, 11> rk_imc_words{};
    std::array<std::array<RLWECiphertext, 4>, 11> rk_words{};
};

EncryptedAesKeyMaterial BuildEncryptedKeyMaterial(const CirBTSContext& cc, const std::shared_ptr<RLWECryptoParams>& rlweParams,
                                                  const LWEPrivateKey& sk,
                                                  const RLWEPrivateKey& sk2, const std::array<uint8_t, 16>& key) {
    EncryptedAesKeyMaterial out;
    out.roundKeys = Aes128KeyExpand(key);
    out.rk10_bits.reserve(16 * 8);
    out.rk0_bits.reserve(16 * 8);
    for (uint32_t i = 0; i < 16; ++i) {
        const uint8_t byte10 = out.roundKeys[10][i];
        const uint8_t byte0 = out.roundKeys[0][i];
        for (uint32_t b = 0; b < 8; ++b) {
            out.rk10_bits.emplace_back(cc.Encrypt(sk, static_cast<LWEPlaintext>((byte10 >> b) & 1u), /*p=*/2));
            out.rk0_bits.emplace_back(cc.Encrypt(sk, static_cast<LWEPlaintext>((byte0 >> b) & 1u), /*p=*/2));
        }
    }
    for (uint32_t col = 0; col < 4; ++col) {
        const uint32_t w = pack_u32_be(out.roundKeys[0][AesIndex(0, col)], out.roundKeys[0][AesIndex(1, col)], out.roundKeys[0][AesIndex(2, col)],
                                       out.roundKeys[0][AesIndex(3, col)]);
        out.rk0_words[col] = EncryptWord32RLWE(rlweParams, sk2, w, /*p=*/2);
    }
    for (uint32_t round = 0; round <= 10; ++round) {
        for (uint32_t col = 0; col < 4; ++col) {
            const uint32_t w = pack_u32_be(out.roundKeys[round][AesIndex(0, col)], out.roundKeys[round][AesIndex(1, col)],
                                           out.roundKeys[round][AesIndex(2, col)], out.roundKeys[round][AesIndex(3, col)]);
            out.rk_words[round][col] = EncryptWord32RLWE(rlweParams, sk2, w, /*p=*/2);
        }
    }
    for (uint32_t round = 1; round <= 9; ++round) {
        for (uint32_t col = 0; col < 4; ++col) {
            const uint32_t w = pack_u32_be(out.roundKeys[round][AesIndex(0, col)], out.roundKeys[round][AesIndex(1, col)],
                                           out.roundKeys[round][AesIndex(2, col)], out.roundKeys[round][AesIndex(3, col)]);
            out.rk_imc_words[round][col] = EncryptWord32RLWE(rlweParams, sk2, InvMixColumnsWord(w), /*p=*/2);
        }
    }
    return out;
}
