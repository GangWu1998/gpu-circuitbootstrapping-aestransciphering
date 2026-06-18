// Plain AES-128, CTR, and AES T-table construction helpers.

constexpr uint32_t rotr32(uint32_t x, uint32_t r) {
    return (x >> (r & 31u)) | (x << ((32u - r) & 31u));
}
constexpr std::array<uint8_t, 256> kAesSbox = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, 0xca, 0x82,
    0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26,
    0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96,
    0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb,
    0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f,
    0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff,
    0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32,
    0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
    0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6,
    0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e,
    0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f,
    0xb0, 0x54, 0xbb, 0x16};

constexpr uint8_t xtime(uint8_t x) {
    return static_cast<uint8_t>((x << 1) ^ ((x & 0x80u) ? 0x1bu : 0x00u));
}
constexpr uint8_t gf_mul2(uint8_t x) {
    return xtime(x);
}
constexpr uint8_t gf_mul3(uint8_t x) {
    return static_cast<uint8_t>(xtime(x) ^ x);
}
constexpr uint8_t gf_mul9(uint8_t x) {
    return static_cast<uint8_t>(xtime(xtime(xtime(x))) ^ x);
}

struct AesTestVector {
    std::array<uint8_t, 16> key{};
    std::array<uint8_t, 16> pt{};
    std::array<uint8_t, 16> ct{};
};

const std::array<AesTestVector, 1> kNistAes128 = {
    AesTestVector{
        {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f},
        {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
        {0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a}},
};
constexpr uint8_t gf_mul11(uint8_t x) {
    return static_cast<uint8_t>(xtime(xtime(xtime(x))) ^ xtime(x) ^ x);
}
constexpr uint8_t gf_mul13(uint8_t x) {
    return static_cast<uint8_t>(xtime(xtime(xtime(x))) ^ xtime(xtime(x)) ^ x);
}
constexpr uint8_t gf_mul14(uint8_t x) {
    return static_cast<uint8_t>(xtime(xtime(xtime(x))) ^ xtime(xtime(x)) ^ xtime(x));
}

constexpr uint32_t pack_u32_be(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    return (static_cast<uint32_t>(b0) << 24) | (static_cast<uint32_t>(b1) << 16) | (static_cast<uint32_t>(b2) << 8) |
           static_cast<uint32_t>(b3);
}

constexpr uint32_t AesIndex(uint32_t row, uint32_t col) {
    return col * 4 + row;  // column-major
}

using RoundKeys = std::array<std::array<uint8_t, 16>, 11>;

uint32_t RotWord(uint32_t w) {
    return (w << 8) | (w >> 24);
}

std::array<uint8_t, 256> BuildInvSbox() {
    std::array<uint8_t, 256> inv{};
    for (uint32_t i = 0; i < 256; ++i) {
        inv[kAesSbox[i]] = static_cast<uint8_t>(i);
    }
    return inv;
}

uint32_t SubWord(uint32_t w) {
    const uint8_t b0 = kAesSbox[(w >> 24) & 0xFFu];
    const uint8_t b1 = kAesSbox[(w >> 16) & 0xFFu];
    const uint8_t b2 = kAesSbox[(w >> 8) & 0xFFu];
    const uint8_t b3 = kAesSbox[w & 0xFFu];
    return pack_u32_be(b0, b1, b2, b3);
}

RoundKeys Aes128KeyExpand(const std::array<uint8_t, 16>& key) {
    constexpr std::array<uint8_t, 10> rcon = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36};
    std::array<uint32_t, 44> w{};
    for (uint32_t i = 0; i < 4; ++i) {
        w[i] = pack_u32_be(key[4 * i + 0], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]);
    }
    for (uint32_t i = 4; i < 44; ++i) {
        uint32_t temp = w[i - 1];
        if ((i & 3u) == 0u) {
            temp = SubWord(RotWord(temp)) ^ (static_cast<uint32_t>(rcon[(i / 4) - 1]) << 24);
        }
        w[i] = w[i - 4] ^ temp;
    }

    RoundKeys rk{};
    for (uint32_t round = 0; round < 11; ++round) {
        for (uint32_t col = 0; col < 4; ++col) {
            const uint32_t word = w[round * 4 + col];
            rk[round][AesIndex(0, col)] = static_cast<uint8_t>((word >> 24) & 0xFFu);
            rk[round][AesIndex(1, col)] = static_cast<uint8_t>((word >> 16) & 0xFFu);
            rk[round][AesIndex(2, col)] = static_cast<uint8_t>((word >> 8) & 0xFFu);
            rk[round][AesIndex(3, col)] = static_cast<uint8_t>(word & 0xFFu);
        }
    }
    return rk;
}

void AesAddRoundKey(std::array<uint8_t, 16>& state, const std::array<uint8_t, 16>& rk) {
    for (uint32_t i = 0; i < 16; ++i) {
        state[i] ^= rk[i];
    }
}

void AesSubBytes(std::array<uint8_t, 16>& state) {
    for (uint32_t i = 0; i < 16; ++i) {
        state[i] = kAesSbox[state[i]];
    }
}

void AesShiftRows(std::array<uint8_t, 16>& state) {
    std::array<uint8_t, 16> tmp{};
    for (uint32_t row = 0; row < 4; ++row) {
        for (uint32_t col = 0; col < 4; ++col) {
            tmp[AesIndex(row, col)] = state[AesIndex(row, (col + row) & 3u)];
        }
    }
    state = tmp;
}

void AesMixColumns(std::array<uint8_t, 16>& state) {
    for (uint32_t col = 0; col < 4; ++col) {
        const uint8_t a0 = state[AesIndex(0, col)];
        const uint8_t a1 = state[AesIndex(1, col)];
        const uint8_t a2 = state[AesIndex(2, col)];
        const uint8_t a3 = state[AesIndex(3, col)];
        state[AesIndex(0, col)] = gf_mul2(a0) ^ gf_mul3(a1) ^ a2 ^ a3;
        state[AesIndex(1, col)] = a0 ^ gf_mul2(a1) ^ gf_mul3(a2) ^ a3;
        state[AesIndex(2, col)] = a0 ^ a1 ^ gf_mul2(a2) ^ gf_mul3(a3);
        state[AesIndex(3, col)] = gf_mul3(a0) ^ a1 ^ a2 ^ gf_mul2(a3);
    }
}

std::array<uint8_t, 16> Aes128EncryptPlain(const std::array<uint8_t, 16>& in, const RoundKeys& rk) {
    std::array<uint8_t, 16> state = in;
    AesAddRoundKey(state, rk[0]);
    for (uint32_t round = 1; round <= 9; ++round) {
        AesSubBytes(state);
        AesShiftRows(state);
        AesMixColumns(state);
        AesAddRoundKey(state, rk[round]);
    }
    AesSubBytes(state);
    AesShiftRows(state);
    AesAddRoundKey(state, rk[10]);
    return state;
}

void AesInvShiftRows(std::array<uint8_t, 16>& state) {
    std::array<uint8_t, 16> tmp{};
    for (uint32_t row = 0; row < 4; ++row) {
        for (uint32_t col = 0; col < 4; ++col) {
            tmp[AesIndex(row, col)] = state[AesIndex(row, (col + 4u - row) & 3u)];
        }
    }
    state = tmp;
}

void AesInvSubBytes(std::array<uint8_t, 16>& state, const std::array<uint8_t, 256>& invSbox) {
    for (uint32_t i = 0; i < 16; ++i) {
        state[i] = invSbox[state[i]];
    }
}

void AesInvMixColumns(std::array<uint8_t, 16>& state) {
    for (uint32_t col = 0; col < 4; ++col) {
        const uint8_t a0 = state[AesIndex(0, col)];
        const uint8_t a1 = state[AesIndex(1, col)];
        const uint8_t a2 = state[AesIndex(2, col)];
        const uint8_t a3 = state[AesIndex(3, col)];
        state[AesIndex(0, col)] = gf_mul14(a0) ^ gf_mul11(a1) ^ gf_mul13(a2) ^ gf_mul9(a3);
        state[AesIndex(1, col)] = gf_mul9(a0) ^ gf_mul14(a1) ^ gf_mul11(a2) ^ gf_mul13(a3);
        state[AesIndex(2, col)] = gf_mul13(a0) ^ gf_mul9(a1) ^ gf_mul14(a2) ^ gf_mul11(a3);
        state[AesIndex(3, col)] = gf_mul11(a0) ^ gf_mul13(a1) ^ gf_mul9(a2) ^ gf_mul14(a3);
    }
}

std::array<uint8_t, 16> Aes128DecryptPlain(const std::array<uint8_t, 16>& in, const RoundKeys& rk) {
    const auto invSbox = BuildInvSbox();
    std::array<uint8_t, 16> state = in;
    AesAddRoundKey(state, rk[10]);
    for (uint32_t round = 9; round >= 1; --round) {
        AesInvShiftRows(state);
        AesInvSubBytes(state, invSbox);
        AesAddRoundKey(state, rk[round]);
        AesInvMixColumns(state);
    }
    AesInvShiftRows(state);
    AesInvSubBytes(state, invSbox);
    AesAddRoundKey(state, rk[0]);
    return state;
}

// ---------------------------
// Decryption T-tables: Td0..Td3 and final-round Td4_*.
// ---------------------------
using Table256 = std::array<uint32_t, 256>;

struct AesDecTables {
    Table256 td0{};
    Table256 td1{};
    Table256 td2{};
    Table256 td3{};
    Table256 td4_0{};
    Table256 td4_1{};
    Table256 td4_2{};
    Table256 td4_3{};
};

AesDecTables BuildAesDecTables() {
    const auto invSbox = BuildInvSbox();
    AesDecTables t{};
    for (uint32_t x = 0; x < 256; ++x) {
        const uint8_t s = invSbox[x];
        const uint32_t td0 = pack_u32_be(gf_mul14(s), gf_mul9(s), gf_mul13(s), gf_mul11(s));
        t.td0[x] = td0;
        t.td1[x] = rotr32(td0, 8);
        t.td2[x] = rotr32(td0, 16);
        t.td3[x] = rotr32(td0, 24);

        t.td4_0[x] = pack_u32_be(s, 0, 0, 0);
        t.td4_1[x] = pack_u32_be(0, s, 0, 0);
        t.td4_2[x] = pack_u32_be(0, 0, s, 0);
        t.td4_3[x] = pack_u32_be(0, 0, 0, s);
    }
    return t;
}

// ---------------------------
// Encryption T-tables: Te0..Te3 and final-round Te4_* (CTR uses AES-ENC).
// ---------------------------
struct AesEncTables {
    Table256 te0{};
    Table256 te1{};
    Table256 te2{};
    Table256 te3{};
    Table256 te4_0{};
    Table256 te4_1{};
    Table256 te4_2{};
    Table256 te4_3{};
};

AesEncTables BuildAesEncTables() {
    AesEncTables t{};
    for (uint32_t x = 0; x < 256; ++x) {
        const uint8_t s = kAesSbox[x];
        const uint32_t te0 = pack_u32_be(gf_mul2(s), s, s, gf_mul3(s));
        t.te0[x] = te0;
        t.te1[x] = rotr32(te0, 8);
        t.te2[x] = rotr32(te0, 16);
        t.te3[x] = rotr32(te0, 24);

        t.te4_0[x] = pack_u32_be(s, 0, 0, 0);
        t.te4_1[x] = pack_u32_be(0, s, 0, 0);
        t.te4_2[x] = pack_u32_be(0, 0, s, 0);
        t.te4_3[x] = pack_u32_be(0, 0, 0, s);
    }
    return t;
}

bool ParseHex128(const std::string& hex, std::array<uint8_t, 16>& out) {
    if (hex.size() != 32) {
        return false;
    }
    auto hexval = [](char c) -> int {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'a' && c <= 'f')
            return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F')
            return 10 + (c - 'A');
        return -1;
    };
    for (size_t i = 0; i < 16; ++i) {
        const int hi = hexval(hex[2 * i]);
        const int lo = hexval(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            return false;
        }
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

std::string Hex128(const std::array<uint8_t, 16>& in) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(32, '0');
    for (size_t i = 0; i < in.size(); ++i) {
        out[2 * i] = kHex[in[i] >> 4];
        out[2 * i + 1] = kHex[in[i] & 0x0Fu];
    }
    return out;
}

uint64_t LoadBE64(const std::array<uint8_t, 16>& in, size_t offset) {
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) {
        v = (v << 8) | static_cast<uint64_t>(in[offset + i]);
    }
    return v;
}

void StoreBE64(uint64_t v, std::array<uint8_t, 16>& out, size_t offset) {
    for (size_t i = 0; i < 8; ++i) {
        out[offset + 7 - i] = static_cast<uint8_t>(v & 0xFFu);
        v >>= 8;
    }
}

std::array<uint8_t, 16> MakeCtrBlock(const std::array<uint8_t, 16>& base, uint64_t block_idx) {
    std::array<uint8_t, 16> ctr = base;
    const uint64_t lo = LoadBE64(ctr, 8);
    const uint64_t lo_new = lo + block_idx;
    StoreBE64(lo_new, ctr, 8);
    return ctr;
}

uint32_t InvMixColumnsWord(uint32_t w) {
    const uint8_t b0 = static_cast<uint8_t>((w >> 24) & 0xFFu);
    const uint8_t b1 = static_cast<uint8_t>((w >> 16) & 0xFFu);
    const uint8_t b2 = static_cast<uint8_t>((w >> 8) & 0xFFu);
    const uint8_t b3 = static_cast<uint8_t>(w & 0xFFu);
    const uint8_t o0 = gf_mul14(b0) ^ gf_mul11(b1) ^ gf_mul13(b2) ^ gf_mul9(b3);
    const uint8_t o1 = gf_mul9(b0) ^ gf_mul14(b1) ^ gf_mul11(b2) ^ gf_mul13(b3);
    const uint8_t o2 = gf_mul13(b0) ^ gf_mul9(b1) ^ gf_mul14(b2) ^ gf_mul11(b3);
    const uint8_t o3 = gf_mul11(b0) ^ gf_mul13(b1) ^ gf_mul9(b2) ^ gf_mul14(b3);
    return pack_u32_be(o0, o1, o2, o3);
}
