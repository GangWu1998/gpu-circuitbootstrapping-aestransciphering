#include "kernel.cuh"
#include "kernels_fft.cuh"
#include "rgsw-acc.cuh"

#include "math/distributiongenerator.h"

#include "arith/big_integer_modop.h"

using namespace lbcrypto;

namespace phantom::bitwise
{
    void GPURingGSWAccumulator::GPUKeyGenAcc(GPURingGSWBTKey& gpu_ek,
                                             const std::shared_ptr<RingGSWCryptoParams>& params,
                                             const NativeVector& sk,
                                             ConstLWEPrivateKey& LWEsk,
                                             const cudaStream_t& stream)
    {
        // initialize NTT
        const size_t N = params->GetN();
        const BasicInteger Q = params->GetQ().ConvertToInt();
        const BasicInteger P = params->GetP().ConvertToInt();
        const BasicInteger PQ = params->GetPQ().ConvertToInt();

        const auto& polyParams = (params->IsCompositeNTT())
                                     ? params->GetCompositePolyParams()
                                     : params->GetPolyParams();

        if (params->IsCompositeNTT())
            ntt_ = std::make_shared<FourStepNTT>(N, P, Q, stream);
        else
            ntt_ = std::make_shared<FourStepNTT>(N, Q, stream);

        // initialize monic polynomials for CGGI
        if (params->GetMethod() == lbcrypto::BINFHE_METHOD::GINX)
        {
            // Precomputed polynomials in Format::EVALUATION representation for X^m - 1
            // (used only for CGGI bootstrapping)
            std::vector<NativePoly> monomials;

            constexpr NativeInteger one{1};
            monomials.reserve(2 * N);
            for (uint32_t i = 0; i < N; ++i)
            {
                NativePoly aPoly(polyParams, Format::COEFFICIENT, true);
                if (params->IsCompositeNTT())
                {
                    // composite NTT
                    aPoly[0].ModSubFastEq(one, PQ); // -1
                    aPoly[i].ModAddFastEq(one, PQ); // X^m
                }
                else
                {
                    // gadget decompose
                    aPoly[0].ModSubFastEq(one, Q); // -1
                    aPoly[i].ModAddFastEq(one, Q); // X^m
                }
                monomials.push_back(std::move(aPoly));
            }
            for (uint32_t i = 0; i < N; ++i)
            {
                NativePoly aPoly(polyParams, Format::COEFFICIENT, true);
                if (params->IsCompositeNTT())
                {
                    // composite NTT
                    aPoly[0].ModSubFastEq(one, PQ); // -1
                    aPoly[i].ModSubFastEq(one, PQ); // -X^m
                }
                else
                {
                    // gadget decompose
                    aPoly[0].ModSubFastEq(one, Q); // -1
                    aPoly[i].ModSubFastEq(one, Q); // -X^m
                }
                monomials.push_back(std::move(aPoly));
            }

            d_monic_polys_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * N, stream);
            for (size_t i = 0; i < 2 * N; i++)
            {
                cudaMemcpyAsync(d_monic_polys_.get() + i * N, &monomials[i].GetValues().at(0),
                                N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);
            }

            if (use_fft_)
            {
                d_monic_polys_fft_ = phantom::util::make_cuda_auto_ptr<complex_t>(2 * N * (N / 2), stream);
                if (N == 1024)
                    fft_1024_->i2c_forward(d_monic_polys_fft_.get(), d_monic_polys_.get(), 2 * N, stream);
                else if (N == 2048)
                    fft_2048_->i2c_forward(d_monic_polys_fft_.get(), d_monic_polys_.get(), 2 * N, stream);
                else if (N == 4096)
                    fft_4096_->i2c_forward(d_monic_polys_fft_.get(), d_monic_polys_.get(), 2 * N, stream);
                else
                    throw std::invalid_argument("ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");
            }

            ntt_->forward_shoup(d_monic_polys_.get(), d_monic_polys_.get(), 256, 2 * N, stream);
            // to Montgomery domain
            ntt_->to_mont(d_monic_polys_.get(), d_monic_polys_.get(), 256, 2 * N, stream);
        }

        auto d_skNTT = phantom::util::make_cuda_auto_ptr<BasicInteger>(N, stream);
        cudaMemcpyAsync(d_skNTT.get(), &sk.at(0), N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);
        ntt_->forward_shoup(d_skNTT.get(), d_skNTT.get(), 256, 1, stream);

        switch (params->GetMethod())
        {
        case BINFHE_METHOD::AP:
            GPUKeyGenAccDM(gpu_ek, params, d_skNTT, LWEsk, stream);
            return;
        case BINFHE_METHOD::GINX:
            GPUKeyGenAccCGGI(gpu_ek, params, d_skNTT, LWEsk, stream);
            return;
        //            case BINFHE_METHOD::LMKCDEY:
        //                GPUKeyGenLMKCDEY(gpu_ek, params, d_skNTT, LWEsk, stream);
        //            return;
        default:
            throw std::invalid_argument("ERROR: Invalid ACC method");
        }
    }

    // Key generation as described in Section 4 of https://eprint.iacr.org/2014/816
    void GPURingGSWAccumulator::GPUKeyGenAccDM(GPURingGSWBTKey& gpu_ek,
                                               const std::shared_ptr<RingGSWCryptoParams>& params,
                                               const phantom::util::cuda_auto_ptr<BasicInteger>& d_skNTT,
                                               ConstLWEPrivateKey& LWEsk,
                                               const cudaStream_t& stream) const
    {
        auto sv{LWEsk->GetElement()};
        const auto mod{sv.GetModulus().ConvertToInt<int32_t>()};
        const auto modHalf{mod >> 1};
        const uint32_t n(sv.GetLength());
        const int32_t baseR(params->GetBaseR());
        const auto& digitsR = params->GetDigitsR();
        const uint32_t N = params->GetN();
        const uint32_t digitsG2{(params->GetDigitsG() - 1) << 1};

        gpu_ek.RGSWACCKey.resize(
            n, std::vector(
                baseR, std::vector(
                    digitsR.size(), util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * 2 * N, stream))));

        if (use_fft_)
            gpu_ek.RGSWACCKey_fft.resize(
                n, std::vector(
                    baseR, std::vector(
                        digitsR.size(), util::make_cuda_auto_ptr<complex_t>(digitsG2 * 2 * N / 2, stream))));

        for (uint32_t i = 0; i < n; ++i)
        {
            for (int32_t j = 0; j < baseR; ++j)
            {
                for (size_t k = 0; k < digitsR.size(); ++k)
                {
                    auto s{sv[i].ConvertToInt<int32_t>()};
                    LWEPlaintext m = (s > modHalf ? s - mod : s) * j * digitsR[k].ConvertToInt<int32_t>();
                    GPUKeyGenDM(gpu_ek.RGSWACCKey[i][j][k], gpu_ek.RGSWACCKey_fft[i][j][k],
                                params, d_skNTT, m, stream);
                }
            }
        }
    }

    // Key generation as described in Section 4 of https://eprint.iacr.org/2014/816
    void GPURingGSWAccumulator::GPUKeyGenAccCGGI(GPURingGSWBTKey& gpu_ek,
                                                 const std::shared_ptr<RingGSWCryptoParams>& params,
                                                 const phantom::util::cuda_auto_ptr<BasicInteger>& d_skNTT,
                                                 ConstLWEPrivateKey& LWEsk,
                                                 const cudaStream_t& stream) const
    {
        auto sv = LWEsk->GetElement();
        const auto neg = sv.GetModulus().ConvertToInt() - 1;
        const uint32_t n = sv.GetLength();
        const uint32_t N = params->GetN();
        const uint32_t digitsG2{(params->GetDigitsG() - 1) << 1};

        gpu_ek.RGSWACCKey.resize(1, std::vector(
                                     2, std::vector(
                                         n, util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * 2 * N, stream))));

        if (use_fft_)
            gpu_ek.RGSWACCKey_fft.resize(1, std::vector(
                                             2, std::vector(
                                                 n, util::make_cuda_auto_ptr<complex_t>(
                                                     digitsG2 * 2 * N / 2, stream))));

        if (params->GetKeyDist() == UNIFORM_TERNARY)
        {
            // handles ternary secrets using signed mod 3 arithmetic
            // 0 -> {0,0}, 1 -> {1,0}, -1 -> {0,1}
            for (uint32_t i = 0; i < n; ++i)
            {
                const auto s = sv[i].ConvertToInt();
                GPUKeyGenCGGI(gpu_ek.RGSWACCKey[0][0][i], gpu_ek.RGSWACCKey_fft[0][0][i],
                              params, d_skNTT, s == 1 ? 1 : 0, stream);
                GPUKeyGenCGGI(gpu_ek.RGSWACCKey[0][1][i], gpu_ek.RGSWACCKey_fft[0][1][i],
                              params, d_skNTT, s == neg ? 1 : 0, stream);
            }
        }
        else if (params->GetKeyDist() == UNIFORM_BINARY)
        {
            for (uint32_t i = 0; i < n; ++i)
            {
                const auto s = sv[i].ConvertToInt();
                GPUKeyGenCGGI(gpu_ek.RGSWACCKey[0][0][i], gpu_ek.RGSWACCKey_fft[0][0][i],
                              params, d_skNTT, s, stream);
            }
        }
        else
        {
            throw std::invalid_argument("ERROR: Invalid key distribution for CGGI");
        }
    }

    // Encryption as described in Section 5 of https://eprint.iacr.org/2014/816
    // skNTT corresponds to the secret key z

    void GPURingGSWAccumulator::GPUKeyGenDM(phantom::util::cuda_auto_ptr<BasicInteger>& d_keyNTT,
                                            phantom::util::cuda_auto_ptr<complex_t>& d_keyFFT,
                                            const std::shared_ptr<RingGSWCryptoParams>& params,
                                            const phantom::util::cuda_auto_ptr<BasicInteger>& d_skNTT,
                                            LWEPlaintext m,
                                            const cudaStream_t& stream) const
    {
        // composite NTT
        const auto& polyParams = params->IsCompositeNTT()
                                     ? params->GetCompositePolyParams()
                                     : params->GetPolyParams();

        DiscreteUniformGeneratorImpl<NativeVector> dug;
        NativeInteger Q{params->GetQ()};
        NativeInteger P{params->GetP()};
        NativeInteger PQ{params->GetPQ()};

        // Reduce mod q (dealing with negative number as well)
        uint64_t q = params->Getq().ConvertToInt();
        uint32_t N = params->GetN();
        int64_t mm = (((m % q) + q) % q) * (2 * N / q);
        bool isReducedMM;
        if ((isReducedMM = (mm >= N)))
            mm -= N;

        const uint32_t digitsG2{(params->GetDigitsG() - 1) << 1};
        RingGSWEvalKeyImpl result(digitsG2, 2);

        // generate GPowers
        std::vector<NativeInteger> Gpow;
        Gpow.reserve(params->GetDigitsG());
        NativeInteger vTemp = 1;
        for (uint32_t i = 0; i < params->GetDigitsG(); ++i)
        {
            Gpow.push_back(vTemp);
            vTemp = vTemp.ModMulFast(NativeInteger(params->GetBaseG()), polyParams->GetModulus());
        }

        for (uint32_t i = 0; i < digitsG2; ++i)
        {
            result[i][0] = NativePoly(dug, polyParams, Format::COEFFICIENT);
            result[i][1] = NativePoly(params->GetDgg(), polyParams, Format::COEFFICIENT);

            auto d_tempA = phantom::util::make_cuda_auto_ptr<BasicInteger>(N, stream);
            cudaMemcpyAsync(d_tempA.get(), &result[i][0].GetValues().at(0),
                            N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);

            // 2: hybrid; 1: composite; 0: gadget decompose
            if (!isReducedMM)
            {
                if (params->IsCompositeNTT() == 2)
                    result[i][i & 0x1][mm].ModAddFastEq(Gpow[(i >> 1) + 1], PQ);
                else if (params->IsCompositeNTT() == 1)
                    result[i][i & 0x1][mm].ModAddFastEq(P, PQ);
                else
                    result[i][i & 0x1][mm].ModAddFastEq(Gpow[(i >> 1) + 1], Q);
            }
            else
            {
                if (params->IsCompositeNTT() == 2)
                    result[i][i & 0x1][mm].ModSubFastEq(Gpow[(i >> 1) + 1], PQ);
                else if (params->IsCompositeNTT() == 1)
                    result[i][i & 0x1][mm].ModSubFastEq(P, PQ);
                else
                    result[i][i & 0x1][mm].ModSubFastEq(Gpow[(i >> 1) + 1], Q);
            }

            auto d_key_i = util::make_cuda_auto_ptr<BasicInteger>(2 * N, stream);
            BasicInteger* d_key_i_0 = d_key_i.get();
            BasicInteger* d_key_i_1 = d_key_i_0 + N;
            cudaMemcpyAsync(d_key_i_0, &result[i][0].GetValues().at(0),
                            N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_key_i_1, &result[i][1].GetValues().at(0),
                            N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);

            BasicInteger* d_keyNTT_i = d_keyNTT.get() + i * 2 * N;
            BasicInteger* d_keyNTT_i_0 = d_keyNTT_i;
            BasicInteger* d_keyNTT_i_1 = d_keyNTT_i_0 + N;

            ntt_->forward_shoup(d_keyNTT_i, d_key_i.get(), 256, 2, stream);
            ntt_->forward_shoup(d_tempA.get(), d_tempA.get(), 256, 1, stream);
            ntt_->multiply_and_accumulate(d_keyNTT_i_1, d_tempA.get(), d_skNTT.get(), stream);

            if (use_fft_)
            {
                complex_t* d_keyFFT_i = d_keyFFT.get() + i * 2 * N / 2;
                complex_t* d_keyFFT_i_0 = d_keyFFT_i;
                complex_t* d_keyFFT_i_1 = d_keyFFT_i_0 + N / 2;
                if (N == 1024)
                    fft_1024_->i2c_forward(d_keyFFT_i_0, d_key_i_0, 1, stream);
                else if (N == 2048)
                    fft_2048_->i2c_forward(d_keyFFT_i_0, d_key_i_0, 1, stream);
                else if (N == 4096)
                    fft_4096_->i2c_forward(d_keyFFT_i_0, d_key_i_0, 1, stream);
                else
                    throw std::invalid_argument("ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");

                auto d_key_i_1_tmp = util::make_cuda_auto_ptr<BasicInteger>(N, stream);
                ntt_->inverse_shoup(d_key_i_1_tmp.get(), d_keyNTT_i_1, 256, 1, stream);

                if (N == 1024)
                    fft_1024_->i2c_forward(d_keyFFT_i_1, d_key_i_1_tmp.get(), 1, stream);
                else if (N == 2048)
                    fft_2048_->i2c_forward(d_keyFFT_i_1, d_key_i_1_tmp.get(), 1, stream);
                else if (N == 4096)
                    fft_4096_->i2c_forward(d_keyFFT_i_1, d_key_i_1_tmp.get(), 1, stream);
                else
                    throw std::invalid_argument("ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");
            }

            // to Montgomery domain
            ntt_->to_mont(d_keyNTT_i, d_keyNTT_i, 256, 2, stream);
        }

        cudaStreamSynchronize(stream);
    }

    // Encryption for the CGGI variant, as described in https://eprint.iacr.org/2020/086
    void GPURingGSWAccumulator::GPUKeyGenCGGI(phantom::util::cuda_auto_ptr<BasicInteger>& d_keyNTT,
                                              phantom::util::cuda_auto_ptr<complex_t>& d_keyFFT,
                                              const std::shared_ptr<RingGSWCryptoParams>& params,
                                              const phantom::util::cuda_auto_ptr<BasicInteger>& d_skNTT,
                                              LWEPlaintext m,
                                              const cudaStream_t& stream) const
    {
        // composite NTT
        const auto& polyParams = params->IsCompositeNTT()
                                     ? params->GetCompositePolyParams()
                                     : params->GetPolyParams();

        DiscreteUniformGeneratorImpl<NativeVector> dug;
        NativeInteger Q{params->GetQ()};
        NativeInteger P{params->GetP()};
        NativeInteger PQ{params->GetPQ()};
        uint32_t N = params->GetN();

        uint32_t digitsG2{(params->GetDigitsG() - 1) << 1};
        RingGSWEvalKeyImpl result(digitsG2, 2);

        // generate GPowers
        std::vector<NativeInteger> Gpow;
        Gpow.reserve(params->GetDigitsG());
        NativeInteger vTemp = 1;
        for (uint32_t i = 0; i < params->GetDigitsG(); ++i)
        {
            Gpow.push_back(vTemp);
            vTemp = vTemp.ModMulFast(NativeInteger(params->GetBaseG()), polyParams->GetModulus());
        }

        for (uint32_t i = 0; i < digitsG2; ++i)
        {
            result[i][0] = NativePoly(dug, polyParams, Format::COEFFICIENT);
            result[i][1] = NativePoly(params->GetDgg(), polyParams, Format::COEFFICIENT);

            auto d_tempA = phantom::util::make_cuda_auto_ptr<BasicInteger>(N, stream);
            cudaMemcpyAsync(d_tempA.get(), &result[i][0].GetValues().at(0),
                            N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);

            // 2: hybrid; 1: composite; 0: gadget decompose
            if (m)
            {
                if (params->IsCompositeNTT() == 2)
                {
                    result[i][i & 0x1][0].ModAddFastEq(Gpow[(i >> 1) + 1], PQ);
                }
                else if (params->IsCompositeNTT() == 1)
                    result[i][i & 0x1][0].ModAddFastEq(P, PQ);
                else
                    result[i][i & 0x1][0].ModAddFastEq(Gpow[(i >> 1) + 1], Q);
            }

            auto d_key_i = util::make_cuda_auto_ptr<BasicInteger>(2 * N, stream);
            BasicInteger* d_key_i_0 = d_key_i.get();
            BasicInteger* d_key_i_1 = d_key_i_0 + N;
            cudaMemcpyAsync(d_key_i_0, &result[i][0].GetValues().at(0),
                            N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_key_i_1, &result[i][1].GetValues().at(0),
                            N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);

            BasicInteger* d_keyNTT_i = d_keyNTT.get() + i * 2 * N;
            BasicInteger* d_keyNTT_i_0 = d_keyNTT_i;
            BasicInteger* d_keyNTT_i_1 = d_keyNTT_i_0 + N;

            ntt_->forward_shoup(d_keyNTT_i, d_key_i.get(), 256, 2, stream);
            ntt_->forward_shoup(d_tempA.get(), d_tempA.get(), 256, 1, stream);
            ntt_->multiply_and_accumulate(d_keyNTT_i_1, d_tempA.get(), d_skNTT.get(), stream);

            if (use_fft_)
            {
                complex_t* d_keyFFT_i = d_keyFFT.get() + i * 2 * N / 2;
                complex_t* d_keyFFT_i_0 = d_keyFFT_i;
                complex_t* d_keyFFT_i_1 = d_keyFFT_i_0 + N / 2;
                if (N == 1024)
                    fft_1024_->i2c_forward(d_keyFFT_i_0, d_key_i_0, 1, stream);
                else if (N == 2048)
                    fft_2048_->i2c_forward(d_keyFFT_i_0, d_key_i_0, 1, stream);
                else if (N == 4096)
                    fft_4096_->i2c_forward(d_keyFFT_i_0, d_key_i_0, 1, stream);
                else
                    throw std::invalid_argument("ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");

                auto d_key_i_1_tmp = util::make_cuda_auto_ptr<BasicInteger>(N, stream);
                ntt_->inverse_shoup(d_key_i_1_tmp.get(), d_keyNTT_i_1, 256, 1, stream);

                if (N == 1024)
                    fft_1024_->i2c_forward(d_keyFFT_i_1, d_key_i_1_tmp.get(), 1, stream);
                else if (N == 2048)
                    fft_2048_->i2c_forward(d_keyFFT_i_1, d_key_i_1_tmp.get(), 1, stream);
                else if (N == 4096)
                    fft_4096_->i2c_forward(d_keyFFT_i_1, d_key_i_1_tmp.get(), 1, stream);
                else
                    throw std::invalid_argument("ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");
            }

            // to Montgomery domain
            ntt_->to_mont(d_keyNTT_i, d_keyNTT_i, 256, 2, stream);
        }

        cudaStreamSynchronize(stream);
    }

    void GPURingGSWAccumulator::GPUEvalAcc(const std::shared_ptr<BinFHECryptoParams>& params,
                                           const GPURingGSWBTKey& EK,
                                           const NativeVector& a,
                                           const phantom::util::cuda_auto_ptr<BasicInteger>& d_acc,
                                           const bool use_fft,
                                           const cudaStream_t& s) const
    {
        switch (params->GetRingGSWParams()->GetMethod())
        {
        case BINFHE_METHOD::AP:
            GPUEvalAccDM(params, EK, a, d_acc, s, use_fft);
            break;
        case BINFHE_METHOD::GINX:
            GPUEvalAccCGGI(params, EK, a, d_acc, s, use_fft);
            break;
        //            case BINFHE_METHOD::LMKCDEY:
        //                GPUEvalAccLMKCDEY(params, EK, a, d_acc, s);
        //                break;
        default:
            throw std::invalid_argument("ERROR: Invalid ACC method");
        }
    }

    void GPURingGSWAccumulator::GPUEvalAccDM(const std::shared_ptr<BinFHECryptoParams>& params,
                                             const GPURingGSWBTKey& EK,
                                             const NativeVector& a,
                                             const phantom::util::cuda_auto_ptr<BasicInteger>& d_acc,
                                             const cudaStream_t& s,
                                             const bool use_fft) const
    {
        auto& LWEParams = params->GetLWEParams();
        auto& RGSWParams = params->GetRingGSWParams();
        auto polyParams = RGSWParams->GetPolyParams();
        const size_t n = LWEParams->Getn();
        auto q = LWEParams->Getq().ConvertToInt();
        auto Q = RGSWParams->GetQ().ConvertToInt();
        auto P = RGSWParams->GetP().ConvertToInt();
        auto PQ = RGSWParams->GetPQ().ConvertToInt();
        const size_t N = RGSWParams->GetN();
        NativeInteger baseR = RGSWParams->GetBaseR();
        const auto& digitsR = RGSWParams->GetDigitsR();
        uint32_t digitsG2 = (RGSWParams->GetDigitsG() - 1) << 1;

        // approximate gadget decomposition is used; the first digit is ignored
        auto d_dct = phantom::util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * N, s);

        for (size_t i = 0; i < n; ++i)
        {
            auto aI = NativeInteger(0).ModSubFast(a[i], q);
            for (size_t k = 0; k < digitsR.size(); ++k, aI /= baseR)
            {
                const auto a0 = (aI.Mod(baseR)).ConvertToInt<uint32_t>();
                if (a0)
                {
                    if (use_fft)
                    {
                        auto d_dct_fft = phantom::util::make_cuda_auto_ptr<complex_t>(digitsG2 * N / 2, s);
                        auto d_acc_fft = phantom::util::make_cuda_auto_ptr<complex_t>(2 * N / 2, s);

                        kernel_SignedDigitDecompose<<<digitsG2, 256, 0, s>>>(
                            d_dct.get(), d_acc.get(),
                            RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N);

                        if (N == 1024)
                            fft_1024_->i2c_forward(d_dct_fft.get(), d_dct.get(), digitsG2, s);
                        else if (N == 2048)
                            fft_2048_->i2c_forward(d_dct_fft.get(), d_dct.get(), digitsG2, s);
                        else if (N == 4096)
                            fft_4096_->i2c_forward(d_dct_fft.get(), d_dct.get(), digitsG2, s);
                        else
                            throw std::invalid_argument(
                                "ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");

                        // acc = dct * ek (matrix product);
                        const auto* d_ACCKey_FFT = EK.RGSWACCKey_fft[i][a0][k].get();

                        kernel_EvalAccCoreDM_fft<<<2, 256, 0, s>>>(
                            d_acc_fft.get(), d_dct_fft.get(), d_ACCKey_FFT, N / 2, digitsG2);

                        if (N == 1024)
                            fft_1024_->c2i_inverse(d_acc.get(), d_acc_fft.get(), Q, 2, s);
                        else if (N == 2048)
                            fft_2048_->c2i_inverse(d_acc.get(), d_acc_fft.get(), Q, 2, s);
                        else if (N == 4096)
                            fft_4096_->c2i_inverse(d_acc.get(), d_acc_fft.get(), Q, 2, s);
                        else
                            throw std::invalid_argument(
                                "ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");
                    }
                    else // use NTT
                    {
                        auto d_acc_ntt = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N, s);

                        // AP Accumulation as described in https://eprint.iacr.org/2020/086
                        if (RGSWParams->IsCompositeNTT() == 2)
                        {
                            // hybrid
                            // temporary storage for hybrid method
                            auto d_acc_temp = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N, s);
                            ntt_->multiply_scalar(d_acc_temp.get(), d_acc.get(), P, s);
                            kernel_SignedDigitDecompose<<<digitsG2, 256, 0, s>>>(
                                d_dct.get(), d_acc_temp.get(),
                                RGSWParams->GetPQ().ConvertToInt(), RGSWParams->GetBaseG(), N);
                            ntt_->forward_shoup(d_dct.get(), d_dct.get(), 256, digitsG2, s);
                        }
                        else if (RGSWParams->IsCompositeNTT() == 1)
                        {
                            // composite NTT
                            ntt_->forward_shoup(d_dct.get(), d_acc.get(), 256, 2, s);
                        }
                        else
                        {
                            kernel_SignedDigitDecompose<<<digitsG2, 256, 0, s>>>(
                                d_dct.get(), d_acc.get(),
                                RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N);
                            ntt_->forward_shoup(d_dct.get(), d_dct.get(), 256, digitsG2, s);
                        }

                        // acc = dct * ek (matrix product);
                        BasicInteger* d_ACCKey = EK.RGSWACCKey[i][a0][k].get();

                        kernel_EvalAccCoreDM<<<2, 256, 0, s>>>(
                            d_acc_ntt.get(), d_dct.get(), d_ACCKey,
                            N, ntt_->getMod(), ntt_->getMont(), digitsG2);

                        ntt_->inverse_shoup(d_acc.get(), d_acc_ntt.get(), 256, 2, s);

                        // composite NTT
                        if (RGSWParams->IsCompositeNTT())
                        {
                            // scale P
                            kernel_scale_by_p<<<2, 256, 0, s>>>(
                                d_acc.get(), d_acc.get(), N, Q, PQ);
                        }
                    }
                }
            }
        }
    }

    void GPURingGSWAccumulator::GPUEvalAccCGGI(const std::shared_ptr<BinFHECryptoParams>& params,
                                               const GPURingGSWBTKey& EK,
                                               const NativeVector& a,
                                               const phantom::util::cuda_auto_ptr<BasicInteger>& d_acc,
                                               const cudaStream_t& s,
                                               const bool use_fft) const
    {
        auto& LWEParams = params->GetLWEParams();
        auto& RGSWParams = params->GetRingGSWParams();
        auto polyParams = RGSWParams->GetPolyParams();
        const size_t n = LWEParams->Getn();
        const size_t N = RGSWParams->GetN();
        auto Q = RGSWParams->GetQ().ConvertToInt();
        auto P = RGSWParams->GetP().ConvertToInt();
        auto PQ = RGSWParams->GetPQ().ConvertToInt();
        uint32_t digitsG2 = (RGSWParams->GetDigitsG() - 1) << 1;
        const auto& mod = a.GetModulus();
        const NativeInteger M{2 * RGSWParams->GetN()};
        const auto MbyMod{2 * RGSWParams->GetN() / a.GetModulus()};

        // approximate gadget decomposition is used; the first digit is ignored
        auto d_dct = phantom::util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * N, s);

        // for temporary storage of accumulator
        auto d_acc_tmp = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N, s);

        for (size_t n_idx = 0; n_idx < n; ++n_idx)
        {
            if (use_fft)
            {
                auto d_dct_fft = phantom::util::make_cuda_auto_ptr<complex_t>(digitsG2 * N / 2, s);

                kernel_SignedDigitDecompose<<<digitsG2, 256, 0, s>>>(
                    d_dct.get(), d_acc.get(),
                    RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N);
                if (N == 1024)
                    fft_1024_->i2c_forward(d_dct_fft.get(), d_dct.get(), digitsG2, s);
                else if (N == 2048)
                    fft_2048_->i2c_forward(d_dct_fft.get(), d_dct.get(), digitsG2, s);
                else if (N == 4096)
                    fft_4096_->i2c_forward(d_dct_fft.get(), d_dct.get(), digitsG2, s);
                else
                    throw std::invalid_argument(
                        "ERROR: Invalid FFT size for CGGI gadget decomposition, must be 1024, 2048 or 4096");

                if (params->GetLWEParams()->GetKeyDist() == UNIFORM_TERNARY)
                {
                    // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                    // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                    NativeInteger ai = NativeInteger(0).ModSubFast(a[n_idx], mod) * MbyMod;
                    const auto indexPos = ai.ConvertToInt<uint32_t>();
                    const auto indexNeg = NativeInteger(0).ModSubFast(ai, M).ConvertToInt<uint32_t>();
                    if (indexPos >= 2 * N || indexNeg >= 2 * N)
                        throw std::invalid_argument("ERROR: indexPos or indexNeg out of bound");

                    // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                    // Needs to be done using two loops for ternary secrets.
                    const auto* d_ACCKey0 = EK.RGSWACCKey_fft[0][0][n_idx].get();
                    const auto* d_ACCKey1 = EK.RGSWACCKey_fft[0][1][n_idx].get();

                    auto d_acc0_fft = phantom::util::make_cuda_auto_ptr<complex_t>(2 * N / 2, s);
                    auto d_acc1_fft = phantom::util::make_cuda_auto_ptr<complex_t>(2 * N / 2, s);

                    kernel_EvalAccCoreCGGI_fft<<<2, 256, 0, s>>>(
                        d_acc0_fft.get(), d_dct_fft.get(), d_ACCKey0, N / 2, digitsG2);

                    kernel_EvalAccCoreCGGI_fft<<<2, 256, 0, s>>>(
                        d_acc1_fft.get(), d_dct_fft.get(), d_ACCKey1, N / 2, digitsG2);

                    if (N == 1024)
                        fft_1024_->c2i_inverse(d_acc_tmp.get(), d_acc0_fft.get(), Q, 2, s);
                    else if (N == 2048)
                        fft_2048_->c2i_inverse(d_acc_tmp.get(), d_acc0_fft.get(), Q, 2, s);
                    else if (N == 4096)
                        fft_4096_->c2i_inverse(d_acc_tmp.get(), d_acc0_fft.get(), Q, 2, s);
                    else
                        throw std::invalid_argument("ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");

                    auto d_acc1_tmp = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N, s);

                    if (N == 1024)
                        fft_1024_->c2i_inverse(d_acc1_tmp.get(), d_acc1_fft.get(), Q, 2, s);
                    else if (N == 2048)
                        fft_2048_->c2i_inverse(d_acc1_tmp.get(), d_acc1_fft.get(), Q, 2, s);
                    else if (N == 4096)
                        fft_4096_->c2i_inverse(d_acc1_tmp.get(), d_acc1_fft.get(), Q, 2, s);
                    else
                        throw std::invalid_argument("ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");

                    kernel_EvalAccCoreCGGI_mac_monic<<<2, 256, 0, s>>>(
                        d_acc.get(), d_acc_tmp.get(), N, Q, indexPos);
                    kernel_EvalAccCoreCGGI_mac_monic<<<2, 256, 0, s>>>(
                        d_acc.get(), d_acc1_tmp.get(), N, Q, indexNeg);
                }
                else if (params->GetLWEParams()->GetKeyDist() == UNIFORM_BINARY)
                {
                    // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                    // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                    NativeInteger ai = NativeInteger(0).ModSubFast(a[n_idx], mod) * MbyMod;
                    const auto indexPos = ai.ConvertToInt<uint32_t>();
                    if (indexPos >= 2 * N)
                        throw std::invalid_argument("ERROR: indexPos out of bound");

                    // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                    // Needs to be done using two loops for ternary secrets.
                    const complex_t* d_ACCKey = EK.RGSWACCKey_fft[0][0][n_idx].get();

                    auto d_acc_fft = phantom::util::make_cuda_auto_ptr<complex_t>(2 * N / 2, s);

                    kernel_EvalAccCoreCGGI_fft<<<2, 256, 0, s>>>(
                        d_acc_fft.get(), d_dct_fft.get(), d_ACCKey, N / 2, digitsG2);

                    if (N == 1024)
                        fft_1024_->c2i_inverse(d_acc_tmp.get(), d_acc_fft.get(), Q, 2, s);
                    else if (N == 2048)
                        fft_2048_->c2i_inverse(d_acc_tmp.get(), d_acc_fft.get(), Q, 2, s);
                    else if (N == 4096)
                        fft_4096_->c2i_inverse(d_acc_tmp.get(), d_acc_fft.get(), Q, 2, s);
                    else
                        throw std::invalid_argument("ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");

                    kernel_EvalAccCoreCGGI_mac_monic<<<2, 256, 0, s>>>(
                        d_acc.get(), d_acc_tmp.get(), N, Q, indexPos);
                }
                else
                {
                    throw std::invalid_argument("ERROR: Invalid key distribution for CGGI");
                }
            }
            else // use NTT
            {
                // handles -a*E(1) and handles -a*E(-1) = a*E(1)
                // CGGI Accumulation as described in https://eprint.iacr.org/2020/086
                // Added ternary MUX introduced in paper https://eprint.iacr.org/2022/074.pdf section 5
                // We optimize the algorithm by multiplying the monomial after the external product
                // This reduces the number of polynomial multiplications which further reduces the runtime
                if (RGSWParams->IsCompositeNTT() == 2)
                {
                    // hybrid
                    // temporary storage for hybrid method
                    auto d_acc_temp = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N, s);
                    ntt_->multiply_scalar(d_acc_temp.get(), d_acc.get(), P, s);
                    kernel_SignedDigitDecompose<<<digitsG2, 256, 0, s>>>(
                        d_dct.get(), d_acc_temp.get(),
                        RGSWParams->GetPQ().ConvertToInt(), RGSWParams->GetBaseG(), N);
                    ntt_->forward_shoup(d_dct.get(), d_dct.get(), 256, digitsG2, s);
                }
                else if (RGSWParams->IsCompositeNTT() == 1)
                {
                    // composite NTT
                    ntt_->forward_shoup(d_dct.get(), d_acc.get(), 256, 2, s);
                }
                else
                {
                    kernel_SignedDigitDecompose<<<digitsG2, 256, 0, s>>>(
                        d_dct.get(), d_acc.get(),
                        RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N);
                    ntt_->forward_shoup(d_dct.get(), d_dct.get(), 256, digitsG2, s);
                }

                if (params->GetLWEParams()->GetKeyDist() == UNIFORM_TERNARY)
                {
                    // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                    // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                    NativeInteger ai = NativeInteger(0).ModSubFast(a[n_idx], mod) * MbyMod;
                    const auto indexPos = ai.ConvertToInt<uint32_t>();
                    const auto indexNeg = NativeInteger(0).ModSubFast(ai, M).ConvertToInt<uint32_t>();
                    if (indexPos >= 2 * N || indexNeg >= 2 * N)
                        throw std::invalid_argument("ERROR: indexPos or indexNeg out of bound");

                    // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                    // Needs to be done using two loops for ternary secrets.
                    const BasicInteger* d_ACCKey0 = EK.RGSWACCKey[0][0][n_idx].get();
                    const BasicInteger* d_ACCKey1 = EK.RGSWACCKey[0][1][n_idx].get();

                    kernel_EvalAccCoreCGGI<<<2, 256, 0, s>>>(
                        d_acc_tmp.get(), d_dct.get(), d_ACCKey0, d_ACCKey1, d_monic_polys_.get(), N,
                        ntt_->getMod(), ntt_->getMont(), digitsG2, indexPos, indexNeg);
                }
                else if (params->GetLWEParams()->GetKeyDist() == UNIFORM_BINARY)
                {
                    // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                    // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                    NativeInteger ai = NativeInteger(0).ModSubFast(a[n_idx], mod) * MbyMod;
                    const auto indexPos = ai.ConvertToInt<uint32_t>();
                    if (indexPos >= 2 * N)
                        throw std::invalid_argument("ERROR: indexPos out of bound");

                    // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                    // Needs to be done using two loops for ternary secrets.
                    const BasicInteger* d_ACCKey = EK.RGSWACCKey[0][0][n_idx].get();

                    kernel_EvalAccCoreCGGI_binary<<<2, 256, 0, s>>>(
                        d_acc_tmp.get(), d_dct.get(), d_ACCKey, d_monic_polys_.get(), N,
                        ntt_->getMod(), ntt_->getMont(), digitsG2, indexPos);
                }
                else
                {
                    throw std::invalid_argument("ERROR: Invalid key distribution for CGGI");
                }

                ntt_->inverse_shoup(d_acc_tmp.get(), d_acc_tmp.get(), 256, 2, s);

                if (RGSWParams->IsCompositeNTT())
                {
                    // composite NTT
                    kernel_scale_by_p<<<2, 256, 0, s>>>(
                        d_acc_tmp.get(), d_acc_tmp.get(), N, Q, PQ);
                }

                // accumulate to acc
                kernel_element_add<<<2, 256, 0, s>>>(
                    d_acc.get(), d_acc.get(), d_acc_tmp.get(), N, Q);
            }
        }
    }
}
