#include "kernel.cuh"
#include "rgsw-acc.cuh"

#include "cuda_wrapper.cuh"
#include "kernels_fft.cuh"

#include "math/distributiongenerator.h"

#include "arith/big_integer_modop.h"

using namespace lbcrypto;

namespace phantom::bitwise
{
    void GPURingGSWAccumulator::BatchGPUEvalAcc(const std::shared_ptr<BinFHECryptoParams>& params,
                                                const GPURingGSWBTKey& EK,
                                                const std::vector<NativeVector>& v_a,
                                                const phantom::util::cuda_auto_ptr<BasicInteger>& d_acc,
                                                const bool use_fft,
                                                const cudaStream_t& s) const
    {
        switch (params->GetRingGSWParams()->GetMethod())
        {
        case BINFHE_METHOD::AP:
            BatchGPUEvalAccDM(params, EK, v_a, d_acc, s, use_fft);
            break;
        case BINFHE_METHOD::GINX:
            BatchGPUCGGI(params, EK, v_a, d_acc, s, use_fft);
            break;
        //            case BINFHE_METHOD::LMKCDEY:
        //                BatchGPUEvalAccLMKCDEY(params, EK, v_a, d_acc, s);
        //                break;
        default:
            throw std::invalid_argument("ERROR: Invalid ACC method");
        }
    }

    void GPURingGSWAccumulator::BatchGPUEvalAccDM(const std::shared_ptr<BinFHECryptoParams>& params,
                                                  const GPURingGSWBTKey& EK,
                                                  const std::vector<NativeVector>& v_a,
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

        size_t batch_size = v_a.size();

        std::vector<BasicInteger*> v_d_ACCKey(n * digitsR.size() * batch_size);
        std::vector<complex_t*> v_d_ACCKey_fft(n * digitsR.size() * batch_size);
        for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++)
        {
            for (size_t n_idx = 0; n_idx < n; ++n_idx)
            {
                auto aI = NativeInteger(0).ModSubFast(v_a[batch_idx][n_idx], q);
                for (size_t k = 0; k < digitsR.size(); ++k, aI /= baseR)
                {
                    const auto a0 = (aI.Mod(baseR)).ConvertToInt<uint32_t>();
                    BasicInteger* d_ACCKey = EK.RGSWACCKey[n_idx][a0][k].get();
                    complex_t* d_ACCKey_fft = EK.RGSWACCKey_fft[n_idx][a0][k].get();
                    v_d_ACCKey[n_idx * digitsR.size() * batch_size + k * batch_size + batch_idx] = d_ACCKey;
                    v_d_ACCKey_fft[n_idx * digitsR.size() * batch_size + k * batch_size + batch_idx] = d_ACCKey_fft;
                }
            }
        }

        auto d_v_d_ACCKey = phantom::util::make_cuda_auto_ptr<BasicInteger*>(n * digitsR.size() * batch_size, s);
        cudaMemcpyAsync(d_v_d_ACCKey.get(), v_d_ACCKey.data(), n * digitsR.size() * batch_size * sizeof(BasicInteger*),
                        cudaMemcpyHostToDevice, s);

        auto d_v_d_ACCKey_fft = phantom::util::make_cuda_auto_ptr<complex_t*>(n * digitsR.size() * batch_size, s);
        cudaMemcpyAsync(d_v_d_ACCKey_fft.get(), v_d_ACCKey_fft.data(),
                        n * digitsR.size() * batch_size * sizeof(complex_t*),
                        cudaMemcpyHostToDevice, s);

        // approximate gadget decomposition is used; the first digit is ignored
        auto d_dct = phantom::util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * N * batch_size, s);

        for (size_t n_idx = 0; n_idx < n; ++n_idx)
        {
            for (size_t k = 0; k < digitsR.size(); ++k)
            {
                if (use_fft)
                {
                    auto d_dct_fft = phantom::util::make_cuda_auto_ptr<complex_t>(batch_size * digitsG2 * N / 2, s);
                    auto d_acc_fft = phantom::util::make_cuda_auto_ptr<complex_t>(batch_size * 2 * N / 2, s);

                    kernel_SignedDigitDecompose<<<dim3(digitsG2, batch_size), 256, 0, s>>>(
                        d_dct.get(), d_acc.get(), RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N);

                    if (N == 1024)
                        fft_1024_->i2c_forward(d_dct_fft.get(), d_dct.get(), digitsG2 * batch_size, s);
                    else if (N == 2048)
                        fft_2048_->i2c_forward(d_dct_fft.get(), d_dct.get(), digitsG2 * batch_size, s);
                    else if (N == 4096)
                        fft_4096_->i2c_forward(d_dct_fft.get(), d_dct.get(), digitsG2 * batch_size, s);
                    else
                        throw std::invalid_argument(
                            "ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");

                    complex_t** d_ACCKeys_fft = d_v_d_ACCKey_fft.get() + n_idx * digitsR.size() * batch_size + k *
                        batch_size;

                    kernel_EvalAccCoreDM_fft_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                        d_acc_fft.get(), d_dct_fft.get(), d_ACCKeys_fft,
                        N / 2, digitsG2);

                    if (N == 1024)
                        fft_1024_->c2i_inverse(d_acc.get(), d_acc_fft.get(), Q, 2 * batch_size, s);
                    else if (N == 2048)
                        fft_2048_->c2i_inverse(d_acc.get(), d_acc_fft.get(), Q, 2 * batch_size, s);
                    else if (N == 4096)
                        fft_4096_->c2i_inverse(d_acc.get(), d_acc_fft.get(), Q, 2 * batch_size, s);
                    else
                        throw std::invalid_argument(
                            "ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");
                }
                else // use NTT
                {
                    auto d_acc_ntt = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * batch_size, s);

                    // AP Accumulation as described in https://eprint.iacr.org/2020/086
                    if (RGSWParams->IsCompositeNTT() == 1)
                    {
                        // composite NTT
                        ntt_->forward_shoup(d_dct.get(), d_acc.get(), 256, 2 * batch_size, s);
                    }
                    else if (RGSWParams->IsCompositeNTT() == 0)
                    {
                        // gadget decompose
                        //                            if (N == 1024) {
                        //                                constexpr size_t n1 = 32;
                        //                                constexpr size_t n2 = 32;
                        //                                size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                        //                                kernel_SignedDigitDecompose_fuse_1024<<<digitsG2, 1024, sMemSize, s>>>(
                        //                                        d_dct.get() + digitsG2 * N * batch_idx, d_acc.get() + 2 * N * batch_idx,
                        //                                        RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(),
                        //                                        ntt_->getTwRoot2n1(), ntt_->getTwRoot2n1Shoup(),
                        //                                        ntt_->getTwRoot2n(), ntt_->getTwRoot2nShoup());
                        //                            } else {
                        kernel_SignedDigitDecompose<<<dim3(digitsG2, batch_size), 256, 0, s>>>(
                            d_dct.get(), d_acc.get(), RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N);
                        ntt_->forward_shoup(d_dct.get(), d_dct.get(), 256, digitsG2 * batch_size, s);
                        //                            }
                    }
                    else
                    {
                        throw std::invalid_argument("ERROR: Unsupported ACC technique");
                    }

                    // acc = dct * ek (matrix product);
                    BasicInteger** d_ACCKeys = d_v_d_ACCKey.get() + n_idx * digitsR.size() * batch_size + k *
                        batch_size;

                    if (N == 1024)
                    {
                        constexpr size_t n1 = 32;
                        constexpr size_t n2 = 32;
                        size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                        kernel_EvalAccCoreDM_1024_batch_fuse<<<dim3(2, batch_size), 256, sMemSize, s>>>(
                            d_acc.get(), d_dct.get(), d_ACCKeys,
                            N, ntt_->getMod(), ntt_->getMont(), digitsG2,
                            ntt_->getTwInvRoot2n(), ntt_->getTwInvRoot2nShoup(),
                            ntt_->getTwInvRoot2n1(), ntt_->getTwInvRoot2n1Shoup(),
                            ntt_->getInvn(), ntt_->getInvnShoup(),
                            RGSWParams->IsCompositeNTT(), P);
                    }
                    else
                    {
                        kernel_EvalAccCoreDM_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                            d_acc_ntt.get(), d_dct.get(), d_ACCKeys,
                            N, ntt_->getMod(), ntt_->getMont(), digitsG2);

                        ntt_->inverse_shoup(d_acc.get(), d_acc_ntt.get(), 256, 2 * batch_size, s);

                        // composite NTT
                        if (RGSWParams->IsCompositeNTT())
                        {
                            // scale P
                            kernel_scale_by_p<<<dim3(2, batch_size), 256, 0, s>>>(
                                d_acc.get(), d_acc.get(), N, Q, PQ);
                        }
                    }
                }
            }
        }
    }

    void GPURingGSWAccumulator::BatchGPUCGGI(const std::shared_ptr<BinFHECryptoParams>& params,
                                             const GPURingGSWBTKey& EK,
                                             const std::vector<NativeVector>& v_a,
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
        const auto& mod = v_a[0].GetModulus();
        const NativeInteger M{2 * RGSWParams->GetN()};
        const auto MbyMod{2 * RGSWParams->GetN() / v_a[0].GetModulus()};

        size_t batch_size = v_a.size();

        std::vector<uint32_t> v_indexPos(n * batch_size);
        std::vector<uint32_t> v_indexNeg(n * batch_size);
        for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++)
        {
            for (size_t n_idx = 0; n_idx < n; ++n_idx)
            {
                // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                NativeInteger ai = NativeInteger(0).ModSubFast(v_a[batch_idx][n_idx], mod) * MbyMod;
                auto indexPos = ai.ConvertToInt<uint32_t>();
                auto indexNeg = NativeInteger(0).ModSubFast(ai, M).ConvertToInt<uint32_t>();
                if (indexPos >= 2 * N || indexNeg >= 2 * N)
                    throw std::invalid_argument("ERROR: indexPos or indexNeg out of bound");
                v_indexPos[n_idx * batch_size + batch_idx] = indexPos;
                if (params->GetLWEParams()->GetKeyDist() == UNIFORM_TERNARY)
                    v_indexNeg[n_idx * batch_size + batch_idx] = indexNeg;
            }
        }

        auto d_v_indexPos = phantom::util::make_cuda_auto_ptr<uint32_t>(n * batch_size, s);
        auto d_v_indexNeg = phantom::util::make_cuda_auto_ptr<uint32_t>(n * batch_size, s);
        cudaMemcpyAsync(d_v_indexPos.get(), v_indexPos.data(), n * batch_size * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_v_indexNeg.get(), v_indexNeg.data(), n * batch_size * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, s);

        // approximate gadget decomposition is used; the first digit is ignored
        auto d_dct = phantom::util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * N * batch_size, s);

        for (size_t n_idx = 0; n_idx < n; ++n_idx)
        {
            if (use_fft)
            {
                auto d_dct_fft = phantom::util::make_cuda_auto_ptr<complex_t>(batch_size * digitsG2 * N / 2, s);
                auto d_acc_fft = phantom::util::make_cuda_auto_ptr<complex_t>(batch_size * 2 * N / 2, s);
                auto d_acc_tmp = phantom::util::make_cuda_auto_ptr<BasicInteger>(batch_size * 2 * N, s);

                kernel_SignedDigitDecompose<<<dim3(digitsG2, batch_size), 256, 0, s>>>(
                    d_dct.get(), d_acc.get(), RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N);

                if (N == 1024)
                    fft_1024_->i2c_forward(d_dct_fft.get(), d_dct.get(), digitsG2 * batch_size, s);
                else if (N == 2048)
                    fft_2048_->i2c_forward(d_dct_fft.get(), d_dct.get(), digitsG2 * batch_size, s);
                else if (N == 4096)
                    fft_4096_->i2c_forward(d_dct_fft.get(), d_dct.get(), digitsG2 * batch_size, s);
                else
                    throw std::invalid_argument(
                        "ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");

                if (params->GetLWEParams()->GetKeyDist() == UNIFORM_TERNARY)
                {
                    // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                    // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                    uint32_t* d_indexPos = d_v_indexPos.get() + n_idx * batch_size;
                    uint32_t* d_indexNeg = d_v_indexNeg.get() + n_idx * batch_size;

                    // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                    // Needs to be done using two loops for ternary secrets.
                    const complex_t* d_ACCKey0_fft = EK.RGSWACCKey_fft[0][0][n_idx].get();
                    const complex_t* d_ACCKey1_fft = EK.RGSWACCKey_fft[0][1][n_idx].get();

                    auto d_acc0_fft = phantom::util::make_cuda_auto_ptr<complex_t>(2 * N / 2 * batch_size, s);
                    auto d_acc1_fft = phantom::util::make_cuda_auto_ptr<complex_t>(2 * N / 2 * batch_size, s);

                    kernel_EvalAccCoreCGGI_fft_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                        d_acc0_fft.get(), d_dct_fft.get(), d_ACCKey0_fft, N / 2, digitsG2);

                    kernel_EvalAccCoreCGGI_fft_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                        d_acc1_fft.get(), d_dct_fft.get(), d_ACCKey1_fft, N / 2, digitsG2);

                    if (N == 1024)
                        fft_1024_->c2i_inverse(d_acc_tmp.get(), d_acc0_fft.get(), Q, 2 * batch_size, s);
                    else if (N == 2048)
                        fft_2048_->c2i_inverse(d_acc_tmp.get(), d_acc0_fft.get(), Q, 2 * batch_size, s);
                    else if (N == 4096)
                        fft_4096_->c2i_inverse(d_acc_tmp.get(), d_acc0_fft.get(), Q, 2 * batch_size, s);
                    else
                        throw std::invalid_argument(
                            "ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");

                    auto d_acc1_tmp = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * batch_size, s);

                    if (N == 1024)
                        fft_1024_->c2i_inverse(d_acc1_tmp.get(), d_acc1_fft.get(), Q, 2 * batch_size, s);
                    else if (N == 2048)
                        fft_2048_->c2i_inverse(d_acc1_tmp.get(), d_acc1_fft.get(), Q, 2 * batch_size, s);
                    else if (N == 4096)
                        fft_4096_->c2i_inverse(d_acc1_tmp.get(), d_acc1_fft.get(), Q, 2 * batch_size, s);
                    else
                        throw std::invalid_argument("ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");

                    kernel_EvalAccCoreCGGI_mac_monic_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                        d_acc.get(), d_acc_tmp.get(), N, Q, d_indexPos);
                    kernel_EvalAccCoreCGGI_mac_monic_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                        d_acc.get(), d_acc1_tmp.get(), N, Q, d_indexNeg);
                }
                else if (params->GetLWEParams()->GetKeyDist() == UNIFORM_BINARY)
                {
                    // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                    // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                    uint32_t* d_indexPos = d_v_indexPos.get() + n_idx * batch_size;

                    // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                    // Needs to be done using two loops for ternary secrets.
                    const complex_t* d_ACCKey_fft = EK.RGSWACCKey_fft[0][0][n_idx].get();

                    kernel_EvalAccCoreCGGI_fft_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                        d_acc_fft.get(), d_dct_fft.get(), d_ACCKey_fft, N / 2, digitsG2);

                    if (N == 1024)
                        fft_1024_->c2i_inverse(d_acc_tmp.get(), d_acc_fft.get(), Q, 2 * batch_size, s);
                    else if (N == 2048)
                        fft_2048_->c2i_inverse(d_acc_tmp.get(), d_acc_fft.get(), Q, 2 * batch_size, s);
                    else if (N == 4096)
                        fft_4096_->c2i_inverse(d_acc_tmp.get(), d_acc_fft.get(), Q, 2 * batch_size, s);
                    else
                        throw std::invalid_argument(
                            "ERROR: Unsupported length for FFT, must be 1024, 2048 or 4096");

                    kernel_EvalAccCoreCGGI_mac_monic_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                        d_acc.get(), d_acc_tmp.get(), N, Q, d_indexPos);
                }
                else
                {
                    throw std::invalid_argument("ERROR: Invalid key distribution for CGGI");
                }
            }
            else // use NTT
            {
                auto d_acc_ntt = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * batch_size, s);

                // handles -a*E(1) and handles -a*E(-1) = a*E(1)
                // CGGI Accumulation as described in https://eprint.iacr.org/2020/086
                // Added ternary MUX introduced in paper https://eprint.iacr.org/2022/074.pdf section 5
                // We optimize the algorithm by multiplying the monomial after the external product
                // This reduces the number of polynomial multiplications which further reduces the runtime
                if (RGSWParams->IsCompositeNTT() == 1)
                {
                    // composite NTT
                    ntt_->forward_shoup(d_dct.get(), d_acc.get(), 256, 2 * batch_size, s);
                }
                else if (RGSWParams->IsCompositeNTT() == 0)
                {
                    // gadget decompose
                    //                    if (N == 1024) {
                    //                        constexpr size_t n1 = 32;
                    //                        constexpr size_t n2 = 32;
                    //                        size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                    //                        kernel_SignedDigitDecompose_fuse_1024<<<digitsG2, 1024, sMemSize, s>>>(
                    //                                d_dct.get() + digitsG2 * N * batch_idx, d_acc.get() + 2 * N * batch_idx,
                    //                                RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(),
                    //                                ntt_->getTwRoot2n1(), ntt_->getTwRoot2n1Shoup(),
                    //                                ntt_->getTwRoot2n(), ntt_->getTwRoot2nShoup());
                    //                    } else {
                    kernel_SignedDigitDecompose<<<dim3(digitsG2, batch_size), 256, 0, s>>>(
                        d_dct.get(), d_acc.get(), RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N);
                    ntt_->forward_shoup(d_dct.get(), d_dct.get(), 256, digitsG2 * batch_size, s);
                    //                    }
                }
                else
                {
                    throw std::invalid_argument("ERROR: Unsupported ACC technique");
                }

                if (params->GetLWEParams()->GetKeyDist() == UNIFORM_TERNARY)
                {
                    // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                    // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                    uint32_t* d_indexPos = d_v_indexPos.get() + n_idx * batch_size;
                    uint32_t* d_indexNeg = d_v_indexNeg.get() + n_idx * batch_size;

                    // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                    // Needs to be done using two loops for ternary secrets.
                    const BasicInteger* d_ACCKey0 = EK.RGSWACCKey[0][0][n_idx].get();
                    const BasicInteger* d_ACCKey1 = EK.RGSWACCKey[0][1][n_idx].get();

                    if (N == 1024)
                    {
                        constexpr size_t n1 = 32;
                        constexpr size_t n2 = 32;
                        size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                        kernel_EvalAccCoreCGGI_1024_batch_fuse<<<dim3(2, batch_size), 256, sMemSize, s>>>(
                            d_acc.get(), d_dct.get(), d_ACCKey0, d_ACCKey1, d_monic_polys_.get(),
                            N, ntt_->getMod(), ntt_->getMont(), digitsG2, d_indexPos, d_indexNeg,
                            ntt_->getTwInvRoot2n(), ntt_->getTwInvRoot2nShoup(),
                            ntt_->getTwInvRoot2n1(), ntt_->getTwInvRoot2n1Shoup(),
                            ntt_->getInvn(), ntt_->getInvnShoup(),
                            RGSWParams->IsCompositeNTT(), P, Q);
                    }
                    else
                    {
                        kernel_EvalAccCoreCGGI_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                            d_acc_ntt.get(), d_dct.get(), d_ACCKey0, d_ACCKey1, d_monic_polys_.get(), N,
                            ntt_->getMod(), ntt_->getMont(), digitsG2, d_indexPos, d_indexNeg);

                        ntt_->inverse_shoup(d_acc_ntt.get(), d_acc_ntt.get(), 256, 2 * batch_size, s);

                        if (RGSWParams->IsCompositeNTT())
                        {
                            // composite NTT
                            kernel_scale_by_p<<<dim3(2, batch_size), 256, 0, s>>>(
                                d_acc_ntt.get(), d_acc_ntt.get(), N, Q, PQ);
                        }

                        // accumulate to acc
                        kernel_element_add<<<dim3(2, batch_size), 256, 0, s>>>(
                            d_acc.get(), d_acc.get(), d_acc_ntt.get(), N, Q);
                    }
                }
                else if (params->GetLWEParams()->GetKeyDist() == UNIFORM_BINARY)
                {
                    // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                    // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                    uint32_t* d_indexPos = d_v_indexPos.get() + n_idx * batch_size;

                    // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                    // Needs to be done using two loops for ternary secrets.
                    const BasicInteger* d_ACCKey = EK.RGSWACCKey[0][0][n_idx].get();

                    if (N == 1024)
                    {
                        constexpr size_t n1 = 32;
                        constexpr size_t n2 = 32;
                        size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                        kernel_EvalAccCoreCGGI_1024_binary_batch_fuse<<<dim3(2, batch_size), 256, sMemSize, s>>>(
                            d_acc.get(), d_dct.get(), d_ACCKey, d_monic_polys_.get(),
                            N, ntt_->getMod(), ntt_->getMont(), digitsG2, d_indexPos,
                            ntt_->getTwInvRoot2n(), ntt_->getTwInvRoot2nShoup(),
                            ntt_->getTwInvRoot2n1(), ntt_->getTwInvRoot2n1Shoup(),
                            ntt_->getInvn(), ntt_->getInvnShoup(),
                            RGSWParams->IsCompositeNTT(), P, Q);
                    }
                    else
                    {
                        kernel_EvalAccCoreCGGI_binary_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                            d_acc_ntt.get(), d_dct.get(), d_ACCKey, d_monic_polys_.get(), N,
                            ntt_->getMod(), ntt_->getMont(), digitsG2, d_indexPos);

                        ntt_->inverse_shoup(d_acc_ntt.get(), d_acc_ntt.get(), 256, 2 * batch_size, s);

                        if (RGSWParams->IsCompositeNTT())
                        {
                            // composite NTT
                            kernel_scale_by_p<<<dim3(2, batch_size), 256, 0, s>>>(
                                d_acc_ntt.get(), d_acc_ntt.get(), N, Q, PQ);
                        }

                        // accumulate to acc
                        kernel_element_add<<<dim3(2, batch_size), 256, 0, s>>>(
                            d_acc.get(), d_acc.get(), d_acc_ntt.get(), N, Q);
                    }
                }
                else
                {
                    throw std::invalid_argument("ERROR: Invalid key distribution for CGGI");
                }
            }
        }
    }
}
