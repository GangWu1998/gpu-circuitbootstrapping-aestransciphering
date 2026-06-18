#include "ciphertext.cuh"

PhantomCiphertext::PhantomCiphertext(const PhantomContext& context,
                                     const lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly>& openfhe_ct)
{
    auto& key_context_data = context.get_context_data(0);
    auto& key_parms = key_context_data.parms();
    auto& key_modulus = key_parms.coeff_modulus();
    const auto poly_degree = key_parms.poly_modulus_degree();
    const size_t size_P = key_parms.special_modulus_size();
    const size_t size_QP = key_modulus.size();
    const size_t size_Q = size_QP - size_P;
    const size_t size_DCRT = openfhe_ct->GetElements().at(0).GetAllElements().size();
    const auto scheme = key_parms.scheme();

    if (scheme == phantom::scheme_type::bfv || scheme == phantom::scheme_type::bgv)
    {
        throw std::invalid_argument(
            "Unsupported scheme type for PhantomCiphertext construction from OpenFHE ciphertext");
    }

    size_ = openfhe_ct->NumberCiphertextElements();
    poly_modulus_degree_ = poly_degree;
    coeff_modulus_size_ = size_DCRT;
    is_ntt_form_ = openfhe_ct->GetElements().at(0).GetFormat() == Format::EVALUATION;

    noiseScaleDeg_ = openfhe_ct->GetNoiseScaleDeg();
    scalingFactor_ = openfhe_ct->GetScalingFactor();
    scalingFactorInt_ = openfhe_ct->GetScalingFactorInt();
    chain_index_ = openfhe_ct->GetLevel() + 1;
    // OpenFHE Ciphertext level starts from 0, Phantom Ciphertext level starts from 1
    hopslevel_ = openfhe_ct->GetHopLevel();
    slots_ = openfhe_ct->GetSlots();
    data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size_ * coeff_modulus_size_ * poly_modulus_degree_,
                                                        cudaStreamPerThread);
    for (size_t i = 0; i < size_; ++i)
    {
        for (size_t j = 0; j < coeff_modulus_size_; ++j)
        {
            const auto& poly = openfhe_ct->GetElements().at(i).GetElementAtIndex(j);
            cudaMemcpyAsync(data_.get() + i * coeff_modulus_size_ * poly_degree + j * poly_degree,
                            &poly.at(0), poly_degree * sizeof(uint64_t), cudaMemcpyHostToDevice,
                            cudaStreamPerThread);
        }
    }
}

void PhantomCiphertext::WriteToOpenFHECiphertext(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& openfhe_cc,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& openfhe_ct) const
{
    std::vector<lbcrypto::DCRTPoly> ct_elements(size_);

    const auto format = is_ntt_form_ ? Format::EVALUATION : Format::COEFFICIENT;

    for (size_t i = 0; i < size_; ++i)
    {
        std::vector<lbcrypto::NativePoly> DCRT_polys(coeff_modulus_size_);

        for (size_t j = 0; j < coeff_modulus_size_; ++j)
        {
            DCRT_polys.at(j) = lbcrypto::NativePoly(
                openfhe_cc->GetCryptoParameters()->GetElementParams()->GetParams().at(j), format, true);
            cudaMemcpyAsync(&DCRT_polys.at(j).at(0),
                            data_.get() + i * coeff_modulus_size_ * poly_modulus_degree_ + j * poly_modulus_degree_,
                            poly_modulus_degree_ * sizeof(uint64_t), cudaMemcpyDeviceToHost,
                            cudaStreamPerThread);
        }

        ct_elements.at(i) = lbcrypto::DCRTPoly(DCRT_polys);
    }
    cudaStreamSynchronize(cudaStreamPerThread);

    openfhe_ct->SetElements(ct_elements);
    openfhe_ct->SetNoiseScaleDeg(noiseScaleDeg_);
    openfhe_ct->SetScalingFactor(scalingFactor_);
    openfhe_ct->SetScalingFactorInt(scalingFactorInt_);
    openfhe_ct->SetLevel(chain_index_ - 1);
    // Phantom Ciphertext level starts from 1, OpenFHE Ciphertext level starts from 0
    openfhe_ct->SetHopLevel(hopslevel_);
    openfhe_ct->SetSlots(slots_);
}
