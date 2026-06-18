#pragma once

#include <cassert>

#include "context.cuh"
#include "polymath.cuh"

#include "openfhe.h"

class PhantomPlaintext {

    friend class PhantomBatchEncoder;

    friend class PhantomCKKSEncoder;

    friend class PhantomSecretKey;

private:

    size_t poly_modulus_degree_ = 0;
    size_t coeff_modulus_size_ = 0;

    // align with OpenFHE
    double scalingFactor_ = 1.0;
    NativeInteger scalingFactorInt_ = 1;
    size_t chain_index_ = 0;
    size_t noiseScaleDeg_ = 1;
    size_t slots_ = 0;

    phantom::util::cuda_auto_ptr<uint64_t> data_;

public:

    PhantomPlaintext() = default;

    explicit PhantomPlaintext(const PhantomContext &context, const lbcrypto::ConstPlaintext& openfhe_pt)
    {
        auto &key_context_data = context.get_context_data(0);
        auto &key_parms = key_context_data.parms();
        auto &key_modulus = key_parms.coeff_modulus();
        const auto poly_degree = key_parms.poly_modulus_degree();
        const size_t size_P = key_parms.special_modulus_size();
        const size_t size_QP = key_modulus.size();
        const size_t size_Q = size_QP - size_P;
        const lbcrypto::DCRTPoly& plainElement = openfhe_pt->GetElement<lbcrypto::DCRTPoly>();
        const std::shared_ptr<lbcrypto::ILDCRTParams<lbcrypto::BigInteger>> bigParams = plainElement.GetParams();
        const std::vector<std::shared_ptr<lbcrypto::ILNativeParams>>& nativeParams = bigParams->GetParams();
        const size_t size_DCRT = nativeParams.size();
        const auto scheme = key_parms.scheme();

        if (scheme == phantom::scheme_type::bfv || scheme == phantom::scheme_type::bgv) {
            throw std::invalid_argument("Unsupported scheme type for PhantomPlaintext construction from OpenFHE plaintext");
        }

        poly_modulus_degree_ = poly_degree;
        coeff_modulus_size_ = size_DCRT;
        scalingFactor_ = openfhe_pt->GetScalingFactor();
        scalingFactorInt_ = openfhe_pt->GetScalingFactorInt();
        chain_index_ = openfhe_pt->GetLevel() + 1;
        noiseScaleDeg_ = openfhe_pt->GetNoiseScaleDeg();
        slots_ = openfhe_pt->GetSlots();
        data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(coeff_modulus_size_ * poly_modulus_degree_, cudaStreamPerThread);
        for (size_t i = 0; i < coeff_modulus_size_; ++i) {
            const auto &poly = plainElement.GetElementAtIndex(i);
            cudaMemcpyAsync(data_.get() + i * poly_degree,
                        &poly.at(0), poly_degree * sizeof(uint64_t), cudaMemcpyHostToDevice,
                        cudaStreamPerThread);
        }
    }

    PhantomPlaintext(const PhantomPlaintext &) = default;

    PhantomPlaintext &operator=(const PhantomPlaintext &) = default;

    PhantomPlaintext(PhantomPlaintext &&) = default;

    PhantomPlaintext &operator=(PhantomPlaintext &&) = default;

    ~PhantomPlaintext() = default;

    void resize(const size_t coeff_modulus_size, const size_t poly_modulus_degree, const cudaStream_t &stream) {
        data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(coeff_modulus_size * poly_modulus_degree, stream);

        coeff_modulus_size_ = coeff_modulus_size;
        poly_modulus_degree_ = poly_modulus_degree;
    }

    [[nodiscard]] std::size_t get_poly_modulus_degree() const noexcept {
        return poly_modulus_degree_;
    }

    [[nodiscard]] std::size_t get_coeff_modulus_size() const noexcept {
        return coeff_modulus_size_;
    }

    [[nodiscard]] auto &chain_index() const noexcept {
        return chain_index_;
    }

    void set_chain_index(const size_t chain_index) {
        chain_index_ = chain_index;
    }

    [[nodiscard]] auto &GetScalingFactor() const {
        return scalingFactor_;
    }

    void SetScalingFactor(const double sf) {
        scalingFactor_ = sf;
    }

    [[nodiscard]] auto &GetScalingFactorInt() const {
        return scalingFactorInt_;
    }

    void SetScalingFactorInt(const NativeInteger &sfi) {
        scalingFactorInt_ = sfi;
    }

    [[nodiscard]] size_t GetLevel() const {
        return chain_index_ - 1;
    }

    void SetLevel(const size_t level) {
        chain_index_ = level + 1;
    }

    [[nodiscard]] auto &GetNoiseScaleDeg() const {
        return noiseScaleDeg_;
    }

    void SetNoiseScaleDeg(const size_t noiseScaleDeg) {
        noiseScaleDeg_ = noiseScaleDeg;
    }

    [[nodiscard]] unsigned long GetSlots() const {
        return slots_;
    }

    void SetSlots(const uint32_t slots) {
        slots_ = slots;
    }

    [[nodiscard]] auto data() const noexcept {
        return data_.get();
    }

    [[nodiscard]] auto &data_ptr() noexcept {
        return data_;
    }

    [[nodiscard]] auto &data_ptr() const noexcept {
        return data_;
    }

    void save(std::ostream &stream) const {
        stream.write(reinterpret_cast<const char *>(&poly_modulus_degree_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&coeff_modulus_size_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&scalingFactor_), sizeof(double));
        stream.write(reinterpret_cast<const char *>(&scalingFactorInt_), sizeof(NativeInteger));
        stream.write(reinterpret_cast<const char *>(&chain_index_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&noiseScaleDeg_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&slots_), sizeof(size_t));

        uint64_t *h_data;
        cudaMallocHost(&h_data, coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaMemcpy(h_data, data_.get(), coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        stream.write(reinterpret_cast<char *>(h_data), coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaFreeHost(h_data);
    }

    void load(std::istream &stream) {
        stream.read(reinterpret_cast<char *>(&poly_modulus_degree_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&coeff_modulus_size_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&scalingFactor_), sizeof(double));
        stream.read(reinterpret_cast<char *>(&scalingFactorInt_), sizeof(NativeInteger));
        stream.read(reinterpret_cast<char *>(&chain_index_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&noiseScaleDeg_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&slots_), sizeof(size_t));

        uint64_t *h_data;
        cudaMallocHost(&h_data, coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        stream.read(reinterpret_cast<char *>(h_data),
                    coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(coeff_modulus_size_ * poly_modulus_degree_,
                                                            cudaStreamPerThread);
        cudaMemcpyAsync(data_.get(), h_data, coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);
        cudaFreeHost(h_data);
    }
};
