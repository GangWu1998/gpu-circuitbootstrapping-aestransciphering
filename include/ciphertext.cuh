#pragma once

#include "context.cuh"
#include "prng.cuh"

#include "openfhe.h"

class PhantomCiphertext {

    friend class PhantomPublicKey;

    friend class PhantomSecretKey;

private:

    size_t size_ = 0; // The number of poly in ciphertext
    size_t poly_modulus_degree_ = 0; // The poly degree
    size_t coeff_modulus_size_ = 0; // The coeff prime number
    uint64_t correction_factor_ = 1; // The correction factor for BGV decryption
    bool is_ntt_form_ = true;
    bool is_asymmetric_ = false;

    // align with OpenFHE
    size_t noiseScaleDeg_ = 1; // the degree of the scaling factor for the encrypted message
    double scalingFactor_ = 1.0; // The scale this ciphertext corresponding to
    NativeInteger scalingFactorInt_ = 1;
    size_t chain_index_ = 0; // The index this ciphertext corresponding
    size_t hopslevel_ = 0; // Parameter for re-encryption to store the number of times the ciphertext has been re-encrypted.
    size_t slots_ = 0;

    phantom::util::cuda_auto_ptr<uint64_t> data_;
    std::vector<uint8_t> seed_; // only for symmetric encryption

public:

    PhantomCiphertext() = default;

    explicit PhantomCiphertext(const PhantomContext &context, const lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly>& openfhe_ct);

    PhantomCiphertext(const PhantomCiphertext &) = default;

    PhantomCiphertext &operator=(const PhantomCiphertext &) = default;

    PhantomCiphertext(PhantomCiphertext &&) = default;

    PhantomCiphertext &operator=(PhantomCiphertext &&) = default;

    ~PhantomCiphertext() = default;

    /* Reset and malloc the ciphertext
     * @notice: when size is larger, the previous data is copied.
     */
    void resize(const PhantomContext &context, size_t chain_index, size_t size, const cudaStream_t &stream) {
        auto &context_data = context.get_context_data(chain_index);
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        auto coeff_modulus_size = coeff_modulus.size();
        auto poly_modulus_degree = parms.poly_modulus_degree();

        size_t old_size = size_ * coeff_modulus_size_ * poly_modulus_degree_;
        size_t new_size = size * coeff_modulus_size * poly_modulus_degree;

        if (new_size == 0) {
            data_.reset();
            return;
        }

        if (new_size != old_size) {
            auto prev_data(std::move(data_));
            data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(new_size, stream);
            size_t copy_size = std::min(old_size, new_size);
            cudaMemcpyAsync(data_.get(), prev_data.get(), copy_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
                            stream);
        }

        size_ = size;
        chain_index_ = chain_index;
        poly_modulus_degree_ = poly_modulus_degree;
        coeff_modulus_size_ = coeff_modulus_size;
    }

    void resize(size_t size, size_t coeff_modulus_size, size_t poly_modulus_degree, const cudaStream_t &stream) {
        size_t old_size = size_ * coeff_modulus_size_ * poly_modulus_degree_;
        size_t new_size = size * coeff_modulus_size * poly_modulus_degree;

        if (new_size == 0) {
            data_.reset();
            return;
        }

        if (new_size != old_size) {
            auto prev_data(std::move(data_));
            data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(new_size, stream);
            size_t copy_size = std::min(old_size, new_size);
            cudaMemcpyAsync(data_.get(), prev_data.get(), copy_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
                            stream);
        }

        size_ = size;
        coeff_modulus_size_ = coeff_modulus_size;
        poly_modulus_degree_ = poly_modulus_degree;
    }

    void WriteToOpenFHECiphertext(
        const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& openfhe_cc,
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &openfhe_ct) const;

    [[nodiscard]] auto &size() const noexcept {
        return size_;
    }

    [[nodiscard]] auto &poly_modulus_degree() const noexcept {
        return poly_modulus_degree_;
    }
    void set_poly_modulus_degree(std::size_t poly_modulus_degree) {
        poly_modulus_degree_ = poly_modulus_degree;
    }

    [[nodiscard]] auto &coeff_modulus_size() const noexcept {
        return coeff_modulus_size_;
    }
    void set_coeff_modulus_size(std::size_t coeff_modulus_size) {
        coeff_modulus_size_ = coeff_modulus_size;
    }

    [[nodiscard]] auto &correction_factor() const noexcept {
        return correction_factor_;
    }
    void set_correction_factor(const uint64_t correction_factor) {
        correction_factor_ = correction_factor;
    }

    [[nodiscard]] auto &is_ntt_form() const noexcept {
        return is_ntt_form_;
    }
    void set_ntt_form(const bool is_ntt_form) {
        is_ntt_form_ = is_ntt_form;
    }

    [[nodiscard]] bool is_asymmetric() const noexcept {
        return is_asymmetric_;
    }

    [[nodiscard]] auto &GetNoiseScaleDeg() const {
        return noiseScaleDeg_;
    }
    void SetNoiseScaleDeg(const size_t noiseScaleDeg) {
        noiseScaleDeg_ = noiseScaleDeg;
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
    [[nodiscard]] auto &chain_index() const noexcept {
        return chain_index_;
    }
    void set_chain_index(const size_t chain_index) {
        chain_index_ = chain_index;
    }

    [[nodiscard]] size_t GetHopLevel() const {
        return hopslevel_;
    }
    void SetHopLevel(const uint32_t hoplevel) {
        hopslevel_ = hoplevel;
    }

    [[nodiscard]] unsigned long GetSlots() const {
        return slots_;
    }
    void SetSlots(const uint32_t slots) {
        slots_ = slots;
    }

    [[nodiscard]] auto data() const {
        return data_.get();
    }

    [[nodiscard]] auto &data_ptr() {
        return data_;
    }

    [[nodiscard]] auto &seed_ptr() {
        return seed_;
    }

    void save(std::ostream &stream) const {
        stream.write(reinterpret_cast<const char *>(&size_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&poly_modulus_degree_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&coeff_modulus_size_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&correction_factor_), sizeof(uint64_t));
        stream.write(reinterpret_cast<const char *>(&is_ntt_form_), sizeof(bool));
        stream.write(reinterpret_cast<const char *>(&is_asymmetric_), sizeof(bool));
        stream.write(reinterpret_cast<const char *>(&noiseScaleDeg_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&scalingFactor_), sizeof(double));
        stream.write(reinterpret_cast<const char *>(&scalingFactorInt_), sizeof(NativeInteger));
        stream.write(reinterpret_cast<const char *>(&chain_index_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&hopslevel_), sizeof(uint32_t));
        stream.write(reinterpret_cast<const char *>(&slots_), sizeof(uint32_t));

        uint64_t *h_data;
        cudaMallocHost(&h_data, size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaMemcpy(h_data, data_.get(), size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        stream.write(reinterpret_cast<char *>(h_data),
                     size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaFreeHost(h_data);
    }

    void load(std::istream &stream) {
        stream.read(reinterpret_cast<char *>(&size_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&poly_modulus_degree_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&coeff_modulus_size_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&correction_factor_), sizeof(uint64_t));
        stream.read(reinterpret_cast<char *>(&is_ntt_form_), sizeof(bool));
        stream.read(reinterpret_cast<char *>(&is_asymmetric_), sizeof(bool));
        stream.read(reinterpret_cast<char *>(&noiseScaleDeg_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&scalingFactor_), sizeof(double));
        stream.read(reinterpret_cast<char *>(&scalingFactorInt_), sizeof(NativeInteger));
        stream.read(reinterpret_cast<char *>(&chain_index_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&hopslevel_), sizeof(uint32_t));
        stream.read(reinterpret_cast<char *>(&slots_), sizeof(uint32_t));

        uint64_t *h_data;
        cudaMallocHost(&h_data, size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        stream.read(reinterpret_cast<char *>(h_data),
                    size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size_ * coeff_modulus_size_ * poly_modulus_degree_,
                                                            cudaStreamPerThread);
        cudaMemcpyAsync(data_.get(), h_data, size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);
        cudaFreeHost(h_data);
        cudaStreamSynchronize(cudaStreamPerThread);
    }

    void save_symmetric(std::ostream &stream) const {
        if (is_asymmetric_)
            throw std::runtime_error("Asymmetric ciphertext does not have seed.");

        if (size_ != 2)
            throw std::runtime_error("This method is only for 2-polynomial ciphertext.");

        stream.write(reinterpret_cast<const char *>(&size_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&poly_modulus_degree_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&coeff_modulus_size_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&correction_factor_), sizeof(uint64_t));
        stream.write(reinterpret_cast<const char *>(&is_ntt_form_), sizeof(bool));
        stream.write(reinterpret_cast<const char *>(&is_asymmetric_), sizeof(bool));
        stream.write(reinterpret_cast<const char *>(&noiseScaleDeg_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&scalingFactor_), sizeof(double));
        stream.write(reinterpret_cast<const char *>(&scalingFactorInt_), sizeof(NativeInteger));
        stream.write(reinterpret_cast<const char *>(&chain_index_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&hopslevel_), sizeof(uint32_t));
        stream.write(reinterpret_cast<const char *>(&slots_), sizeof(uint32_t));

        // Only save c0
        uint64_t *h_c0;
        cudaMallocHost(&h_c0, coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaMemcpy(h_c0, data_.get(), coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        stream.write(reinterpret_cast<char *>(h_c0), coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaFreeHost(h_c0);

        // Save seed of a instead of c1
        stream.write(reinterpret_cast<const char *>(seed_.data()),
                     phantom::util::global_variables::prng_seed_byte_count);
    }

    void load_symmetric(const PhantomContext &context, std::istream &stream) {
        stream.read(reinterpret_cast<char *>(&size_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&poly_modulus_degree_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&coeff_modulus_size_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&correction_factor_), sizeof(uint64_t));
        stream.read(reinterpret_cast<char *>(&is_ntt_form_), sizeof(bool));
        stream.read(reinterpret_cast<char *>(&is_asymmetric_), sizeof(bool));
        stream.read(reinterpret_cast<char *>(&noiseScaleDeg_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&scalingFactor_), sizeof(double));
        stream.read(reinterpret_cast<char *>(&scalingFactorInt_), sizeof(NativeInteger));
        stream.read(reinterpret_cast<char *>(&chain_index_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&hopslevel_), sizeof(uint32_t));
        stream.read(reinterpret_cast<char *>(&slots_), sizeof(uint32_t));

        if (is_asymmetric_)
            throw std::runtime_error("Asymmetric ciphertext does not have seed.");

        if (size_ != 2)
            throw std::runtime_error("This method is only for 2-polynomial ciphertext.");

        data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(2 * coeff_modulus_size_ * poly_modulus_degree_,
                                                            cudaStreamPerThread);
        auto *d_c0 = data_.get();
        auto *d_c1 = data_.get() + coeff_modulus_size_ * poly_modulus_degree_;

        // Load c0 directly from stream
        uint64_t *h_c0;
        cudaMallocHost(&h_c0, coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        stream.read(reinterpret_cast<char *>(h_c0), coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaMemcpyAsync(d_c0, h_c0, coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);

        // Load c1 by generating from seed
        seed_.resize(phantom::util::global_variables::prng_seed_byte_count);
        stream.read(reinterpret_cast<char *>(seed_.data()), phantom::util::global_variables::prng_seed_byte_count);

        auto d_seed = phantom::util::make_cuda_auto_ptr<uint8_t>(
                phantom::util::global_variables::prng_seed_byte_count, cudaStreamPerThread);
        cudaMemcpyAsync(d_seed.get(), seed_.data(), phantom::util::global_variables::prng_seed_byte_count,
                        cudaMemcpyHostToDevice, cudaStreamPerThread);

        // uniform random generator
        auto &first_context_data = context.get_context_data(context.get_first_index());
        auto &first_parms = first_context_data.parms();
        auto &first_coeff_modulus = first_parms.coeff_modulus();
        auto first_coeff_mod_size = first_coeff_modulus.size();

        if (first_coeff_mod_size != coeff_modulus_size_) {
            throw std::runtime_error("Only support ciphertext without modulus switching.");
        }

        auto base_rns = context.gpu_rns_tables().modulus();
        sample_uniform_poly_wrap(
                d_c1, d_seed.get(), base_rns, poly_modulus_degree_, coeff_modulus_size_, cudaStreamPerThread);

        if (!is_ntt_form_) {
            // Transform c1 to non-NTT form
            nwt_2d_radix8_backward_inplace(d_c1, context.gpu_rns_tables(), coeff_modulus_size_, 0, cudaStreamPerThread);
        }

        cudaStreamSynchronize(cudaStreamPerThread);

        // cleanup h_c0
        cudaFreeHost(h_c0);
    }
};
