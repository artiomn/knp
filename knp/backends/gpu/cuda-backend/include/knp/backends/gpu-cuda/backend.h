/**
 * @file backend.h
 * @brief Class definition for CUDA GPU backend.
 * @kaspersky_support Artiom N.
 * @date 24.02.2025
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <knp/core/backend.h>
#include <knp/core/impexp.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/devices/gpu_cuda.h>
#include <knp/neuron-traits/all_traits.h>
#include <knp/synapse-traits/all_traits.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/config.hpp>
#include <boost/dll/alias.hpp>
#include <boost/mp11.hpp>


/**
 * @brief Namespace for single-threaded backend.
 */
namespace knp::backends::gpu
{

/**
 * @brief The CUDAPopulation class is a definition of a CUDA neurons population.
 */
template <typename NeuronType>
struct CUDAPopulation
{
    /**
     * @brief Type of the population neurons.
     */
    using PopulationNeuronType = NeuronType;
    /**
     * @brief Population of neurons with the specified neuron type.
     */
    using PopulationType = CUDAPopulation<NeuronType>;
    /**
     * @brief Neuron parameters and their values for the specified neuron type.
     */
    using NeuronParameters = neuron_traits::neuron_parameters<NeuronType>;

    /**
     * @brief Constructor.
     * @param population source population.
     */
    explicit CUDAPopulation(const knp::core::Population<NeuronType> &population)
        : uid_{population.get_uid().tag.begin(), population.get_uid().tag.end()},
          neurons_{population.get_neurons_parameters()}
    {
    }

    /**
     * @brief UID.
     */
    thrust::device_vector<std::uint8_t> uid_{16};
    /**
     * @brief Neurons.
     */
    thrust::device_vector<NeuronParameters> neurons_;
};


/**
 * @brief The CUDAProjection class is a definition of a CUDA synapses.
 */
template <typename SynapseType>
struct CUDAProjection
{
    /**
     * @brief Type of the projection synapses.
     */
    using ProjectionSynapseType = SynapseType;
    /**
     * @brief Projection of synapses with the specified synapse type.
     */
    using ProjectionType = CUDAProjection<SynapseType>;
    /**
     * @brief Parameters of the specified synapse type.
     */
    using SynapseParameters = typename synapse_traits::synapse_parameters<SynapseType>;

    /**
     * @brief Synapse description structure that contains synapse parameters and indexes of the associated neurons.
     */
    using Synapse = std::tuple<SynapseParameters, uint32_t, uint32_t>;

    /**
     * @brief Constructor.
     * @param projection source projection.
     */
    explicit CUDAProjection(const knp::core::Projection<SynapseType> &projection)
        : uid_(projection.get_uid().tag.begin(), projection.get_uid().tag.end()),
          presynaptic_uid_(projection.get_presynaptic().tag.begin(), projection.get_presynaptic().tag.end()),
          postsynaptic_uid_(projection.get_postsynaptic().tag.begin(), projection.get_postsynaptic().tag.end()),
          is_locked_(thrust::device_malloc<bool>(1))
    {
        *is_locked_ = projection.is_locked();
    }
    /**
     * @brief Destructor.
     */
    ~CUDAProjection() { thrust::device_free(is_locked_); }

    /**
     * @brief UID.
     */
    thrust::device_vector<std::uint8_t> uid_{16};

    /**
     * @brief UID of the population that sends spikes to the projection (presynaptic population)
     */
    thrust::device_vector<std::uint8_t> presynaptic_uid_{16};

    /**
     * @brief UID of the population that receives synapse responses from this projection (postsynaptic population).
     */
    thrust::device_vector<std::uint8_t> postsynaptic_uid_{16};

    /**
     * @brief Return `false` if the weight change for synapses is not locked.
     */
    thrust::device_ptr<bool> is_locked_;

    /**
     * @brief Container of synapse parameters.
     */
    thrust::device_vector<SynapseParameters> synapses_;

    /**
     * @brief Messages container.
     */
    // cppcheck-suppress unusedStructMember
    std::unordered_map<uint64_t, knp::core::messaging::SynapticImpactMessage> messages_;
};


/**
 * @brief The CUDABackend class is a definition of an interface to the CUDA GPU backend.
 */
class KNP_DECLSPEC CUDABackend : public knp::core::Backend
{
public:
    /**
     * @brief List of neuron types supported by the single-threaded CPU backend.
     */
    using SupportedNeurons = boost::mp11::mp_list<knp::neuron_traits::BLIFATNeuron>;

    /**
     * @brief List of synapse types supported by the single-threaded CPU backend.
     */
    using SupportedSynapses = boost::mp11::mp_list<knp::synapse_traits::DeltaSynapse>;

    /**
     * @brief List of supported population types based on neuron types specified in `SupportedNeurons`.
     */
    using SupportedPopulations = boost::mp11::mp_transform<knp::core::Population, SupportedNeurons>;

    /**
     * @brief List of supported projection types based on synapse types specified in `SupportedSynapses`.
     */
    using SupportedProjections = boost::mp11::mp_transform<knp::core::Projection, SupportedSynapses>;

    /**
     * @brief Population variant that contains any population type specified in `SupportedPopulations`.
     * @details `PopulationVariants` takes the value of `std::variant<PopulationType_1,..., PopulationType_n>`, where
     * `PopulationType_[1..n]` is the population type specified in `SupportedPopulations`. \n
     * For example, if `SupportedPopulations` contains BLIFATNeuron and IzhikevichNeuron types,
     * then `PopulationVariants = std::variant<BLIFATNeuron, IzhikevichNeuron>`. \n
     * `PopulationVariants` retains the same order of message types as defined in `SupportedPopulations`.
     * @see ALL_NEURONS.
     */
    using PopulationVariants = boost::mp11::mp_rename<SupportedPopulations, std::variant>;

    /**
     * @brief Projection variant that contains any projection type specified in `SupportedProjections`.
     * @details `ProjectionVariants` takes the value of `std::variant<ProjectionType_1,..., ProjectionType_n>`, where
     * `ProjectionType_[1..n]` is the projection type specified in `SupportedProjections`. \n
     * For example, if `SupportedProjections` contains DeltaSynapse and AdditiveSTDPSynapse types,
     * then `ProjectionVariants = std::variant<DeltaSynapse, AdditiveSTDPSynapse>`. \n
     * `ProjectionVariants` retains the same order of message types as defined in `SupportedProjections`.
     * @see ALL_SYNAPSES.
     */
    using ProjectionVariants = boost::mp11::mp_rename<SupportedProjections, std::variant>;

private:
    using SupportedCUDAPopulations = boost::mp11::mp_transform<CUDAPopulation, SupportedNeurons>;
    using CUDAPopulationVariants = boost::mp11::mp_rename<SupportedCUDAPopulations, std::variant>;

    using SupportedCUDAProjections = boost::mp11::mp_transform<CUDAProjection, SupportedSynapses>;
    using CUDAProjectionVariants = boost::mp11::mp_rename<SupportedCUDAProjections, std::variant>;

public:
    /**
     * @brief Type of population container.
     */
    using PopulationContainer = thrust::host_vector<PopulationVariants>;
    /**
     * @brief Type of projection container.
     */
    using ProjectionContainer = thrust::host_vector<ProjectionVariants>;

    /**
     * @brief Types of non-constant population iterators.
     */
    using PopulationIterator = typename PopulationContainer::iterator;

    /**
     * @brief Types of non-constant projection iterators.
     */
    using ProjectionIterator = typename ProjectionContainer::iterator;

    /**
     * @brief Types of constant population iterators.
     */
    using PopulationConstIterator = typename PopulationContainer::const_iterator;
    /**
     * @brief Types of constant projection iterators.
     */
    using ProjectionConstIterator = typename ProjectionContainer::const_iterator;

public:
    /**
     * @brief Default constructor for CUDA backend.
     */
    CUDABackend();
    /**
     * @brief Destructor for CUDA backend.
     */
    ~CUDABackend() override = default;

public:
    /**
     * @brief Create an object of the single-threaded CUDA backend.
     * @return shared pointer to backend object.
     */
    static std::shared_ptr<CUDABackend> create();

public:
    /**
     * @brief Define if plasticity is supported.
     * @return `true` if plasticity is supported, `false` if plasticity is not supported.
     */
    [[nodiscard]] bool plasticity_supported() const override { return true; }
    /**
     * @brief Get type names of supported neurons.
     * @return vector of supported neuron type names.
     */
    [[nodiscard]] std::vector<std::string> get_supported_neurons() const override;
    /**
     * @brief Get type names of supported synapses.
     * @return vector of supported synapse type names.
     */
    [[nodiscard]] std::vector<std::string> get_supported_synapses() const override;
    /**
     * @brief Get indexes of supported projections.
     * @return type indexes.
     */
    [[nodiscard]] std::vector<size_t> get_supported_projection_indexes() const override;
    /**
     * @brief Get indexes of supported populations.
     * @return type indexes.
     */
    [[nodiscard]] std::vector<size_t> get_supported_population_indexes() const override;

public:
    /**
     * @brief Load populations to the backend.
     * @param populations vector of populations to load.
     */
    void load_populations(const std::vector<PopulationVariants> &populations);

    /**
     * @brief Load projections to the backend.
     * @param projections vector of projections to load.
     */
    void load_projections(const std::vector<ProjectionVariants> &projections);

    /**
     * @brief Add projections to backend.
     * @throw exception if the `projections` parameter contains unsupported projection types.
     * @param projections projections to add.
     */
    void load_all_projections(const std::vector<knp::core::AllProjectionsVariant> &projections) override;

    /**
     * @brief Add populations to backend.
     * @throw exception if the `populations` parameter contains unsupported population types.
     * @param populations populations to add.
     */
    void load_all_populations(const std::vector<knp::core::AllPopulationsVariant> &populations) override;

public:
    /**
     * @brief Get an iterator pointing to the first element of the population loaded to backend.
     * @return population iterator.
     */
    PopulationIterator begin_populations();

    /**
     * @brief Get an iterator pointing to the first element of the population loaded to backend.
     * @return constant population iterator.
     */
    PopulationConstIterator begin_populations() const;
    /**
     * @brief Get an iterator pointing to the last element of the population.
     * @return iterator.
     */
    PopulationIterator end_populations();
    /**
     * @brief Get a constant iterator pointing to the last element of the population.
     * @return iterator.
     */
    PopulationConstIterator end_populations() const;

    /**
     * @todo Make iterator which returns projections, but not a wrapper.
     */
    /**
     * @brief Get an iterator pointing to the first element of the projection loaded to backend.
     * @return projection iterator.
     */
    ProjectionIterator begin_projections();
    /**
     * @brief Get an iterator pointing to the first element of the projection loaded to backend.
     * @return constant projection iterator.
     */
    ProjectionConstIterator begin_projections() const;
    /**
     * @brief Get an iterator pointing to the last element of the projection.
     * @return iterator.
     */
    ProjectionIterator end_projections();
    /**
     * @brief Get a constant iterator pointing to the last element of the projection.
     * @return iterator.
     */
    ProjectionConstIterator end_projections() const;

public:
    /**
     * @brief Remove projections with given UIDs from the backend.
     * @param uids UIDs of projections to remove.
     */
    void remove_projections(const std::vector<knp::core::UID> &uids) override {}

    /**
     * @brief Remove populations with given UIDs from the backend.
     * @param uids UIDs of populations to remove.
     */
    void remove_populations(const std::vector<knp::core::UID> &uids) override {}

public:
    /**
     * @brief Get a list of devices supported by the backend.
     * @return list of devices.
     * @see Device.
     */
    [[nodiscard]] std::vector<std::unique_ptr<knp::core::Device>> get_devices() const override;

public:
    /**
     * @copydoc knp::core::Backend::_step()
     */
    void _step() override;

    /**
     * @brief Stop training by locking all projections.
     */
    void stop_learning() override
    {
        for (auto &proj : projections_) std::visit([](auto &entity) { entity.lock_weights(); }, proj);
    }

    /**
     * @brief Resume training by unlocking all projections.
     */
    void start_learning() override
    {
        /**
         * @todo Probably only need to use `start_learning` for some of projections: the ones that were locked with
         * `lock()`.
         */
        for (auto &proj : projections_) std::visit([](auto &entity) { entity.unlock_weights(); }, proj);
    }

    /**
     * @brief Get a set of iterators for projections and populations.
     * @return `DataRanges` structure containing iterators.
     */
    [[nodiscard]] DataRanges get_network_data() const override { return {}; }

protected:
    /**
     * @brief Map used for message construction. It maps a message to its future output step.
     */
    using SynapticMessageQueue = std::unordered_map<uint64_t, core::messaging::SynapticImpactMessage>;

    /**
     * @copydoc knp::core::Backend::_init()
     */
    void _init() override;

    /**
     * @brief Calculate population of BLIFAT neurons.
     * @note Population will be changed during calculation.
     * @param population population to calculate.
     * @return copy of a spike message if population is emitting one.
     */
    std::optional<core::messaging::SpikeMessage> calculate_population(
        knp::core::Population<knp::neuron_traits::BLIFATNeuron> &population);

    /**
     * @brief Calculate population of `SynapticResourceSTDPNeuron` neurons.
     * @note Population will be changed during calculation.
     * @param population population to calculate.
     * @return optional `SpikeMessage`.
     */
    std::optional<core::messaging::SpikeMessage> calculate_population(
        knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population);
    /**
     * @brief Calculate projection of delta synapses.
     * @note Projection will be changed during calculation.
     * @param projection projection to calculate.
     * @param message_queue message queue to send to projection for calculation.
     */
    void calculate_projection(
        knp::core::Projection<knp::synapse_traits::DeltaSynapse> &projection, SynapticMessageQueue &message_queue);
    /**
     * @brief Calculate projection of `AdditiveSTDPDeltaSynapse` synapses.
     * @note Projection will be changed during calculation.
     * @param projection projection to calculate.
     * @param message_queue message queue to send to projection for calculation.
     */
    void calculate_projection(
        knp::core::Projection<knp::synapse_traits::AdditiveSTDPDeltaSynapse> &projection,
        SynapticMessageQueue &message_queue);
    /**
     * @brief Calculate projection of `SynapticResourceSTDPDeltaSynapse` synapses.
     * @note Projection will be changed during calculation.
     * @param projection projection to calculate.
     * @param message_queue message queue to send to projection for calculation.
     */
    void calculate_projection(
        knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> &projection,
        SynapticMessageQueue &message_queue);

private:
    // cppcheck-suppress unusedStructMember
    PopulationContainer populations_;
    ProjectionContainer projections_;

    // cppcheck-suppress unusedStructMember
    std::vector<CUDAPopulationVariants> device_populations_;
    // cppcheck-suppress unusedStructMember
    std::vector<CUDAProjectionVariants> device_projections_;
};

}  // namespace knp::backends::gpu
