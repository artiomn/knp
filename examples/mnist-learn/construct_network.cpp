/**
 * @file construct_network.cpp
 * @brief Functions for network construction.
 * @kaspersky_support A. Vartenkov
 * @date 03.12.2024
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

#include "construct_network.h"

#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/framework/sonata/network_io.h>
#include <knp/neuron-traits/all_traits.h>
#include <knp/synapse-traits/all_traits.h>

#include "generators.h"

// A list of short type names to make reading easier.
using BlifatPopulation = knp::core::Population<knp::neuron_traits::BLIFATNeuron>;
using ResourceBlifatPopulation = knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;
using ResourceNeuron = knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron;
using ResourceNeuronData = knp::neuron_traits::neuron_parameters<ResourceNeuron>;

// Network hyperparameters. You may want to fine-tune these.
constexpr float default_threshold = 8.571F;
constexpr float min_synaptic_weight = -0.7;
constexpr float max_synaptic_weight = 0.864249F;
constexpr float base_weight_value = 0.000F;
constexpr int neuron_dopamine_period = 10;
constexpr int synapse_dopamine_period = 10;
constexpr float l_neuron_potential_decay = 1.0 - 1.0 / 3.0;
constexpr float dopamine_parameter = 0.042F;
constexpr float dopamine_value = dopamine_parameter;
constexpr float threshold_weight_coeff = 0.023817F;

//
// Network geometry.
//

// Number of neurons reserved per a single digit.
constexpr int neurons_per_column = 15;

// Ten possible digits, one column per each.
constexpr int num_possible_labels = classes_in_mnist;

// All columns are a part of the same population.
constexpr int num_input_neurons = neurons_per_column * num_possible_labels;

// Number of pixels for a single MNIST image.
constexpr int input_size = 28 * 28;

// Dense input projection from 28 * 28 image to population of 150 neurons.
constexpr int input_projection_size = input_size * num_input_neurons;


// Intermediate population neurons.
template <class Neuron>
struct PopulationData
{
    size_t size_;
    knp::neuron_traits::neuron_parameters<Neuron> neuron_;
};


enum PopIndexes
{
    INPUT = 0,
    DOPAMINE = 1,
    OUTPUT = 2,
    GATE = 3
};


// Calculate synaptic resource value given synapse weight.
float resource_from_weight(float weight, float min_weight, float max_weight)
{
    // Max weight is only possible with infinite resource, so we should select a value less than that.
    float eps = 1e-6;
    if (min_weight > max_weight) std::swap(min_weight, max_weight);
    if (weight < min_weight || weight >= max_weight - eps)
        throw std::logic_error("Weight should not be less than min_weight, more than max_weight or too close to it.");
    double diff = max_weight - min_weight;
    double over = weight - min_weight;
    return static_cast<float>(over * diff / (diff - over));
}


// Add populations to the network.
auto add_subnetwork_populations(AnnotatedNetwork &result)
{
    result.data_.wta_data_.push_back({});
    // Parameters for a default neuron.
    ResourceNeuronData default_neuron{{}};
    default_neuron.activation_threshold_ = default_threshold;
    ResourceNeuronData l_neuron = default_neuron;
    l_neuron.potential_decay_ = l_neuron_potential_decay;
    l_neuron.d_h_ = -dopamine_value;
    l_neuron.dopamine_plasticity_time_ = neuron_dopamine_period;
    l_neuron.synapse_sum_threshold_coefficient_ = threshold_weight_coeff;
    l_neuron.isi_max_ = 10;

    struct PopulationRole
    {
        PopulationData<ResourceNeuron> pd_;
        bool for_inference_;
        bool output_;
        std::string name_;
    };
    auto dopamine_neuron = default_neuron;
    dopamine_neuron.total_blocking_period_ = 0;
    // Create initial neuron data for populations. There are four of them.
    std::vector<PopulationRole> pop_data{
        {{num_input_neurons, l_neuron}, true, false, "INPUT"},
        {{num_input_neurons, dopamine_neuron}, false, false, "DOPAMINE"},
        {{num_possible_labels, default_neuron}, true, true, "OUTPUT"},
        {{num_possible_labels, default_neuron}, false, false, "GATE"}};

    // Creating a population. It's usually very simple as all neurons are usually the same.
    std::vector<knp::core::UID> population_uids;
    for (auto &pop_init_data : pop_data)
    {
        // A very simple neuron generator returning a default neuron.
        auto neuron_generator = [&pop_init_data](size_t index) { return pop_init_data.pd_.neuron_; };

        knp::core::UID uid;
        result.network_.add_population(ResourceBlifatPopulation{uid, neuron_generator, pop_init_data.pd_.size_});
        population_uids.push_back(uid);
        result.data_.population_names_[uid] = pop_init_data.name_;
        if (pop_init_data.for_inference_) result.data_.inference_population_uids_.insert(uid);
        if (pop_init_data.output_) result.data_.output_uids_.push_back(uid);
    }

    result.data_.wta_data_.back().first.push_back(population_uids[INPUT]);
    return std::make_pair(population_uids, pop_data);
}


// Create network for MNIST.
AnnotatedNetwork create_example_network(int num_compound_networks)
{
    AnnotatedNetwork result;
    for (int i = 0; i < num_compound_networks; ++i)
    {
        auto [population_uids, pop_data] = add_subnetwork_populations(result);

        // Now that we added all the populations we need, we have to connect them with projections.
        // Creating a projection is more tricky, as all the connection logic should be described in a generator.
        // Create a default synapse.
        ResourceSynapseParams default_synapse;
        auto afferent_synapse = default_synapse;
        afferent_synapse.rule_.synaptic_resource_ =
            resource_from_weight(base_weight_value, min_synaptic_weight, max_synaptic_weight);
        afferent_synapse.rule_.dopamine_plasticity_period_ = synapse_dopamine_period;
        afferent_synapse.rule_.w_min_ = min_synaptic_weight;
        afferent_synapse.rule_.w_max_ = max_synaptic_weight;

        // 1. Trainable input projection.
        ResourceDeltaProjection input_projection{
            knp::core::UID{false}, population_uids[INPUT], make_dense_generator(input_size, afferent_synapse),
            input_projection_size};
        result.data_.projections_from_raster_.push_back(input_projection.get_uid());
        input_projection.unlock_weights();  // Trainable
        result.network_.add_projection(input_projection);
        result.data_.inference_internal_projection_.insert(input_projection.get_uid());

        default_synapse.weight_ = 9;

        // 2. Activating projection. It sends signals from labels to dopamine population.
        const DeltaSynapseData default_activating_synapse{1, 1, knp::synapse_traits::OutputType::BLOCKING};
        DeltaProjection projection_2{
            knp::core::UID{false}, population_uids[DOPAMINE],
            make_aligned_generator(pop_data[INPUT].pd_.size_, pop_data[DOPAMINE].pd_.size_, default_activating_synapse),
            pop_data[INPUT].pd_.size_};
        result.network_.add_projection(projection_2);
        result.data_.wta_data_[i].second.push_back(projection_2.get_uid());

        // 3. Dopamine projection, it goes from dopamine population to input population.
        const DeltaSynapseData default_dopamine_synapse{dopamine_value, 1, knp::synapse_traits::OutputType::DOPAMINE};
        DeltaProjection projection_3{
            population_uids[DOPAMINE], population_uids[INPUT],
            make_aligned_generator(pop_data[DOPAMINE].pd_.size_, pop_data[INPUT].pd_.size_, default_dopamine_synapse),
            pop_data[INPUT].pd_.size_};

        result.network_.add_projection(projection_3);
        result.data_.inference_internal_projection_.insert(projection_3.get_uid());

        // 4. Strong excitatory projection going to output neurons.
        DeltaProjection projection_4{
            knp::core::UID{false}, population_uids[OUTPUT],
            make_aligned_generator(pop_data[INPUT].pd_.size_, pop_data[OUTPUT].pd_.size_, default_synapse),
            pop_data[INPUT].pd_.size_};
        result.data_.wta_data_[i].second.push_back(projection_4.get_uid());

        result.network_.add_projection(projection_4);
        result.data_.inference_internal_projection_.insert(projection_4.get_uid());

        // 5. Blocking projection.
        const DeltaSynapseData default_blocking_synapse{-20, 1, knp::synapse_traits::OutputType::BLOCKING};
        DeltaProjection projection_5{
            population_uids[OUTPUT], population_uids[GATE],
            make_aligned_generator(pop_data[OUTPUT].pd_.size_, pop_data[GATE].pd_.size_, default_blocking_synapse),
            num_possible_labels};
        result.network_.add_projection(projection_5);
        result.data_.inference_internal_projection_.insert(projection_5.get_uid());

        // 6. Strong excitatory projection going from ground truth classes.
        DeltaProjection projection_6{
            knp::core::UID{false}, population_uids[DOPAMINE],
            make_aligned_generator(num_possible_labels, pop_data[DOPAMINE].pd_.size_, default_synapse),
            pop_data[DOPAMINE].pd_.size_};
        result.network_.add_projection(projection_6);
        result.data_.projections_from_classes_.push_back(projection_6.get_uid());

        // 7. Strong slow excitatory projection going from ground truth classes.
        auto slow_synapse = default_synapse;
        slow_synapse.delay_ = 10;
        DeltaProjection projection_7{
            knp::core::UID{false}, population_uids[GATE],
            make_aligned_generator(num_possible_labels, pop_data[GATE].pd_.size_, slow_synapse),
            pop_data[GATE].pd_.size_};
        result.network_.add_projection(projection_7);
        result.data_.projections_from_classes_.push_back(projection_7.get_uid());

        // 8. Strong inhibitory projection from ground truth input.
        auto inhibitory_synapse = default_synapse;
        inhibitory_synapse.weight_ = -30;
        DeltaProjection projection_8{
            knp::core::UID{false}, population_uids[GATE],
            make_exclusive_generator(num_possible_labels, inhibitory_synapse),
            num_possible_labels * (pop_data[GATE].pd_.size_ - 1)};
        result.data_.projections_from_classes_.push_back(projection_8.get_uid());
        result.network_.add_projection(projection_8);

        // 9. Weak excitatory projection.
        auto weak_excitatory_synapse = default_synapse;
        weak_excitatory_synapse.weight_ = 3;
        DeltaProjection projection_9{
            population_uids[GATE], population_uids[INPUT],
            make_aligned_generator(pop_data[GATE].pd_.size_, pop_data[INPUT].pd_.size_, weak_excitatory_synapse),
            pop_data[INPUT].pd_.size_};
        result.network_.add_projection(projection_9);
        result.data_.inference_internal_projection_.insert(projection_9.get_uid());
    }

    // Return created network.
    return result;
}
