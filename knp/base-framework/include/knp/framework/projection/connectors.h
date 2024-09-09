/**
 * @file generators.h
 * @brief Population generators.
 * @author Artiom N.
 * @date 10.08.2024
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
 */
#pragma once

#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <exception>
#include <functional>
#include <optional>
#include <random>
#include <tuple>

/**
 * @brief Projection namespace.
 */
namespace knp::framework::projection
{

/**
 * @brief Namespace for framework projection connectors.
 */
namespace connectors
{

/**
 * @brief Default generator of synapse parameters.
 * @tparam SynapseType synapse type.
 * @return synapse parameters.
 */
template <typename SynapseType>
typename knp::core::Projection<SynapseType>::SynapseParameters default_synapse_gen(size_t)  // NOLINT
{
    return typename knp::core::Projection<SynapseType>::SynapseParameters();
}


/**
 * @brief Default generator of synapse parameters.
 * @tparam SynapseType type of the synapse.
 * @return synapse parameters.
 */
template <typename SynapseType>
typename knp::core::Projection<SynapseType>::SynapseParameters default_synapse_gen1(size_t, size_t)
{
    return typename knp::core::Projection<SynapseType>::SynapseParameters();
}


/**
 * @brief Make connections between each presynaptic population (source) neuron to each postsynaptic population (destination) neuron.
 * @details Simple connector that generates connections from source neuron index to all destination indexes and
 * otherwise. For populations of size `N x M` the connector generates connections such as: `0 -> 0`, `0 -> 1`, `0 -> 2`, ..., 
 * `0 -> M`, `1 -> 0`, `1 -> 1`, ..., `1 -> M`, ..., `N -> M`.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population neuron count.
 * @param postsynaptic_pop_size postsynaptic population neuron count.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> all_to_all(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t presynaptic_pop_size,
    size_t postsynaptic_pop_size,
    std::function<typename knp::core::Projection<SynapseType>::SynapseParameters(size_t, size_t)> syn_gen =
        default_synapse_gen1<SynapseType>)
{
    const auto proj_size = presynaptic_pop_size * postsynaptic_pop_size;
    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid,
        [presynaptic_pop_size, postsynaptic_pop_size,
         syn_gen](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
        {
            const size_t index0 = index % presynaptic_pop_size;
            const size_t index1 = index / presynaptic_pop_size;

            return std::make_tuple(syn_gen(index0, index1), index0, index1);
        },
        proj_size);
}


/**
 * @brief Make one-to-one connections between neurons of presynaptic and postsynaptic populations. 
 * @details Simple connector that generates connections from source neuron index to the same destination index.
 * For the populations of size `N x N` the connector generates connections such as: `0 -> 0`, `1 -> 1`, `2 -> 2`, ..., `N -> N`.
 * @pre Population sizes must be equal.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param population_size neuron count in populations.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> one_to_one(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t population_size,
    std::function<typename knp::core::Projection<SynapseType>::SynapseParameters(size_t)> syn_gen =
        default_synapse_gen<SynapseType>)
{
    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid,
        [syn_gen](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
        { return std::make_tuple(syn_gen(index), index, index); },
        population_size);
}


/**
 * @brief Generate projection from container.
 * @details Container must contain synapses as `(parameters, from_index, to_index)` tuples, 
 * where `parameters` are synapse parameters, `from_index` is presynaptic neuron index, 
 * and `to_index` is postsynaptic neuron index. 
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param container container with synapses.
 * @tparam SynapseType neuron type.
 * @tparam Container container type.
 * @return projection.
 */
template <typename SynapseType, template <typename...> class Container>
[[nodiscard]] knp::core::Projection<SynapseType> from_container(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid,
    const Container<typename core::Projection<SynapseType>::Synapse> &container)
{
    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid,
        [&container](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
        { return container[index]; },
        container.size());
}


/**
 * @brief Generate projection from `std::map` object.
 * @details 'std::map' object must contain synapse parameters as values and `(from_index, to_index)` tuples as keys.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param synapses_map map with tuples containing indexes of presynaptic and postsynaptic neurons as keys
 *  and synapse parameters as values.
 * @tparam SynapseType projection synapse type.
 * @return projection.
 */
template <typename SynapseType, template <typename, typename, typename...> class Map>
[[nodiscard]] knp::core::Projection<SynapseType> from_map(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid,
    const Map<typename std::tuple<size_t, size_t>, typename knp::core::Projection<SynapseType>::SynapseParameters>
        &synapses_map)
{
    auto iter = synapses_map.begin();

    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid,
        [&iter](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
        {
            const auto [from_index, to_index] = iter->first;
            auto synapse = std::make_tuple(iter->second, from_index, to_index);
            std::advance(iter, 1);
            return synapse;
        },
        synapses_map.size());
}


/**
 * @brief Make connections with some probability between each presynaptic population (source) neuron 
 * to each postsynaptic population (destination) neuron.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population neuron count.
 * @param postsynaptic_pop_size postsynaptic population neuron count.
 * @param connection_probability connection probability.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> fixed_probability(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t presynaptic_pop_size,
    size_t postsynaptic_pop_size, double connection_probability,
    std::function<typename knp::core::Projection<SynapseType>::SynapseParameters(size_t, size_t)> syn_gen =
        &default_synapse_gen1<SynapseType>)
{
    if (connection_probability > 1 || connection_probability < 0)
        throw std::logic_error("Incorrect probability, set probability between 0..1.");

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0, 1);
    const auto proj_size = presynaptic_pop_size * postsynaptic_pop_size;

    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid,
        [presynaptic_pop_size, postsynaptic_pop_size, &connection_probability, &dist, &mt,
         syn_gen](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
        {
            const size_t index0 = index % presynaptic_pop_size;
            const size_t index1 = index / presynaptic_pop_size;

            if (dist(mt) < connection_probability) return std::make_tuple(syn_gen(index0, index1), index0, index1);
            return std::nullopt;
        },
        proj_size);
}


/**
 * @brief Make connections between neurons of presynaptic and postsynaptic populations 
 * based on the synapse generation function result.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population neuron count.
 * @param postsynaptic_pop_size postsynaptic population neuron count.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> index_based(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t presynaptic_pop_size,
    size_t postsynaptic_pop_size,
    std::function<typename std::optional<typename knp::core::Projection<SynapseType>::SynapseParameters>(
        size_t index0, size_t index1)>
        syn_gen)
{
    const auto proj_size = presynaptic_pop_size * postsynaptic_pop_size;

    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid,
        [presynaptic_pop_size, postsynaptic_pop_size,
         syn_gen](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
        {
            const size_t index0 = index % presynaptic_pop_size;
            const size_t index1 = index / presynaptic_pop_size;
            auto opt_result = syn_gen(index0, index1);

            if (opt_result.has_value()) return std::make_tuple(opt_result.value(), index0, index1);
            return std::nullopt;
        },
        proj_size);
}


/**
 * @brief Make connections between each presynaptic neuron and a fixed number of random postsynaptic neurons.
 * @details This connector uses MT19937 generator with uniform integer distribution.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population neuron count.
 * @param postsynaptic_pop_size postsynaptic population neuron count.
 * @param neurons_count number of postsynaptic neurons.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> fixed_number_post(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t presynaptic_pop_size,
    size_t postsynaptic_pop_size, size_t neurons_count,
    std::function<typename knp::core::Projection<SynapseType>::SynapseParameters(size_t index0, size_t index1)>
        syn_gen = default_synapse_gen1<SynapseType>)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<size_t> dist(0, postsynaptic_pop_size - 1);
    const auto proj_size = presynaptic_pop_size * neurons_count;

    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid,
        [presynaptic_pop_size, postsynaptic_pop_size, neurons_count, &mt, &dist,
         syn_gen](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
        {
            const size_t index0 = index % presynaptic_pop_size;
            const size_t index1 = dist(mt);

            return std::make_tuple(syn_gen(index0, index1), index0, index1);
        },
        proj_size);
}


/**
 * @brief Make connections between each postsynaptic neuron and a fixed number of random presynaptic neurons.
 * @details This connector uses MT19937 generator with uniform integer distribution.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population neuron count.
 * @param postsynaptic_pop_size postsynaptic population neuron count.
 * @param neurons_count number of presynaptic neurons.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> fixed_number_pre(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t presynaptic_pop_size,
    size_t postsynaptic_pop_size, size_t neurons_count,
    std::function<typename knp::core::Projection<SynapseType>::SynapseParameters(size_t index0, size_t index1)>
        syn_gen = default_synapse_gen1<SynapseType>)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<size_t> dist(0, presynaptic_pop_size - 1);
    const auto proj_size = postsynaptic_pop_size * neurons_count;

    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid,
        [presynaptic_pop_size, postsynaptic_pop_size, neurons_count, &mt, &dist,
         syn_gen](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
        {
            const size_t index0 = dist(mt);
            const size_t index1 = index % postsynaptic_pop_size;

            return std::make_tuple(syn_gen(index0, index1), index0, index1);
        },
        proj_size);
}


/**
 * @brief Generate a projection which synapses are duplicated from another projection.
 * @details Source and target projections can have different types.
 * @todo Clone synapse parameters when projection types are the same.
 * @param source_proj source projection.
 * @param presynaptic_uid optional presynaptic population UID.
 * @param postsynaptic_uid optional postsynaptic population UID.
 * @param syn_gen generator of synapse parameters.
 * @tparam DestinationSynapseType generator of target synapse parameters.
 * @tparam SourceSynapseType source projection synapse type.
 * @return projection of the `DestinationSynapseType` synapses.
 */
template <typename DestinationSynapseType, typename SourceSynapseType>
[[nodiscard]] knp::core::Projection<DestinationSynapseType> clone_projection(
    knp::core::Projection<SourceSynapseType> source_proj,
    std::function<typename knp::core::Projection<DestinationSynapseType>::SynapseParameters(size_t)> syn_gen =
        default_synapse_gen<DestinationSynapseType>,
    const std::optional<knp::core::UID> &presynaptic_uid = std::nullopt,
    const std::optional<knp::core::UID> &postsynaptic_uid = std::nullopt)
{
    return knp::core::Projection<DestinationSynapseType>(
        presynaptic_uid.value_or(source_proj.get_presynaptic()),
        postsynaptic_uid.value_or(source_proj.get_postsynaptic()),
        [&source_proj,
         syn_gen](size_t index) -> std::optional<typename knp::core::Projection<DestinationSynapseType>::Synapse>
        {
            auto const &synapse = source_proj[index];
            return std::make_tuple(
                syn_gen(index), std::get<knp::core::source_neuron_id>(synapse),
                std::get<knp::core::target_neuron_id>(synapse));
        },
        source_proj.size());
}

}  // namespace connectors

}  // namespace knp::framework::projection