//
// Created by vartenkov on 01.04.25.
//

#pragma once
#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/population.h>

#include <optional>

#include "impl/altai_lif_population_impl.h"
#include "impl/lif_population_impl.h"


/**
 * @brief Namespace for CPU backends.
 */
namespace knp::backends::cpu
{
template <class LifNeuron>
std::optional<knp::core::messaging::SpikeMessage> calculate_lif_population(
    knp::core::Population<LifNeuron> &population, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    return calculate_lif_population_impl(population, endpoint, step_n);
}
}  // namespace knp::backends::cpu
