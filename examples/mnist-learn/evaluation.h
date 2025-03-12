//
// Created by an_vartenkov on 11.03.25.
//
#pragma once
#include <knp/core/messaging/messaging.h>

#include <string>
#include <utility>
#include <vector>


/**
 * @brief A class used for accuracy evaluation.
 */
class Target
{
    struct TargetClass
    {
        std::string str;
    };

    const std::vector<int> &states_;

    std::vector<std::pair<int, int>> predicted_states_;
    size_t tact = 0;
    const int state_duration_ = 20;
    std::string prediction_file_;
    std::vector<double> possible_predictions_;
    std::vector<int> prediction_votes_;
    std::vector<int> max_vote_;
    std::vector<TargetClass> vtc_;
    int index_offset_ = 0;

public:
    enum Criterion
    {
        absolute_error,
        weighted_error,
        averaged_f
    };

    Target(int nTargetClasses, const std::vector<int> &classes)
        : prediction_votes_(nTargetClasses, 0), states_(classes), max_vote_(nTargetClasses, 0)
    {
    }

    void obtain_output_spikes(const knp::core::messaging::SpikeData &firing_neuron_indices);

    [[nodiscard]] int get_ntargets() const { return static_cast<int>(prediction_votes_.size()); }

    [[nodiscard]] int finalize(
        enum Criterion criterion = absolute_error, std::string strPredictionFile = std::string()) const;
};
