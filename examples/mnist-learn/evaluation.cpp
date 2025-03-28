/**
 * @file evaluation.cpp
 * @brief Functions for network quality estimation.
 * @kaspersky_support A. Vartenkov
 * @date 12.03.2025
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
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
#include "evaluation.h"

#include <knp/core/messaging/messaging.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>


constexpr int num_possible_labels = 10;

/**
 * @brief Prediction result structure.
 */
struct Result
{
    int real_ = 0;
    int predicted_ = 0;
    int correcty_predicted_ = 0;
};


/**
 * @brief A class used for accuracy evaluation.
 */
class Target
{
public:
    Target(int num_target_classes, const std::vector<int> &classes)
        : prediction_votes_(num_target_classes, 0), states_(classes), max_vote_(num_target_classes, 0)
    {
    }

    void obtain_output_spikes(const knp::core::messaging::SpikeData &firing_neuron_indices);

    [[nodiscard]] int get_num_targets() const { return static_cast<int>(prediction_votes_.size()); }

    [[nodiscard]] int finalize(const std::filesystem::path &strPredictionFile = "") const;

private:
    void write_predictions_to_file(
        const std::filesystem::path &out_file_path, const std::vector<Result> &prediction_results,
        const std::vector<int> &predictions) const;

    const std::vector<int> &states_;
    std::vector<std::pair<int, int>> predicted_states_;
    size_t tact_ = 0;
    const int state_duration_ = 20;
    std::vector<int> prediction_votes_;
    std::vector<int> max_vote_;
    int index_offset_ = 0;
};


void Target::obtain_output_spikes(const knp::core::messaging::SpikeData &firing_neuron_indices)
{
    for (auto i : firing_neuron_indices) ++prediction_votes_[i % get_num_targets()];
    if (!((tact_ + 1) % state_duration_))
    {
        int starting_index = index_offset_;
        int j = starting_index;
        int n_max = 0;
        int predicted_state = -1;
        do
        {
            if (prediction_votes_[j] > n_max)
            {
                n_max = prediction_votes_[j];
                predicted_state = j;
            }
            if (++j == get_num_targets()) j = 0;
        } while (j != starting_index);
        if (++index_offset_ == get_num_targets()) index_offset_ = 0;
        predicted_states_.push_back(std::make_pair(predicted_state, n_max));
        if (n_max) max_vote_[predicted_state] = std::max(max_vote_[predicted_state], n_max);
        std::fill(prediction_votes_.begin(), prediction_votes_.end(), 0);
    }
    ++tact_;
}


double get_precision(const Result &prediction_result)
{
    if (prediction_result.predicted_ == 0) return 0.F;
    return static_cast<double>(prediction_result.correcty_predicted_) / prediction_result.predicted_;
}


double get_recall(const Result &prediction_result)
{
    if (prediction_result.real_ == 0) return 0.0F;
    return static_cast<double>(prediction_result.correcty_predicted_) / prediction_result.real_;
}


double get_f_measure(double precision, double recall)
{
    if (precision * recall == 0) return 0.0F;
    return 2.0F * precision * recall / (precision + recall);
}


void Target::write_predictions_to_file(
    const std::filesystem::path &out_file_path, const std::vector<Result> &prediction_results,
    const std::vector<int> &predictions) const
{
    if (out_file_path.empty()) return;
    std::ofstream out_stream(out_file_path);
    out_stream << "TARGET,PRECISION,RECALL,F\n";
    for (int label = 0; label < get_num_targets(); ++label)
    {
        double precision = get_precision(prediction_results[label]);
        double recall = get_recall(prediction_results[label]);
        double f_measure = get_f_measure(precision, recall);
        out_stream << label << ',' << precision << ',' << recall << ',' << f_measure << std::endl;
    }
    for (size_t i = 0; i < predicted_states_.size(); ++i)
        out_stream << states_[i] << ',' << predicted_states_[i].first << ',' << predicted_states_[i].second << ','
                   << predictions[i] << std::endl;
}


int Target::finalize(const std::filesystem::path &out_file_path) const
{
    if (none_of(max_vote_.begin(), max_vote_.end(), std::identity()))  // No predictions at all...
        return 0;
    std::vector<int> predictions;
    for (size_t i = 0; i < predicted_states_.size(); ++i)
        predictions.push_back(
            predicted_states_[i].first == -1 || predicted_states_[i].second < 1 ? -1 : predicted_states_[i].first);
    int num_errors = 0;

    std::vector<Result> prediction_results(get_num_targets());
    int num_true_negatives = 0;
    for (size_t i = 0; i < predicted_states_.size(); ++i)
    {
        int predicted = predictions[i];
        if (states_[i] != -1) ++prediction_results[states_[i]].real_;
        if (predicted != -1) ++prediction_results[predicted].predicted_;
        if (predicted != states_[i])
            ++num_errors;
        else if (predicted != -1)
            ++prediction_results[predicted].correcty_predicted_;
        else
            ++num_true_negatives;
    }
    write_predictions_to_file(out_file_path, prediction_results, predictions);
    return static_cast<int>(std::lround(10000.0 * (1 - static_cast<double>(num_errors) / predicted_states_.size())));
}


void process_inference_results(
    const std::vector<knp::core::messaging::SpikeMessage> &spikes, const std::vector<int> &classes_for_testing,
    int testing_period)
{
    auto j = spikes.begin();
    Target target(num_possible_labels, classes_for_testing);
    for (int tact = 0; tact < testing_period; ++tact)
    {
        knp::core::messaging::SpikeData firing_neuron_indices;
        while (j != spikes.end() && j->header_.send_time_ == tact)
        {
            firing_neuron_indices.insert(
                firing_neuron_indices.end(), j->neuron_indexes_.begin(), j->neuron_indexes_.end());
            ++j;
        }
        target.obtain_output_spikes(firing_neuron_indices);
    }
    auto res = target.finalize("mnist.log");
    std::cout << "ACCURACY: " << res / 100.F << "%\n";
}
