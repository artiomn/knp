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
    struct Result
    {
        int real = 0;
        int predicted = 0;
        int correcty_predicted = 0;
    };

    [[nodiscard]] std::vector<int> calculate_winning_predictions() const;
    void write_predictions_to_file(
        const std::filesystem::path &out_file_path, const std::vector<Result> &prediction_results,
        const std::vector<int> &winning_predictions, const std::vector<int> &predictions) const;

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
};


void Target::obtain_output_spikes(const knp::core::messaging::SpikeData &firing_neuron_indices)
{
    for (auto i : firing_neuron_indices) ++prediction_votes_[i % get_num_targets()];
    if (!((tact + 1) % state_duration_))
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
    ++tact;
}


std::vector<int> Target::calculate_winning_predictions() const
{
    auto parameters = max_vote_;
    auto best_parameters = parameters;
    int num_correct_max = 0;
    int current_class = 0;
    do
    {
        // Fill predictions from states
        std::vector<int> predictions;
        predictions.reserve(predicted_states_.size());
        for (size_t i = 0; i < predicted_states_.size(); ++i)
        {
            int predicted_value = -1;
            if (predicted_states_[i].first != -1 &&
                predicted_states_[i].second >= parameters[predicted_states_[i].first])
                predicted_value = predicted_states_[i].first;

            predictions.push_back(predicted_value);
        }
        // Calculate correct predictions
        int num_correct = 0;
        for (size_t i = 0; i < predictions.size(); ++i)
            if (predictions[i] == states_[i]) ++num_correct;

        if (num_correct > num_correct_max)
        {
            num_correct_max = num_correct;
            best_parameters = parameters;
        }
        if (parameters[current_class])
            --parameters[current_class];
        else
        {
            parameters = best_parameters;
            ++current_class;
        }
    } while (current_class < get_num_targets());

    return best_parameters;
}


void Target::write_predictions_to_file(
    const std::filesystem::path &out_file_path, const std::vector<Result> &prediction_results,
    const std::vector<int> &winning_predictions, const std::vector<int> &predictions) const
{
    if (out_file_path.empty()) return;
    std::ofstream out_stream(out_file_path);
    out_stream << "TARGET,THRESHOLD,PRECISION,RECAL"
                  "L,F\n";
    for (int label = 0; label < get_num_targets(); ++label)
    {
        float precision =
            prediction_results[label].predicted
                ? prediction_results[label].correcty_predicted / static_cast<float>(prediction_results[label].predicted)
                : 0.F;

        float recall = prediction_results[label].real ? prediction_results[label].correcty_predicted /
                                                            static_cast<float>(prediction_results[label].real)
                                                      : 0.F;
        out_stream << label << ',' << winning_predictions[label] << ',' << precision << ',' << recall << ','
                   << (precision && recall ? 2 / (1 / precision + 1 / recall) : 0.F) << std::endl;
    }
    for (size_t i = 0; i < predicted_states_.size(); ++i)
        out_stream << states_[i] << ',' << predicted_states_[i].first << ',' << predicted_states_[i].second << ','
                   << predictions[i] << std::endl;
}


int Target::finalize(const std::filesystem::path &out_file_path) const
{
    if (none_of(max_vote_.begin(), max_vote_.end(), std::identity()))  // No predictions at all...
        return 0;
    auto winning_predictions = calculate_winning_predictions();
    std::vector<int> predictions;
    for (size_t i = 0; i < predicted_states_.size(); ++i)
        predictions.push_back(
            predicted_states_[i].first == -1 ||
                    predicted_states_[i].second < winning_predictions[predicted_states_[i].first]
                ? -1
                : predicted_states_[i].first);
    int num_errors = 0;

    std::vector<Result> prediction_results(get_num_targets());
    int num_true_negatives = 0;
    for (size_t i = 0; i < predicted_states_.size(); ++i)
    {
        int predicted = predictions[i];
        if (states_[i] != -1) ++prediction_results[states_[i]].real;
        if (predicted != -1) ++prediction_results[predicted].predicted;
        if (predicted != states_[i])
            ++num_errors;
        else if (predicted != -1)
            ++prediction_results[predicted].correcty_predicted;
        else
            ++num_true_negatives;
    }
    write_predictions_to_file(out_file_path, prediction_results, winning_predictions, predictions);
    return static_cast<int>(std::lround(10000.0 * (1 - static_cast<double>(num_errors) / predicted_states_.size())));
}


void process_inference_results(
    const std::vector<knp::core::messaging::SpikeMessage> &spikes, const std::vector<int> &classes_for_testing,
    int testing_period)
{
    auto j = spikes.begin();
    Target target(10, classes_for_testing);
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
