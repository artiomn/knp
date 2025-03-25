/**
 * @file evaluation.cpp
 * @brief Functions for network quality estimation.
 * @kaspersky_support A. Vartenkov
 * @date 30.08.2024
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
    enum class Criterion
    {
        absolute_error,
        weighted_error,
        averaged_f
    };

    Target(int num_target_classes, const std::vector<int> &classes)
        : prediction_votes_(num_target_classes, 0), states_(classes), max_vote_(num_target_classes, 0)
    {
    }

    void obtain_output_spikes(const knp::core::messaging::SpikeData &firing_neuron_indices);

    [[nodiscard]] int get_num_targets() const { return static_cast<int>(prediction_votes_.size()); }

    [[nodiscard]] int finalize(
        enum Criterion criterion = Criterion::absolute_error,
        const std::filesystem::path &strPredictionFile = "") const;

private:
    struct Result
    {
        int real = 0;
        int predicted = 0;
        int correcty_predicted = 0;
    };

    int finalize_absolute_err(const std::vector<Result> &prediction_results) const;
    int finalize_weighted_err(const std::vector<Result> &prediction_results) const;
    int finalize_averaged_f(const std::vector<Result> &prediction_results) const;

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


int Target::finalize(enum Criterion criterion, const std::filesystem::path &out_file_path) const
{
    int num_existing_classes, num_no_class;
    if (none_of(max_vote_.begin(), max_vote_.end(), std::identity()))  // No predictions at all...
        return 0;
    std::vector<int> predictions;
    auto parameters = max_vote_;
    auto best_parameters = parameters;
    int num_correct_max = 0;
    int current_class = 0;
    do
    {
        predictions.clear();
        for (size_t i = 0; i < predicted_states_.size(); ++i)
            predictions.push_back(
                predicted_states_[i].first == -1 || predicted_states_[i].second < parameters[predicted_states_[i].first]
                    ? -1
                    : predicted_states_[i].first);
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

    predictions.clear();
    for (size_t i = 0; i < predicted_states_.size(); ++i)
        predictions.push_back(
            predicted_states_[i].first == -1 ||
                    predicted_states_[i].second < best_parameters[predicted_states_[i].first]
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
    if (!out_file_path.empty())
    {
        std::ofstream out_stream(out_file_path);
        out_stream << "TARGET,THRESHOLD,PRECISION,RECAL"
                      "L,F\n";
        for (int label = 0; label < get_num_targets(); ++label)
        {
            float precision =
                prediction_results[label].predicted
                    ? target[label].correcty_predicted / static_cast<float>(prediction_results[label].predicted)
                    : 0.F;

            float recall = prediction_results[label].real ? prediction_results[label].correcty_predicted /
                                                                static_cast<float>(prediction_results[label].real)
                                                          : 0.F;
            out_stream << label << ',' << best_parameters[label] << ',' << precision << ',' << recall << ','
                       << (precision && recall ? 2 / (1 / precision + 1 / recall) : 0.F) << std::endl;
        }
        for (size_t i = 0; i < predicted_states_.size(); ++i)
            out_stream << states_[i] << ',' << predicted_states_[i].first << ',' << predicted_states_[i].second << ','
                       << predictions[i] << std::endl;
    }
    double num_correct_weighted;
    switch (criterion)
    {
        case Criterion::absolute_error:
            return static_cast<int>(
                std::lround(10000.0 * (1 - static_cast<double>(num_errors) / predicted_states_.size())));

        case Criterion::weighted_error:
            num_existing_classes = 0;
            num_correct_weighted = 0.;
            num_no_class = static_cast<int>(predicted_states_.size());
            for (int cla = 0; cla < get_num_targets(); ++cla)
                if (prediction_results[cla].real)
                {
                    num_correct_weighted +=
                        prediction_results[cla].correcty_predicted / static_cast<double>(prediction_results[cla].real);
                    ++num_existing_classes;
                    num_no_class -= prediction_results[cla].real;
                }
            if (num_no_class)
            {
                ++num_existing_classes;
                num_correct_weighted += num_true_negatives / static_cast<double>(num_no_class);
            }
            return static_cast<int>(std::lround(10000.0 * num_correct_weighted / num_existing_classes));

        default:
            return finalize_averaged_f(prediction_results);
    }
}


int Target::finalize_averaged_f(const std::vector<Result> &prediction_results) const
{
    double weighted_f_measure = 0.;
    int ndef = 0;
    for (int target_index = 0; target_index < get_num_targets(); ++target_index)
    {
        if (prediction_results[target_index].real)
        {
            double precision = prediction_results[target_index].predicted
                                   ? prediction_results[target_index].correcty_predicted /
                                         static_cast<double>(prediction_results[target_index].predicted)
                                   : 0.F;
            double recall = prediction_results[target_index].correcty_predicted /
                            static_cast<double>(prediction_results[target_index].real);
            double f_measure = precision && recall ? 2 / (1 / precision + 1 / recall) : 0.;
            weighted_f_measure += f_measure * prediction_results[target_index].real;
            ndef += prediction_results[target_index].real;
        }
    }
    return static_cast<int>(std::lround(10000.0 * weighted_f_measure / ndef));
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
    auto res = target.finalize(Target::Criterion::absolute_error, "mnist.log");
    std::cout << "ACCURACY: " << res / 100.F << "%\n";
}
