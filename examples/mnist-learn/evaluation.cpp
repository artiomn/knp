//
// Created by an_vartenkov on 11.03.25.
//
#include "evaluation.h"

#include <knp/core/messaging/messaging.h>

#include <algorithm>
#include <fstream>


void Target::obtain_output_spikes(const knp::core::messaging::SpikeData &firing_neuron_indices)
{
    for (auto i : firing_neuron_indices) ++prediction_votes_[i % get_ntargets()];
    if (!((tact + 1) % state_duration_))
    {
        int starting_index = index_offset_;
        int j = starting_index;
        int n_max = 0;
        int PredictedState = -1;
        do
        {
            if (prediction_votes_[j] > n_max)
            {
                n_max = prediction_votes_[j];
                PredictedState = j;
            }
            if (++j == get_ntargets()) j = 0;
        } while (j != starting_index);
        if (++index_offset_ == get_ntargets()) index_offset_ = 0;
        predicted_states_.push_back(std::make_pair(PredictedState, n_max));
        if (n_max) max_vote_[PredictedState] = std::max(max_vote_[PredictedState], n_max);
        std::fill(prediction_votes_.begin(), prediction_votes_.end(), 0);
    }
    ++tact;
}


int Target::finalize(enum Criterion criterion, std::string strPredictionFile) const
{
    int nExistingClasses, nNoClass, ndef;
    if (none_of(max_vote_.begin(), max_vote_.end(), std::identity()))  // No predictions at all...
        return 1;
    std::vector<int> predictions;
    auto v_lims = max_vote_;
    auto v_pars = v_lims;
    auto v_parbest = v_pars;
    int ncorrbest = 0;
    size_t i;
    int curcla = 0;
    do
    {
        predictions.clear();
        for (i = 0; i < predicted_states_.size(); ++i)
            predictions.push_back(
                predicted_states_[i].first == -1 || predicted_states_[i].second < v_pars[predicted_states_[i].first]
                    ? -1
                    : predicted_states_[i].first);
        int ncorr = 0;
        for (i = 0; i < predictions.size(); ++i)
            if (predictions[i] == states_[i]) ++ncorr;
        if (ncorr > ncorrbest)
        {
            ncorrbest = ncorr;
            v_parbest = v_pars;
        }
        if (v_pars[curcla])
            --v_pars[curcla];
        else
        {
            v_pars = v_parbest;
            ++curcla;
        }
    } while (curcla < get_ntargets());

    predictions.clear();
    for (i = 0; i < predicted_states_.size(); ++i)
        predictions.push_back(
            predicted_states_[i].first == -1 || predicted_states_[i].second < v_parbest[predicted_states_[i].first]
                ? -1
                : predicted_states_[i].first);
    int nerr = 0;
    struct classres
    {
        int real = 0;
        int predicted = 0;
        int correcty_predicted = 0;
    };
    std::vector<classres> vcr_(get_ntargets());
    int num_true_negatives = 0;
    for (i = 0; i < predicted_states_.size(); ++i)
    {
        int predicted = predictions[i];
        if (states_[i] != -1) ++vcr_[states_[i]].real;
        if (predicted != -1) ++vcr_[predicted].predicted;
        if (predicted != states_[i])
            ++nerr;
        else if (predicted != -1)
            ++vcr_[predicted].correcty_predicted;
        else
            ++num_true_negatives;
    }
    if (strPredictionFile.length())
    {
        std::ofstream out_stream(strPredictionFile);
        out_stream << "TARGET,THRESHOLD,PRECISION,RECAL"
                      "L,F\n";
        for (int cla = 0; cla < get_ntargets(); ++cla)
        {
            float precision =
                vcr_[cla].predicted ? vcr_[cla].correcty_predicted / static_cast<float>(vcr_[cla].predicted) : 0.F;

            float recall = vcr_[cla].real ? vcr_[cla].correcty_predicted / static_cast<float>(vcr_[cla].real) : 0.F;
            out_stream << cla << ',' << v_parbest[cla] << ',' << precision << ',' << recall << ','
                       << (precision && recall ? 2 / (1 / precision + 1 / recall) : 0.F) << std::endl;
        }
        for (i = 0; i < predicted_states_.size(); ++i)
            out_stream << states_[i] << ',' << predicted_states_[i].first << ',' << predicted_states_[i].second << ','
                       << predictions[i] << std::endl;
    }
    double dWeightedCorrect, dWeightedF;
    switch (criterion)
    {
        case absolute_error:
            return static_cast<int>(std::lround(10000.0 * (1 - static_cast<double>(nerr) / predicted_states_.size())));
        case weighted_error:
            nExistingClasses = 0;
            dWeightedCorrect = 0.;
            nNoClass = static_cast<int>(predicted_states_.size());
            for (int cla = 0; cla < get_ntargets(); ++cla)
                if (vcr_[cla].real)
                {
                    dWeightedCorrect += vcr_[cla].correcty_predicted / static_cast<double>(vcr_[cla].real);
                    ++nExistingClasses;
                    nNoClass -= vcr_[cla].real;
                }
            if (nNoClass)
            {
                ++nExistingClasses;
                dWeightedCorrect += num_true_negatives / static_cast<double>(nNoClass);
            }
            return static_cast<int>(std::lround(10000.0 * dWeightedCorrect / nExistingClasses));
        default:
            dWeightedF = 0.;
            ndef = 0;
            for (int cla = 0; cla < get_ntargets(); ++cla)
                if (vcr_[cla].real)
                {
                    double precision = vcr_[cla].predicted
                                           ? vcr_[cla].correcty_predicted / static_cast<double>(vcr_[cla].predicted)
                                           : 0.F;
                    double recall = vcr_[cla].correcty_predicted / static_cast<double>(vcr_[cla].real);
                    double f_measure = precision && recall ? 2 / (1 / precision + 1 / recall) : 0.;
                    dWeightedF += f_measure * vcr_[cla].real;
                    ndef += vcr_[cla].real;
                }
            return static_cast<int>(std::lround(10000.0 * dWeightedF / ndef));
    }
}
