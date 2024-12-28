//
// Created by an_vartenkov on 30.08.24.
//
#include <knp/core/projection.h>
#include <knp/framework/model.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/monitoring/observer.h>
#include <knp/framework/network.h>
#include <knp/framework/sonata/network_io.h>
#include <knp/synapse-traits/all_traits.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <functional>
#include <iostream>
#include <optional>

#include "construct_network.h"
#include "mnist-learn.h"

namespace fs = std::filesystem;

using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using ResourceDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;

constexpr int numSubNetworks = 1;
constexpr int nClasses = 10;
constexpr int LearningPeriod = 1200000;
constexpr int TestingPeriod = 200000;

constexpr int logging_aggregation_period = 4000;


// Create a spike message generator from an array of boolean frames.
auto make_input_generator(const std::vector<std::vector<bool>> &spike_frames, size_t offset)
{
    auto generator = [&spike_frames, offset](knp::core::Step step)
    {
        knp::core::messaging::SpikeData message;

        if (step >= spike_frames.size()) return message;

        for (size_t i = 0; i < spike_frames[step + offset].size(); ++i)
        {
            if (spike_frames[step + offset][i]) message.push_back(i);
        }
        return message;
    };

    return generator;
}


void read_classes(
    std::string classes_file, std::vector<std::vector<bool>> &spike_classes, std::vector<int> &classes_for_testing)
{
    std::ifstream file_stream(classes_file);
    int cla;
    while (file_stream.good())
    {
        std::string str;
        if (!std::getline(file_stream, str).good()) break;
        std::stringstream ss(str);
        ss >> cla;
        std::vector<bool> buffer(input_size, false);
        buffer[cla] = true;
        if (spike_classes.size() >= TestingPeriod) classes_for_testing.push_back(cla);
        for (int i = 0; i < frames_per_image; ++i) spike_classes.push_back(buffer);
    }
}


/**
 * @brief Structure that stores inference results from a single population.
 */
struct InferenceResult
{
    /**
     * @brief Response creation step.
     */
    size_t step_ = 0;

    /**
     * @brief Indexes of spiking neurons.
     */
    std::vector<int> indexes_{};
};


// Create an observer function that outputs resulting spikes to terminal.
auto make_observer_function(std::vector<InferenceResult> &result)
{
    auto observer_func = [&result](const std::vector<knp::core::messaging::SpikeMessage> &messages)
    {
        if (messages.empty() || messages[0].neuron_indexes_.empty()) return;
        InferenceResult result_buf;
        result_buf.step_ = messages[0].header_.send_time_;
        for (auto index : messages[0].neuron_indexes_)
        {
            std::cout << index << " ";
            result_buf.indexes_.push_back(index);
        }
        result.push_back(result_buf);
        std::cout << std::endl;
    };
    return observer_func;
}


auto make_log_observer_function(std::ofstream &log_stream, const std::map<knp::core::UID, std::string> &pop_names)
{
    auto observer_func = [&log_stream, pop_names](const std::vector<knp::core::messaging::SpikeMessage> &messages)
    {
        for (const auto &msg : messages)
        {
            const std::string name = pop_names.find(msg.header_.sender_uid_)->second;
            log_stream << "Step: " << msg.header_.send_time_ << "\nSender: " << name << std::endl;
            for (auto spike : msg.neuron_indexes_)
            {
                log_stream << spike << " ";
            }
            log_stream << std::endl;
        }
        log_stream << "-----------------" << std::endl;
    };
    return observer_func;
}


auto make_projection_observer_function(
    std::ofstream &weights_log, size_t frequency, knp::framework::ModelExecutor &model_executor,
    const knp::core::UID &uid)
{
    auto observer_func =
        [&weights_log, frequency, &model_executor, uid](const std::vector<knp::core::messaging::SpikeMessage> &)
    {
        if (!weights_log.good()) return;
        size_t step = model_executor.get_backend()->get_step();
        if (step % frequency != 0) return;
        weights_log << "Step: " << step << std::endl;
        const auto ranges = model_executor.get_backend()->get_network_data();
        for (auto &iter = *ranges.projection_range.first; iter != *ranges.projection_range.second; ++iter)
        {
            const knp::core::UID curr_proj_uid = std::visit([](const auto &proj) { return proj.get_uid(); }, *iter);
            if (curr_proj_uid == uid)
            {
                const knp::core::AllProjectionsVariant proj_variant = *iter;
                const auto &proj = std::get<ResourceDeltaProjection>(proj_variant);
                std::vector<std::tuple<int, int, float>> weights_by_receiver_sender;
                for (const auto &synapse_data : proj)
                {
                    float weight = std::get<0>(synapse_data).weight_;
                    size_t sender = std::get<1>(synapse_data);
                    size_t receiver = std::get<2>(synapse_data);
                    weights_by_receiver_sender.push_back({receiver, sender, weight});
                }
                std::sort(
                    weights_by_receiver_sender.begin(), weights_by_receiver_sender.end(),
                    [](const auto &v1, const auto &v2)
                    {
                        if (std::get<0>(v1) != std::get<0>(v2)) return std::get<0>(v1) < std::get<0>(v2);
                        return std::get<1>(v1) < std::get<1>(v2);
                    });
                size_t neuron = -1;
                for (const auto &syn_data : weights_by_receiver_sender)
                {
                    size_t new_neuron = std::get<0>(syn_data);
                    if (neuron != new_neuron)
                    {
                        neuron = new_neuron;
                        weights_log << std::endl << "Neuron " << neuron << std::endl;
                    }
                    weights_log << std::get<2>(syn_data) << " ";
                }
                weights_log << std::endl;
                return;
            }
        }
    };
    return observer_func;
}


void write_aggregated_log_header(std::ofstream &log_stream, const std::map<knp::core::UID, std::string> &pop_names)
{
    std::vector<std::string> vec(pop_names.size());
    std::transform(pop_names.begin(), pop_names.end(), vec.begin(), [](const auto &val) { return val.second; });
    std::sort(vec.begin(), vec.end());
    log_stream << "Index";
    for (const auto &name : vec) log_stream << ", " << name;
    log_stream << std::endl;
}


void save_aggregated_log(std::ofstream &log_stream, const std::map<std::string, size_t> &values, size_t index)
{
    // Write values in order. Map is sorted by key value, that means by population name.
    log_stream << index;
    for (const auto &name_count_pair : values)
    {
        log_stream << ", " << name_count_pair.second;
    }
    log_stream << std::endl;
}


auto make_aggregate_observer(
    std::ofstream &log_stream, int period, const std::map<knp::core::UID, std::string> &pop_names,
    std::map<std::string, size_t> &accumulator, size_t &curr_index)
{
    // Initialize accumulator
    accumulator.clear();
    for (const auto &val : pop_names) accumulator.insert({val.second, 0});
    curr_index = 0;

    auto observer_func = [&log_stream, &accumulator, pop_names, period,
                          &curr_index](const std::vector<knp::core::messaging::SpikeMessage> &messages)
    {
        if (curr_index != 0 && curr_index % period == 0)
        {
            // Write container to log
            save_aggregated_log(log_stream, accumulator, curr_index);
            // Reset container
            accumulator.clear();
            for (const auto &val : pop_names) accumulator.insert({val.second, 0});
        }

        // Add spike numbers to accumulator
        for (const auto &msg : messages)
        {
            auto name_iter = pop_names.find(msg.header_.sender_uid_);
            if (name_iter == pop_names.end()) continue;
            std::string population_name = name_iter->second;
            accumulator[population_name] += msg.neuron_indexes_.size();
        }
        ++curr_index;
    };
    return observer_func;
}


knp::framework::Network get_network_for_inference(
    const knp::core::Backend &backend, const std::set<knp::core::UID> &inference_population_uids,
    const std::set<knp::core::UID> &inference_internal_projection)
{
    auto data_ranges = backend.get_network_data();
    knp::framework::Network res_network;
    for (auto &iter = *data_ranges.population_range.first; iter != *data_ranges.population_range.second; ++iter)
    {
        auto population = *iter;
        knp::core::UID pop_uid = std::visit([](const auto &p) { return p.get_uid(); }, population);
        if (inference_population_uids.find(pop_uid) != inference_population_uids.end())
            res_network.add_population(std::move(population));
    }
    for (auto &iter = *data_ranges.projection_range.first; iter != *data_ranges.projection_range.second; ++iter)
    {
        auto projection = *iter;
        knp::core::UID proj_uid = std::visit([](const auto &p) { return p.get_uid(); }, projection);
        if (inference_internal_projection.find(proj_uid) != inference_internal_projection.end())
            res_network.add_projection(std::move(projection));
    }
    return res_network;
}


class Target
{
    struct TargetClass
    {
        std::string str;
    };
    const std::vector<int> &v_States;
    std::vector<std::pair<int, int>> PredictedStates;
    size_t tact = 0;
    const int StateDuration = 20;
    std::string strPredictionFile;
    std::vector<double> vd_PossiblePredictions;
    std::vector<int> vn_PredictionVotes;
    std::vector<int> v_maxVote;
    std::vector<TargetClass> vtc_;
    int indran = 0;

public:
    enum criterion
    {
        absolute_error,
        weighted_error,
        averaged_F
    };

    Target(int nTargetClasses, const std::vector<int> &classes)
        : vn_PredictionVotes(nTargetClasses, 0), v_States(classes), v_maxVote(nTargetClasses, 0)
    {
    }
    void ObtainOutputSpikes(const knp::core::messaging::SpikeData &firing_neuron_indices);
    [[nodiscard]] int get_ntargets() const { return static_cast<int>(vn_PredictionVotes.size()); }
    [[nodiscard]] int Finalize(
        enum criterion cri = absolute_error, std::string strPredictionFile = std::string()) const;
};


void Target::ObtainOutputSpikes(const knp::core::messaging::SpikeData &firing_neuron_indices)
{
    for (auto i : firing_neuron_indices) ++vn_PredictionVotes[i % get_ntargets()];
    if (!((tact + 1) % StateDuration))
    {
        int indstart = indran;
        int j = indstart;
        int nmax = 0;
        int PredictedState = -1;
        do
        {
            if (vn_PredictionVotes[j] > nmax)
            {
                nmax = vn_PredictionVotes[j];
                PredictedState = j;
            }
            if (++j == get_ntargets()) j = 0;
        } while (j != indstart);
        if (++indran == get_ntargets()) indran = 0;
        PredictedStates.push_back(std::make_pair(PredictedState, nmax));
        if (nmax) v_maxVote[PredictedState] = std::max(v_maxVote[PredictedState], nmax);
        std::fill(vn_PredictionVotes.begin(), vn_PredictionVotes.end(), 0);
    }
    ++tact;
}


int Target::Finalize(enum criterion cri, std::string strPredictionFile) const
{
    int nExistingClasses, nNoClass, ndef;
    double dWeightedCorrect, dWeightedF;
    if (none_of(v_maxVote.begin(), v_maxVote.end(), std::identity()))  // No predictions at all...
        return 1;
    std::vector<int> v_pred;
    auto v_lims = v_maxVote;
    auto v_pars = v_lims;
    auto v_parbest = v_pars;
    int ncorrbest = 0;
    size_t i;
    int curcla = 0;
    do
    {
        v_pred.clear();
        for (i = 0; i < PredictedStates.size(); ++i)
            v_pred.push_back(
                PredictedStates[i].first == -1 || PredictedStates[i].second < v_pars[PredictedStates[i].first]
                    ? -1
                    : PredictedStates[i].first);
        int ncorr = 0;
        for (i = 0; i < v_pred.size(); ++i)
            if (v_pred[i] == v_States[i]) ++ncorr;
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

    v_pred.clear();
    for (i = 0; i < PredictedStates.size(); ++i)
        v_pred.push_back(
            PredictedStates[i].first == -1 || PredictedStates[i].second < v_parbest[PredictedStates[i].first]
                ? -1
                : PredictedStates[i].first);
    int nerr = 0;
    struct classres
    {
        int real = 0;
        int predicted = 0;
        int correcty_predicted = 0;
    };
    std::vector<classres> vcr_(get_ntargets());
    int nTrueNegatives = 0;
    for (i = 0; i < PredictedStates.size(); ++i)
    {
        int predicted = v_pred[i];
        if (v_States[i] != -1) ++vcr_[v_States[i]].real;
        if (predicted != -1) ++vcr_[predicted].predicted;
        if (predicted != v_States[i])
            ++nerr;
        else if (predicted != -1)
            ++vcr_[predicted].correcty_predicted;
        else
            ++nTrueNegatives;
    }
    if (strPredictionFile.length())
    {
        std::ofstream ofsPredictionFile(strPredictionFile);
        ofsPredictionFile << "TARGET,THRESHOLD,PRECISION,RECALL,F\n";
        for (int cla = 0; cla < get_ntargets(); ++cla)
        {
            float rPrecision =
                vcr_[cla].predicted ? vcr_[cla].correcty_predicted / static_cast<float>(vcr_[cla].predicted) : 0.F;
            float rRecall = vcr_[cla].real ? vcr_[cla].correcty_predicted / static_cast<float>(vcr_[cla].real) : 0.F;
            ofsPredictionFile << cla << ',' << v_parbest[cla] << ',' << rPrecision << ',' << rRecall << ','
                              << (rPrecision && rRecall ? 2 / (1 / rPrecision + 1 / rRecall) : 0.F) << std::endl;
        }
        for (i = 0; i < PredictedStates.size(); ++i)
            ofsPredictionFile << v_States[i] << ',' << PredictedStates[i].first << ',' << PredictedStates[i].second
                              << ',' << v_pred[i] << std::endl;
    }
    switch (cri)
    {
        case absolute_error:
            return 10000 * (1 - static_cast<float>(nerr) / PredictedStates.size());
        case weighted_error:
            nExistingClasses = 0;
            dWeightedCorrect = 0.;
            nNoClass = static_cast<int>(PredictedStates.size());
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
                dWeightedCorrect += nTrueNegatives / static_cast<double>(nNoClass);
            }
            return 10000 * dWeightedCorrect / nExistingClasses;
        default:
            dWeightedF = 0.;
            ndef = 0;
            for (int cla = 0; cla < get_ntargets(); ++cla)
                if (vcr_[cla].real)
                {
                    double dPrecision = vcr_[cla].predicted
                                            ? vcr_[cla].correcty_predicted / static_cast<double>(vcr_[cla].predicted)
                                            : 0.F;
                    double dRecall = vcr_[cla].correcty_predicted / static_cast<double>(vcr_[cla].real);
                    double dF = dPrecision && dRecall ? 2 / (1 / dPrecision + 1 / dRecall) : 0.;
                    dWeightedF += dF * vcr_[cla].real;
                    ndef += vcr_[cla].real;
                }
            return static_cast<int>(std::lround(10000.0 * dWeightedF / ndef));
    }
}


std::string get_time_string()
{
    auto time_now = std::chrono::system_clock::now();
    std::time_t c_time = std::chrono::system_clock::to_time_t(time_now);
    std::string result(std::ctime(&c_time));
    return result;
}


std::vector<knp::core::UID> add_wta_handlers(const AnnotatedNetwork &network, knp::framework::ModelExecutor &executor)
{
    std::vector<size_t> borders;
    std::vector<knp::core::UID> result;
    for (size_t i = 0; i < 10; ++i) borders.push_back(15 * i);
    for (const auto &senders_receivers : network.data_.wta_data)
    {
        knp::core::UID handler_uid;
        executor.add_spike_message_handler(
            knp::framework::modifier::KWtaPerGroup{borders}, senders_receivers.first, senders_receivers.second,
            handler_uid);
        result.push_back(handler_uid);
    }
    return result;
}


std::vector<knp::core::UID> add_wta_handlers_inference(
    const AnnotatedNetwork &network, knp::framework::ModelExecutor &executor)
{
    std::vector<size_t> borders;
    std::vector<knp::core::UID> result;
    for (size_t i = 0; i < 10; ++i) borders.push_back(15 * i);
    for (const auto &senders_receivers : network.data_.wta_data)
    {
        knp::core::UID handler_uid;
        executor.add_spike_message_handler(
            knp::framework::modifier::GroupWtaRandomHandler{borders}, senders_receivers.first, senders_receivers.second,
            handler_uid);
        result.push_back(handler_uid);
    }
    return result;
}


AnnotatedNetwork train_mnist_network(
    const fs::path &path_to_backend, const std::vector<std::vector<bool>> &spike_frames,
    const std::vector<std::vector<bool>> &spike_classes, const fs::path &log_path = "")
{
    AnnotatedNetwork example_network = create_example_network_new(numSubNetworks);
    std::filesystem::create_directory("mnist_network");
    knp::framework::sonata::save_network(example_network.network_, "mnist_network");
    knp::framework::Model model(std::move(example_network.network_));

    knp::core::UID input_image_channel_raster;
    knp::core::UID input_image_channel_classses;

    for (auto image_proj_uid : example_network.data_.projections_from_raster)
        model.add_input_channel(input_image_channel_raster, image_proj_uid);
    for (auto target_proj_uid : example_network.data_.projections_from_classes)
        model.add_input_channel(input_image_channel_classses, target_proj_uid);

    knp::framework::ModelLoader::InputChannelMap channel_map;

    channel_map.insert({input_image_channel_raster, make_input_generator(spike_frames, 0)});
    channel_map.insert({input_image_channel_classses, make_input_generator(spike_classes, 0)});

    knp::framework::BackendLoader backend_loader;
    knp::framework::ModelExecutor model_executor(model, backend_loader.load(path_to_backend), std::move(channel_map));
    std::vector<InferenceResult> result;

    // Add observer.
    model_executor.add_observer<knp::core::messaging::SpikeMessage>(
        make_observer_function(result), example_network.data_.output_uids);

    // Add all spikes observer.
    // These variables should have the same lifetime as model_executor, or else UB.
    std::ofstream log_stream, weight_stream, all_spikes_stream;
    std::map<std::string, size_t> spike_accumulator;
    // cppcheck-suppress variableScope
    size_t current_index = 0;
    std::vector<knp::core::UID> wta_uids = add_wta_handlers(example_network, model_executor);
    // All loggers go here
    if (!log_path.empty())
    {
        std::vector<knp::core::UID> all_populations_uids;
        for (const auto &pop : model.get_network().get_populations())
        {
            knp::core::UID pop_uid = std::visit([](const auto &p) { return p.get_uid(); }, pop);
            all_populations_uids.push_back(pop_uid);
        }
        log_stream.open(log_path / "spikes_training.csv", std::ofstream::out);
        auto all_names = example_network.data_.population_names;
        for (const auto &uid : wta_uids) all_names.insert({uid, "WTA"});

        if (log_stream.is_open())
        {
            write_aggregated_log_header(log_stream, all_names);
            model_executor.add_observer<knp::core::messaging::SpikeMessage>(
                make_aggregate_observer(
                    log_stream, logging_aggregation_period, example_network.data_.population_names, spike_accumulator,
                    current_index),
                all_populations_uids);
        }
        else
            std::cout << "Couldn't open log file at " << log_path << std::endl;
        all_spikes_stream.open(log_path / "all_spikes_training.log", std::ofstream::out);
        if (all_spikes_stream.is_open())
        {
            auto all_senders = all_populations_uids;
            all_senders.insert(all_senders.end(), wta_uids.begin(), wta_uids.end());
            model_executor.add_observer<knp::core::messaging::SpikeMessage>(
                make_log_observer_function(all_spikes_stream, example_network.data_.population_names), all_senders);
        }
    }

    // Start model.
    std::cout << get_time_string() << ": learning started\n";

    model_executor.start(
        [](size_t step)
        {
            if (step % 20 == 0) std::cout << "Step: " << step << std::endl;
            return step != LearningPeriod;
        });

    std::cout << get_time_string() << ": learning finished\n";
    example_network.network_ = get_network_for_inference(
        *model_executor.get_backend(), example_network.data_.inference_population_uids,
        example_network.data_.inference_internal_projection);
    return example_network;
}


std::vector<knp::core::messaging::SpikeMessage> run_mnist_inference(
    const fs::path &path_to_backend, AnnotatedNetwork &described_network,
    const std::vector<std::vector<bool>> &spike_frames, const fs::path &log_path = "")
{
    knp::framework::BackendLoader backend_loader;
    knp::framework::Model model(std::move(described_network.network_));

    // Creates arbitrary o_channel_uid identifier for the output channel.
    knp::core::UID o_channel_uid;
    // Passes to the model object the created output channel ID (o_channel_uid)
    // and the population IDs.
    knp::framework::ModelLoader::InputChannelMap channel_map;
    knp::core::UID input_image_channel_uid;
    channel_map.insert({input_image_channel_uid, make_input_generator(spike_frames, LearningPeriod)});

    for (auto i : described_network.data_.output_uids) model.add_output_channel(o_channel_uid, i);
    for (auto image_proj_uid : described_network.data_.projections_from_raster)
        model.add_input_channel(input_image_channel_uid, image_proj_uid);
    knp::framework::ModelExecutor model_executor(model, backend_loader.load(path_to_backend), std::move(channel_map));

    // Receives a link to the output channel object (out_channel) from
    // the model executor (model_executor) by the output channel ID (o_channel_uid).
    auto &out_channel = model_executor.get_loader().get_output_channel(o_channel_uid);

    model_executor.get_backend()->stop_learning();
    std::vector<InferenceResult> result;

    // Add observer.
    model_executor.add_observer<knp::core::messaging::SpikeMessage>(
        make_observer_function(result), described_network.data_.output_uids);
    std::ofstream log_stream;
    // These two variables should have the same lifetime as model_executor, or else UB.
    std::map<std::string, size_t> spike_accumulator;
    // cppcheck-suppress variableScope
    size_t current_index = 0;
    auto wta_uids = add_wta_handlers_inference(described_network, model_executor);
    auto all_senders_names = described_network.data_.population_names;
    for (const auto &uid : wta_uids)
    {
        all_senders_names.insert({uid, "WTA"});
    }

    // All loggers go here
    if (!log_path.empty())
    {
        log_stream.open(log_path / "spikes_inference.csv", std::ofstream::out);
        if (!log_stream.is_open()) std::cout << "Couldn't open log file : " << log_path << std::endl;
    }
    if (log_stream.is_open())
    {
        std::vector<knp::core::UID> all_senders_uids;
        for (const auto &pop : model.get_network().get_populations())
        {
            knp::core::UID pop_uid = std::visit([](const auto &p) { return p.get_uid(); }, pop);
            all_senders_uids.push_back(pop_uid);
        }
        write_aggregated_log_header(log_stream, all_senders_names);
        model_executor.add_observer<knp::core::messaging::SpikeMessage>(
            make_aggregate_observer(
                log_stream, logging_aggregation_period, described_network.data_.population_names, spike_accumulator,
                current_index),
            all_senders_uids);
    }
    // Add all spikes observer
    std::vector<knp::core::UID> all_populations_uids;
    for (const auto &pop : model.get_network().get_populations())
    {
        knp::core::UID pop_uid = std::visit([](const auto &p) { return p.get_uid(); }, pop);
        all_populations_uids.push_back(pop_uid);
    }

    std::ofstream all_spikes_stream;
    all_spikes_stream.open(log_path / "all_spikes_inference.log", std::ofstream::out);
    if (all_spikes_stream.is_open())
    {
        auto all_senders = all_populations_uids;
        all_senders.insert(all_senders.end(), wta_uids.begin(), wta_uids.end());
        model_executor.add_observer<knp::core::messaging::SpikeMessage>(
            make_log_observer_function(all_spikes_stream, all_senders_names), all_senders);
    }

    // Start model.
    std::cout << get_time_string() << ": inference started\n";
    model_executor.start(
        [&spike_frames](size_t step)
        {
            if (step % 20 == 0) std::cout << "Inference step: " << step << std::endl;
            return step != TestingPeriod;
        });
    // Creates the results vector that contains the indices of the spike steps.
    std::vector<knp::core::Step> results;
    // Updates the output channel.
    auto spikes = out_channel.update();
    std::sort(
        spikes.begin(), spikes.end(),
        [](const auto &sm1, const auto &sm2) { return sm1.header_.send_time_ < sm2.header_.send_time_; });
    return spikes;
}


void process_inference_results(
    const std::vector<knp::core::messaging::SpikeMessage> &spikes, const std::vector<int> &classes_for_testing)
{
    auto j = spikes.begin();
    Target tar(10, classes_for_testing);
    for (int tact = 0; tact < TestingPeriod; ++tact)
    {
        knp::core::messaging::SpikeData firing_neuron_indices;
        while (j != spikes.end() && j->header_.send_time_ == tact)
        {
            firing_neuron_indices.insert(
                firing_neuron_indices.end(), j->neuron_indexes_.begin(), j->neuron_indexes_.end());
            ++j;
        }
        tar.ObtainOutputSpikes(firing_neuron_indices);
    }
    auto res = tar.Finalize(Target::absolute_error, "mnist.log");
    std::cout << "ACCURACY: " << res / 100.F << "%\n";
}


int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Not enough parameters to run script: paths to both frames and targets are required" << std::endl;
        return EXIT_FAILURE;
    }
    fs::path log_path;
    if (argc >= 4) log_path = argv[3];

    // Defines path to backend, on which to run a network.
    std::filesystem::path path_to_backend =
        std::filesystem::path(argv[0]).parent_path() / "knp-cpu-single-threaded-backend";
    auto spike_frames = read_spike_frames(argv[1]);
    std::vector<std::vector<bool>> spike_classes;
    std::vector<int> classes_for_testing;
    read_classes(argv[2], spike_classes, classes_for_testing);
    AnnotatedNetwork trained_network = train_mnist_network(path_to_backend, spike_frames, spike_classes, log_path);

    auto spikes = run_mnist_inference(path_to_backend, trained_network, spike_frames, log_path);
    std::cout << get_time_string() << ": inference finished  -- output spike count is " << spikes.size() << std::endl;

    process_inference_results(spikes, classes_for_testing);
    return EXIT_SUCCESS;
}
