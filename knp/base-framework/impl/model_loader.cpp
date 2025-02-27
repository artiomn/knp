/**
 * @file model_loader.cpp
 * @brief Model loader class implementation.
 * @kaspersky_support Artiom N.
 * @date 14.06.2024
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

#include <knp/framework/model_loader.h>

#include <spdlog/spdlog.h>

#include <exception>


namespace knp::framework
{

template <typename GenType>
void ModelLoader::init_channels(
    knp::framework::Model &model, const std::unordered_multimap<core::UID, core::UID, core::uid_hash> &channels,
    GenType channel_gen)
{
    for (size_t i = 0; i < channels.bucket_count(); ++i)
    {
        auto bucket_iter = channels.begin(i);
        if (channels.end(i) == bucket_iter) continue;
        auto channel_uid = bucket_iter->first;

        std::vector<core::UID> p_uids;
        p_uids.reserve(channels.bucket_size(i));

        std::transform(
            bucket_iter, channels.end(i), std::back_inserter(p_uids),
#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wunused-lambda-capture"
#endif
                        [&channel_uid](const auto &bucket)
                        {
                            SPDLOG_TRACE(
                                "Inserting channel {} peer bucket UID = {}...", std::string(channel_uid),
                                std::string(bucket.second));
                            return bucket.second;
                        });
#ifdef __clang__
#    pragma clang diagnostic pop
#endif
        (this->*channel_gen)(model, channel_uid, p_uids);
    }
}


void ModelLoader::gen_input_channel(
    knp::framework::Model &model, const core::UID &channel_uid, const std::vector<core::UID> &p_uids)
{
    try
    {
        in_channels_.emplace_back(channel_uid, backend_->get_message_bus().create_endpoint(), i_map_.at(channel_uid));
    }
    catch (const std::out_of_range &)
    {
        const std::string msg = "Incorrect input channel UID = " + std::string(channel_uid) + ".";
        SPDLOG_ERROR("{}", msg);
        throw std::logic_error(msg);
    }
    auto &network = model.get_network();

    for (const auto &proj_uid : p_uids)
    {
        SPDLOG_TRACE(
            "Input projection {} subscribing to channel {}...", std::string(proj_uid), std::string(channel_uid));
        backend_->get_message_endpoint().subscribe<knp::core::messaging::SpikeMessage>(proj_uid, {channel_uid});

        std::visit(
            [](auto &proj)
            {
                SPDLOG_TRACE("Tagging input projection {}...", std::string(proj.get_uid()));

                proj.get_tags()[core::tags::io_type_tag] = core::tags::IOType::input;
            },
            network.get_projection(proj_uid));
    }
}


void ModelLoader::gen_output_channel(
    knp::framework::Model &model, const core::UID &channel_uid, const std::vector<core::UID> &p_uids)
{
    auto endpoint = backend_->get_message_bus().create_endpoint();
    endpoint.subscribe<knp::core::messaging::SpikeMessage>(channel_uid, p_uids);
    out_channels_.emplace_back(channel_uid, std::move(endpoint));

    auto &network = model.get_network();

    for (const auto &pop_uid : p_uids)
    {
        std::visit(
            [](auto &pop)
            {
                SPDLOG_TRACE("Tagging output population {}...", std::string(pop.get_uid()));

                pop.get_tags()[core::tags::io_type_tag] = core::tags::IOType::output;
            },
            network.get_population(pop_uid));
    }
}


void ModelLoader::load(knp::framework::Model &model)
{
    SPDLOG_DEBUG("Model loader initializing...");

    // Create input.
    SPDLOG_TRACE("Input channels initializing...");
    init_channels(model, model.get_input_channels(), &ModelLoader::gen_input_channel);

    // Create output.
    SPDLOG_TRACE("Output channels initializing...");
    init_channels(model, model.get_output_channels(), &ModelLoader::gen_output_channel);

    const auto &network = model.get_network();

    // Must be used after channel initialization, because the `init_channels()` function adds tags to
    // the input projections and output populations.
    backend_->load_all_populations(network.get_populations());
    backend_->load_all_projections(network.get_projections());
}


const io::input::InputChannel &ModelLoader::get_input_channel(const core::UID &channel_uid) const
{
    auto result = std::find_if(
        in_channels_.cbegin(), in_channels_.cend(),
        [&channel_uid](const auto &input_channel) { return input_channel.get_uid() == channel_uid; });
    if (in_channels_.cend() == result) throw std::runtime_error("Wrong input channel UID.");
    return *result;
}


io::input::InputChannel &ModelLoader::get_input_channel(const core::UID &channel_uid)
{
    auto result = std::find_if(
        in_channels_.begin(), in_channels_.end(),
        [&channel_uid](const auto &input_channel) { return input_channel.get_uid() == channel_uid; });
    if (in_channels_.end() == result) throw std::runtime_error("Wrong input channel UID.");
    return *result;
}


io::output::OutputChannel &ModelLoader::get_output_channel(const core::UID &channel_uid)
{
    auto result = std::find_if(
        out_channels_.begin(), out_channels_.end(),
        [&channel_uid](const auto &output_channel) { return output_channel.get_uid() == channel_uid; });
    if (out_channels_.end() == result) throw std::runtime_error("Wrong output channel UID.");
    return *result;
}


const io::output::OutputChannel &ModelLoader::get_output_channel(const core::UID &channel_uid) const
{
    auto result = std::find_if(
        out_channels_.cbegin(), out_channels_.cend(),
        [&channel_uid](const auto &output_channel) { return output_channel.get_uid() == channel_uid; });
    if (out_channels_.cend() == result) throw std::runtime_error("Wrong output channel UID.");
    return *result;
}

}  // namespace knp::framework
