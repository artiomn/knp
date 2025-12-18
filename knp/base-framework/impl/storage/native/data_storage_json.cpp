/**
 * @file data_storage_json.cpp
 * @brief Saving and loading data from JSON format.
 * @kaspersky_support An. Vartenkov
 * @date 16.04.2024
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

#include <knp/framework/io/storage/native/data_storage_json.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <istream>
#include <vector>

#include <boost/format.hpp>

#include "data_storage_common.h"


namespace knp::framework::io::storage::native
{
namespace fs = std::filesystem;

constexpr char header_string[] =
    R"--("attributes": [
{
    "name": "magic",
            "type": {
        "class": "Integer (unsigned)",
                "size": 32,
                "endianness": "little-endian"
    },
    "value": 2682
},
{
    "name": "version",
            "shape": [2],
    "type": {
        "class": "Integer (unsigned)",
        "size": 32,
        "endianness": "little-endian"
    },
    "value": [0, 1]
}
])--";


constexpr char spike_attributes[] =
    R"--("attributes": [
  {
    "name": "sorting",
    "type": {
      "class": "Enumeration",
      "mapping": {
        "by_id": 1,
        "by_time": 2,
        "none": 0
      }
    },
    "value": "by_time"
  }
])--";


constexpr char node_structure[] =
    R"--("node_ids": {
  "shape": [%d],
  "type": {
    "class": "Integer (unsigned)",
    "size": 64,
    "endianness": "little-endian"
  },
  "value": [%s]
})--";


constexpr char timestamp_structure[] =
    R"--("timestamps": {
  "attributes": [
    {
      "name": "units",
      "type": {
        "class": "String",
        "charSet": "ASCII"
      },
      "value": "step"
    }
  ],
  "shape": [%d],
  "type": {
    "class": "Float",
    "endianness": "little-endian"
  },
  "value": [%s]
})--";


constexpr char whole_file_string[] =
    R"--(
{
    %s,
    "spikes" :
        {
            %s,
            %s,
            %s
        }
    }
)--";


// clang-format off // No const operator[] for simdjson document, so no const parameter.
bool is_json_has_magic(const rapidjson::Document &doc)  // cppcheck-suppress constParameterReference
// clang-format on
{
    const std::string magic = "magic";
    if (!doc.HasMember("attributes")) return false;
    const auto &attributes = doc["attributes"].GetArray();

    bool has_magic = false;
    for (auto group_iter = attributes.Begin(); group_iter != attributes.End(); ++group_iter)
    {
        const auto &group = *group_iter;
        if (group.HasMember("name") && group["name"].GetString() == magic)
        {
            if (group.HasMember("value") && group["value"].GetInt() != MAGIC_NUMBER) break;
            has_magic = true;
            break;
        }
    }
    return has_magic;
}


// clang-format off // No const operator[] for simdjson document, so no const parameter.
bool is_correct_version(const rapidjson::Document &doc)  // cppcheck-suppress constParameterReference
// clang-format on
{
    const std::string version_str = "version";

    if (!doc.HasMember("attributes")) return false;
    const auto &attributes = doc["attributes"].GetArray();

    for (auto group_iter = attributes.Begin(); group_iter != attributes.End(); ++group_iter)
    {
        const auto &group = *group_iter;

        if (!group.HasMember("name") || group["name"].GetString() != version_str) continue;
        if (!group.HasMember("value") || !group["value"].IsArray()) return false;

        const auto &version_arr = group["value"].GetArray();
        if (VERSION.size() != version_arr.Size()) return false;

        std::vector<int64_t> version;
        version.reserve(VERSION.size());

        for (auto val_iter = version_arr.Begin(); val_iter != version_arr.End(); ++val_iter)
        {
            // std::transform isn't compiled by MSVC.
            version.push_back(val_iter->GetInt());  // cppcheck-suppress useStlAlgorithm
        }

        return std::equal(version.begin(), version.end(), VERSION.begin());
    }
    return false;
}


auto read_nodes(const rapidjson::Document::Object &spikes_group)
{
    // Reading node IDs.
    if (!spikes_group.HasMember("node_ids") || !spikes_group["node_ids"].IsObject())
        throw std::runtime_error("No \"node_ids\" array in \"spikes\" group.");
    const auto &nodes_ids = spikes_group["node_ids"];

    if (!nodes_ids.HasMember("value") || !nodes_ids["value"].IsArray())
        throw std::runtime_error("Missing node data in JSON data file.");
    const auto &nodes_array = nodes_ids["value"].GetArray();

    std::vector<int64_t> nodes;
    nodes.reserve(nodes_array.Size());

    // No const reference val possible.
    for (auto val_iter = nodes_array.Begin(); val_iter != nodes_array.End(); ++val_iter)
    {
        // std::transform isn't compiled by MSVC.
        nodes.push_back(val_iter->GetInt());  // cppcheck-suppress useStlAlgorithm
    }

    return nodes;
}


auto read_timestamps(const rapidjson::Document::Object &spikes_group)
{
    // Reading timestamps.
    if (!spikes_group.HasMember("timestamps") || !spikes_group["timestamps"].IsObject())
        throw std::runtime_error("No \"timestamps\" array in \"spikes\" group.");
    const auto &timestamps_group = spikes_group["timestamps"];

    if (!timestamps_group.HasMember("value") || !timestamps_group["value"].IsArray())
        throw std::runtime_error("No \"value\" array in \"timestamps\" group.");
    const auto &timestamps_array = timestamps_group["value"].GetArray();

    std::vector<float> timestamps;
    timestamps.reserve(timestamps_array.Size());
    // No const reference val possible.
    for (auto val_iter = timestamps_array.Begin(); val_iter != timestamps_array.End(); ++val_iter)
    {
        // std::transform isn't compiled by MSVC.
        timestamps.push_back(static_cast<float>(val_iter->GetDouble()));  // cppcheck-suppress useStlAlgorithm
    }

    return timestamps;
}


KNP_DECLSPEC std::vector<core::messaging::SpikeMessage> load_messages_from_json(
    std::istream &input_stream, const knp::core::UID &uid, bool strict_format)
{
    rapidjson::Document doc;
    rapidjson::IStreamWrapper isw{input_stream};
    if (doc.ParseStream(isw).HasParseError()) throw std::runtime_error("Cannot parse stream.");

    if (!is_json_has_magic(doc))
    {
        if (strict_format)
            throw std::runtime_error("Unable to find magic number: wrong file format or version.");
        else
            SPDLOG_WARN("Unable to find magic number: wrong file format or version.");
    }
    if (!is_correct_version(doc)) SPDLOG_WARN("Unable to verify file version.");

    if (!doc.HasMember("spikes") || !doc["spikes"].IsObject())
        throw std::runtime_error("Unable to find \"spikes\" group in data file.");
    const auto &spikes_group = doc["spikes"].GetObj();

    auto nodes = read_nodes(spikes_group);
    auto timestamps = read_timestamps(spikes_group);

    return convert_node_time_arrays_to_messages(nodes, timestamps, uid, 1);
}


KNP_DECLSPEC std::vector<core::messaging::SpikeMessage> load_messages_from_json(
    const fs::path &path_to_json, const knp::core::UID &uid, bool strict_format)
{
    std::ifstream json_stream(path_to_json, std::ios::in);

    return load_messages_from_json(json_stream, uid, strict_format);
}


KNP_DECLSPEC void save_messages_to_json(
    const std::vector<core::messaging::SpikeMessage> &messages, const std::filesystem::path &path_to_save)
{
    boost::format format_nodes(node_structure);
    boost::format format_times(timestamp_structure);
    boost::format format_file(whole_file_string);
    std::ostringstream node_stream;
    std::ostringstream time_stream;
    std::vector<core::messaging::SpikeMessage> sorted_messages(messages.size());

    partial_sort_copy(
        messages.begin(), messages.end(), sorted_messages.begin(), sorted_messages.end(),
        [](const auto &msg1, const auto &msg2) { return msg1.header_.send_time_ < msg2.header_.send_time_; });

    size_t count = 0;
    for (const auto &msg : sorted_messages)
    {
        for (auto index : msg.neuron_indexes_)
        {
            node_stream << index << ", ";
            time_stream << msg.header_.send_time_ << ", ";
            ++count;
        }
    }
    std::string node_string = node_stream.str();
    std::string time_string = time_stream.str();
    if (node_string.size() > 2)
    {
        // Removing last comma.
        node_string.resize(node_string.size() - 2);
        time_string.resize(time_string.size() - 2);
    }
    std::string nodes_res = (format_nodes % count % node_string).str();
    std::string times_res = (format_times % count % time_string).str();
    std::string file_string = (format_file % header_string % spike_attributes % nodes_res % times_res).str();
    std::ofstream out_file(path_to_save, std::ofstream::out);
    out_file << file_string;
}
}  // namespace knp::framework::io::storage::native
