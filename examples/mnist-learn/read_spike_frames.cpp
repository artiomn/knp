//
// Created by Mike on 10/15/2024.
//

#include <algorithm>
#include <fstream>

#include "mnist-learn.h"

/**
 * @brief Read buffers from binary data file.
 * @param path_to_data path to binary data file.
 */
std::vector<std::vector<unsigned char>> read_images_from_file(const std::string &path_to_data)
{
    std::ifstream file_stream(path_to_data, std::ios::binary);
    std::vector<unsigned char> buffer;
    std::vector<std::vector<unsigned char>> result;
    while (file_stream.good())
    {
        buffer.resize(input_size);
        file_stream.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(input_size));
        result.push_back(std::move(buffer));
        buffer.clear();
    }
    return result;
}

std::vector<std::vector<bool>> image_to_spikes(std::vector<unsigned char> &buf, std::vector<double> &state)
{
    std::vector<std::vector<bool>> ret;
    int i;
    for (i = 0; i < intensity_levels; ++i)
    {
        std::vector<bool> spikes(input_size, false);
        for (int l = 0; l < input_size; ++l)
        {
            state[l] += state_increment_factor * buf[l];
            if (state[l] >= 1.)
            {
                spikes[l] = true;
                --state[l];
            }
        }
        ret.push_back(spikes);
    }
    for (; i < frames_per_image; ++i) ret.push_back(std::vector<bool>(input_size, false));
    return ret;
}
// Read image dataset from a binary file and trasnform it into a vector of boolean frames.
std::vector<std::vector<bool>> read_spike_frames(const std::string &path_to_data)
{
    auto images = read_images_from_file(path_to_data);
    std::vector<std::vector<bool>> result;
    std::vector<double> state(input_size, 0.);
    for (size_t img_num = 0; img_num < images.size(); ++img_num)
    {
        std::vector<std::vector<bool>> spikes_per_image = image_to_spikes(images[img_num], state);
        std::transform(
            spikes_per_image.begin(), spikes_per_image.end(), std::back_inserter(result),
            [](auto &v) { return std::move(v); });
    }
    return result;
}
