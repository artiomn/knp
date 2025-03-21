/**
 * @author M. V. Kiselev
 * @date {DAY}.{MONTH}.{YEAR}
 */

#pragma once
#include <filesystem>
#include <vector>

constexpr int intensity_levels = 10;
constexpr int frames_per_image = 20;
constexpr size_t input_size = 28 * 28;
constexpr double state_increment_factor = 1. / 255;

std::vector<std::vector<bool>> read_spike_frames(const std::filesystem::path &path_to_data);


struct Labels
{
    std::vector<int> test_;
    std::vector<std::vector<bool>> train_;
};


Labels read_labels(const std::filesystem::path &classes_file, int learning_period, int offset = 0);
