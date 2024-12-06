/**
 * @author M. V. Kiselev
 * @date {DAY}.{MONTH}.{YEAR}
 */

//
// Created by Mike on 10/15/2024.
//

#pragma once
#include <vector>

constexpr int intensity_levels = 10;
constexpr int frames_per_image = 20;
constexpr size_t input_size = 28 * 28;
constexpr double state_increment_factor = 1. / 255;

std::vector<std::vector<bool>> read_spike_frames(const std::string &path_to_data);
