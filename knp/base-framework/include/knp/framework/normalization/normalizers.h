/**
 * @file normalizers.h
 * @brief Normalization functions.
 * @kaspersky_support Artiom N.
 * @date 23.07.2025
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

#pragma once

#include <knp/core/core.h>


/**
 * @brief Framework namespace.
 */
namespace knp::framework
{

/**
 * @brief The Rescaler class need to rescale one interval to another.
 */
template <typename ValueType>
class Rescaler
{
public:
    /**
     * @brief Rescaler constructor.
     * @param in_interval_start first interval minimum value.
     * @param in_interval_end first interval maximum value.
     * @param out_interval_start second interval minimum value.
     * @param out_interval_end second interval maximum value.
     */
    Rescaler(
        ValueType in_interval_start, ValueType in_interval_end, ValueType out_interval_start,
        ValueType out_interval_end)
        : in_interval_start_(in_interval_start),
          in_interval_end_(in_interval_end),
          out_interval_start_(out_interval_start),
          out_interval_end_(out_interval_end),
          intervals_rescaler_((out_interval_end - out_interval_start) / (in_interval_end - in_interval_start))
    {
    }

    /**
     * @brief operator () makes rescaling.
     * @param value value to rescale.
     * @return rescaled value.
     */
    ValueType operator()(ValueType value)
    {
        return out_interval_start_ + (value - in_interval_start_) * intervals_rescaler_;
    }

private:
    ValueType in_interval_start_;
    ValueType in_interval_end_;
    ValueType out_interval_start_;
    ValueType out_interval_end_;
    double intervals_rescaler_;
};

}  // namespace knp::framework
