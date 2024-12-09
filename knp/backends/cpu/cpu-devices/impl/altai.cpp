/**
 * @file altai.cpp
 * @brief AltAI devices class implementation.
 * @author Liubiakin A.
 * @date 17.10.2024
 * @license Apache 2.0
 * @copyright © 2024 Motiv NT
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

#include <knp/devices/altai.h>

#include <boost/uuid/name_generator.hpp>
#include <boost/uuid/string_generator.hpp>
#include <boost/uuid/uuid_generators.hpp>

namespace knp::devices::altai
{

static bool is_generated = false;
static boost::uuids::uuid altai_gm_uid;

static constexpr const int compact_board_columns = 16;
static constexpr const int compact_board_rows = 16;
static const std::vector<Port> compact_board_ports{
    Port{4, 4, Port::port_side::kNorth},
};

Port make_standart_north_port(size_t begin_core)
{
    return Port{begin_core, 4, Port::port_side::kNorth};
}

std::pair<size_t, size_t> AltAI::get_grid_cols_rows() const
{
    return std::pair<size_t, size_t>(columns_, rows_);
}


std::vector<Port> AltAI::get_grid_ports() const
{
    return ports_;
}

knp::core::DeviceType AltAI::get_type() const
{
    return knp::core::DeviceType::AltAI1_NPU;
}


AltAI_GM::AltAI_GM()
{
    altai_name_ = "AltAIv1_GoldenModel";
    if (!is_generated)
    {
        boost::uuids::random_generator generator;
        altai_gm_uid = generator();
        is_generated = true;
    }
    Device::base_.uid_ = knp::core::UID(altai_gm_uid);
    columns_ = 4;
    rows_ = 4;
    ports_ = std::vector<Port>{make_standart_north_port(0)};
}


AltAI_GM::AltAI_GM(AltAI_GM&& other) : altai_name_{std::move(other.altai_name_)} {}


AltAI_GM::~AltAI_GM() {}


AltAI_GM& AltAI_GM::operator=(AltAI_GM&& other) noexcept
{
    if (this != &other)
    {
        altai_name_.swap(other.altai_name_);
    }
    return *this;
}


const std::string& AltAI_GM::get_name() const
{
    return altai_name_;
}


float AltAI_GM::get_power() const
{
    return 0.;
}

void AltAI_GM::load_core_grid_params(size_t rows, size_t columns, const std::vector<Port>& ports)
{
    columns_ = columns;
    rows_ = rows;
    ports_ = ports;
}

AltAI_HW::AltAI_HW(size_t rows, size_t columns, const std::vector<Port>& ports)
{
    altai_name_ = "AltAIv1_Compact_v1.0";
    if (!is_generated)
    {
        boost::uuids::random_generator generator;
        altai_gm_uid = generator();
        is_generated = true;
    }
    Device::base_.uid_ =
        knp::core::UID(boost::uuids::name_generator(knp::core::UID(altai_gm_uid))(altai_name_.c_str()));
    columns_ = columns;
    rows_ = rows;
    ports_ = ports;
}

AltAI_HW::AltAI_HW(AltAI_HW&& other) : altai_name_(std::move(other.altai_name_)) {}

AltAI_HW::~AltAI_HW() {}

AltAI_HW& AltAI_HW::operator=(AltAI_HW&& other) noexcept
{
    if (this != &other)
    {
        altai_name_.swap(other.altai_name_);
    }
    return *this;
}


const std::string& AltAI_HW::get_name() const
{
    return altai_name_;
}


float AltAI_HW::get_power() const
{
    return 0.;
}

std::vector<std::unique_ptr<AltAI>> list_altai_devices()
{
    std::vector<std::unique_ptr<AltAI>> devices;
    devices.push_back(std::unique_ptr<AltAI>(new AltAI_GM()));
    devices.push_back(
        std::unique_ptr<AltAI>(new AltAI_HW(compact_board_rows, compact_board_columns, compact_board_ports)));
    return devices;
}

}  // namespace knp::devices::altai
