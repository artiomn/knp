/**
 * @file altai.h
 * @brief Class definitions for AltAI, AltAI_GM and AltAI_HW devices.
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

#pragma once

#include <knp/core/device.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/**
 * @namespace knp::devices
 * @brief Namespace for implementation of concrete devices.
 */

/**
 * @brief AltAI device namespace.
 */
namespace knp::devices::altai
{

/**
 * @brief Structure used to define location of input and output ports in a grid of
 * AltAI cores.
 */
struct Port
{
    /**
     * @brief The `port_side` class is an enumeration class that represents sides of port connections to a grid of AltAI
     * cores.
     */
    enum class port_side
    {
        /**
         * @brief The north side of the grid (y = 0).
         */
        kNorth,
        /**
         * @brief The south side of the grid (y = y_max).
         * @note Only the north side for ports is currently supported.
         */
        kSouth,
        /**
         * @brief The east side of the grid (x = 0).
         * @note Only the north side for ports is currently supported.
         */
        kEast,
        /**
         * @brief The west side of the grid (x = x_max).
         * @note Only the north side for ports is currently supported.
         */
        kWest
    };

    /**
     * @brief Start core.
     */
    // cppcheck-suppress unusedStructMember
    size_t start_core;
    /**
     * @brief Port length from start core.
     */
    // cppcheck-suppress unusedStructMember
    size_t port_length;
    /**
     * @brief Side: kNorth, kSouth, kEast, or kWest.
     */
    // cppcheck-suppress unusedStructMember
    port_side side;
};

/**
 * @brief Create port of standard size (4) on the north side of the AltAI grid.
 * @details The function creates a port with standard length and size, which are used on most of existing AltAI devices.
 * @param begin_core position of a core, from which the port starts.
 * @return created port.
 */
Port make_standart_north_port(size_t begin_core);

/**
 * @brief Base class for AltAI devices.
 */
class AltAI : public knp::core::Device
{
public:
    /**
     * @brief AltAI destructor.
     */
    virtual ~AltAI() = default;

    /**
     * @brief Get device type.
     * @return device type.
     */
    [[nodiscard]] knp::core::DeviceType get_type() const override;

    /**
     * @brief Get number of columns and rows in device grid.
     * @return pair of AltAI grid columns and rows.
     */
    [[nodiscard]] std::pair<size_t, size_t> get_grid_cols_rows() const;

    /**
     * @brief Get port vector of device grid.
     * @return `Port` vector of a given AltAI device.
     */
    [[nodiscard]] std::vector<Port> get_grid_ports() const;

protected:
    /**
     * @brief Number of columns in core grid of AltAI device.
     */
    // cppcheck-suppress unusedStructMember
    size_t columns_;
    /**
     * @brief Number of rows in core grid of AltAI device.
     */
    // cppcheck-suppress unusedStructMember
    size_t rows_;
    /**
     * @brief Ports in core grid of AltAI device.
     */
    // cppcheck-suppress unusedStructMember
    std::vector<Port> ports_;
};


/**
 * @brief The `AltAI_GM` class is a definition of an interface to the AltAI golden model device.
 */
class AltAI_GM : public AltAI
{
public:
    /**
     * @brief Avoid copy of a `AltAI_GM` device.
     */
    AltAI_GM(const AltAI_GM &) = delete;

    /**
     * @brief Avoid copy assignment of a `AltAI_GM` device.
     */
    AltAI_GM &operator=(const AltAI_GM &) = delete;

    /**
     * @brief `AltAI_GM` device move constructor.
     */
    AltAI_GM(AltAI_GM &&);

    /**
     * @brief `AltAI_GM` device move operator.
     * @return reference to `AltAI_GM` instance.
     */
    AltAI_GM &operator=(AltAI_GM &&) noexcept;

    /**
     * @brief `AltAI_GM` device destructor.
     */
    ~AltAI_GM() override;

public:
    /**
     * @brief Get device name.
     * @return device name in the arbitrary format.
     */
    [[nodiscard]] const std::string &get_name() const override;

    /**
     * @brief Get power consumption details for the device.
     * @return amount of consumed power.
     */
    [[nodiscard]] float get_power() const override;

public:
    /**
     * @brief Load parameters to initialization grid of the AltAI core network.
     * @param rows number of rows in grid of AltAI cores.
     * @param columns number of columns in grid of AltAI cores.
     * @param ports port configuration in grid of AltAI cores.
     */
    void load_core_grid_params(
        size_t rows, size_t columns, const std::vector<Port> &ports = {make_standart_north_port(0)});

private:
    /**
     * @brief `AltAI_GM` device constructor.
     * @details Create `AltAI_GM` device with 4-by-4 core grid and standard port on the north side.
     */
    AltAI_GM();
    friend std::vector<std::unique_ptr<AltAI>> list_altai_devices();

private:
    // cppcheck-suppress unusedStructMember
    std::string altai_name_;
};

/**
 * @brief The `AltAI_HW` class is a definition of an interface to AltAI hardware device.
 */
class AltAI_HW : public AltAI
{
public:
    /**
     * @brief `AltAI_HW` device constructor is deleted.
     */
    AltAI_HW() = delete;

    /**
     * @brief Avoid copy of a `AltAI_HW` device.
     */
    AltAI_HW(const AltAI_HW &) = delete;

    /**
     * @brief Avoid copy assignment of a `AltAI_HW` device.
     */
    AltAI_HW &operator=(const AltAI_HW &) = delete;

    /**
     * @brief `AltAI_HW` device move constructor.
     */
    AltAI_HW(AltAI_HW &&);

    /**
     * @brief `AltAI_HW` device move operator.
     * @return reference to `AltAI_HW` instance.
     */
    AltAI_HW &operator=(AltAI_HW &&) noexcept;

    /**
     * @brief `AltAI_HW` device destructor.
     */
    ~AltAI_HW() override;

public:
    /**
     * @brief Get device name.
     * @return device name in the arbitrary format.
     */
    [[nodiscard]] const std::string &get_name() const override;

    /**
     * @brief Get power consumption details for the device.
     * @return amount of consumed power.
     */
    [[nodiscard]] float get_power() const override;

private:
    /**
     * @brief `AltAI_HW` device constructor.
     * @param rows number of rows in core grid of the given AltAI hardware.
     * @param columns number of columns in core grid of the given AltAI hardware.
     * @param ports input and output ports in core grid of the given AltAI hardware.
     */
    AltAI_HW(size_t rows, size_t columns, const std::vector<Port> &ports);
    friend std::vector<std::unique_ptr<AltAI>> list_altai_devices();

private:
    // cppcheck-suppress unusedStructMember
    std::string altai_name_;
};

/**
 * @brief List all AltAI devices on which backend can be initialized.
 * @return vector of unique pointers to base `AltAI` device class,
 *  which are `AltAI_HW` or `AltAI_GM` devices.
 */
std::vector<std::unique_ptr<AltAI>> list_altai_devices();
}  // namespace knp::devices::altai
