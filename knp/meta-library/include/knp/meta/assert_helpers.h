/**
 * @file assert_helpers.h
 * @brief Some helpers that are useful for assertions.
 * @kaspersky_support Artiom N.
 * @date 31.08.2023
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

#pragma once


/**
 * @brief Metaprogramming library namespace.
 */
namespace knp::meta
{

/**
 * @brief Template used to generate a fixed `false` value for static asserts.
 */
template <class>
inline constexpr bool always_false_v = false;

}  // namespace knp::meta
