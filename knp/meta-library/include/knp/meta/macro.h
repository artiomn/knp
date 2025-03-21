/**
 * @file macro.h
 * @brief Some wide used macros.
 * @kaspersky_support Artiom N.
 * @date 01.03.2024
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

#include <boost/predef.h>


#if defined(__GNUC__)
#    define KNP_UNROLL_LOOP() _Pragma("GCC unroll 16")
#elif defined(__clang__)
#    define KNP_UNROLL_LOOP() _Pragma("unroll")
#elif defined(_MSC_VER)
#    define KNP_UNROLL_LOOP() _Pragma("loop(hint_parallel(16))")
#endif
