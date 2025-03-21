/**
 * @file tests_common.h
 * @brief Common routines used for tests.
 * @kaspersky_support Artiom N.
 * @date 31.12.2022
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <gtest/internal/gtest-filepath.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <string>
#include <vector>


namespace testing::internal
{
/// Command line arguments.
::std::vector<std::string> GetArgvs();

#if GTEST_HAS_FILE_SYSTEM
//// Return the current application name, removing directory path if it
//// is present.
FilePath GetCurrentExecutableName();
#endif
}  // namespace testing::internal


namespace knp::testing
{

/// Return tester executable path.
std::filesystem::path get_exe_path();

/// Return backend path.
std::filesystem::path get_backend_path(const std::string &backend_name = "knp-cpu-single-threaded-backend");

}  // namespace knp::testing
