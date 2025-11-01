/**
 * @file altai_lif.cpp
 * @brief Python bindings for AltaAI LIF neuron.
 * @kaspersky_support Artiom N.
 * @date 31.10.25
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

#if defined(_KNP_IN_NEURON_TRAITS)

#    include "common.h"

using alif_params = knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF>;

py::class_<alif_params>("AltAILIFNeuronParameters", "Structure for BLIFAT neuron parameters")
    .def(py::init<>())
    .add_property(
        "is_diff", &alif_params::is_diff_,
        "If `is_diff_` flag is set to `true` and neuron potential exceeds one of its threshold value after the neuron "
        "receives a spike, the `potential_` parameter takes a value by which the potential threshold is exceeded.")
    .add_property(
        "is_reset", &alif_params::is_diff_,
        "If `is_reset_` flag is set to `true` and neuron potential exceeds its threshold value after the neuron "
        "receives a spike, the `potential_` parameter")
    .add_property(
        "leak_rev", &alif_params::leak_rev_,
        "If `leak_rev_` flag is set to `true`, the `potential_leak_` sign automatically changes along with the change "
        "of the `potential_` value sign.")
    .add_property(
        "saturate", &alif_params::saturate_,
        "If `saturate_` flag is set to `true` and the neuron potential is less than a negative "
        "`negative_activation_threshold_` value after the neuron receives a spike, the `potential_` parameter takes "
        "the `negative_activation_threshold_` value.")
    .add_property(
        "do_not_save", &alif_params::do_not_save_,
        "If `do_not_save_` flag is set to `false`, the `potential_` value is stored with each timestamp.")
    .add_property("potential", &alif_params::potential_, "The parameter defines the neuron potential value.")
    .add_property(
        "activation_threshold", &alif_params::activation_threshold_,
        "The parameter defines the threshold value of neuron potential, after exceeding which a positive spike can be "
        "emitted.")
    .add_property(
        "negative_activation_threshold", &alif_params::negative_activation_threshold_,
        "The parameter defines the threshold value of neuron potential, below which a negative spike can be emitted.")
    .add_property(
        "potential_leak", &alif_params::potential_leak_,
        "The parameter defines the constant leakage of the neuron potential.")
    .add_property(
        "potential_reset_value", &alif_params::potential_reset_value_,
        "The parameter defines a reset value of the neuron potential after one of the thresholds has been exceeded.");

#endif
