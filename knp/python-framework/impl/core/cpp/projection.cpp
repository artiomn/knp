/**
 * @file projection.cpp
 * @brief Python bindings for Projection.
 * @kaspersky_support Artiom N.
 * @date 16.02.2024
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


// "Construct an empty projection.")
//"Construct a projection by running a synapse generator a given number of times."
//py::arg("presynaptic_uid"), py::arg("postsynaptic_uid"))

#if defined(KNP_IN_CORE)

#    include "projection.h"

#    include <tuple>
#    include <vector>

#    include "common.h"
#    include "tuple_converter.h"


namespace st = knp::synapse_traits;

/*

.def("get_shared_parameters",
   (core::SharedSynapseParameters(core::Projection::*)()) &
   core::Projection::get_shared_parameters,
   "Get parameters shared between all synapses.")
.def("get_shared_parameters",
   (core::SharedSynapseParameters(core::Projection::*)()) &
   core::Projection::get_shared_parameters,
   "Get parameters shared between all synapses.")


py::class_<core::Synapse>(
    "Synapse",
    "Synapse description structure that contains synapse parameters and indexes of the associated neurons.")

    py::class_<core::SharedSynapseParametersT>(
        "SharedSynapseParametersT", "Shared synapse parameters for the non-STDP variant of the projection.")

        py::class_<SharedSynapseParametersT<core::synapse_traits::STDP<Rule, SynapseT>>>(
            "SharedSynapseParametersT<knp::synapse_traits::STDP<Rule,SynapseT>>",
            "Structure for the parameters shared between synapses for STDP.")

            py::enum_<SharedSynapseParametersT<core::synapse_traits::STDP<Rule, SynapseT>>::ProcessingType>(
                "ProcessingType")
                .value(
                    "STDPOnly",
                    SharedSynapseParametersT<core::synapse_traits::STDP<Rule, SynapseT>>::ProcessingType::0)
                .value(
                    "STDPAndSpike",
                    SharedSynapseParametersT<core::synapse_traits::STDP<Rule, SynapseT>>::ProcessingType::1);
*/
//py::register_tuple<typename core::Projection<st::synapse_type>::Synapse>();

#    define INSTANCE_PY_PROJECTIONS(n, template_for_instance, synapse_type)                                            \
        py::implicitly_convertible<core::Projection<st::synapse_type>, core::AllProjectionsVariant>();                 \
        py::class_<typename core::Projection<st::synapse_type>::Synapse>(                                              \
            BOOST_PP_STRINGIZE(BOOST_PP_CAT(synapse_type, Parameters)));                                               \
                                                                                                                       \
        py::class_<core::Projection<st::synapse_type>>(                                                                \
            BOOST_PP_STRINGIZE(                                                                                        \
                BOOST_PP_CAT(synapse_type, Projection)),                                                               \
                "The Projection class is a definition of similar connections between the neurons of two populations.", \
                py::no_init)                                                                                           \
                .def(py::init<core::UID, core::UID>())                                                                 \
                .def(py::init<core::UID, core::UID, core::UID>())                                                      \
                .def(                                                                                                  \
                    "__init__",                                                                                        \
                    py::make_constructor(static_cast<std::shared_ptr<core::Projection<st::synapse_type>> (*)(          \
                                             core::UID, core::UID, core::UID, const py::object &, size_t)>(            \
                        &projection_constructor_wrapper<st::synapse_type>)),                                           \
                    "Construct a projection by running a synapse generator a given number of times.")                  \
                .def(                                                                                                  \
                    "__init__",                                                                                        \
                    py::make_constructor(static_cast<std::shared_ptr<core::Projection<st::synapse_type>> (*)(          \
                                             core::UID, core::UID, const py::object &, size_t)>(                       \
                        &projection_constructor_wrapper<st::synapse_type>)),                                           \
                    "Construct a projection by running a synapse generator a given number of times.")                  \
                .def(                                                                                                  \
                    "add_synapses", &projection_synapses_add_wrapper<st::synapse_type>,                                \
                    "Append connections to the existing projection.")                                                  \
                .add_property(                                                                                         \
                    "uid", make_handler([](core::Projection<st::synapse_type> &proj) { return proj.get_uid(); }),      \
                    "Get projection UID.")                                                                             \
                .def(                                                                                                  \
                    "__iter__", py::iterator<core::Projection<st::synapse_type>>(),                                    \
                    "Get an iterator of the projection.")                                                              \
                .def(                                                                                                  \
                    "__len__", &core::Projection<st::synapse_type>::size,                                              \
                    "Count number of synapses in the projection.")                                                     \
                .def(                                                                                                  \
                    "__getitem__",                                                                                     \
                    static_cast<core::Projection<st::synapse_type>::Synapse &(                                         \
                        core::Projection<st::synapse_type>::*)(size_t)>(                                               \
                        &core::Projection<st::synapse_type>::operator[]),                                              \
                    py::return_internal_reference<>(),                                                                 \
                    "Get parameter values of a synapse with the given index.");  // NOLINT

BOOST_PP_SEQ_FOR_EACH(INSTANCE_PY_PROJECTIONS, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_SYNAPSES))  //!OCLINT(Parameters used)

#endif
