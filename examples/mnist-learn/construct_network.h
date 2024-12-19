//
// Created by an_vartenkov on 03.12.24.
//

#pragma once

#include <knp/framework/network.h>

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>


struct AnnotatedNetwork
{
    knp::framework::Network network_;
    struct Annotation
    {
        std::vector<knp::core::UID> output_uids;
        std::vector<knp::core::UID> projections_from_raster;
        std::vector<knp::core::UID> projections_from_classes;
        std::set<knp::core::UID> inference_population_uids;
        std::set<knp::core::UID> inference_internal_projection;

        // For each compound network: a vector of senders and a vector of receivers.
        std::vector<std::pair<std::vector<knp::core::UID>, std::vector<knp::core::UID>>> wta_data;
        std::map<knp::core::UID, std::string> population_names;
    } data_;
};

AnnotatedNetwork create_example_network(int num_compound_networks);

AnnotatedNetwork create_example_network_new(int num_compound_networks);
