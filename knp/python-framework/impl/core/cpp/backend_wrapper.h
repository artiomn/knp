#pragma once

#include "common.h"


struct BackendWrapper : core::Backend, py::wrapper<core::Backend>
{
    BackendWrapper() = default;
    virtual ~BackendWrapper() = default;
    bool plasticity_supported() const override { return this->get_override("plasticity_supported")(); }

    std::vector<std::string> get_supported_neurons() const override
    {
        return this->get_override("get_supported_neurons")();
    }

    std::vector<std::string> get_supported_synapses() const override
    {
        return this->get_override("get_supported_synapses")();
    }

    std::vector<size_t> get_supported_population_indexes() const override
    {
        return this->get_override("get_supported_population_indexes")();
    }

    std::vector<size_t> get_supported_projection_indexes() const override
    {
        return this->get_override("get_supported_projection_indexes")();
    }

    void load_all_projections(const std::vector<core::AllProjectionsVariant> &projections) override
    {
        this->get_override("load_all_projections")(projections);
    }

    void load_all_populations(const std::vector<core::AllPopulationsVariant> &populations) override
    {
        this->get_override("load_all_populations")(populations);
    }

    void remove_projections(const std::vector<core::UID> &uids) override
    {
        this->get_override("remove_projections")(uids);
    }

    void remove_populations(const std::vector<core::UID> &uids) override
    {
        this->get_override("remove_populations")(uids);
    }

    std::vector<std::unique_ptr<core::Device>> get_devices_vector(const std::string &method_name) const
    {
        auto py_result = boost::python::list(this->get_override(method_name.c_str())());
        const auto dev_count = boost::python::len(py_result);

        std::vector<std::unique_ptr<core::Device>> f_result;
        f_result.reserve(dev_count);

        for (std::remove_const_t<decltype(dev_count)> i = 0; i < dev_count; ++i)
        {
            boost::python::extract<core::Device *> extractor(py_result[i]);
            if (extractor.check())
            {
                core::Device *dev = extractor();
                // py_result[i].release();
                f_result.emplace_back(dev);
            }
            else
            {
                // Error.
            }
        }

        return f_result;
    }

    std::vector<std::unique_ptr<core::Device>> get_devices() const override
    {
        // return get_devices_vector("get_devices");
        return this->get_devices_vector("get_devices");
    }
    /*
    std::vector<std::unique_ptr<knp::core::Device>> &get_current_devices()
    {
        // return get_devices_vector("get_current_devices");
        return this->get_override("get_current_devices");
    }

    const std::vector<std::unique_ptr<core::Device>> &get_current_devices() const
    {
        // return get_devices_vector("get_current_devices");
        return this->get_override("get_current_devices");
    }
*/
    void select_devices(const std::set<core::UID> &uids) override { this->get_override("select_devices")(uids); }
    void select_device(std::unique_ptr<core::Device> &&device) override { this->get_override("select_device")(device); }

    /*
        const core::MessageEndpoint &get_message_endpoint() const override
        {
            // Warning: possibly error.
            // return reinterpret_cast<const knp::core::MessageEndpoint
       &>(this->get_override("get_message_endpoint")()); return reinterpret_cast<const knp::core::MessageEndpoint
       &>(this->get_override("get_message_endpoint")());
        }

        knp::core::MessageEndpoint &get_message_endpoint()
        {
            // Warning: possibly error.
            return reinterpret_cast<knp::core::MessageEndpoint &>(this->get_override("get_message_endpoint")());
        }
    */

    DataRanges get_network_data() const override
    {
        return {};  //std::move(this->get_override("get_network_data")());
    }

    void stop_learning() override { this->get_override("stop_learning")(); }
    void start_learning() override { this->get_override("start_learning")(); }
    void _step() override { this->get_override("step")(); }
    void _init() override { this->get_override("init")(); }
};
