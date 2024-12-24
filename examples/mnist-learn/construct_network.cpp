//
// Created by an_vartenkov on 03.12.24.
//

#include "construct_network.h"

#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/neuron-traits/all_traits.h>
#include <knp/synapse-traits/all_traits.h>


using DeltaSynapseData = knp::synapse_traits::synapse_parameters<knp::synapse_traits::DeltaSynapse>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using BlifatPopulation = knp::core::Population<knp::neuron_traits::BLIFATNeuron>;
using ResourceBlifatPopulation = knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;
using ResourceNeuron = knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron;
using ResourceNeuronData = knp::neuron_traits::neuron_parameters<ResourceNeuron>;
using ResourceSynapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
using ResourceDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
using ResourceSynapseData = ResourceDeltaProjection::Synapse;
using ResourceSynapseGenerator = std::function<ResourceSynapseData(size_t)>;
using ResourceSynapseParams = knp::synapse_traits::synapse_parameters<ResourceSynapse>;
using ResourceNeuronGenerator = std::function<ResourceNeuronData(size_t)>;


// A dense projection generator from a default synapse.
ResourceSynapseGenerator make_dense_generator(size_t from_size, const ResourceSynapseParams &default_synapse)
{
    ResourceSynapseGenerator synapse_generator = [from_size, default_synapse](size_t index)
    {
        size_t from_index = index % from_size;
        size_t to_index = index / from_size;
        // Если нужно, можно модифицировать параметры синапса по умолчанию в зависимости от индекса.
        return ResourceSynapseData{default_synapse, from_index, to_index};
    };
    return synapse_generator;
}


// A 1 to 1 simple synapse generator.
ResourceSynapseGenerator make_1_to_1_generator(const ResourceSynapseParams &default_synapse)
{
    ResourceSynapseGenerator synapse_generator = [default_synapse](size_t index) {
        return ResourceSynapseData{default_synapse, index, index};
    };
    return synapse_generator;
}


DeltaProjection::SynapseGenerator make_all_to_all_sections_generator(
    size_t section_size, const DeltaSynapseData &default_synapse)
{
    DeltaProjection::SynapseGenerator generator =
        [section_size, default_synapse](size_t index) -> std::optional<DeltaProjection::Synapse>
    {
        size_t section = index / (section_size * section_size);
        size_t index_in_section = index % (section_size * section_size);
        size_t input_neuron = index_in_section % section_size;
        size_t output_neuron = index_in_section / section_size;
        if (input_neuron == output_neuron) return {};  // Сам к себе: связи нет.
        return DeltaProjection::Synapse{
            default_synapse, section * section_size + input_neuron, section * section_size + output_neuron};
    };
    return generator;
}


DeltaProjection::SynapseGenerator make_aligned_generator(
    size_t prepopulation_size, size_t postpopulation_size, const DeltaSynapseData &default_synapse)
{
    DeltaProjection::SynapseGenerator synapse_generator =
        [prepopulation_size, postpopulation_size, default_synapse](size_t index)
    {
        size_t from_index;
        size_t pack_size;
        size_t to_index;
        if (prepopulation_size >= postpopulation_size)
        {
            from_index = index;
            pack_size = prepopulation_size / postpopulation_size;
            to_index = index / pack_size;
        }
        else
        {
            to_index = index;
            pack_size = postpopulation_size / prepopulation_size;
            from_index = index / pack_size;
        }
        return DeltaProjection::Synapse{default_synapse, from_index, to_index};
    };
    return synapse_generator;
}


DeltaProjection::SynapseGenerator make_exclusive_generator(
    size_t population_size, const DeltaSynapseData &default_synapse)
{
    DeltaProjection::SynapseGenerator synapse_generator = [population_size, default_synapse](size_t index)
    {
        size_t from_index;
        size_t to_index;
        from_index = index / (population_size - 1);
        to_index = index % (population_size - 1);
        if (to_index >= from_index) ++to_index;
        return DeltaProjection::Synapse{default_synapse, from_index, to_index};
    };
    return synapse_generator;
}


template <class Neuron>
struct PopulationData
{
    size_t size_;
    knp::neuron_traits::neuron_parameters<Neuron> neuron_;
};


AnnotatedNetwork create_example_network(int num_compound_networks)
{
    // Сеть, которую мы наполним и вернём
    AnnotatedNetwork result;
    enum ProjectionIndexes
    {
        INPUT = 0,
        WTA = 1,
        DOPAMINE = 2,
        OUTPUT = 3,
        GATE = 4
    };

    for (int i = 0; i < num_compound_networks; ++i)
    {
        // Параметры нейрона по умолчанию. Здесь можно задать те параметры, которые нужны.
        ResourceNeuronData default_neuron{{}};
        default_neuron.activation_threshold_ = 8.571;
        ResourceNeuronData L_neuron = default_neuron;
        L_neuron.potential_decay_ = 1 - 1 / 3.;
        L_neuron.d_h_ = -0.042F;
        L_neuron.dopamine_plasticity_time_ = 10;
        // L_neuron.w_min_ = -0.7F;
        // L_neuron.w_max_ = 0.864249F;
        L_neuron.isi_max_ = 10;
        // L_neuron.synaptic_resource_threshold_ = 10;

        struct PopulationRole
        {
            PopulationData<ResourceNeuron> pd;
            bool for_inference;
            bool output;
            std::string name;
        };
        auto dopamine_neuron = default_neuron;
        dopamine_neuron.total_blocking_period_ = 0;
        // Инициируем данные для создания популяций. Их будет 5.
        std::vector<PopulationRole> pop_data{
            {{150, L_neuron}, true, false, "INPUT"},
            {{150, default_neuron}, true, false, "WTA"},
            {{150, dopamine_neuron}, false, false, "DOPAMINE"},
            {{10, default_neuron}, true, true, "OUTPUT"},
            {{10, default_neuron}, false, false, "GATE"}};

        // Создаём популяции: это обычно очень просто, нейроны скорее всего исходно все одинаковые, связей в популяции
        // нет.
        std::vector<knp::core::UID> population_uids;
        for (auto &pop_init_data : pop_data)
        {
            // Очень простой генератор нейронов "по образцу".
            auto neuron_generator = [&pop_init_data](size_t index) { return pop_init_data.pd.neuron_; };

            knp::core::UID uid;
            result.network_.add_population(ResourceBlifatPopulation{uid, neuron_generator, pop_init_data.pd.size_});
            population_uids.push_back(uid);
            result.data_.population_names[uid] = pop_init_data.name;
            if (pop_init_data.for_inference) result.data_.inference_population_uids.insert(uid);
            if (pop_init_data.output) result.data_.output_uids.push_back(uid);
        }

        // В сеть добавлены все нужные популяции, теперь связываем их проекциями. Пусть последовательно.
        // Проекции создавать несколько сложнее, так как вся логика связей сети расположена в них.

        // Создаём образец "обучаемого" синапса. Здесь же можно задать ему нужные параметры.
        ResourceSynapseParams default_synapse;
        auto afferent_synapse = default_synapse;
        afferent_synapse.rule_.synaptic_resource_ = 1.267F;
        afferent_synapse.rule_.dopamine_plasticity_period_ = 10;
        afferent_synapse.rule_.w_min_ = -0.7F;
        afferent_synapse.rule_.w_max_ = 0.864249F;
        ResourceDeltaProjection input_projection{
            knp::core::UID{false}, population_uids[INPUT], make_dense_generator(28 * 28, afferent_synapse),
            28 * 28 * 150};
        result.data_.projections_from_raster.push_back(input_projection.get_uid());

        default_synapse.weight_ = 9;
        input_projection.unlock_weights();  // Обучаемая
        result.network_.add_projection(input_projection);
        result.data_.inference_internal_projection.insert(input_projection.get_uid());

        // Проекция между первой и второй популяцией.

        ResourceDeltaProjection projection_1{
            population_uids[INPUT], population_uids[WTA], make_1_to_1_generator(default_synapse), 150};
        result.network_.add_projection(projection_1);
        if (result.data_.inference_population_uids.find(population_uids[INPUT]) !=
                result.data_.inference_population_uids.end() &&
            result.data_.inference_population_uids.find(population_uids[WTA]) !=
                result.data_.inference_population_uids.end())
        {
            result.data_.inference_internal_projection.insert(projection_1.get_uid());
        }

        // Блокирующая wta-проекция на себя, с задержкой 1.
        const DeltaSynapseData default_wta_synapse{-10, 1, knp::synapse_traits::OutputType::BLOCKING};
        // Обратите внимание: количество созданных синапсов здесь будет меньше, чем число итераций с попытками их
        // создать. Это связано с логикmake_wta_generatorой работы генератора, который пропускает итерацию, где должен
        // создаться синапс "сам к себе".
        DeltaProjection wta_projection{
            population_uids[WTA], population_uids[WTA], make_all_to_all_sections_generator(15, default_wta_synapse),
            pop_data[WTA].pd.size_ * pop_data[WTA].pd.size_};
        result.network_.add_projection(wta_projection);
        if (result.data_.inference_population_uids.find(population_uids[WTA]) !=
            result.data_.inference_population_uids.end())
        {
            result.data_.inference_internal_projection.insert(wta_projection.get_uid());
        }

        // Активирующая проекция
        const DeltaSynapseData default_activating_synapse{1, 1, knp::synapse_traits::OutputType::BLOCKING};
        DeltaProjection projection_2{
            population_uids[WTA], population_uids[DOPAMINE],
            make_aligned_generator(pop_data[WTA].pd.size_, pop_data[DOPAMINE].pd.size_, default_activating_synapse),
            pop_data[WTA].pd.size_};
        result.network_.add_projection(projection_2);
        if (result.data_.inference_population_uids.find(population_uids[WTA]) !=
                result.data_.inference_population_uids.end() &&
            result.data_.inference_population_uids.find(population_uids[DOPAMINE]) !=
                result.data_.inference_population_uids.end())
        {
            result.data_.inference_internal_projection.insert(projection_2.get_uid());
        }

        // Дофаминовая проекция
        const DeltaSynapseData default_dopamine_synapse{0.042F, 1, knp::synapse_traits::OutputType::DOPAMINE};
        DeltaProjection projection_3{
            population_uids[DOPAMINE], population_uids[INPUT],
            make_aligned_generator(pop_data[DOPAMINE].pd.size_, pop_data[INPUT].pd.size_, default_dopamine_synapse),
            pop_data[INPUT].pd.size_};

        result.network_.add_projection(projection_3);

        if (result.data_.inference_population_uids.find(population_uids[INPUT]) !=
                result.data_.inference_population_uids.end() &&
            result.data_.inference_population_uids.find(population_uids[DOPAMINE]) !=
                result.data_.inference_population_uids.end())
        {
            result.data_.inference_internal_projection.insert(projection_3.get_uid());
        }

        // Strong excitatory projection going to output neurons.
        DeltaProjection projection_4{
            population_uids[WTA], population_uids[OUTPUT],
            make_aligned_generator(pop_data[WTA].pd.size_, pop_data[OUTPUT].pd.size_, default_synapse),
            pop_data[WTA].pd.size_};

        result.network_.add_projection(projection_4);
        if (result.data_.inference_population_uids.find(population_uids[WTA]) !=
                result.data_.inference_population_uids.end() &&
            result.data_.inference_population_uids.find(population_uids[OUTPUT]) !=
                result.data_.inference_population_uids.end())
        {
            result.data_.inference_internal_projection.insert(projection_4.get_uid());
        }

        // Blocking 1-to-1 projection.
        const DeltaSynapseData default_blocking_synapse{-20, 1, knp::synapse_traits::OutputType::BLOCKING};
        DeltaProjection projection_5{
            population_uids[OUTPUT], population_uids[GATE],
            make_aligned_generator(pop_data[OUTPUT].pd.size_, pop_data[GATE].pd.size_, default_blocking_synapse), 10};
        result.network_.add_projection(projection_5);
        if (result.data_.inference_population_uids.find(population_uids[OUTPUT]) !=
                result.data_.inference_population_uids.end() &&
            result.data_.inference_population_uids.find(population_uids[GATE]) !=
                result.data_.inference_population_uids.end())
        {
            result.data_.inference_internal_projection.insert(projection_5.get_uid());
        }

        // Strong excitatory projection going from ground truth classes.
        DeltaProjection projection_6{
            knp::core::UID{false}, population_uids[DOPAMINE],
            make_aligned_generator(10, pop_data[DOPAMINE].pd.size_, default_synapse), pop_data[DOPAMINE].pd.size_};
        result.network_.add_projection(projection_6);
        result.data_.projections_from_classes.push_back(projection_6.get_uid());

        // Strong slow excitatory projection going from ground truth classes.
        auto slow_synapse = default_synapse;
        slow_synapse.delay_ = 10;
        DeltaProjection projection_7{
            knp::core::UID{false}, population_uids[GATE],
            make_aligned_generator(10, pop_data[GATE].pd.size_, slow_synapse), pop_data[GATE].pd.size_};
        result.network_.add_projection(projection_7);
        result.data_.projections_from_classes.push_back(projection_7.get_uid());

        // Strong inhibitory projection from ground truth input.
        auto inhibitory_synapse = default_synapse;
        inhibitory_synapse.weight_ = -30;
        DeltaProjection projection_8{
            knp::core::UID{false}, population_uids[GATE], make_exclusive_generator(10, inhibitory_synapse),
            10 * (pop_data[GATE].pd.size_ - 1)};
        result.data_.projections_from_classes.push_back(projection_8.get_uid());
        result.network_.add_projection(projection_8);

        // Weak excitatory projection.
        auto weak_excitatory_synapse = default_synapse;
        weak_excitatory_synapse.weight_ = 3;
        DeltaProjection projection_9{
            population_uids[GATE], population_uids[INPUT],
            make_aligned_generator(pop_data[GATE].pd.size_, pop_data[INPUT].pd.size_, weak_excitatory_synapse),
            pop_data[INPUT].pd.size_};
        result.network_.add_projection(projection_9);
        if (result.data_.inference_population_uids.find(population_uids[INPUT]) !=
                result.data_.inference_population_uids.end() &&
            result.data_.inference_population_uids.find(population_uids[GATE]) !=
                result.data_.inference_population_uids.end())
        {
            result.data_.inference_internal_projection.insert(projection_9.get_uid());
        }
    }

    // Сеть готова, возвращаем её.
    return result;
}


AnnotatedNetwork create_example_network_new(int num_compound_networks)
{
    // Сеть, которую мы наполним и вернём
    AnnotatedNetwork result;
    enum PopIndexes
    {
        INPUT = 0,
        DOPAMINE = 1,
        OUTPUT = 2,
        GATE = 3,
    };
    result.data_.wta_data.resize(num_compound_networks);
    for (int i = 0; i < num_compound_networks; ++i)
    {
        // Параметры нейрона по умолчанию. Здесь можно задать те параметры, которые нужны.
        ResourceNeuronData default_neuron{{}};
        default_neuron.activation_threshold_ = 8.571;
        ResourceNeuronData L_neuron = default_neuron;
        L_neuron.potential_decay_ = 1 - 1 / 3.;
        L_neuron.d_h_ = -0.042F;
        L_neuron.dopamine_plasticity_time_ = 10;
        // L_neuron.w_min_ = -0.7F;
        // L_neuron.w_max_ = 0.864249F;
        L_neuron.isi_max_ = 10;
        // L_neuron.synaptic_resource_threshold_ = 10;

        struct PopulationRole
        {
            PopulationData<ResourceNeuron> pd;
            bool for_inference;
            bool output;
            std::string name;
        };
        auto dopamine_neuron = default_neuron;
        dopamine_neuron.total_blocking_period_ = 0;
        // Инициируем данные для создания популяций. Их будет 5.
        std::vector<PopulationRole> pop_data{
            {{150, L_neuron}, true, false, "INPUT"},
            //{{150, default_neuron}, true, false, "WTA"}, // Using WTA message handler instead.
            {{150, dopamine_neuron}, false, false, "DOPAMINE"},
            {{10, default_neuron}, true, true, "OUTPUT"},
            {{10, default_neuron}, false, false, "GATE"}};

        // Создаём популяции: это обычно очень просто, нейроны скорее всего исходно все одинаковые, связей в популяции
        // нет.
        std::vector<knp::core::UID> population_uids;
        for (auto &pop_init_data : pop_data)
        {
            // Очень простой генератор нейронов "по образцу".
            auto neuron_generator = [&pop_init_data](size_t index) { return pop_init_data.pd.neuron_; };

            knp::core::UID uid;
            result.network_.add_population(ResourceBlifatPopulation{uid, neuron_generator, pop_init_data.pd.size_});
            population_uids.push_back(uid);
            result.data_.population_names[uid] = pop_init_data.name;
            if (pop_init_data.for_inference) result.data_.inference_population_uids.insert(uid);
            if (pop_init_data.output) result.data_.output_uids.push_back(uid);
        }
        result.data_.wta_data[i].first.push_back(population_uids[INPUT]);

        // В сеть добавлены все нужные популяции, теперь связываем их проекциями. Пусть последовательно.
        // Проекции создавать несколько сложнее, так как вся логика связей сети расположена в них.

        // Создаём образец "обучаемого" синапса. Здесь же можно задать ему нужные параметры.
        ResourceSynapseParams default_synapse;
        auto afferent_synapse = default_synapse;
        afferent_synapse.rule_.synaptic_resource_ = 1.267F;
        afferent_synapse.rule_.dopamine_plasticity_period_ = 10;
        afferent_synapse.rule_.w_min_ = -0.7F;
        afferent_synapse.rule_.w_max_ = 0.864249F;
        ResourceDeltaProjection input_projection{
            knp::core::UID{false}, population_uids[INPUT], make_dense_generator(28 * 28, afferent_synapse),
            28 * 28 * 150};
        result.data_.projections_from_raster.push_back(input_projection.get_uid());

        default_synapse.weight_ = 9;
        input_projection.unlock_weights();  // Обучаемая
        result.network_.add_projection(input_projection);
        result.data_.inference_internal_projection.insert(input_projection.get_uid());

        // Активирующая проекция
        const DeltaSynapseData default_activating_synapse{1, 1, knp::synapse_traits::OutputType::BLOCKING};
        DeltaProjection projection_2{
            knp::core::UID{false}, population_uids[DOPAMINE],
            make_aligned_generator(pop_data[INPUT].pd.size_, pop_data[DOPAMINE].pd.size_, default_activating_synapse),
            pop_data[INPUT].pd.size_};
        result.network_.add_projection(projection_2);
        result.data_.wta_data[i].second.push_back(projection_2.get_uid());

        // Дофаминовая проекция
        // const DeltaSynapseData default_dopamine_synapse{0.042F, 1, knp::synapse_traits::OutputType::DOPAMINE};
        const DeltaSynapseData default_dopamine_synapse{20.042F, 1, knp::synapse_traits::OutputType::DOPAMINE};
        DeltaProjection projection_3{
            population_uids[DOPAMINE], population_uids[INPUT],
            make_aligned_generator(pop_data[DOPAMINE].pd.size_, pop_data[INPUT].pd.size_, default_dopamine_synapse),
            pop_data[INPUT].pd.size_};

        result.network_.add_projection(projection_3);

        if (result.data_.inference_population_uids.find(population_uids[INPUT]) !=
                result.data_.inference_population_uids.end() &&
            result.data_.inference_population_uids.find(population_uids[DOPAMINE]) !=
                result.data_.inference_population_uids.end())
        {
            result.data_.inference_internal_projection.insert(projection_3.get_uid());
        }

        // Strong excitatory projection going to output neurons.
        DeltaProjection projection_4{
            knp::core::UID{false}, population_uids[OUTPUT],
            make_aligned_generator(pop_data[INPUT].pd.size_, pop_data[OUTPUT].pd.size_, default_synapse),
            pop_data[INPUT].pd.size_};
        result.data_.wta_data[i].second.push_back(projection_4.get_uid());

        result.network_.add_projection(projection_4);
        result.data_.inference_internal_projection.insert(projection_4.get_uid());

        // Blocking 1-to-1 projection.
        const DeltaSynapseData default_blocking_synapse{-20, 1, knp::synapse_traits::OutputType::BLOCKING};
        DeltaProjection projection_5{
            population_uids[OUTPUT], population_uids[GATE],
            make_aligned_generator(pop_data[OUTPUT].pd.size_, pop_data[GATE].pd.size_, default_blocking_synapse), 10};
        result.network_.add_projection(projection_5);
        if (result.data_.inference_population_uids.find(population_uids[OUTPUT]) !=
                result.data_.inference_population_uids.end() &&
            result.data_.inference_population_uids.find(population_uids[GATE]) !=
                result.data_.inference_population_uids.end())
        {
            result.data_.inference_internal_projection.insert(projection_5.get_uid());
        }

        // Strong excitatory projection going from ground truth classes.
        DeltaProjection projection_6{
            knp::core::UID{false}, population_uids[DOPAMINE],
            make_aligned_generator(10, pop_data[DOPAMINE].pd.size_, default_synapse), pop_data[DOPAMINE].pd.size_};
        result.network_.add_projection(projection_6);
        result.data_.projections_from_classes.push_back(projection_6.get_uid());

        // Strong slow excitatory projection going from ground truth classes.
        auto slow_synapse = default_synapse;
        slow_synapse.delay_ = 10;
        DeltaProjection projection_7{
            knp::core::UID{false}, population_uids[GATE],
            make_aligned_generator(10, pop_data[GATE].pd.size_, slow_synapse), pop_data[GATE].pd.size_};
        result.network_.add_projection(projection_7);
        result.data_.projections_from_classes.push_back(projection_7.get_uid());

        // Strong inhibitory projection from ground truth input.
        auto inhibitory_synapse = default_synapse;
        inhibitory_synapse.weight_ = -30;
        DeltaProjection projection_8{
            knp::core::UID{false}, population_uids[GATE], make_exclusive_generator(10, inhibitory_synapse),
            10 * (pop_data[GATE].pd.size_ - 1)};
        result.data_.projections_from_classes.push_back(projection_8.get_uid());
        result.network_.add_projection(projection_8);

        // Weak excitatory projection.
        auto weak_excitatory_synapse = default_synapse;
        weak_excitatory_synapse.weight_ = 3;
        DeltaProjection projection_9{
            population_uids[GATE], population_uids[INPUT],
            make_aligned_generator(pop_data[GATE].pd.size_, pop_data[INPUT].pd.size_, weak_excitatory_synapse),
            pop_data[INPUT].pd.size_};
        result.network_.add_projection(projection_9);
        if (result.data_.inference_population_uids.find(population_uids[INPUT]) !=
                result.data_.inference_population_uids.end() &&
            result.data_.inference_population_uids.find(population_uids[GATE]) !=
                result.data_.inference_population_uids.end())
        {
            result.data_.inference_internal_projection.insert(projection_9.get_uid());
        }
    }

    // Сеть готова, возвращаем её.
    return result;
}
