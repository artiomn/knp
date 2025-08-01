/**
 * @file classification.h
 * @brief Evaluation of how good model performs by inference results
 * @kaspersky_support D. Postnikov
 * @date 16.07.2025
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

#pragma once

#include <knp/core/impexp.h>
#include <knp/core/messaging/messaging.h>
#include <knp/framework/data_processing/classification/image.h>

#include <vector>


namespace knp::framework::inference_evaluation::classification
{

/**
 * @brief Processed inference result for single class.
 */
class KNP_DECLSPEC InferenceResultForClass
{
public:
    /**
     * @brief Get correctly predicted.
     * @ret Correctly predicted.
     */
    [[nodiscard]] size_t get_correctly_predicted() const { return correctly_predicted_; }

    /**
     * @brief Get incorrectly predicted.
     * @ret Incorrectly predicted.
     */
    [[nodiscard]] size_t get_incorrectly_predicted() const { return incorrectly_predicted_; }

    /**
     * @brief Get how much times prediction happened with no votes.
     * @details Prediction with no votes can happen when model did no votes, so we cant decide what model actually.
     * predicted. So we use first possible class, and increment no votes counter.
     * @ret Amount of times prediction happened with no votes.
     */
    [[nodiscard]] size_t get_no_votes() const { return no_votes_; }

    /**
     * @brief Get total amount of predictions.
     * @ret Total amount of predictions.
     */
    [[nodiscard]] size_t get_total() const { return correctly_predicted_ + incorrectly_predicted_ + no_votes_; }

    /**
     * @detail A class to process inference results.
     */
    class KNP_DECLSPEC InferenceResultsProcessor
    {
    public:
        /**
         * @brief Process inference results. Suited for classification models.
         * @param spikes All spikes from inference.
         * @param dataset Dataset.
         * @return processed inference results for each class.
         */
        void process_inference_results(
            const std::vector<knp::core::messaging::SpikeMessage> &spikes,
            const knp::framework::data_processing::classification::Dataset &dataset);

        /**
         * @brief Put inference results for each class to a stream in form of csv.
         * @param results_stream stream for output.
         */
        void write_inference_results_to_stream_as_csv(std::ostream &results_stream);

        /**
         * @brief Get inference results.
         * @ret Inference results.
         */
        [[nodiscard]] auto const &get_inference_results() const { return inference_results_; }

    private:
        /**
         * @brief Processed inference results.
         */
        std::vector<InferenceResultForClass> inference_results_;

        /**
         * @brief An internal class to help with evaluation.
         */
        class EvaluationHelper;
    };

private:
    /**
     * @brief Amount of correct model predictions.
     */
    size_t correctly_predicted_ = 0;

    /**
     * @brief Amount of incorrect model predictions.
     */
    size_t incorrectly_predicted_ = 0;

    /**
     * @brief Amount of model predictions when this class won, without any votes.
     */
    size_t no_votes_ = 0;
};

}  //namespace knp::framework::inference_evaluation::classification
