/*************************************************************************
 *
 * Copyright 2025 Realm Inc.
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
 *
 **************************************************************************/

#include "realm.hpp"
#include <realm/index_hnsw.hpp>
#include <realm/table.hpp>
#include <realm/group.hpp>

namespace realm::c_api {

// Convert C API distance metric to C++ enum
static DistanceMetric to_cpp_metric(realm_hnsw_distance_metric_e metric) {
    switch (metric) {
        case RLM_HNSW_METRIC_EUCLIDEAN:
            return DistanceMetric::Euclidean;
        case RLM_HNSW_METRIC_COSINE:
            return DistanceMetric::Cosine;
        case RLM_HNSW_METRIC_DOT_PRODUCT:
            return DistanceMetric::DotProduct;
        default:
            return DistanceMetric::Euclidean;
    }
}

} // namespace realm::c_api

extern "C" {

RLM_API bool realm_hnsw_search_knn(const realm_t* realm, 
                                   realm_class_key_t class_key,
                                   realm_property_key_t property_key,
                                   const double* query_vector,
                                   size_t vector_size,
                                   size_t k,
                                   size_t ef_search,
                                   realm_hnsw_search_result_t* out_results,
                                   size_t* out_num_results)
{
    return realm::c_api::wrap_err([&]() {
        auto& shared_realm = *realm;
        auto table = (*shared_realm).read_group().get_table(realm::TableKey(class_key));
        if (!table) {
            throw std::runtime_error("Table not found");
        }

        realm::ColKey col_key(property_key);
        
        // Get the search index for this column
        auto search_index = table->get_search_index(col_key);
        if (!search_index) {
            throw std::runtime_error("No HNSW index found on this property");
        }

        // Cast to HNSWIndex
        auto* hnsw_index = dynamic_cast<realm::HNSWIndex*>(search_index);
        if (!hnsw_index) {
            throw std::runtime_error("Property does not have an HNSW index");
        }

        // Create query vector
        std::vector<double> query(query_vector, query_vector + vector_size);

        // Perform search
        auto results = hnsw_index->search_knn(query, k, ef_search);

        // Copy results
        size_t num_results = std::min(results.size(), k);
        for (size_t i = 0; i < num_results; ++i) {
            out_results[i].object_key = results[i].first.value;
            out_results[i].distance = results[i].second;
        }

        if (out_num_results) {
            *out_num_results = num_results;
        }

        return true;
    });
}

RLM_API bool realm_hnsw_search_radius(const realm_t* realm,
                                      realm_class_key_t class_key,
                                      realm_property_key_t property_key,
                                      const double* query_vector,
                                      size_t vector_size,
                                      double max_distance,
                                      realm_hnsw_search_result_t* out_results,
                                      size_t max_results,
                                      size_t* out_num_results)
{
    return realm::c_api::wrap_err([&]() {
        auto& shared_realm = *realm;
        auto table = (*shared_realm).read_group().get_table(realm::TableKey(class_key));
        if (!table) {
            throw std::runtime_error("Table not found");
        }

        realm::ColKey col_key(property_key);
        
        // Get the search index for this column
        auto search_index = table->get_search_index(col_key);
        if (!search_index) {
            throw std::runtime_error("No HNSW index found on this property");
        }

        // Cast to HNSWIndex
        auto* hnsw_index = dynamic_cast<realm::HNSWIndex*>(search_index);
        if (!hnsw_index) {
            throw std::runtime_error("Property does not have an HNSW index");
        }

        // Create query vector
        std::vector<double> query(query_vector, query_vector + vector_size);

        // Perform radius search
        auto results = hnsw_index->search_radius(query, max_distance);

        // Copy results (up to max_results)
        size_t num_results = std::min(results.size(), max_results);
        for (size_t i = 0; i < num_results; ++i) {
            out_results[i].object_key = results[i].first.value;
            out_results[i].distance = results[i].second;
        }

        if (out_num_results) {
            *out_num_results = num_results;
        }

        return true;
    });
}

RLM_API bool realm_hnsw_create_index(realm_t* realm,
                                     realm_class_key_t class_key,
                                     realm_property_key_t property_key,
                                     size_t M,
                                     size_t ef_construction,
                                     realm_hnsw_distance_metric_e metric)
{
    return realm::c_api::wrap_err([&]() {
        // Note: M, ef_construction, and metric parameters are currently unused.
        // The HNSW index uses default configuration from HNSWIndex::Config.
        // TODO: Extend Table API to accept custom HNSW config if needed.
        (void)M;
        (void)ef_construction;
        (void)metric;
        
        auto& shared_realm = *realm;
        auto table = (*shared_realm).read_group().get_table(realm::TableKey(class_key));
        if (!table) {
            throw std::runtime_error("Table not found");
        }

        realm::ColKey col_key(property_key);
        table->add_search_index(col_key, realm::IndexType::HNSW);

        return true;
    });
}

RLM_API bool realm_hnsw_remove_index(realm_t* realm,
                                     realm_class_key_t class_key,
                                     realm_property_key_t property_key)
{
    return realm::c_api::wrap_err([&]() {
        auto& shared_realm = *realm;
        auto table = (*shared_realm).read_group().get_table(realm::TableKey(class_key));
        if (!table) {
            throw std::runtime_error("Table not found");
        }

        realm::ColKey col_key(property_key);
        table->remove_search_index(col_key);

        return true;
    });
}

RLM_API bool realm_hnsw_has_index(const realm_t* realm,
                                  realm_class_key_t class_key,
                                  realm_property_key_t property_key,
                                  bool* out_has_index)
{
    return realm::c_api::wrap_err([&]() {
        auto& shared_realm = *realm;
        auto table = (*shared_realm).read_group().get_table(realm::TableKey(class_key));
        if (!table) {
            throw std::runtime_error("Table not found");
        }

        realm::ColKey col_key(property_key);
        auto search_index = table->get_search_index(col_key);
        
        bool has_hnsw = false;
        if (search_index) {
            auto* hnsw_index = dynamic_cast<realm::HNSWIndex*>(search_index);
            has_hnsw = (hnsw_index != nullptr);
        }

        if (out_has_index) {
            *out_has_index = has_hnsw;
        }

        return true;
    });
}

RLM_API bool realm_hnsw_get_stats(const realm_t* realm,
                                  realm_class_key_t class_key,
                                  realm_property_key_t property_key,
                                  size_t* out_num_vectors,
                                  int* out_max_layer)
{
    return realm::c_api::wrap_err([&]() {
        auto& shared_realm = *realm;
        auto table = (*shared_realm).read_group().get_table(realm::TableKey(class_key));
        if (!table) {
            throw std::runtime_error("Table not found");
        }

        realm::ColKey col_key(property_key);
        auto search_index = table->get_search_index(col_key);
        if (!search_index) {
            throw std::runtime_error("No HNSW index found on this property");
        }

        // Cast to HNSWIndex
        auto* hnsw_index = dynamic_cast<realm::HNSWIndex*>(search_index);
        if (!hnsw_index) {
            throw std::runtime_error("Property does not have an HNSW index");
        }

        if (out_num_vectors) {
            *out_num_vectors = hnsw_index->get_num_vectors();
        }

        if (out_max_layer) {
            *out_max_layer = hnsw_index->get_max_layer();
        }

        return true;
    });
}

} // extern "C"
