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

#include "util/test_file.hpp"
#include "util/test_utils.hpp"

#include <realm.h>
#include <realm/object-store/c_api/types.hpp>
#include <realm/object-store/shared_realm.hpp>
#include <realm/object-store/property.hpp>
#include <realm/object-store/results.hpp>
#include <realm/query.hpp>
#include <realm/list.hpp>

#include <catch2/catch_all.hpp>

using namespace realm;

TEST_CASE("C API: HNSW Comprehensive - Production Readiness", "[c_api][hnsw][comprehensive]") {
    TestFile config;
    config.cache = false;
    config.automatic_change_notifications = false;
    config.schema = Schema{
        {"Document", {
            {"_id", PropertyType::Int, Property::IsPrimary{true}},
            {"embedding", PropertyType::Array | PropertyType::Double},
            {"category", PropertyType::String | PropertyType::Nullable},
            {"score", PropertyType::Double}
        }}
    };

    SECTION("Insert and Delete Operations - Index Updates") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_Document");
        auto embed_col = table->get_column_key("embedding");
        
        // Create HNSW index
        realm_hnsw_create_index(c_realm, table->get_key().value, embed_col.value,
                               16, 200, RLM_HNSW_METRIC_EUCLIDEAN);
        
        // Insert initial vectors
        std::vector<ObjKey> obj_keys;
        for (int i = 0; i < 10; i++) {
            auto obj = table->create_object_with_primary_key(i);
            obj_keys.push_back(obj.get_key());
            auto list = obj.get_list<double>(embed_col);
            list.add(i * 1.0);
            list.add(i * 2.0);
            list.add(i * 0.5);
        }
        
        realm->commit_transaction();
        
        // Verify initial count
        size_t num_vectors = 0;
        realm_hnsw_get_stats(c_realm, table->get_key().value, embed_col.value, 
                            &num_vectors, nullptr);
        REQUIRE(num_vectors == 10);
        
        // Delete some objects
        realm->begin_transaction();
        for (int i = 0; i < 3; i++) {
            table->remove_object(obj_keys[i]);
        }
        realm->commit_transaction();
        
        // Verify count after deletion
        realm_hnsw_get_stats(c_realm, table->get_key().value, embed_col.value,
                            &num_vectors, nullptr);
        REQUIRE(num_vectors == 7);
        
        // Search should only find remaining objects
        std::vector<double> query_vec = {5.0, 10.0, 2.5};
        realm_hnsw_search_result_t results[10];
        size_t num_results = 0;
        
        bool success = realm_hnsw_search_knn(c_realm, table->get_key().value, embed_col.value,
                                            query_vec.data(), query_vec.size(),
                                            10, 50, results, &num_results);
        
        REQUIRE(success);
        REQUIRE(num_results == 7);
        
        // Verify no deleted objects in results
        for (size_t i = 0; i < num_results; i++) {
            int64_t id = results[i].object_key;
            REQUIRE(id >= 3); // All deleted IDs were < 3
        }
        
        delete c_realm;
    }
    
    SECTION("Update Operations - Vector Modification") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_Document");
        auto embed_col = table->get_column_key("embedding");
        
        realm_hnsw_create_index(c_realm, table->get_key().value, embed_col.value,
                               16, 200, RLM_HNSW_METRIC_EUCLIDEAN);
        
        // Insert object with initial vector
        auto obj = table->create_object_with_primary_key(1);
        auto list = obj.get_list<double>(embed_col);
        list.add(1.0);
        list.add(2.0);
        list.add(3.0);
        
        realm->commit_transaction();
        
        // Search for initial vector
        std::vector<double> query1 = {1.0, 2.0, 3.0};
        realm_hnsw_search_result_t results[1];
        size_t num_results = 0;
        
        realm_hnsw_search_knn(c_realm, table->get_key().value, embed_col.value,
                             query1.data(), query1.size(), 1, 50, results, &num_results);
        
        REQUIRE(num_results == 1);
        REQUIRE(results[0].distance < 0.01); // Should be exact match
        
        // Update the vector
        realm->begin_transaction();
        obj = table->get_object(obj.get_key());
        list = obj.get_list<double>(embed_col);
        list.clear();
        list.add(10.0);
        list.add(20.0);
        list.add(30.0);
        realm->commit_transaction();
        
        // Search with new vector should find it
        std::vector<double> query2 = {10.0, 20.0, 30.0};
        realm_hnsw_search_knn(c_realm, table->get_key().value, embed_col.value,
                             query2.data(), query2.size(), 1, 50, results, &num_results);
        
        REQUIRE(num_results == 1);
        REQUIRE(results[0].distance < 0.01);
        
        // Search with old vector should not find exact match
        realm_hnsw_search_knn(c_realm, table->get_key().value, embed_col.value,
                             query1.data(), query1.size(), 1, 50, results, &num_results);
        
        REQUIRE(num_results == 1);
        REQUIRE(results[0].distance > 1.0); // Should be far from old vector
        
        delete c_realm;
    }
    
    SECTION("Filtered Search - Integration with Query") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_Document");
        auto embed_col = table->get_column_key("embedding");
        auto cat_col = table->get_column_key("category");
        auto score_col = table->get_column_key("score");
        
        realm_hnsw_create_index(c_realm, table->get_key().value, embed_col.value,
                               16, 200, RLM_HNSW_METRIC_EUCLIDEAN);
        
        // Insert vectors with different categories
        for (int i = 0; i < 20; i++) {
            auto obj = table->create_object_with_primary_key(i);
            auto list = obj.get_list<double>(embed_col);
            list.add(i * 1.0);
            list.add(i * 2.0);
            list.add(i * 0.5);
            obj.set(cat_col, (i < 10) ? StringData("CategoryA") : StringData("CategoryB"));
            obj.set(score_col, i * 0.5);
        }
        
        realm->commit_transaction();
        
        // Note: C API doesn't directly expose filtered vector search
        // In Dart, this will be done via Query.vectorSearchKnn() on filtered results
        // Here we verify the index works and can be queried from filtered table views
        
        // Get all vectors from CategoryA using standard query
        Query q = table->where().equal(cat_col, StringData("CategoryA"));
        TableView tv = q.find_all();
        REQUIRE(tv.size() == 10);
        
        // Verify vector search works on full table
        std::vector<double> query_vec = {5.0, 10.0, 2.5};
        realm_hnsw_search_result_t results[5];
        size_t num_results = 0;
        
        bool success = realm_hnsw_search_knn(c_realm, table->get_key().value, embed_col.value,
                                            query_vec.data(), query_vec.size(),
                                            5, 50, results, &num_results);
        
        REQUIRE(success);
        REQUIRE(num_results == 5);
        
        // Closest match should be ID=5 (vector [5.0, 10.0, 2.5])
        REQUIRE(results[0].object_key == 5);
        
        delete c_realm;
    }
    
    SECTION("Edge Cases - Empty Index and Single Vector") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_Document");
        auto embed_col = table->get_column_key("embedding");
        
        realm_hnsw_create_index(c_realm, table->get_key().value, embed_col.value,
                               16, 200, RLM_HNSW_METRIC_EUCLIDEAN);
        
        realm->commit_transaction();
        
        // Search on empty index
        std::vector<double> query_vec = {1.0, 2.0, 3.0};
        realm_hnsw_search_result_t results[10];
        size_t num_results = 0;
        
        bool success = realm_hnsw_search_knn(c_realm, table->get_key().value, embed_col.value,
                                            query_vec.data(), query_vec.size(),
                                            5, 50, results, &num_results);
        
        REQUIRE(success);
        REQUIRE(num_results == 0);
        
        // Add single vector
        realm->begin_transaction();
        auto obj = table->create_object_with_primary_key(1);
        auto list = obj.get_list<double>(embed_col);
        list.add(1.0);
        list.add(2.0);
        list.add(3.0);
        realm->commit_transaction();
        
        // Search with k > num_vectors
        success = realm_hnsw_search_knn(c_realm, table->get_key().value, embed_col.value,
                                       query_vec.data(), query_vec.size(),
                                       10, 50, results, &num_results);
        
        REQUIRE(success);
        REQUIRE(num_results == 1);
        
        delete c_realm;
    }
    
    SECTION("Error Handling - Non-Indexed Column") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_Document");
        auto embed_col = table->get_column_key("embedding");
        auto cat_col = table->get_column_key("category");
        
        realm_hnsw_create_index(c_realm, table->get_key().value, embed_col.value,
                               16, 200, RLM_HNSW_METRIC_EUCLIDEAN);
        
        realm->commit_transaction();
        
        // Test: Has index returns false on non-indexed column
        bool has_index = true;
        bool success = realm_hnsw_has_index(c_realm, table->get_key().value,
                                           cat_col.value, &has_index);
        
        REQUIRE(success);
        REQUIRE_FALSE(has_index);
        
        // Test: Has index returns true on indexed column
        success = realm_hnsw_has_index(c_realm, table->get_key().value,
                                      embed_col.value, &has_index);
        
        REQUIRE(success);
        REQUIRE(has_index);
        
        delete c_realm;
    }
    
    SECTION("High Dimensional Vectors - 128D") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_Document");
        auto embed_col = table->get_column_key("embedding");
        
        realm_hnsw_create_index(c_realm, table->get_key().value, embed_col.value,
                               16, 200, RLM_HNSW_METRIC_EUCLIDEAN);
        
        // Insert 128-dimensional vectors
        const int dim = 128;
        for (int i = 0; i < 20; i++) {
            auto obj = table->create_object_with_primary_key(i);
            auto list = obj.get_list<double>(embed_col);
            for (int d = 0; d < dim; d++) {
                list.add(std::sin(i + d * 0.1));
            }
        }
        
        realm->commit_transaction();
        
        // Search with 128D query
        std::vector<double> query_vec;
        for (int d = 0; d < dim; d++) {
            query_vec.push_back(std::sin(10 + d * 0.1));
        }
        
        realm_hnsw_search_result_t results[5];
        size_t num_results = 0;
        
        bool success = realm_hnsw_search_knn(c_realm, table->get_key().value, embed_col.value,
                                            query_vec.data(), query_vec.size(),
                                            5, 50, results, &num_results);
        
        REQUIRE(success);
        REQUIRE(num_results == 5);
        
        // Closest should be ID=10
        REQUIRE(results[0].object_key == 10);
        REQUIRE(results[0].distance < 0.01);
        
        delete c_realm;
    }
    
    SECTION("Transaction Rollback - Index Consistency") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_Document");
        auto embed_col = table->get_column_key("embedding");
        
        realm_hnsw_create_index(c_realm, table->get_key().value, embed_col.value,
                               16, 200, RLM_HNSW_METRIC_EUCLIDEAN);
        
        // Insert initial vectors
        for (int i = 0; i < 5; i++) {
            auto obj = table->create_object_with_primary_key(i);
            auto list = obj.get_list<double>(embed_col);
            list.add(i * 1.0);
            list.add(i * 2.0);
        }
        
        realm->commit_transaction();
        
        // Verify initial state
        size_t num_vectors = 0;
        realm_hnsw_get_stats(c_realm, table->get_key().value, embed_col.value,
                            &num_vectors, nullptr);
        REQUIRE(num_vectors == 5);
        
        // Start transaction and add more
        realm->begin_transaction();
        for (int i = 5; i < 10; i++) {
            auto obj = table->create_object_with_primary_key(i);
            auto list = obj.get_list<double>(embed_col);
            list.add(i * 1.0);
            list.add(i * 2.0);
        }
        
        // Rollback
        realm->cancel_transaction();
        
        // Verify index wasn't corrupted by rollback
        realm_hnsw_get_stats(c_realm, table->get_key().value, embed_col.value,
                            &num_vectors, nullptr);
        REQUIRE(num_vectors == 5); // Should still be 5
        
        // Search should still work
        std::vector<double> query_vec = {2.0, 4.0};
        realm_hnsw_search_result_t results[5];
        size_t num_results = 0;
        
        bool success = realm_hnsw_search_knn(c_realm, table->get_key().value, embed_col.value,
                                            query_vec.data(), query_vec.size(),
                                            5, 50, results, &num_results);
        
        REQUIRE(success);
        REQUIRE(num_results == 5);
        
        delete c_realm;
    }
    
    SECTION("Radius Search - Distance Threshold") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_Document");
        auto embed_col = table->get_column_key("embedding");
        
        realm_hnsw_create_index(c_realm, table->get_key().value, embed_col.value,
                               16, 200, RLM_HNSW_METRIC_EUCLIDEAN);
        
        // Insert vectors at known distances
        for (int i = 0; i < 10; i++) {
            auto obj = table->create_object_with_primary_key(i);
            auto list = obj.get_list<double>(embed_col);
            list.add(i * 1.0); // Distance from origin increases linearly
            list.add(0.0);
        }
        
        realm->commit_transaction();
        
        // Search with radius 3.5 from origin
        std::vector<double> query_vec = {0.0, 0.0};
        realm_hnsw_search_result_t results[10];
        size_t num_results = 0;
        
        bool success = realm_hnsw_search_radius(c_realm, table->get_key().value, embed_col.value,
                                               query_vec.data(), query_vec.size(),
                                               3.5, results, 10, &num_results);
        
        REQUIRE(success);
        REQUIRE(num_results <= 4); // Should find IDs 0,1,2,3 (distances 0,1,2,3)
        
        // Verify all are within radius
        for (size_t i = 0; i < num_results; i++) {
            REQUIRE(results[i].distance <= 3.5);
        }
        
        delete c_realm;
    }
}
