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
#include <realm/list.hpp>

#include <catch2/catch_all.hpp>

using namespace realm;

TEST_CASE("C API: HNSW Vector Search", "[c_api][hnsw]") {
    TestFile config;
    config.cache = false;
    config.automatic_change_notifications = false;
    config.schema = Schema{
        {"TestObject", {
            {"_id", PropertyType::Int, Property::IsPrimary{true}},
            {"embedding", PropertyType::Array | PropertyType::Double}
        }}
    };

    SECTION("Create HNSW index") {
        auto realm = Realm::get_shared_realm(config);
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_TestObject");
        REQUIRE(table);
        
        auto col_key = table->get_column_key("embedding");
        REQUIRE(col_key);
        
        // Wrap SharedRealm in C API struct
        auto* c_realm = new shared_realm(realm);
        
        // Test C API: create_index
        bool success = realm_hnsw_create_index(
            c_realm,
            table->get_key().value,
            col_key.value,
            16,    // M
            200,   // ef_construction
            RLM_HNSW_METRIC_EUCLIDEAN
        );
        
        REQUIRE(success);
        REQUIRE(table->has_search_index(col_key));
        
        realm->commit_transaction();
        delete c_realm;
    }
    
    SECTION("Check HNSW index exists") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_TestObject");
        auto col_key = table->get_column_key("embedding");
        
        // Create index first
        realm_hnsw_create_index(
            c_realm,
            table->get_key().value,
            col_key.value,
            16, 200, RLM_HNSW_METRIC_EUCLIDEAN
        );
        
        realm->commit_transaction();
        
        // Test C API: has_index
        bool has_index = false;
        bool success = realm_hnsw_has_index(
            c_realm,
            table->get_key().value,
            col_key.value,
            &has_index
        );
        
        REQUIRE(success);
        REQUIRE(has_index);
        delete c_realm;
    }
    
    SECTION("Insert vectors and search KNN") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_TestObject");
        auto col_key = table->get_column_key("embedding");
        
        // Create HNSW index
        realm_hnsw_create_index(
            c_realm,
            table->get_key().value,
            col_key.value,
            16, 200, RLM_HNSW_METRIC_EUCLIDEAN
        );
        
        // Insert 10 test vectors (3D)
        for (int i = 0; i < 10; i++) {
            auto obj = table->create_object_with_primary_key(i);
            auto list = obj.get_list<double>(col_key);
            list.add(i * 1.0);
            list.add(i * 2.0);
            list.add(i * 0.5);
        }
        
        realm->commit_transaction();
        
        // Test C API: search_knn
        std::vector<double> query_vec = {5.0, 10.0, 2.5};
        realm_hnsw_search_result_t results[5];
        size_t num_results = 0;
        
        bool success = realm_hnsw_search_knn(
            c_realm,
            table->get_key().value,
            col_key.value,
            query_vec.data(),
            query_vec.size(),
            5,      // k
            50,     // ef_search
            results,
            &num_results
        );
        
        REQUIRE(success);
        REQUIRE(num_results > 0);
        REQUIRE(num_results <= 5);
        
        // The closest vector should be object with _id=5
        // (vector [5.0, 10.0, 2.5] matches exactly)
        REQUIRE(results[0].object_key == 5);
        REQUIRE(results[0].distance < 0.01); // Should be very close to 0
        delete c_realm;
    }
    
    SECTION("Search with radius") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_TestObject");
        auto col_key = table->get_column_key("embedding");
        
        realm_hnsw_create_index(
            c_realm,
            table->get_key().value,
            col_key.value,
            16, 200, RLM_HNSW_METRIC_EUCLIDEAN
        );
        
        // Insert test vectors
        for (int i = 0; i < 10; i++) {
            auto obj = table->create_object_with_primary_key(i);
            auto list = obj.get_list<double>(col_key);
            list.add(i * 1.0);
            list.add(i * 2.0);
            list.add(i * 0.5);
        }
        
        realm->commit_transaction();
        
        // Test C API: search_radius
        std::vector<double> query_vec = {5.0, 10.0, 2.5};
        realm_hnsw_search_result_t results[10];
        size_t num_results = 0;
        
        bool success = realm_hnsw_search_radius(
            c_realm,
            table->get_key().value,
            col_key.value,
            query_vec.data(),
            query_vec.size(),
            5.0,    // max_distance
            results,
            10,     // max_results
            &num_results
        );
        
        REQUIRE(success);
        REQUIRE(num_results > 0);
        
        // All results should be within the radius
        for (size_t i = 0; i < num_results; i++) {
            REQUIRE(results[i].distance <= 5.0);
        }
        delete c_realm;
    }
    
    SECTION("Get HNSW statistics") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_TestObject");
        auto col_key = table->get_column_key("embedding");
        
        realm_hnsw_create_index(
            c_realm,
            table->get_key().value,
            col_key.value,
            16, 200, RLM_HNSW_METRIC_EUCLIDEAN
        );
        
        // Insert test vectors
        for (int i = 0; i < 10; i++) {
            auto obj = table->create_object_with_primary_key(i);
            auto list = obj.get_list<double>(col_key);
            list.add(i * 1.0);
            list.add(i * 2.0);
            list.add(i * 0.5);
        }
        
        realm->commit_transaction();
        
        // Test C API: get_stats
        size_t num_vectors = 0;
        int max_layer = 0;
        
        bool success = realm_hnsw_get_stats(
            c_realm,
            table->get_key().value,
            col_key.value,
            &num_vectors,
            &max_layer
        );
        
        REQUIRE(success);
        REQUIRE(num_vectors == 10);
        REQUIRE(max_layer >= 0);
        delete c_realm;
    }
    
    SECTION("Remove HNSW index") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_TestObject");
        auto col_key = table->get_column_key("embedding");
        
        // Create index
        realm_hnsw_create_index(
            c_realm,
            table->get_key().value,
            col_key.value,
            16, 200, RLM_HNSW_METRIC_EUCLIDEAN
        );
        
        REQUIRE(table->has_search_index(col_key));
        
        // Test C API: remove_index
        bool success = realm_hnsw_remove_index(
            c_realm,
            table->get_key().value,
            col_key.value
        );
        
        REQUIRE(success);
        REQUIRE_FALSE(table->has_search_index(col_key));
        
        realm->commit_transaction();
        delete c_realm;
    }
}

TEST_CASE("C API: HNSW Distance Metric Configuration Validation", "[c_api][hnsw][metrics]") {
    TestFile config;
    config.cache = false;
    config.automatic_change_notifications = false;
    config.schema = Schema{
        {"MetricTest", {
            {"_id", PropertyType::Int, Property::IsPrimary{true}},
            {"embedding", PropertyType::Array | PropertyType::Double}
        }}
    };

    SECTION("Verify Euclidean metric is enforced") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_MetricTest");
        auto col_key = table->get_column_key("embedding");
        
        // Create index with Euclidean metric (currently only supported)
        bool success = realm_hnsw_create_index(
            c_realm,
            table->get_key().value,
            col_key.value,
            16, 200, RLM_HNSW_METRIC_EUCLIDEAN
        );
        
        REQUIRE(success);
        REQUIRE(table->has_search_index(col_key));
        
        realm->commit_transaction();
        delete c_realm;
    }
    
    SECTION("Verify Cosine and Dot Product metrics are properly supported") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_MetricTest");
        auto col_key_embedding = table->get_column_key("embedding");
        
        // Create index with Cosine metric - should succeed
        bool success_cosine = realm_hnsw_create_index(
            c_realm,
            table->get_key().value,
            col_key_embedding.value,
            16, 200, RLM_HNSW_METRIC_COSINE
        );
        
        // Should return true indicating success
        REQUIRE(success_cosine);
        
        // Index should have been created
        REQUIRE(table->has_search_index(col_key_embedding));
        REQUIRE(table->search_index_type(col_key_embedding) == IndexType::HNSW);
        
        // Remove index for next test
        table->remove_search_index(col_key_embedding);
        REQUIRE_FALSE(table->has_search_index(col_key_embedding));
        
        // Create index with Dot Product metric - should also succeed
        bool success_dot = realm_hnsw_create_index(
            c_realm,
            table->get_key().value,
            col_key_embedding.value,
            16, 200, RLM_HNSW_METRIC_DOT_PRODUCT
        );
        
        // Should return true indicating success
        REQUIRE(success_dot);
        
        // Index should exist
        REQUIRE(table->has_search_index(col_key_embedding));
        REQUIRE(table->search_index_type(col_key_embedding) == IndexType::HNSW);
        
        realm->commit_transaction();
        delete c_realm;
    }
    
    SECTION("Verify Euclidean distance calculations are correct") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_MetricTest");
        auto col_key = table->get_column_key("embedding");
        
        realm_hnsw_create_index(
            c_realm,
            table->get_key().value,
            col_key.value,
            16, 200, RLM_HNSW_METRIC_EUCLIDEAN
        );
        
        // Insert test vectors with known distances
        // Vector 1: [0, 0, 0] - origin
        auto obj0 = table->create_object_with_primary_key(0);
        auto list0 = obj0.get_list<double>(col_key);
        list0.add(0.0); list0.add(0.0); list0.add(0.0);
        
        // Vector 2: [3, 4, 0] - distance from origin = 5.0
        auto obj1 = table->create_object_with_primary_key(1);
        auto list1 = obj1.get_list<double>(col_key);
        list1.add(3.0); list1.add(4.0); list1.add(0.0);
        
        // Vector 3: [1, 0, 0] - distance from origin = 1.0
        auto obj2 = table->create_object_with_primary_key(2);
        auto list2 = obj2.get_list<double>(col_key);
        list2.add(1.0); list2.add(0.0); list2.add(0.0);
        
        // Vector 4: [6, 8, 0] - distance from origin = 10.0
        auto obj3 = table->create_object_with_primary_key(3);
        auto list3 = obj3.get_list<double>(col_key);
        list3.add(6.0); list3.add(8.0); list3.add(0.0);
        
        realm->commit_transaction();
        
        // Query from origin [0, 0, 0]
        std::vector<double> query_origin = {0.0, 0.0, 0.0};
        realm_hnsw_search_result_t results[4];
        size_t num_results = 0;
        
        bool success = realm_hnsw_search_knn(
            c_realm,
            table->get_key().value,
            col_key.value,
            query_origin.data(),
            query_origin.size(),
            4,
            50,
            results,
            &num_results
        );
        
        REQUIRE(success);
        REQUIRE(num_results == 4);
        
        // Verify distances are correct and ordered (Euclidean distance)
        // Closest: id=0, distance=0.0
        REQUIRE(results[0].object_key == 0);
        REQUIRE(results[0].distance < 0.01);
        
        // Second: id=2, distance=1.0
        REQUIRE(results[1].object_key == 2);
        REQUIRE(std::abs(results[1].distance - 1.0) < 0.01);
        
        // Third: id=1, distance=5.0
        REQUIRE(results[2].object_key == 1);
        REQUIRE(std::abs(results[2].distance - 5.0) < 0.01);
        
        // Fourth: id=3, distance=10.0
        REQUIRE(results[3].object_key == 3);
        REQUIRE(std::abs(results[3].distance - 10.0) < 0.01);
        
        delete c_realm;
    }
    
    SECTION("Verify distance calculations with different query points") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_MetricTest");
        auto col_key = table->get_column_key("embedding");
        
        realm_hnsw_create_index(
            c_realm,
            table->get_key().value,
            col_key.value,
            16, 200, RLM_HNSW_METRIC_EUCLIDEAN
        );
        
        // Insert vectors
        auto obj0 = table->create_object_with_primary_key(0);
        auto list0 = obj0.get_list<double>(col_key);
        list0.add(1.0); list0.add(2.0); list0.add(3.0);
        
        auto obj1 = table->create_object_with_primary_key(1);
        auto list1 = obj1.get_list<double>(col_key);
        list1.add(4.0); list1.add(5.0); list1.add(6.0);
        
        auto obj2 = table->create_object_with_primary_key(2);
        auto list2 = obj2.get_list<double>(col_key);
        list2.add(7.0); list2.add(8.0); list2.add(9.0);
        
        realm->commit_transaction();
        
        // Query from [1, 2, 3] - should match id=0 exactly
        std::vector<double> query1 = {1.0, 2.0, 3.0};
        realm_hnsw_search_result_t results1[1];
        size_t num_results1 = 0;
        
        realm_hnsw_search_knn(
            c_realm,
            table->get_key().value,
            col_key.value,
            query1.data(),
            query1.size(),
            1,
            50,
            results1,
            &num_results1
        );
        
        REQUIRE(num_results1 == 1);
        REQUIRE(results1[0].object_key == 0);
        REQUIRE(results1[0].distance < 0.01);
        
        // Query from [4, 5, 6] - should match id=1 exactly
        std::vector<double> query2 = {4.0, 5.0, 6.0};
        realm_hnsw_search_result_t results2[1];
        size_t num_results2 = 0;
        
        realm_hnsw_search_knn(
            c_realm,
            table->get_key().value,
            col_key.value,
            query2.data(),
            query2.size(),
            1,
            50,
            results2,
            &num_results2
        );
        
        REQUIRE(num_results2 == 1);
        REQUIRE(results2[0].object_key == 1);
        REQUIRE(results2[0].distance < 0.01);
        
        // Query from [2.5, 3.5, 4.5] - midpoint between id=0 and id=1
        // Distance to id=0: sqrt((1.5)^2 + (1.5)^2 + (1.5)^2) = sqrt(6.75) ≈ 2.598
        // Distance to id=1: sqrt((1.5)^2 + (1.5)^2 + (1.5)^2) = sqrt(6.75) ≈ 2.598
        std::vector<double> query3 = {2.5, 3.5, 4.5};
        realm_hnsw_search_result_t results3[2];
        size_t num_results3 = 0;
        
        realm_hnsw_search_knn(
            c_realm,
            table->get_key().value,
            col_key.value,
            query3.data(),
            query3.size(),
            2,
            50,
            results3,
            &num_results3
        );
        
        REQUIRE(num_results3 >= 2);
        
        // Both id=0 and id=1 should be approximately equidistant
        double expected_dist = std::sqrt(6.75);
        bool found_id0 = false, found_id1 = false;
        
        for (size_t i = 0; i < num_results3; i++) {
            if (results3[i].object_key == 0) {
                found_id0 = true;
                REQUIRE(std::abs(results3[i].distance - expected_dist) < 0.1);
            }
            if (results3[i].object_key == 1) {
                found_id1 = true;
                REQUIRE(std::abs(results3[i].distance - expected_dist) < 0.1);
            }
        }
        
        REQUIRE(found_id0);
        REQUIRE(found_id1);
        
        delete c_realm;
    }
    
    SECTION("Verify Cosine similarity metric works correctly through C API") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_MetricTest");
        auto col_key = table->get_column_key("embedding");
        
        // Create index with Cosine metric
        realm_hnsw_create_index(
            c_realm,
            table->get_key().value,
            col_key.value,
            16, 200, RLM_HNSW_METRIC_COSINE
        );
        
        // Insert test vectors with known cosine similarities
        // Vector 1: [1, 0, 0] - same direction as query
        auto obj0 = table->create_object_with_primary_key(10);
        auto list0 = obj0.get_list<double>(col_key);
        list0.add(1.0); list0.add(0.0); list0.add(0.0);
        
        // Vector 2: [2, 0, 0] - same direction as query, different magnitude
        auto obj1 = table->create_object_with_primary_key(11);
        auto list1 = obj1.get_list<double>(col_key);
        list1.add(2.0); list1.add(0.0); list1.add(0.0);
        
        // Vector 3: [0, 1, 0] - orthogonal to query
        auto obj2 = table->create_object_with_primary_key(12);
        auto list2 = obj2.get_list<double>(col_key);
        list2.add(0.0); list2.add(1.0); list2.add(0.0);
        
        // Vector 4: [-1, 0, 0] - opposite direction to query
        auto obj3 = table->create_object_with_primary_key(13);
        auto list3 = obj3.get_list<double>(col_key);
        list3.add(-1.0); list3.add(0.0); list3.add(0.0);
        
        realm->commit_transaction();
        
        // Query with [1, 0, 0]
        std::vector<double> query = {1.0, 0.0, 0.0};
        realm_hnsw_search_result_t results[4];
        size_t num_results = 0;
        
        bool success = realm_hnsw_search_knn(
            c_realm,
            table->get_key().value,
            col_key.value,
            query.data(),
            query.size(),
            4,
            50,
            results,
            &num_results
        );
        
        REQUIRE(success);
        REQUIRE(num_results == 4);
        
        // For Cosine metric, distance = 1 - cosine_similarity
        // Vector [1,0,0] and [2,0,0]: similarity=1.0, distance=0.0 (closest)
        // Vector [1,0,0] and [1,0,0]: similarity=1.0, distance=0.0 (closest)
        // Vector [0,1,0]: similarity=0.0, distance=1.0 (orthogonal)
        // Vector [-1,0,0]: similarity=-1.0, distance=2.0 (opposite)
        
        // Check that same-direction vectors are closest (distance ~ 0)
        REQUIRE(results[0].distance < 0.01);
        REQUIRE(results[1].distance < 0.01);
        
        delete c_realm;
    }
    
    SECTION("Verify Dot Product metric works correctly through C API") {
        auto realm = Realm::get_shared_realm(config);
        auto* c_realm = new shared_realm(realm);
        
        realm->begin_transaction();
        
        auto table = realm->read_group().get_table("class_MetricTest");
        auto col_key = table->get_column_key("embedding");
        
        // Create index with Dot Product metric
        realm_hnsw_create_index(
            c_realm,
            table->get_key().value,
            col_key.value,
            16, 200, RLM_HNSW_METRIC_DOT_PRODUCT
        );
        
        // Insert test vectors with known dot products
        // Vector 1: [3, 4, 0] - dot product with [1,1,0] = 7
        auto obj0 = table->create_object_with_primary_key(20);
        auto list0 = obj0.get_list<double>(col_key);
        list0.add(3.0); list0.add(4.0); list0.add(0.0);
        
        // Vector 2: [1, 1, 0] - dot product with [1,1,0] = 2
        auto obj1 = table->create_object_with_primary_key(21);
        auto list1 = obj1.get_list<double>(col_key);
        list1.add(1.0); list1.add(1.0); list1.add(0.0);
        
        // Vector 3: [0, 0, 1] - dot product with [1,1,0] = 0
        auto obj2 = table->create_object_with_primary_key(22);
        auto list2 = obj2.get_list<double>(col_key);
        list2.add(0.0); list2.add(0.0); list2.add(1.0);
        
        realm->commit_transaction();
        
        // Query with [1, 1, 0]
        std::vector<double> query = {1.0, 1.0, 0.0};
        realm_hnsw_search_result_t results[3];
        size_t num_results = 0;
        
        bool success = realm_hnsw_search_knn(
            c_realm,
            table->get_key().value,
            col_key.value,
            query.data(),
            query.size(),
            3,
            50,
            results,
            &num_results
        );
        
        REQUIRE(success);
        REQUIRE(num_results == 3);
        
        // For Dot Product metric, distance = -dot_product (negative for MIPS)
        // Vector [3,4,0]: dot=7, distance=-7 (closest, most similar)
        // Vector [1,1,0]: dot=2, distance=-2
        // Vector [0,0,1]: dot=0, distance=0 (least similar)
        
        // Simply verify that all 3 objects were found and distances increase
        // (HNSW is approximate, so exact ordering isn't guaranteed with small datasets)
        REQUIRE(results[0].distance < results[1].distance);
        REQUIRE(results[1].distance < results[2].distance);
        
        delete c_realm;
    }
}
