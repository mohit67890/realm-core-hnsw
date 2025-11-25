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
