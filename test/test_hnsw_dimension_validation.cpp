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

#include "testsettings.hpp"

#if defined(TEST_HNSW_REALWORLD)

#include <realm.hpp>
#include <realm/query_expression.hpp>
#include <realm/table.hpp>
#include <realm/db.hpp>
#include <realm/list.hpp>

#include "test.hpp"
#include "test_table_helper.hpp"

using namespace realm;

// Test that HNSW index enforces vector dimensions
TEST(HNSW_DimensionValidation_Basic) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto table = wt->add_table("Vectors");
    auto vector_col = table->add_column_list(type_Double, "vector");
    
    table->add_search_index(vector_col, IndexType::HNSW);
    
    // Insert first vector with 128 dimensions
    auto obj1 = table->create_object();
    auto list1 = obj1.get_list<double>(vector_col);
    for (int i = 0; i < 128; i++) {
        list1.add(i * 0.1);
    }
    
    wt->commit();
    
    // Try to insert vector with different dimension (256) - should fail
    wt = sg->start_write();
    table = wt->get_table("Vectors");
    
    auto obj2 = table->create_object();
    auto list2 = obj2.get_list<double>(vector_col);
    
    bool caught_exception = false;
    try {
        for (int i = 0; i < 256; i++) {
            list2.add(i * 0.1);
        }
        wt->commit();  // This should trigger dimension validation
    } catch (const InvalidArgument& e) {
        caught_exception = true;
        std::string msg = e.what();
        CHECK(msg.find("dimension mismatch") != std::string::npos);
        CHECK(msg.find("expected 128") != std::string::npos);
        CHECK(msg.find("got 256") != std::string::npos);
    }
    
    CHECK(caught_exception);
}

// Test that all vectors must have same dimension
TEST(HNSW_DimensionValidation_MultipleVectors) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto table = wt->add_table("Vectors");
    auto id_col = table->add_column(type_Int, "id");
    auto vector_col = table->add_column_list(type_Double, "vector");
    
    table->add_search_index(vector_col, IndexType::HNSW);
    
    const int DIMENSION = 768;
    
    // Insert 10 vectors with correct dimension
    for (int i = 0; i < 10; i++) {
        auto obj = table->create_object();
        obj.set(id_col, static_cast<int64_t>(i));
        
        auto list = obj.get_list<double>(vector_col);
        for (int j = 0; j < DIMENSION; j++) {
            list.add(i * j * 0.001);
        }
    }
    
    wt->commit();
    
    // Try to insert vector with wrong dimension
    wt = sg->start_write();
    table = wt->get_table("Vectors");
    
    auto obj = table->create_object();
    obj.set(id_col, static_cast<int64_t>(999));
    
    auto list = obj.get_list<double>(vector_col);
    
    bool caught_exception = false;
    try {
        // Wrong dimension: 512 instead of 768
        for (int j = 0; j < 512; j++) {
            list.add(j * 0.001);
        }
        wt->commit();
    } catch (const InvalidArgument& e) {
        caught_exception = true;
        std::string msg = e.what();
        CHECK(msg.find("dimension mismatch") != std::string::npos);
        CHECK(msg.find("768") != std::string::npos);
        CHECK(msg.find("512") != std::string::npos);
    }
    
    CHECK(caught_exception);
}

// Test dimension validation with empty vector
TEST(HNSW_DimensionValidation_EmptyVector) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto table = wt->add_table("Vectors");
    auto vector_col = table->add_column_list(type_Double, "vector");
    
    table->add_search_index(vector_col, IndexType::HNSW);
    
    // Insert vector with 100 dimensions
    auto obj1 = table->create_object();
    auto list1 = obj1.get_list<double>(vector_col);
    for (int i = 0; i < 100; i++) {
        list1.add(i * 0.1);
    }
    
    wt->commit();
    
    // Insert empty vector - should be allowed (not indexed)
    wt = sg->start_write();
    table = wt->get_table("Vectors");
    
    auto obj2 = table->create_object();
    // Don't add anything to the list - leave it empty
    
    wt->commit();  // Should succeed
    
    // Verify only 1 vector in index (empty vector not indexed)
    auto rt = sg->start_read();
    table = rt->get_table("Vectors");
    
    CHECK_EQUAL(table->size(), 2);  // Both objects exist
}

// Test dimension validation with single-element vector
TEST(HNSW_DimensionValidation_SingleElement) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto table = wt->add_table("Vectors");
    auto vector_col = table->add_column_list(type_Double, "vector");
    
    table->add_search_index(vector_col, IndexType::HNSW);
    
    // Insert first vector with 1 dimension
    auto obj1 = table->create_object();
    auto list1 = obj1.get_list<double>(vector_col);
    list1.add(5.0);
    
    wt->commit();
    
    // Try to insert vector with 2 dimensions - should fail
    wt = sg->start_write();
    table = wt->get_table("Vectors");
    
    auto obj2 = table->create_object();
    auto list2 = obj2.get_list<double>(vector_col);
    
    bool caught_exception = false;
    try {
        list2.add(1.0);
        list2.add(2.0);
        wt->commit();
    } catch (const InvalidArgument& e) {
        caught_exception = true;
        std::string msg = e.what();
        CHECK(msg.find("dimension mismatch") != std::string::npos);
        CHECK(msg.find("expected 1") != std::string::npos);
        CHECK(msg.find("got 2") != std::string::npos);
    }
    
    CHECK(caught_exception);
}

// Test dimension validation across transaction boundaries
TEST(HNSW_DimensionValidation_AcrossTransactions) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    const int DIM = 384;
    
    // Transaction 1: Create index and insert first vector
    {
        auto wt = sg->start_write();
        auto table = wt->add_table("Vectors");
        auto vector_col = table->add_column_list(type_Double, "vector");
        
        table->add_search_index(vector_col, IndexType::HNSW);
        
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vector_col);
        for (int i = 0; i < DIM; i++) {
            list.add(i * 0.01);
        }
        
        wt->commit();
    }
    
    // Transaction 2: Insert vector with correct dimension
    {
        auto wt = sg->start_write();
        auto table = wt->get_table("Vectors");
        auto vector_col = table->get_column_key("vector");
        
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vector_col);
        for (int i = 0; i < DIM; i++) {
            list.add(i * 0.02);
        }
        
        wt->commit();  // Should succeed
    }
    
    // Transaction 3: Try to insert vector with wrong dimension
    {
        auto wt = sg->start_write();
        auto table = wt->get_table("Vectors");
        auto vector_col = table->get_column_key("vector");
        
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vector_col);
        
        bool caught_exception = false;
        try {
            for (int i = 0; i < DIM * 2; i++) {  // Wrong dimension
                list.add(i * 0.01);
            }
            wt->commit();
        } catch (const InvalidArgument& e) {
            caught_exception = true;
            std::string msg = e.what();
            CHECK(msg.find("dimension mismatch") != std::string::npos);
        }
        
        CHECK(caught_exception);
    }
}

// Test dimension validation after index creation
TEST(HNSW_DimensionValidation_AfterIndexCreation) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto table = wt->add_table("Vectors");
    auto vector_col = table->add_column_list(type_Double, "vector");
    
    // Insert vectors BEFORE creating index
    for (int i = 0; i < 5; i++) {
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vector_col);
        for (int j = 0; j < 256; j++) {
            list.add(i * j * 0.001);
        }
    }
    
    // Now add index - should detect dimension from existing vectors
    table->add_search_index(vector_col, IndexType::HNSW);
    
    wt->commit();
    
    // Try to insert vector with different dimension
    wt = sg->start_write();
    table = wt->get_table("Vectors");
    
    auto obj = table->create_object();
    auto list = obj.get_list<double>(vector_col);
    
    bool caught_exception = false;
    try {
        for (int j = 0; j < 128; j++) {  // Wrong dimension
            list.add(j * 0.001);
        }
        wt->commit();
    } catch (const InvalidArgument& e) {
        caught_exception = true;
        std::string msg = e.what();
        CHECK(msg.find("dimension mismatch") != std::string::npos);
        CHECK(msg.find("256") != std::string::npos);
        CHECK(msg.find("128") != std::string::npos);
    }
    
    CHECK(caught_exception);
}

// Test dimension validation with update (set operation)
TEST(HNSW_DimensionValidation_Update) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto table = wt->add_table("Vectors");
    auto vector_col = table->add_column_list(type_Double, "vector");
    
    table->add_search_index(vector_col, IndexType::HNSW);
    
    // Insert first vector with 64 dimensions
    auto obj = table->create_object();
    auto list = obj.get_list<double>(vector_col);
    for (int i = 0; i < 64; i++) {
        list.add(i * 0.1);
    }
    
    ObjKey obj_key = obj.get_key();
    
    wt->commit();
    
    // Try to update with vector of different dimension
    wt = sg->start_write();
    table = wt->get_table("Vectors");
    
    auto obj_to_update = table->get_object(obj_key);
    auto list_update = obj_to_update.get_list<double>(vector_col);
    
    bool caught_exception = false;
    try {
        list_update.clear();
        for (int i = 0; i < 32; i++) {  // Wrong dimension
            list_update.add(i * 0.2);
        }
        wt->commit();
    } catch (const InvalidArgument& e) {
        caught_exception = true;
        std::string msg = e.what();
        CHECK(msg.find("dimension mismatch") != std::string::npos);
        CHECK(msg.find("expected 64") != std::string::npos);
        CHECK(msg.find("got 32") != std::string::npos);
    }
    
    CHECK(caught_exception);
}

// Test that search validates query vector dimension
TEST(HNSW_DimensionValidation_SearchQuery) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto table = wt->add_table("Vectors");
    auto vector_col = table->add_column_list(type_Double, "vector");
    
    table->add_search_index(vector_col, IndexType::HNSW);
    
    const int DIM = 128;
    
    // Insert vectors
    for (int i = 0; i < 10; i++) {
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vector_col);
        for (int j = 0; j < DIM; j++) {
            list.add(i * j * 0.01);
        }
    }
    
    wt->commit();
    
    // Search with correct dimension - should work
    auto rt = sg->start_read();
    table = rt->get_table("Vectors");
    
    std::vector<double> query_correct(DIM, 0.5);
    
    Query q = table->where();
    auto results = q.vector_search_knn(vector_col, query_correct, 5);
    
    CHECK(results.size() > 0);
    
    // Search with wrong dimension - should fail
    std::vector<double> query_wrong(DIM * 2, 0.5);
    
    bool caught_exception = false;
    try {
        Query q2 = table->where();
        auto results2 = q2.vector_search_knn(vector_col, query_wrong, 5);
    } catch (const InvalidArgument& e) {
        caught_exception = true;
        std::string msg = e.what();
        CHECK(msg.find("dimension mismatch") != std::string::npos);
    }
    
    CHECK(caught_exception);
}

#endif // TEST_HNSW_REALWORLD
