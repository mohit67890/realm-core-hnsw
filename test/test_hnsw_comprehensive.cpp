/*************************************************************************
 *
 * Copyright 2024 Realm Inc.
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
#ifdef TEST_HNSW_COMPREHENSIVE

#include <realm.hpp>
#include <realm/query_expression.hpp>
#include <realm/table.hpp>
#include <realm/db.hpp>
#include <realm/list.hpp>

#include "test.hpp"
#include "test_table_helper.hpp"

using namespace realm;
using namespace realm::util;
using namespace realm::test_util;

// ===========================
// EDGE CASE TESTS
// ===========================

TEST(HNSW_EdgeCase_EmptyVector)
{
    // Test inserting an object with empty vector - should be ignored
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Vectors");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Insert object with empty vector
    auto obj1 = table->create_object();
    auto list1 = obj1.get_list<double>(vec_col);
    // Don't add any elements - empty list
    
    // Insert object with valid vector
    auto obj2 = table->create_object();
    auto list2 = obj2.get_list<double>(vec_col);
    list2.add(1.0);
    list2.add(2.0);
    list2.add(3.0);
    
    tr->commit();
    
    // Search should only find the valid vector
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Vectors");
    Query q = table2->where();
    std::vector<double> query_vec = {1.1, 2.1, 3.1};
    
    auto results = q.vector_search_knn(vec_col, query_vec, 10);
    CHECK_EQUAL(results.size(), 1);  // Only the valid vector
    CHECK_EQUAL(results.get_key(0), obj2.get_key());
}

TEST(HNSW_EdgeCase_SingleDimension)
{
    // Test 1D vectors
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Vectors");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    for (int i = 0; i < 10; i++) {
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 1.0);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Vectors");
    Query q = table2->where();
    std::vector<double> query_vec = {5.5};
    
    auto results = q.vector_search_knn(vec_col, query_vec, 3);
    CHECK_EQUAL(results.size(), 3);
}

TEST(HNSW_EdgeCase_IdenticalVectors)
{
    // Test multiple identical vectors
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Vectors");
    auto id_col = table->add_column(type_Int, "id");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Insert 5 identical vectors
    for (int i = 0; i < 5; i++) {
        auto obj = table->create_object();
        obj.set(id_col, int64_t(i));
        auto list = obj.get_list<double>(vec_col);
        list.add(1.0);
        list.add(2.0);
        list.add(3.0);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Vectors");
    Query q = table2->where();
    std::vector<double> query_vec = {1.0, 2.0, 3.0};
    
    auto results = q.vector_search_knn(vec_col, query_vec, 3);
    CHECK_EQUAL(results.size(), 3);  // Should return 3 of the 5 identical vectors
}

TEST(HNSW_EdgeCase_NegativeValues)
{
    // Test vectors with negative values
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Vectors");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    for (int i = -5; i < 5; i++) {
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 1.0);
        list.add(i * -2.0);
        list.add(i * 0.5);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Vectors");
    Query q = table2->where();
    std::vector<double> query_vec = {-1.0, 2.0, -0.5};
    
    auto results = q.vector_search_knn(vec_col, query_vec, 5);
    CHECK_EQUAL(results.size(), 5);
}

TEST(HNSW_EdgeCase_LargeValues)
{
    // Test vectors with very large values
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Vectors");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    for (int i = 0; i < 10; i++) {
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 1e6);
        list.add(i * 1e7);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Vectors");
    Query q = table2->where();
    std::vector<double> query_vec = {5e6, 5e7};
    
    auto results = q.vector_search_knn(vec_col, query_vec, 3);
    CHECK_EQUAL(results.size(), 3);
}

// ===========================
// FILTER COMBINATION TESTS
// ===========================

TEST(HNSW_Filter_AND_Conditions)
{
    // Test multiple AND conditions together
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Products");
    auto name_col = table->add_column(type_String, "name");
    auto price_col = table->add_column(type_Double, "price");
    auto category_col = table->add_column(type_String, "category");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Insert test data
    for (int i = 0; i < 20; i++) {
        auto obj = table->create_object();
        obj.set(name_col, util::format("Product%1", i));
        obj.set(price_col, 10.0 + i * 5.0);
        obj.set(category_col, (i < 10) ? "Electronics" : "Books");
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 0.1);
        list.add(i * 0.2);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Products");
    
    // Filter: category == "Electronics" AND price >= 20 AND price <= 50
    Query q = table2->where()
        .equal(category_col, "Electronics")
        .greater_equal(price_col, 20.0)
        .less_equal(price_col, 50.0);
    
    std::vector<double> query_vec = {0.5, 1.0};
    auto results = q.vector_search_knn(vec_col, query_vec, 10);
    
    // Verify all results match ALL conditions
    CHECK(results.size() > 0);
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = table2->get_object(key);
        CHECK_EQUAL(obj.get<String>(category_col), "Electronics");
        double price = obj.get<double>(price_col);
        CHECK(price >= 20.0);
        CHECK(price <= 50.0);
    }
}

TEST(HNSW_Filter_OR_Conditions)
{
    // Test OR conditions (using group_begin/group_end)
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Items");
    auto tag_col = table->add_column(type_String, "tag");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    std::vector<std::string> tags = {"A", "B", "C", "D"};
    for (int i = 0; i < 20; i++) {
        auto obj = table->create_object();
        obj.set(tag_col, tags[i % 4]);
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 0.5);
        list.add(i * 1.0);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Items");
    
    // Filter: tag == "A" OR tag == "B"
    Query q = table2->where();
    q.group();
    q.equal(tag_col, "A");
    q.Or();
    q.equal(tag_col, "B");
    q.end_group();
    
    std::vector<double> query_vec = {5.0, 10.0};
    auto results = q.vector_search_knn(vec_col, query_vec, 15);
    
    // Verify all results match one of the OR conditions
    CHECK(results.size() > 0);
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = table2->get_object(key);
        std::string tag = obj.get<String>(tag_col);
        CHECK(tag == "A" || tag == "B");
    }
}

TEST(HNSW_Filter_NOT_Condition)
{
    // Test NOT condition
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Items");
    auto status_col = table->add_column(type_String, "status");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    std::vector<std::string> statuses = {"active", "inactive", "deleted"};
    for (int i = 0; i < 30; i++) {
        auto obj = table->create_object();
        obj.set(status_col, statuses[i % 3]);
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 0.3);
        list.add(i * 0.6);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Items");
    
    // Filter: status != "deleted"
    Query q = table2->where().not_equal(status_col, "deleted");
    
    std::vector<double> query_vec = {5.0, 10.0};
    auto results = q.vector_search_knn(vec_col, query_vec, 20);
    
    // Verify no results have status == "deleted"
    CHECK(results.size() > 0);
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = table2->get_object(key);
        CHECK_NOT_EQUAL(obj.get<String>(status_col), "deleted");
    }
}

TEST(HNSW_Filter_ComplexNested)
{
    // Test complex nested conditions: (A AND B) OR (C AND D)
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Records");
    auto type_col = table->add_column(type_String, "type");
    auto priority_col = table->add_column(type_Int, "priority");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    for (int i = 0; i < 40; i++) {
        auto obj = table->create_object();
        obj.set(type_col, (i % 2 == 0) ? "urgent" : "normal");
        obj.set(priority_col, int64_t(i % 5));
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 0.25);
        list.add(i * 0.5);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Records");
    
    // Filter: (type == "urgent" AND priority >= 3) OR (type == "normal" AND priority == 0)
    Query q = table2->where();
    q.group();
    q.group();
    q.equal(type_col, "urgent").greater_equal(priority_col, 3);
    q.end_group();
    q.Or();
    q.group();
    q.equal(type_col, "normal").equal(priority_col, 0);
    q.end_group();
    q.end_group();
    
    std::vector<double> query_vec = {5.0, 10.0};
    auto results = q.vector_search_knn(vec_col, query_vec, 15);
    
    CHECK(results.size() > 0);
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = table2->get_object(key);
        std::string type = obj.get<String>(type_col);
        int64_t priority = obj.get<int64_t>(priority_col);
        
        bool matches = (type == "urgent" && priority >= 3) || 
                      (type == "normal" && priority == 0);
        CHECK(matches);
    }
}

TEST(HNSW_Filter_NoResults)
{
    // Test filter that matches no vectors
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Items");
    auto value_col = table->add_column(type_Int, "value");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    for (int i = 0; i < 20; i++) {
        auto obj = table->create_object();
        obj.set(value_col, int64_t(i));
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 1.0);
        list.add(i * 2.0);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Items");
    
    // Filter that matches nothing
    Query q = table2->where().greater(value_col, 100);
    
    std::vector<double> query_vec = {10.0, 20.0};
    auto results = q.vector_search_knn(vec_col, query_vec, 10);
    
    CHECK_EQUAL(results.size(), 0);
}

// ===========================
// PERSISTENCE & TRANSACTION TESTS
// ===========================

TEST(HNSW_MultipleCommits)
{
    // Test multiple commits with incremental insertions
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    {
        auto tr = db->start_write();
        TableRef table = tr->add_table("Vectors");
        auto vec_col = table->add_column_list(type_Double, "embedding");
        table->add_search_index(vec_col, IndexType::HNSW);
        tr->commit();
    }
    
    // First batch
    {
        auto tr = db->start_write();
        TableRef table = tr->get_table("Vectors");
        auto vec_col = table->get_column_key("embedding");
        for (int i = 0; i < 10; i++) {
            auto obj = table->create_object();
            auto list = obj.get_list<double>(vec_col);
            list.add(i * 1.0);
            list.add(i * 2.0);
        }
        tr->commit();
    }
    
    // Second batch
    {
        auto tr = db->start_write();
        TableRef table = tr->get_table("Vectors");
        auto vec_col = table->get_column_key("embedding");
        for (int i = 10; i < 20; i++) {
            auto obj = table->create_object();
            auto list = obj.get_list<double>(vec_col);
            list.add(i * 1.0);
            list.add(i * 2.0);
        }
        tr->commit();
    }
    
    // Third batch
    {
        auto tr = db->start_write();
        TableRef table = tr->get_table("Vectors");
        auto vec_col = table->get_column_key("embedding");
        for (int i = 20; i < 30; i++) {
            auto obj = table->create_object();
            auto list = obj.get_list<double>(vec_col);
            list.add(i * 1.0);
            list.add(i * 2.0);
        }
        tr->commit();
    }
    
    // Verify all 30 vectors are searchable
    {
        auto tr = db->start_read();
        TableRef table = tr->get_table("Vectors");
        auto vec_col = table->get_column_key("embedding");
        Query q = table->where();
        std::vector<double> query_vec = {15.0, 30.0};
        
        auto results = q.vector_search_knn(vec_col, query_vec, 30);
        CHECK_EQUAL(results.size(), 30);
    }
}

TEST(HNSW_DeleteAndReinsert)
{
    // Test deleting objects and reinserting new ones
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    ObjKey key1, key2, key3;
    
    {
        auto tr = db->start_write();
        TableRef table = tr->add_table("Vectors");
        auto id_col = table->add_column(type_Int, "id");
        auto vec_col = table->add_column_list(type_Double, "embedding");
        table->add_search_index(vec_col, IndexType::HNSW);
        
        auto obj1 = table->create_object();
        obj1.set(id_col, 1);
        auto list1 = obj1.get_list<double>(vec_col);
        list1.add(1.0);
        list1.add(2.0);
        key1 = obj1.get_key();
        
        auto obj2 = table->create_object();
        obj2.set(id_col, 2);
        auto list2 = obj2.get_list<double>(vec_col);
        list2.add(3.0);
        list2.add(4.0);
        key2 = obj2.get_key();
        
        auto obj3 = table->create_object();
        obj3.set(id_col, 3);
        auto list3 = obj3.get_list<double>(vec_col);
        list3.add(5.0);
        list3.add(6.0);
        key3 = obj3.get_key();
        
        tr->commit();
    }
    
    // Delete one object
    {
        auto tr = db->start_write();
        TableRef table = tr->get_table("Vectors");
        table->remove_object(key2);
        tr->commit();
    }
    
    // Verify only 2 vectors remain
    {
        auto tr = db->start_read();
        TableRef table = tr->get_table("Vectors");
        auto vec_col = table->get_column_key("embedding");
        Query q = table->where();
        std::vector<double> query_vec = {3.0, 4.0};
        
        auto results = q.vector_search_knn(vec_col, query_vec, 10);
        CHECK_EQUAL(results.size(), 2);
    }
    
    // Add new objects
    {
        auto tr = db->start_write();
        TableRef table = tr->get_table("Vectors");
        auto id_col = table->get_column_key("id");
        auto vec_col = table->get_column_key("embedding");
        
        for (int i = 4; i <= 10; i++) {
            auto obj = table->create_object();
            obj.set(id_col, i);
            auto list = obj.get_list<double>(vec_col);
            list.add(i * 1.0);
            list.add(i * 2.0);
        }
        tr->commit();
    }
    
    // Verify we now have 9 vectors total (2 original + 7 new)
    {
        auto tr = db->start_read();
        TableRef table = tr->get_table("Vectors");
        auto vec_col = table->get_column_key("embedding");
        Query q = table->where();
        std::vector<double> query_vec = {5.0, 10.0};
        
        auto results = q.vector_search_knn(vec_col, query_vec, 20);
        CHECK_EQUAL(results.size(), 9);
    }
}

TEST(HNSW_UpdateVector)
{
    // Test updating vector values within a List
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    ObjKey obj_key;
    
    {
        auto tr = db->start_write();
        TableRef table = tr->add_table("Vectors");
        auto vec_col = table->add_column_list(type_Double, "embedding");
        table->add_search_index(vec_col, IndexType::HNSW);
        
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vec_col);
        list.add(1.0);
        list.add(2.0);
        list.add(3.0);
        obj_key = obj.get_key();
        
        tr->commit();
    }
    
    // Update the vector
    {
        auto tr = db->start_write();
        TableRef table = tr->get_table("Vectors");
        auto vec_col = table->get_column_key("embedding");
        
        auto obj = table->get_object(obj_key);
        auto list = obj.get_list<double>(vec_col);
        list.set(0, 10.0);
        list.set(1, 20.0);
        list.set(2, 30.0);
        
        tr->commit();
    }
    
    // Search should find updated vector
    {
        auto tr = db->start_read();
        TableRef table = tr->get_table("Vectors");
        auto vec_col = table->get_column_key("embedding");
        Query q = table->where();
        std::vector<double> query_vec = {11.0, 21.0, 31.0};
        
        auto results = q.vector_search_knn(vec_col, query_vec, 5);
        CHECK_EQUAL(results.size(), 1);
        CHECK_EQUAL(results.get_key(0), obj_key);
    }
}

// ===========================
// PERFORMANCE & SCALE TESTS
// ===========================

TEST(HNSW_Scale_ManyVectors)
{
    // Test with larger dataset
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Vectors");
    auto id_col = table->add_column(type_Int, "id");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Insert 5000 vectors
    for (int i = 0; i < 5000; i++) {
        auto obj = table->create_object();
        obj.set(id_col, int64_t(i));
        auto list = obj.get_list<double>(vec_col);
        for (int j = 0; j < 10; j++) {
            list.add(sin(i * 0.01 + j * 0.1));
        }
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Vectors");
    Query q = table2->where();
    
    std::vector<double> query_vec;
    for (int i = 0; i < 10; i++) {
        query_vec.push_back(sin(i * 0.1));
    }
    
    auto results = q.vector_search_knn(vec_col, query_vec, 50);
    CHECK_EQUAL(results.size(), 50);
}

TEST(HNSW_Scale_HighDimensional_256D)
{
    // Test very high dimensional vectors (256D)
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Embeddings");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    for (int i = 0; i < 100; i++) {
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vec_col);
        for (int d = 0; d < 256; d++) {
            list.add(sin(i * 0.05 + d * 0.02));
        }
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Embeddings");
    Query q = table2->where();
    
    std::vector<double> query_vec;
    for (int d = 0; d < 256; d++) {
        query_vec.push_back(sin(50 * 0.05 + d * 0.02));
    }
    
    auto results = q.vector_search_knn(vec_col, query_vec, 10);
    CHECK_EQUAL(results.size(), 10);
}

TEST(HNSW_Filter_LowSelectivity)
{
    // Test filter that matches most vectors (low selectivity)
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Items");
    auto active_col = table->add_column(type_Bool, "active");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // 95% active, 5% inactive
    for (int i = 0; i < 1000; i++) {
        auto obj = table->create_object();
        obj.set(active_col, i < 950);
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 0.1);
        list.add(i * 0.2);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Items");
    Query q = table2->where().equal(active_col, true);
    
    std::vector<double> query_vec = {50.0, 100.0};
    auto results = q.vector_search_knn(vec_col, query_vec, 100);
    
    CHECK_EQUAL(results.size(), 100);
    // Verify all are active
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = table2->get_object(key);
        CHECK(obj.get<bool>(active_col));
    }
}

TEST(HNSW_Filter_HighSelectivity)
{
    // Test filter that matches few vectors (high selectivity)
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Items");
    auto premium_col = table->add_column(type_Bool, "premium");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Only 2% premium
    for (int i = 0; i < 1000; i++) {
        auto obj = table->create_object();
        obj.set(premium_col, i % 50 == 0);  // Every 50th is premium
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 0.1);
        list.add(i * 0.2);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Items");
    Query q = table2->where().equal(premium_col, true);
    
    std::vector<double> query_vec = {50.0, 100.0};
    auto results = q.vector_search_knn(vec_col, query_vec, 30);
    
    // Should return <= 20 results (total premium items)
    CHECK(results.size() <= 20);
    // Verify all are premium
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = table2->get_object(key);
        CHECK(obj.get<bool>(premium_col));
    }
}

// ===========================
// RADIUS SEARCH TESTS
// ===========================

TEST(HNSW_Radius_ExactDistance)
{
    // Test radius search with exact distance threshold
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Points");
    auto vec_col = table->add_column_list(type_Double, "coords");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Insert points on a grid
    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 10; y++) {
            auto obj = table->create_object();
            auto list = obj.get_list<double>(vec_col);
            list.add(x * 1.0);
            list.add(y * 1.0);
        }
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Points");
    Query q = table2->where();
    
    // Search from origin with radius 5.0
    std::vector<double> query_vec = {0.0, 0.0};
    auto results = q.vector_search_radius(vec_col, query_vec, 5.0);
    
    // Should find points within distance 5.0
    CHECK(results.size() > 0);
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = table2->get_object(key);
        auto list = obj.get_list<double>(vec_col);
        double x = list.get(0);
        double y = list.get(1);
        double dist = sqrt(x * x + y * y);
        CHECK(dist <= 5.0);
    }
}

TEST(HNSW_Radius_WithFilter)
{
    // Test radius search combined with filters
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Points");
    auto category_col = table->add_column(type_String, "category");
    auto vec_col = table->add_column_list(type_Double, "coords");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    for (int i = 0; i < 50; i++) {
        auto obj = table->create_object();
        obj.set(category_col, (i % 2 == 0) ? "A" : "B");
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 0.5);
        list.add(i * 1.0);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Points");
    Query q = table2->where().equal(category_col, "A");
    
    std::vector<double> query_vec = {10.0, 20.0};
    auto results = q.vector_search_radius(vec_col, query_vec, 15.0);
    
    // Verify all results are category A and within distance
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = table2->get_object(key);
        CHECK_EQUAL(obj.get<String>(category_col), "A");
    }
}

TEST(HNSW_Radius_VerySmall)
{
    // Test with very small radius
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Points");
    auto vec_col = table->add_column_list(type_Double, "coords");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    for (int i = 0; i < 20; i++) {
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 2.0);
        list.add(i * 3.0);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Points");
    Query q = table2->where();
    
    std::vector<double> query_vec = {10.0, 15.0};
    auto results = q.vector_search_radius(vec_col, query_vec, 0.1);
    
    // Very small radius - might find 0 or very few results
    CHECK(results.size() <= 2);
}

TEST(HNSW_Radius_VeryLarge)
{
    // Test with very large radius (should return all)
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Points");
    auto vec_col = table->add_column_list(type_Double, "coords");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    for (int i = 0; i < 30; i++) {
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 1.0);
        list.add(i * 1.0);
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Points");
    Query q = table2->where();
    
    std::vector<double> query_vec = {15.0, 15.0};
    auto results = q.vector_search_radius(vec_col, query_vec, 1000.0);
    
    // Should return all 30 vectors
    CHECK_EQUAL(results.size(), 30);
}

// ===========================
// ACCURACY TESTS
// ===========================

TEST(HNSW_Accuracy_NearestNeighbor)
{
    // Verify that truly nearest neighbors are found
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    ObjKey nearest_key;
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Vectors");
    auto id_col = table->add_column(type_Int, "id");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Insert vectors, with one being very close to our query
    for (int i = 0; i < 100; i++) {
        auto obj = table->create_object();
        obj.set(id_col, int64_t(i));
        auto list = obj.get_list<double>(vec_col);
        
        if (i == 42) {
            // This vector is almost identical to our query
            list.add(10.001);
            list.add(20.001);
            list.add(30.001);
            nearest_key = obj.get_key();
        } else {
            list.add(i * 1.0);
            list.add(i * 2.0);
            list.add(i * 3.0);
        }
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Vectors");
    Query q = table2->where();
    
    std::vector<double> query_vec = {10.0, 20.0, 30.0};
    auto results = q.vector_search_knn(vec_col, query_vec, 5);
    
    // HNSW is approximate - verify the true nearest is in top results
    bool found_nearest = false;
    for (size_t i = 0; i < results.size(); i++) {
        if (results.get_key(i) == nearest_key) {
            found_nearest = true;
            break;
        }
    }
    CHECK(found_nearest);
}

TEST(HNSW_Accuracy_DistanceOrdering)
{
    // Verify results are ordered by distance
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Vectors");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Insert vectors at known distances from origin
    std::vector<ObjKey> keys;
    for (int i = 1; i <= 10; i++) {
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vec_col);
        // Distance from origin will be i * sqrt(2)
        list.add(i * 1.0);
        list.add(i * 1.0);
        keys.push_back(obj.get_key());
    }
    
    tr->commit();
    
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Vectors");
    Query q = table2->where();
    
    std::vector<double> query_vec = {0.0, 0.0};
    auto results = q.vector_search_knn(vec_col, query_vec, 10);
    
    CHECK_EQUAL(results.size(), 10);
    
    // Verify results are ordered by distance (should be keys[0], keys[1], ...)
    for (size_t i = 0; i < results.size(); i++) {
        CHECK_EQUAL(results.get_key(i), keys[i]);
    }
}

#endif // TEST_HNSW_COMPREHENSIVE
